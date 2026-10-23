"""Real-activation capture and synthetic controls for the studies.

Every study in ``STUDIES.md`` runs on two kinds of data:

* **Real activations.** Per-layer, per-head ``(Q, K, V)`` taken from a pretrained
  model at exactly the point where that model computes attention, so that

      softmax(Q K^T / sqrt(d) + causal mask) V

  reproduces the model's own attention output. This is asserted at capture time
  (``verify=True``) and tested in ``tests/test_harness.py``; it is the check that
  catches a wrong scale, a missing RoPE, or a transposed/mis-grouped head axis.
* **Synthetic control.** ``synthetic()`` draws q, k ~ N(0, I_d), v ~ N(0, I_dv)
  from a seeded ``torch.Generator``.

Conventions (``DESIGN.md`` §2): ``q [B,H,m,d]``, ``k [B,H,n,d]``, ``v [B,H,n,dv]``.
Here ``B`` indexes *passages* and ``m == n`` (prefill); the data is causal, so pass
``causal=True`` to every method.

Grouped-query attention
-----------------------
K and V are stored at their **true KV-head count** (``batch.n_kv_heads``), not
broadcast up to the query-head count, so the cache carries the real cache-capacity
ratio that §3.4 and the ``gqa`` method care about. Query head ``h`` reads KV head
``h // batch.kv_groups``. Call :meth:`ActivationBatch.expand_kv` to get the plain
``[B,H,n,*]`` library convention; :meth:`ActivationBatch.units` does the mapping
for you.

Scale
-----
Tensors are stored so that the library's **default** scale ``d ** -0.5`` is the
correct one. If a model applies a different logit scale (e.g. GPT-2's
``scale_attn_by_inverse_layer_idx``) the ratio is folded into ``Q`` at capture
time and recorded in ``meta['scale_fold']``. ``ActivationBatch.scale`` is always
the scale to hand to ``attnbank`` methods.

RoPE
----
Rotary embeddings are applied by the model to Q and K *before* the attention
kernel runs, and the capture point is inside that kernel, so RoPE is already baked
into the captured Qwen tensors. GPT-2 uses learned absolute position embeddings,
so nothing extra is needed there.

Sampling
--------
Long-context captures do not take every layer and head. Layers are a deterministic
even spread over depth (early/middle/late); query heads are a **seeded** sample.
Both the layer list and the head ids are written into the provenance.

Caching
-------
Captures go to ``$ATTNBANK_CACHE`` or ``lib/.cache/activations`` (gitignored)
as one ``.npz`` per layer plus a ``meta.json``. A study never re-runs a model:
``get_activations`` returns from cache when the key matches.

Offline
-------
If the model or dataset cannot be obtained, ``get_activations`` raises
``DataUnavailableError`` with an actionable message. Use
``get_activations_or_synthetic`` (or just ``synthetic``) so that no study is
blocked by the absence of a network.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "ActivationBatch", "DataUnavailableError", "MODELS", "DATASETS",
    "get_activations", "get_activations_or_synthetic", "load_activations",
    "synthetic", "cache_dir", "clear_cache", "describe_sources", "verify_capture",
    "list_captures",
    "sample_heads", "stratified_layers",
]

# --------------------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    n_layers: int
    n_tokens: int = 1024
    n_passages: int = 64
    n_layer_sample: int = 4
    n_head_sample: int | None = None      # None = all query heads
    model_dtype: str = "float32"
    store_dtype: str = "float32"
    batch_size: int = 4
    verify_tol: float = 2e-3
    dataset: str = "wikitext103"
    #: attention kernel the capture delegates to: "eager" materialises the n x n matrix
    #: (and gives a weight-level check), "sdpa" does not (needed at long context)
    base_impl: str = "eager"
    note: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: str
    name: str | None
    split: str
    text_field: str = "text"
    #: join consecutive rows into one stream (line-structured corpora) instead of
    #: treating each row as an independent document
    stream: bool = True
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "gpt2": ModelSpec(
        "gpt2", "gpt2", n_layers=12, n_tokens=1024, n_passages=64, n_layer_sample=4,
        note="GPT-2 small, 124M, learned absolute positions, MHA (12q/12kv), ctx 1024"),
    "qwen2.5-0.5b": ModelSpec(
        "qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B", n_layers=24, n_tokens=1024, n_passages=64,
        n_layer_sample=4,
        note="Qwen2.5 0.5B, RoPE, GQA 14q/2kv, head_dim 64"),
    "qwen2.5-7b-instruct": ModelSpec(
        "qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct", n_layers=28,
        n_tokens=8192, n_passages=8, n_layer_sample=4, n_head_sample=8,
        model_dtype="bfloat16", store_dtype="float16", batch_size=1, verify_tol=5e-2,
        dataset="pg19", base_impl="sdpa",
        note="Qwen2.5 7B Instruct, RoPE, GQA 28q/4kv, head_dim 128, long context"),
}
#: convenience aliases
MODELS["qwen"] = MODELS["qwen2.5-0.5b"]
MODELS["qwen7b"] = MODELS["qwen2.5-7b-instruct"]

DATASETS: dict[str, DatasetSpec] = {
    "wikitext103": DatasetSpec("wikitext103", "wikitext", "wikitext-103-raw-v1", "test",
                               stream=True, note="WikiText-103 raw, test split"),
    "pg19": DatasetSpec("pg19", "emozilla/pg19-test", None, "test", stream=False,
                        note="PG-19 books (Project Gutenberg), long-context source"),
}
DATASETS["longctx"] = DATASETS["pg19"]


class DataUnavailableError(RuntimeError):
    """Raised when a model or dataset cannot be loaded (no network, no cache)."""


def describe_sources() -> dict:
    return {
        "models": {k: {"repo": s.repo, "n_layers": s.n_layers, "n_tokens": s.n_tokens,
                       "n_passages": s.n_passages, "note": s.note}
                   for k, s in MODELS.items()},
        "datasets": {k: {"path": s.path, "name": s.name, "split": s.split, "note": s.note}
                     for k, s in DATASETS.items()},
        "cache_dir": str(cache_dir()),
    }


def cache_dir() -> Path:
    env = os.environ.get("ATTNBANK_CACHE")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[1] / ".cache" / "activations"


def clear_cache(model: str | None = None) -> int:
    """Delete cached captures (all, or only those for ``model``). Returns #dirs removed."""
    import shutil
    root = cache_dir()
    if not root.exists():
        return 0
    n = 0
    for d in sorted(root.iterdir()):
        if d.is_dir() and (model is None or d.name.startswith(_slug(model) + "__")):
            shutil.rmtree(d)
            n += 1
    return n


# --------------------------------------------------------------------------------------
# explicit, seeded sampling
# --------------------------------------------------------------------------------------


def stratified_layers(n_layers: int, k: int = 4) -> tuple[int, ...]:
    """``k`` layers evenly spread over depth, always including the first and the last.

    Deterministic by construction (no RNG): early / middle / late strata are covered
    and the choice is reproducible from ``(n_layers, k)`` alone.
    """
    k = max(1, min(int(k), int(n_layers)))
    if k == 1:
        return (n_layers // 2,)
    return tuple(sorted({round(i * (n_layers - 1) / (k - 1)) for i in range(k)}))


def sample_heads(n_heads: int, k: int | None, seed: int = 0) -> tuple[int, ...]:
    """A seeded sample of ``k`` query heads out of ``n_heads`` (all of them if ``k`` is None).

    Stratified: the head index space is cut into ``k`` contiguous bins and one head is
    drawn uniformly from each, so the sample spans the whole head range instead of
    clumping. Reproducible from ``(n_heads, k, seed)``.
    """
    if k is None or k >= n_heads:
        return tuple(range(n_heads))
    k = max(1, int(k))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    edges = [round(i * n_heads / k) for i in range(k + 1)]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(hi, lo + 1)
        out.append(lo + int(torch.randint(0, hi - lo, (1,), generator=g).item()))
    return tuple(sorted(set(out)))


# --------------------------------------------------------------------------------------
# the data record
# --------------------------------------------------------------------------------------


@dataclass
class ActivationBatch:
    """One layer's worth of ``(Q, K, V)``.

    ``q [B,Hq,m,d]``, ``k [B,Hkv,n,d]``, ``v [B,Hkv,n,dv]`` with ``B`` = passages and
    ``m == n`` = context length. For MHA models ``Hkv == Hq``; for GQA models K and V
    are kept at the true KV-head count and query head ``heads[i]`` reads KV head
    ``heads[i] // kv_groups``. Always causal.
    """

    q: Tensor
    k: Tensor
    v: Tensor
    source: str                       # "gpt2/wikitext103" or "synthetic"
    layer: int
    heads: tuple[int, ...]            # original query-head ids, aligned with q's axis 1
    passages: tuple[int, ...]
    scale: float                      # pass to attnbank methods (== d ** -0.5)
    kv_heads: tuple[int, ...] = ()    # original KV-head ids, aligned with k/v's axis 1
    kv_groups: int = 1                # query heads per KV head
    causal: bool = True
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.kv_heads:
            self.kv_heads = tuple(range(self.k.shape[1]))

    # -- shape helpers -------------------------------------------------------------
    @property
    def B(self) -> int: return self.q.shape[0]

    @property
    def H(self) -> int: return self.q.shape[1]

    @property
    def n_kv_heads(self) -> int: return self.k.shape[1]

    @property
    def m(self) -> int: return self.q.shape[2]

    @property
    def n(self) -> int: return self.k.shape[2]

    @property
    def d(self) -> int: return self.q.shape[3]

    @property
    def dv(self) -> int: return self.v.shape[3]

    @property
    def shape(self) -> dict:
        return {"B": self.B, "H": self.H, "n_kv_heads": self.n_kv_heads,
                "m": self.m, "n": self.n, "d": self.d, "dv": self.dv}

    def to(self, device=None, dtype=None) -> "ActivationBatch":
        f = lambda t: t.to(device=device, dtype=dtype)
        return replace(self, q=f(self.q), k=f(self.k), v=f(self.v))

    # -- GQA ------------------------------------------------------------------------
    def _kv_index_for(self, head_ids: Sequence[int]) -> list[int]:
        idx = []
        for h in head_ids:
            kvh = h // self.kv_groups
            if kvh not in self.kv_heads:
                raise KeyError(f"KV head {kvh} (for query head {h}) is not in this batch")
            idx.append(self.kv_heads.index(kvh))
        return idx

    def expand_kv(self) -> "ActivationBatch":
        """Broadcast K and V so that axis 1 lines up with the query heads ([B,H,n,*])."""
        if self.n_kv_heads == self.H and self.kv_groups == 1:
            return self
        sel = torch.as_tensor(self._kv_index_for(self.heads), device=self.k.device)
        return replace(self, k=self.k.index_select(1, sel), v=self.v.index_select(1, sel),
                       kv_heads=tuple(h // self.kv_groups for h in self.heads), kv_groups=1,
                       meta={**self.meta, "kv_expanded": True})

    # -- slicing ---------------------------------------------------------------------
    def select(self, heads: Sequence[int] | None = None,
               passages: Sequence[int] | None = None,
               n_tokens: int | None = None) -> "ActivationBatch":
        """Sub-select by *label* (head/passage ids as they appear in ``self.heads``)."""
        out = self
        if heads is not None:
            heads = tuple(int(h) for h in heads)
            qi = torch.as_tensor([out.heads.index(h) for h in heads], device=out.q.device)
            ki = torch.as_tensor(sorted({out._kv_index_for(heads)[i]
                                         for i in range(len(heads))}), device=out.k.device)
            out = replace(out, q=out.q.index_select(1, qi),
                          k=out.k.index_select(1, ki), v=out.v.index_select(1, ki),
                          heads=heads,
                          kv_heads=tuple(out.kv_heads[i] for i in ki.tolist()))
        if passages is not None:
            passages = tuple(int(p) for p in passages)
            sel = torch.as_tensor([out.passages.index(p) for p in passages], device=out.q.device)
            out = replace(out, q=out.q.index_select(0, sel), k=out.k.index_select(0, sel),
                          v=out.v.index_select(0, sel), passages=passages)
        if n_tokens is not None and n_tokens < out.n:
            # keep the FIRST n tokens: a prefix of a causal context is itself a valid context
            out = replace(out, q=out.q[..., :n_tokens, :], k=out.k[..., :n_tokens, :],
                          v=out.v[..., :n_tokens, :])
        return out

    def units(self, max_units: int | None = None, heads: Sequence[int] | None = None,
              passages: Sequence[int] | None = None
              ) -> Iterator[tuple[dict, "ActivationBatch"]]:
        """Iterate over (layer, head, passage) triples as ``[1,1,n,*]`` batches.

        Yields ``(tag, unit)``; ``tag`` identifies the triple for result records and
        already carries the KV head, so GQA sharing is visible in the results.
        """
        head_ids = tuple(heads) if heads is not None else self.heads
        pass_ids = tuple(passages) if passages is not None else self.passages
        count = 0
        for p in pass_ids:
            bi = self.passages.index(p)
            for h in head_ids:
                if max_units is not None and count >= max_units:
                    return
                hi = self.heads.index(h)
                ki = self._kv_index_for([h])[0]
                tag = {"source": self.source, "layer": self.layer, "head": h,
                       "kv_head": h // self.kv_groups, "passage": p}
                yield tag, replace(
                    self,
                    q=self.q[bi:bi + 1, hi:hi + 1], k=self.k[bi:bi + 1, ki:ki + 1],
                    v=self.v[bi:bi + 1, ki:ki + 1], heads=(h,),
                    kv_heads=(self.kv_heads[ki],), kv_groups=1, passages=(p,))
                count += 1

    def provenance(self) -> dict:
        """Everything a results file needs to identify this data."""
        return {"source": self.source, "layer": self.layer, "heads": list(self.heads),
                "kv_heads": list(self.kv_heads), "kv_groups": self.kv_groups,
                "passages": list(self.passages), "scale": self.scale, "causal": self.causal,
                **self.shape, **{k: v for k, v in self.meta.items() if _jsonable(v)}}

    def __repr__(self) -> str:
        return (f"ActivationBatch(source={self.source!r}, layer={self.layer}, "
                f"q{list(self.q.shape)} k{list(self.k.shape)} v{list(self.v.shape)}, "
                f"kv_groups={self.kv_groups}, scale={self.scale:.6g}, dtype={self.q.dtype})")


def _jsonable(x) -> bool:
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


# --------------------------------------------------------------------------------------
# synthetic control
# --------------------------------------------------------------------------------------


def synthetic(n: int = 1024, d: int = 64, dv: int | None = None, heads: int = 8,
              passages: int = 4, m: int | None = None, seed: int = 0,
              device: str | torch.device = "cpu", dtype: torch.dtype = torch.float64,
              q_scale: float = 1.0, layer: int = 0, kv_groups: int = 1) -> ActivationBatch:
    """Gaussian control: q, k ~ N(0, I_d), v ~ N(0, I_dv). Seeded, reproducible.

    ``q_scale`` multiplies q, which scales the logits — the knob S3 sweeps to grow
    ``||q|| ||k||``. ``kv_groups > 1`` produces a GQA-shaped control.
    """
    dv = d if dv is None else dv
    m = n if m is None else m
    if heads % kv_groups:
        raise ValueError(f"heads={heads} is not a multiple of kv_groups={kv_groups}")
    n_kv = heads // kv_groups
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    rn = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    q = rn(passages, heads, m, d) * q_scale
    k = rn(passages, n_kv, n, d)
    v = rn(passages, n_kv, n, dv)
    return ActivationBatch(
        q=q, k=k, v=v, source="synthetic", layer=layer,
        heads=tuple(range(heads)), kv_heads=tuple(range(n_kv)), kv_groups=kv_groups,
        passages=tuple(range(passages)), scale=d ** -0.5, causal=True,
        meta={"kind": "synthetic", "seed": int(seed), "q_scale": float(q_scale),
              "distribution": "q,k ~ N(0,I_d); v ~ N(0,I_dv)"})


# --------------------------------------------------------------------------------------
# text passages
# --------------------------------------------------------------------------------------


def _load_tokenizer(repo: str):
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover - import guard
        raise DataUnavailableError(
            "transformers is not installed. `pip install transformers` or use "
            "attnbank.data.synthetic(...) instead.") from e
    try:
        return AutoTokenizer.from_pretrained(repo)
    except Exception as e:
        raise DataUnavailableError(
            f"could not load the tokenizer for {repo!r} ({type(e).__name__}: {e}). "
            f"If this machine is offline, pre-download it on a connected machine with "
            f"`huggingface-cli download {repo}`, or run the study on "
            f"attnbank.data.synthetic(...).") from e


def _load_text_rows(spec: DatasetSpec) -> list[str]:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover - import guard
        raise DataUnavailableError(
            "the `datasets` package is not installed. `pip install datasets` or use "
            "attnbank.data.synthetic(...) instead.") from e
    try:
        ds = load_dataset(spec.path, spec.name, split=spec.split)
    except Exception as e:
        raise DataUnavailableError(
            f"could not load dataset {spec.path!r} (config={spec.name!r}, split={spec.split!r}) "
            f"({type(e).__name__}: {e}). If this machine is offline, pre-download it with "
            f"`python -c \"import datasets; datasets.load_dataset({spec.path!r}, {spec.name!r})\"` "
            f"on a connected machine, or run the study on attnbank.data.synthetic(...).") from e
    return list(ds[spec.text_field])


def _build_passages(spec: DatasetSpec, tokenizer, n_tokens: int, n_passages: int,
                    seed: int = 0) -> tuple[list[list[int]], list[dict]]:
    """Return ``n_passages`` token sequences of exactly ``n_tokens`` ids, plus provenance."""
    rows = _load_text_rows(spec)
    passages: list[list[int]] = []
    prov: list[dict] = []
    if spec.stream:
        # line-structured corpus: concatenate rows into one stream, cut into blocks
        buf: list[int] = []
        row_start = 0
        for i, text in enumerate(rows):
            if not text.strip():
                continue
            buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
            while len(buf) >= n_tokens:
                passages.append(buf[:n_tokens])
                prov.append({"dataset": spec.key, "block": len(passages) - 1,
                             "rows": [row_start, i]})
                buf = buf[n_tokens:]
                row_start = i
                if len(passages) >= n_passages:
                    return passages, prov
    else:
        # document corpus: one block per document (documents are long)
        for i, text in enumerate(rows):
            ids = tokenizer(text[: 60 * n_tokens], add_special_tokens=False)["input_ids"]
            if len(ids) < n_tokens:
                continue
            passages.append(ids[:n_tokens])
            prov.append({"dataset": spec.key, "document": i})
            if len(passages) >= n_passages:
                return passages, prov
    if len(passages) < n_passages:
        raise DataUnavailableError(
            f"{spec.key}: only {len(passages)} passages of {n_tokens} tokens available "
            f"(asked for {n_passages}). Use a smaller n_tokens/n_passages or another dataset.")
    return passages, prov


# --------------------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------------------

CAPTURE_IMPL = "attnbank_capture"
_ACTIVE: "_Capture | None" = None
_BASE_IMPL = "eager"      # which kernel the capture delegates to during the active run


def _causal_mask(m: int, n: int, device=None) -> Tensor:
    """[m,n] bool, True where key j is visible to query i, last query aligned with last key."""
    i = torch.arange(m, device=device)[:, None] + (n - m)
    j = torch.arange(n, device=device)[None, :]
    return j <= i


def _base_impl(module):
    """The attention function the model would have used if we were not capturing.

    ``_BASE_IMPL == "eager"`` prefers the model file's own ``eager_attention_forward``,
    which also hands back the attention matrix so the capture can be checked at the
    weight level. At long context that matrix is too large, and ``"sdpa"`` is used
    instead (output-level check only).
    """
    if _BASE_IMPL == "eager":
        mod = sys.modules.get(type(module).__module__)
        fn = getattr(mod, "eager_attention_forward", None)
        if fn is not None:
            return fn
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    return ALL_ATTENTION_FUNCTIONS["sdpa"]


def _module_scale(module, d: int, dv: int) -> float:
    """The logit scale the model actually applies, as a float."""
    s = getattr(module, "scaling", None)
    if s is not None:
        return float(s)
    scale = 1.0                                              # GPT-2 style
    if getattr(module, "scale_attn_weights", True):
        scale = float(dv) ** -0.5
    if getattr(module, "scale_attn_by_inverse_layer_idx", False):
        scale = scale / float(getattr(module, "layer_idx", 0) + 1)
    return float(scale)


class _Capture:
    """Collects Q, K, V (and the model's own attention output) during a forward pass."""

    #: how many (head, query) elements the in-flight verification is allowed to touch
    VERIFY_HEADS = 2
    VERIFY_QUERIES = 256

    def __init__(self, layers: Sequence[int], store_dtype: torch.dtype, keep_ref: bool,
                 head_ids: Sequence[int] | None = None, verify: bool = True,
                 tol: float = 2e-3):
        self.layers = {int(l) for l in layers}
        self.store_dtype = store_dtype
        self.keep_ref = keep_ref
        self.head_ids = None if head_ids is None else tuple(int(h) for h in head_ids)
        self.verify = verify
        self.tol = tol
        self.buf: dict[int, dict[str, list[Tensor]]] = {
            l: {"q": [], "k": [], "v": [], "y": []} for l in self.layers}
        self.info: dict[int, dict] = {}
        self.report: dict[int, dict] = {}

    # ------------------------------------------------------------------
    def record(self, module, query, key, value, out):
        li = int(getattr(module, "layer_idx", -1))
        if li not in self.layers or getattr(module, "is_cross_attention", False):
            return
        sw = getattr(module, "sliding_window", None)
        if sw is not None:
            raise RuntimeError(
                f"layer {li} uses sliding-window attention (window={sw}); the captured "
                f"(Q,K,V) would not reproduce the model under a plain causal mask. Pick a "
                f"model without sliding windows, or exclude this layer.")

        n_q, n_kv = query.shape[1], key.shape[1]
        groups = n_q // n_kv
        d, dv = query.shape[-1], value.shape[-1]
        model_scale = _module_scale(module, d, dv)
        lib_scale = d ** -0.5
        fold = model_scale / lib_scale

        head_ids = self.head_ids if self.head_ids is not None else tuple(range(n_q))
        hsel = torch.as_tensor(head_ids, device=query.device)
        q_sel = query.index_select(1, hsel)
        if abs(fold - 1.0) > 1e-12:
            q_sel = q_sel * fold

        y_ref = out[0] if isinstance(out, tuple) else out
        y_ref = y_ref.transpose(1, 2)                       # [B, H, n, dv]
        w_ref = out[1] if isinstance(out, tuple) and len(out) > 1 else None

        self.info.setdefault(li, {
            "model_scale": model_scale, "lib_scale": lib_scale, "scale_fold": fold,
            "kv_groups": int(groups), "n_kv_heads": int(n_kv), "n_q_heads": int(n_q),
            "d": int(d), "dv": int(dv), "heads": list(head_ids),
            "kv_heads": list(range(n_kv)),
        })
        self._verify(li, q_sel, key, value, y_ref, w_ref, lib_scale, head_ids, groups)

        b = self.buf[li]
        b["q"].append(q_sel.detach().to("cpu", self.store_dtype))
        b["k"].append(key.detach().to("cpu", self.store_dtype))
        b["v"].append(value.detach().to("cpu", self.store_dtype))
        if self.keep_ref:
            b["y"].append(y_ref.index_select(1, hsel).detach().to("cpu", self.store_dtype))

    # ------------------------------------------------------------------
    def _verify(self, li, q_sel, key, value, y_ref, w_ref, scale, head_ids, groups):
        """Recompute attention from what we are about to store; compare with the model."""
        if not self.verify or li in self.report:
            return
        with torch.no_grad():
            nh = min(self.VERIFY_HEADS, len(head_ids))
            hs = list(head_ids[:nh])
            qi = torch.as_tensor(list(range(nh)), device=q_sel.device)
            ki = torch.as_tensor([h // groups for h in hs], device=key.device)
            n = key.shape[2]
            vq = min(self.VERIFY_QUERIES, q_sel.shape[2])
            q64 = q_sel.index_select(1, qi)[:, :, -vq:].double()
            k64 = key.index_select(1, ki).double()
            v64 = value.index_select(1, ki).double()
            s = (q64 @ k64.transpose(-1, -2)) * scale
            mask = _causal_mask(vq, n, device=s.device)
            pi = torch.softmax(s.masked_fill(~mask, float("-inf")), dim=-1)
            y = pi @ v64
            head_idx = torch.as_tensor(hs, device=y_ref.device)
            yr = y_ref.index_select(1, head_idx)[:, :, -vq:].double()
            rel_y = float((y - yr).norm() / yr.norm().clamp_min(1e-30))
            rel_w = float("nan")
            if w_ref is not None:
                wr = w_ref.index_select(1, torch.as_tensor(hs, device=w_ref.device))[:, :, -vq:, :n]
                rel_w = float((pi - wr.double()).abs().max())
            del s, pi, y, q64, k64, v64
        self.report[li] = {"rel_output_error": rel_y, "max_weight_error": rel_w,
                           "heads_checked": hs, "queries_checked": vq}
        if not (rel_y < self.tol):
            raise RuntimeError(
                f"captured activations do not reproduce the model at layer {li}: relative "
                f"output error {rel_y:.3e} > {self.tol:.1e}. The logit scale, the positional "
                f"embedding (RoPE), the head axis or the GQA group mapping is wrong. "
                f"Refusing to write a bad cache.")


def _capture_attention_forward(module, query, key, value, attention_mask, **kwargs):
    base = _base_impl(module)
    out = base(module, query, key, value, attention_mask, **kwargs)
    if _ACTIVE is not None:
        _ACTIVE.record(module, query, key, value, out)
    return out


def _register_impl() -> None:
    """Register the capture kernel, and the matching *mask* builder.

    The second half matters: transformers picks the attention-mask builder by the
    implementation name, and an unregistered name yields no causal mask at all — the
    model would then attend bidirectionally and the capture would be garbage. We alias
    our name to the eager mask builder so the mask is exactly the one eager would get.
    """
    from transformers.modeling_utils import AttentionInterface
    AttentionInterface.register(CAPTURE_IMPL, _capture_attention_forward)
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
    AttentionMaskInterface.register(CAPTURE_IMPL, ALL_MASK_ATTENTION_FUNCTIONS["eager"])


def _load_model(repo: str, device, dtype):
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:  # pragma: no cover
        raise DataUnavailableError(
            "transformers is not installed. `pip install transformers` or use "
            "attnbank.data.synthetic(...) instead.") from e
    _register_impl()
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype,
                                                     attn_implementation=CAPTURE_IMPL)
    except Exception:
        try:
            model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype)
        except Exception as e2:
            raise DataUnavailableError(
                f"could not load model {repo!r} ({type(e2).__name__}: {e2}). If this machine is "
                f"offline, pre-download it with `huggingface-cli download {repo}`, or run the "
                f"study on attnbank.data.synthetic(...).") from e2
    _force_impl(model)
    return model.to(device).eval()


def _force_impl(model) -> None:
    """Make every attention module route through the capture interface."""
    seen: set[int] = set()
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        for c in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "decoder", None)):
            if c is not None and id(c) not in seen:
                seen.add(id(c))
                try:
                    c._attn_implementation = CAPTURE_IMPL
                except Exception:
                    object.__setattr__(c, "_attn_implementation", CAPTURE_IMPL)


def _n_layers(model) -> int:
    cfg = model.config
    for a in ("num_hidden_layers", "n_layer", "num_layers"):
        if getattr(cfg, a, None):
            return int(getattr(cfg, a))
    raise RuntimeError("cannot determine the number of layers from the model config")


# --------------------------------------------------------------------------------------
# cache plumbing
# --------------------------------------------------------------------------------------


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in s).lower()


def _capture_key(model: str, dataset: str, n_tokens: int, n_passages: int, heads,
                 model_dtype: str, store_dtype: str, seed: int) -> tuple[str, dict]:
    spec = {"model": MODELS[model].repo, "dataset": dataset, "n_tokens": n_tokens,
            "n_passages": n_passages, "heads": list(heads) if heads is not None else None,
            "model_dtype": model_dtype, "store_dtype": store_dtype, "seed": seed,
            "format": 3}
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    name = f"{_slug(MODELS[model].key)}__{_slug(dataset)}__n{n_tokens}_p{n_passages}__{h}"
    return name, spec


def _layer_file(d: Path, layer: int) -> Path:
    return d / f"layer{layer:03d}.npz"


_DTYPES = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64,
           "bfloat16": torch.bfloat16}


def _as_dtype(x) -> torch.dtype:
    return x if isinstance(x, torch.dtype) else _DTYPES[str(x)]


def _dtype_name(x: torch.dtype) -> str:
    return str(x).replace("torch.", "")


# --------------------------------------------------------------------------------------
# public capture API
# --------------------------------------------------------------------------------------


def get_activations(model: str = "gpt2",
                    layers: Sequence[int] | int | None = None,
                    heads: Sequence[int] | int | None = None,
                    n_tokens: int | None = None,
                    n_passages: int | None = None,
                    dataset: str | None = None,
                    device: str | torch.device = "cpu",
                    dtype: torch.dtype = torch.float64,
                    *,
                    capture_device: str | torch.device | None = None,
                    model_dtype: torch.dtype | str | None = None,
                    store_dtype: torch.dtype | str | None = None,
                    batch_size: int | None = None,
                    seed: int = 0,
                    cache: bool = True,
                    verify: bool = True,
                    keep_ref_output: bool = False,
                    expand_kv: bool = False,
                    ) -> Iterator[ActivationBatch]:
    """Yield one :class:`ActivationBatch` per requested layer.

    Parameters
    ----------
    model : ``"gpt2"`` | ``"qwen2.5-0.5b"`` | ``"qwen2.5-7b-instruct"`` (alias ``"qwen7b"``)
    layers : layer indices, an int (that many evenly spread layers), or None (model default)
    heads : query-head indices, an int (that many *seeded* sampled heads), or None
        (the model default: all heads for the small models, 8 for the 7B)
    n_tokens, n_passages : context length and number of passages (the ``n`` and ``B`` axes);
        None uses the model's default (1024x64 for GPT-2 and Qwen-0.5B, 8192x8 for the 7B)
    dataset : ``"wikitext103"`` | ``"pg19"``; None uses the model's default
    device, dtype : where/what the *returned* tensors are. float64 by default — every
        error measurement in STUDIES.md is float64.
    capture_device : where the model runs (default: cuda if available)
    model_dtype : the model's compute dtype during capture (float32 keeps the
        verification tight; lowering it loosens what ``verify`` can prove)
    store_dtype : on-disk dtype of the cache
    verify : assert at capture time that ``softmax(QK^T/sqrt(d)+mask)V`` reproduces the
        model's own attention output
    expand_kv : broadcast K/V up to the query-head count before yielding (default False:
        the true KV-head count is preserved, see :meth:`ActivationBatch.expand_kv`)

    Raises
    ------
    DataUnavailableError : the model or dataset cannot be obtained; use ``synthetic``.
    """
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; available: {sorted(MODELS)}")
    mspec = MODELS[model]
    dataset = dataset or mspec.dataset
    if dataset not in DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; available: {sorted(DATASETS)}")
    dspec = DATASETS[dataset]
    n_tokens = mspec.n_tokens if n_tokens is None else int(n_tokens)
    n_passages = mspec.n_passages if n_passages is None else int(n_passages)
    batch_size = mspec.batch_size if batch_size is None else int(batch_size)
    model_dtype = _as_dtype(mspec.model_dtype if model_dtype is None else model_dtype)
    store_dtype = _as_dtype(mspec.store_dtype if store_dtype is None else store_dtype)

    head_ids = _resolve_heads(heads, mspec, seed)
    want = _resolve_layers(layers, mspec)

    key, spec = _capture_key(model, dataset, n_tokens, n_passages, head_ids,
                             _dtype_name(model_dtype), _dtype_name(store_dtype), seed)
    cdir = cache_dir() / key
    missing = [l for l in want if not _layer_file(cdir, l).exists()] if cache else list(want)

    sink: dict = {}
    if missing:
        _run_capture(mspec, dspec, layers=missing, head_ids=head_ids, n_tokens=n_tokens,
                     n_passages=n_passages, seed=seed, capture_device=capture_device,
                     model_dtype=model_dtype, store_dtype=store_dtype, batch_size=batch_size,
                     verify=verify, keep_ref_output=keep_ref_output,
                     out_dir=cdir if cache else None, spec=spec, sink=sink)
    for layer in want:
        if cache:
            b = load_activations(cdir, layer, device=device, dtype=dtype,
                                 with_ref=keep_ref_output)
        else:
            b = _batch_from_capture(sink[layer], device=device, dtype=dtype)
        yield b.expand_kv() if expand_kv else b


def _resolve_layers(layers, mspec: ModelSpec) -> tuple[int, ...]:
    if layers is None:
        return stratified_layers(mspec.n_layers, mspec.n_layer_sample)
    if isinstance(layers, int):
        return stratified_layers(mspec.n_layers, layers)
    return tuple(int(l) for l in layers)


def _n_q_heads(mspec: ModelSpec) -> int:
    return {"gpt2": 12, "qwen2.5-0.5b": 14, "qwen2.5-7b-instruct": 28}[mspec.key]


def _resolve_heads(heads, mspec: ModelSpec, seed: int) -> tuple[int, ...] | None:
    nh = _n_q_heads(mspec)
    if heads is None:
        return sample_heads(nh, mspec.n_head_sample, seed=seed)
    if isinstance(heads, int):
        return sample_heads(nh, heads, seed=seed)
    return tuple(int(h) for h in heads)


def _run_capture(mspec: ModelSpec, dspec: DatasetSpec, *, layers, head_ids, n_tokens,
                 n_passages, seed, capture_device, model_dtype, store_dtype, batch_size,
                 verify, keep_ref_output, out_dir: Path | None, spec: dict,
                 sink: dict | None = None) -> dict:
    """Run the model once and write (or return) the per-layer captures."""
    global _ACTIVE, _BASE_IMPL
    if capture_device is None:
        capture_device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = _load_tokenizer(mspec.repo)
    passages, prov = _build_passages(dspec, tok, n_tokens, n_passages, seed=seed)
    model = _load_model(mspec.repo, capture_device, model_dtype)
    n_layers = _n_layers(model)
    layers = [l for l in layers if 0 <= l < n_layers]

    cap = _Capture(layers, store_dtype, keep_ref=keep_ref_output, head_ids=head_ids,
                   verify=verify, tol=mspec.verify_tol)
    ids = torch.tensor(passages, dtype=torch.long)
    t0 = time.time()
    _BASE_IMPL = mspec.base_impl
    _ACTIVE = cap
    try:
        with torch.no_grad():
            for i in range(0, ids.shape[0], batch_size):
                chunk = ids[i:i + batch_size].to(capture_device)
                model(input_ids=chunk, use_cache=False)
                print(f"  capture {mspec.key}/{dspec.key} n={n_tokens}: "
                      f"{min(i + batch_size, ids.shape[0])}/{ids.shape[0]} passages",
                      file=sys.stderr, flush=True)
    finally:
        _ACTIVE = None
        _BASE_IMPL = "eager"
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    meta = {
        **spec,
        "model_key": mspec.key, "model_repo": mspec.repo, "dataset_key": dspec.key,
        "dataset_path": dspec.path, "dataset_name": dspec.name, "dataset_split": dspec.split,
        "n_layers": n_layers, "layers": list(layers), "n_tokens": n_tokens,
        "n_passages": n_passages, "head_sample_seed": seed,
        "heads": list(head_ids) if head_ids is not None else None,
        "capture_device": str(capture_device), "base_impl": mspec.base_impl,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "capture_seconds": round(time.time() - t0, 2),
        "verification": {str(k): v for k, v in cap.report.items()},
        "passage_provenance": prov,
        "rope_applied": True,
        "note": ("Q,K,V captured inside the attention kernel: RoPE already applied; K,V kept "
                 "at the true KV-head count (query head h reads KV head h // kv_groups); the "
                 "model's logit scale folded into Q so that scale == d**-0.5."),
    }
    out: dict[int, dict] = {}
    for li in layers:
        b = cap.buf[li]
        if not b["q"]:
            continue
        arrs = {nm: torch.cat(b[nm], dim=0) for nm in ("q", "k", "v")}
        if keep_ref_output and b["y"]:
            arrs["y_ref"] = torch.cat(b["y"], dim=0)
        info = {**meta, **cap.info[li], "layer": li,
                "verification": cap.report.get(li, {})}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            # np.savez appends ".npz" unless the name already ends in it, so the
            # temporary name has to keep the extension last.
            tmp = out_dir / f"layer{li:03d}.partial.npz"
            np.savez(tmp, **{k: v.numpy() for k, v in arrs.items()},
                     info=np.frombuffer(json.dumps(info).encode(), dtype=np.uint8))
            tmp.replace(_layer_file(out_dir, li))
        out[li] = {"arrays": arrs, "info": info}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if sink is not None:
        sink.update(out)
    return out


_META_KEYS = ("model_repo", "dataset_path", "dataset_name", "dataset_split", "n_tokens",
              "n_passages", "model_scale", "scale_fold", "kv_groups", "n_kv_heads",
              "n_q_heads", "captured_at", "store_dtype", "model_dtype", "rope_applied",
              "head_sample_seed", "verification")


def _batch_from_capture(rec: dict, device, dtype, with_ref: bool = False) -> ActivationBatch:
    info, a = rec["info"], rec["arrays"]
    f = lambda t: t.to(device=device, dtype=dtype)
    meta = {k: info[k] for k in _META_KEYS if k in info}
    if with_ref and "y_ref" in a:
        meta["y_ref"] = f(a["y_ref"])
    return ActivationBatch(
        q=f(a["q"]), k=f(a["k"]), v=f(a["v"]),
        source=f"{info['model_key']}/{info['dataset_key']}", layer=int(info["layer"]),
        heads=tuple(info["heads"]), kv_heads=tuple(info["kv_heads"]),
        kv_groups=int(info["kv_groups"]), passages=tuple(range(a["q"].shape[0])),
        scale=float(info["lib_scale"]), causal=True, meta=meta)


def load_activations(capture_dir: str | Path, layer: int, heads: Sequence[int] | None = None,
                     passages: Sequence[int] | None = None,
                     device: str | torch.device = "cpu",
                     dtype: torch.dtype = torch.float64,
                     with_ref: bool = False) -> ActivationBatch:
    """Load one cached layer into an :class:`ActivationBatch`."""
    capture_dir = Path(capture_dir)
    f = _layer_file(capture_dir, layer)
    if not f.exists():
        raise FileNotFoundError(f"no cached capture at {f}")
    with np.load(f) as z:
        info = json.loads(bytes(z["info"]).decode())
        arrays = {k: torch.from_numpy(z[k]) for k in ("q", "k", "v")}
        if with_ref and "y_ref" in z:
            arrays["y_ref"] = torch.from_numpy(z["y_ref"])
    batch = _batch_from_capture({"info": info, "arrays": arrays}, device, dtype, with_ref)
    batch.meta["capture_dir"] = str(capture_dir)
    if heads is not None or passages is not None:
        batch = batch.select(heads=heads, passages=passages)
    return batch


def list_captures() -> list[dict]:
    """Every capture already on disk: one row per cached ``(model, dataset, n, P)``.

    Lets a study pick a slice without re-deriving cache keys, and lets the report of a
    study say exactly which captures it drew on.
    """
    root = cache_dir()
    out = []
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        f = d / "meta.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text())
        out.append({
            "dir": str(d), "model": m.get("model_key"), "dataset": m.get("dataset_key"),
            "n_tokens": m.get("n_tokens"), "n_passages": m.get("n_passages"),
            "layers": [int(p.stem[5:]) for p in sorted(d.glob("layer*.npz"))],
            "heads": m.get("heads"), "store_dtype": m.get("store_dtype"),
            "model_dtype": m.get("model_dtype"), "captured_at": m.get("captured_at"),
            "verification": m.get("verification"),
        })
    return out


def get_activations_or_synthetic(model: str = "gpt2", *, fallback: dict | None = None,
                                 **kw) -> Iterator[ActivationBatch]:
    """``get_activations``, degrading to a matched synthetic control if data is unavailable.

    Lets every study run offline. Warns loudly and sets ``batch.meta['fallback']``.
    """
    try:
        yield from get_activations(model=model, **kw)
    except DataUnavailableError as e:
        warnings.warn(f"real activations unavailable ({e}); falling back to the synthetic "
                      f"control", RuntimeWarning, stacklevel=2)
        mspec = MODELS.get(model)
        fb = dict(n=kw.get("n_tokens") or (mspec.n_tokens if mspec else 1024), d=64, heads=8,
                  passages=min(8, kw.get("n_passages") or 8), seed=kw.get("seed", 0))
        fb.update(fallback or {})
        b = synthetic(**fb)
        b.meta["fallback"] = f"{model}: {e}"
        yield b


def verify_capture(model: str = "gpt2", n_tokens: int = 128, n_passages: int = 2,
                   layers: Sequence[int] | int | None = 3,
                   heads: Sequence[int] | int | None = None,
                   dataset: str | None = None,
                   device: str | torch.device | None = None,
                   model_dtype: torch.dtype | str | None = None,
                   tol: float | None = None) -> dict:
    """Capture a tiny slice *without touching the cache* and check it against the model.

    Recomputes ``softmax(QK^T/sqrt(d) + causal)V`` in float64 from exactly the tensors
    that would be cached, and compares with the attention output the model itself
    produced. Returns ``{layer: {...}}``; raises if they disagree.
    """
    mspec = MODELS[model]
    dspec = DATASETS[dataset or mspec.dataset]
    model_dtype = _as_dtype(model_dtype or mspec.model_dtype)
    tol = mspec.verify_tol if tol is None else tol
    head_ids = _resolve_heads(heads, mspec, 0)
    want = _resolve_layers(layers, mspec)
    sink: dict = {}
    _, spec = _capture_key(model, dspec.key, n_tokens, n_passages, head_ids,
                           _dtype_name(model_dtype), "float32", 0)
    _run_capture(mspec, dspec, layers=want, head_ids=head_ids, n_tokens=n_tokens,
                 n_passages=n_passages, seed=0, capture_device=device,
                 model_dtype=model_dtype, store_dtype=torch.float32, batch_size=n_passages,
                 verify=True, keep_ref_output=True, out_dir=None, spec=spec, sink=sink)

    report = {}
    for li, rec in sink.items():
        b = _batch_from_capture(rec, device="cpu", dtype=torch.float64, with_ref=True)
        e = b.expand_kv()
        s = (e.q @ e.k.transpose(-1, -2)) * e.scale
        pi = torch.softmax(s.masked_fill(~_causal_mask(e.m, e.n), float("-inf")), dim=-1)
        y = pi @ e.v
        yr = b.meta["y_ref"]
        rel = float((y - yr).norm() / yr.norm().clamp_min(1e-30))
        report[li] = {"rel_output_error": rel, "max_abs_output_error": float((y - yr).abs().max()),
                      "scale": e.scale, "kv_groups": b.kv_groups, "n_kv_heads": b.n_kv_heads,
                      "heads": list(b.heads), "shape": list(e.q.shape), "tol": tol}
        if not (rel < tol):
            raise AssertionError(
                f"{model} layer {li}: recomputed attention does not match the model "
                f"(relative error {rel:.3e} > {tol:.1e})")
    return report
