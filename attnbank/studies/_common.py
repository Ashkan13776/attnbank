"""Shared plumbing for the eight studies: data slices, references, figure paths.

Not a study. Everything here is the part that would otherwise be copied eight times:

* :class:`SliceSet` — a dict of ``(source, layer, passage) -> ActivationBatch``, each
  slice a ``[1, H, n, *]`` tensor already on the compute device in float64, plus the
  provenance block that goes into the results file.
* :func:`load_real` / :func:`load_synthetic` — the two data sources every study uses:
  real activations as the primary evidence, the Gaussian control alongside it.
* :func:`reference` — the exact softmax measure, logits and output for a slice.
* :func:`fit_loglog_slope` — slope with a 95% CI, used wherever the paper quotes an
  exponent.
* :data:`FIGURES` — the repo-level ``figures/`` directory of DESIGN.md Sec. 5.

Every study runs its sweep on the real slices *and* on the synthetic control, writes
both into one results file under a ``data`` axis, and plots the real one with the
synthetic as a secondary panel or series.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor

from .. import data as abdata
from ..core import causal_mask
from . import _harness as H

#: figures live at the repository root (DESIGN.md Sec. 5), not inside the package
FIGURES = Path(__file__).resolve().parents[3] / "figures"

#: the two models whose activations carry the primary evidence
PRIMARY_SOURCES = (("gpt2", "wikitext103"), ("qwen2.5-0.5b", "wikitext103"))
#: the long-context source
LONG_SOURCES = (("qwen2.5-0.5b", "pg19"), ("qwen2.5-7b-instruct", "pg19"))


# ======================================================================================
# slices
# ======================================================================================


@dataclass
class SliceSet:
    """A keyed collection of ``[1, H, n, *]`` activation slices plus its provenance."""

    slices: dict[tuple, abdata.ActivationBatch] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.slices)

    def key(self, job) -> tuple:
        return (job["source"], int(job["layer"]), int(job["passage"]))

    def get(self, job) -> abdata.ActivationBatch:
        return self.slices[self.key(job)]

    def jobs(self) -> list[dict]:
        """One job stub per slice: ``{source, layer, passage}``."""
        return [{"source": s, "layer": l, "passage": p} for (s, l, p) in self.slices]

    def n_units(self) -> int:
        return sum(b.H for b in self.slices.values())

    def update(self, other: "SliceSet") -> "SliceSet":
        self.slices.update(other.slices)
        for k, v in other.provenance.items():
            cur = self.provenance.get(k)
            if isinstance(cur, dict) and isinstance(v, dict):
                # Merge one level deeper. `provenance["real"]` is keyed by source, so a
                # plain setdefault would drop every model after the first and leave the
                # results file claiming a narrower sweep than was actually run.
                for sub_k, sub_v in v.items():
                    cur.setdefault(sub_k, sub_v)
            else:
                self.provenance.setdefault(k, v)
        return self

    def describe(self) -> dict:
        any_b = next(iter(self.slices.values())) if self.slices else None
        return {"n_slices": len(self.slices), "n_units": self.n_units(),
                "sources": sorted({k[0] for k in self.slices}),
                "shape": any_b.shape if any_b is not None else None,
                **self.provenance}


def _device(device=None) -> torch.device:
    return torch.device(device) if device is not None else H.default_device()


def load_real(model: str = "gpt2", dataset: str | None = None, *, n_layers: int = 4,
              n_passages: int = 8, n_tokens: int | None = None, heads: Sequence[int] | None = None,
              m_queries: int | None = None, device=None, dtype: torch.dtype = torch.float64,
              passage_stride: int | None = None) -> SliceSet:
    """Real activations as one slice per (layer, passage), ``[1, H, n, *]``.

    ``n_passages`` passages are taken evenly spread through the capture (so the sample
    is not all from one region of the corpus); ``n_tokens`` trims the context and
    ``m_queries`` keeps only the last ``m`` queries, which preserves causal alignment
    and is how the expensive sweeps stay affordable.
    """
    dev = _device(device)
    out = SliceSet()
    got_any = False
    for batch in abdata.get_activations(model=model, dataset=dataset, layers=n_layers,
                                        heads=heads, device="cpu", dtype=torch.float32):
        got_any = True
        stride = passage_stride or max(1, batch.B // max(1, n_passages))
        picks = list(batch.passages)[::stride][:n_passages]
        sub = batch.select(passages=picks).expand_kv()
        if n_tokens is not None:
            sub = sub.select(n_tokens=n_tokens)
        sub = sub.to(device=dev, dtype=dtype)
        for p in picks:
            one = sub.select(passages=[p])
            if m_queries is not None and m_queries < one.m:
                one = _tail(one, m_queries)
            out.slices[(sub.source, batch.layer, p)] = one
        out.provenance.setdefault("real", {}).setdefault(sub.source, {
            "model": model, "dataset": sub.meta.get("dataset_path"),
            "model_repo": sub.meta.get("model_repo"),
            "n_tokens_captured": sub.meta.get("n_tokens"),
            "n_passages_captured": sub.meta.get("n_passages"),
            "kv_groups": sub.meta.get("kv_groups"), "scale": sub.scale,
            "captured_at": sub.meta.get("captured_at"),
            "verification": sub.meta.get("verification"),
            "layers": [], "passages": list(picks),
            "n": one.n, "m": one.m, "H": one.H, "d": one.d, "dv": one.dv,
        })
        out.provenance["real"][sub.source]["layers"].append(batch.layer)
    if not got_any:
        raise abdata.DataUnavailableError(f"no activations for {model}/{dataset}")
    return out


def load_synthetic(n: int = 1024, d: int = 64, dv: int | None = None, heads: int = 12,
                   passages: int = 4, seed: int = 0, m_queries: int | None = None,
                   device=None, dtype: torch.dtype = torch.float64,
                   q_scale: float = 1.0) -> SliceSet:
    """The Gaussian control, shaped like the real slices so the same sweep runs on it."""
    dev = _device(device)
    out = SliceSet()
    full = abdata.synthetic(n=n, d=d, dv=dv, heads=heads, passages=passages, seed=seed,
                            device=dev, dtype=dtype, q_scale=q_scale)
    for p in full.passages:
        one = full.select(passages=[p])
        if m_queries is not None and m_queries < one.m:
            one = _tail(one, m_queries)
        out.slices[("synthetic", 0, p)] = one
    out.provenance["synthetic"] = {"n": n, "d": d, "dv": dv or d, "heads": heads,
                                   "passages": passages, "seed": seed, "q_scale": q_scale,
                                   "distribution": "q,k ~ N(0,I_d); v ~ N(0,I_dv)"}
    return out


def _tail(b: abdata.ActivationBatch, m: int) -> abdata.ActivationBatch:
    """Keep the last ``m`` queries. Causal alignment (last query <-> last key) survives."""
    from dataclasses import replace
    return replace(b, q=b.q[..., -m:, :])


def study_slices(*, real: Iterable[tuple[str, str]] = PRIMARY_SOURCES,
                 synthetic: dict | None = None, allow_fallback: bool = True,
                 **kw) -> SliceSet:
    """Real slices for each ``(model, dataset)`` plus the matched synthetic control."""
    out = SliceSet()
    for model, dataset in real:
        try:
            out.update(load_real(model, dataset, **kw))
        except abdata.DataUnavailableError as e:
            if not allow_fallback:
                raise
            out.provenance.setdefault("unavailable", {})[f"{model}/{dataset}"] = str(e)
    syn = dict(n=kw.get("n_tokens") or 1024, d=64, heads=12, passages=4,
               m_queries=kw.get("m_queries"), device=kw.get("device"),
               dtype=kw.get("dtype", torch.float64))
    syn.update(synthetic or {})
    out.update(load_synthetic(**syn))
    return out


# ======================================================================================
# references
# ======================================================================================


def reference(b: abdata.ActivationBatch, causal: bool = True) -> dict:
    """Exact softmax on a slice: scaled logits ``s``, measure ``pi``, output ``y``."""
    s = (b.q @ b.k.transpose(-1, -2)) * b.scale
    if causal:
        s = s.masked_fill(~causal_mask(b.m, b.n, device=s.device), float("-inf"))
    pi = torch.softmax(s, dim=-1)
    return {"s": s, "pi": pi, "y": pi @ b.v}


def per_head(t: Tensor) -> list[float]:
    """Reduce a ``[1, H, m]`` (or ``[1, H]``) per-query quantity to one median per head."""
    t = t.double()
    if t.dim() >= 3:
        t = t.median(dim=-1).values
    return [float(x) for x in t.flatten()]


# ======================================================================================
# fitting
# ======================================================================================


def fit_loglog_slope(x: Sequence[float], y: Sequence[float]) -> dict:
    """Least-squares slope of ``log y`` on ``log x`` with a 95% CI on the slope.

    Used wherever the paper quotes an exponent (S3's ``O(1/r)`` and ``O(r^{-1/2})``),
    so that the exponent in the text is measured rather than asserted.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    lx, ly = np.log(x[ok]), np.log(y[ok])
    n = lx.size
    if n < 3:
        return {"slope": float("nan"), "intercept": float("nan"), "ci95": [float("nan")] * 2,
                "r2": float("nan"), "n": int(n)}
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    resid = ly - A @ coef
    dof = n - 2
    s2 = float(resid @ resid) / dof
    sxx = float(((lx - lx.mean()) ** 2).sum())
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    t = _t_crit(dof)
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    return {"slope": slope, "intercept": intercept,
            "ci95": [slope - t * se, slope + t * se], "se": se,
            "r2": 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan"),
            "n": int(n)}


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980}


def _t_crit(dof: int) -> float:
    """Two-sided 95% Student-t critical value (table lookup, no scipy dependency)."""
    if dof <= 0:
        return float("nan")
    keys = sorted(_T95)
    for key in keys:
        if dof <= key:
            return _T95[key]
    return 1.960


# ======================================================================================
# small reporting helpers
# ======================================================================================


def med_iqr(values: Sequence[float]) -> dict:
    a = np.asarray([float(x) for x in values], dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    return {"n": int(a.size), "median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75))}


def explode(records: Sequence[dict], list_keys: Sequence[str],
            index_name: str = "unit") -> list[dict]:
    """Turn records whose metrics are per-unit lists into one record per unit.

    Studies run a whole ``[1, H, n, *]`` slice in a single job (that is what makes the
    GPU worth using) but report per-head points; this splits them back apart without
    losing the job's identity.
    """
    out = []
    for r in records:
        if r.get("_status", "ok") != "ok":
            continue
        met = r.get("metrics", {})
        lists = {k: met[k] for k in list_keys if isinstance(met.get(k), list)}
        if not lists:
            continue
        length = min(len(v) for v in lists.values())
        scalars = {k: v for k, v in met.items() if k not in lists}
        for i in range(length):
            row = {**r.get("job", {}), **scalars, index_name: i}
            row.update({k: v[i] for k, v in lists.items()})
            out.append(row)
    return out


def group_rows(rows: Sequence[dict], by: Sequence[str], value: str) -> list[dict]:
    """``summarize``-style median/IQR over already-exploded rows."""
    recs = [{"job": {k: r.get(k) for k in by}, "metrics": {value: r.get(value)},
             "_status": "ok"} for r in rows]
    return H.summarize(recs, by=list(by), value=value)


# ======================================================================================
# a pure-function memo for the random-feature matrices
# ======================================================================================

_FEATURE_CACHE: dict = {}
_FEATURE_BYTES = 0


def enable_feature_cache(max_bytes: float = 6e9) -> None:
    """Memoise ``gaussian_features``.

    It is a pure function of ``(r, d, seed, orthogonal)`` that computes in float64 on
    the CPU and only casts at the end, so caching the float64 result is **bit-identical**
    to recomputing it.  Without this, the orthogonal path redoes ``r/d`` float64 QR
    factorisations on the CPU for every forward pass -- twice per call, once for q and
    once for k -- which for S3's seed sweep costs more CPU than the whole study costs
    GPU, and starves any other study sharing the machine.

    Studies that sweep many seeds at fixed ``r`` should call this; nothing else changes.
    """
    from ..methods.interaction import performer as _perf
    from ..methods.interaction import trigrff as _trig

    if getattr(_perf, "_attnbank_feature_cache_installed", False):
        return
    original = _perf.gaussian_features

    def cached(r, d, seed, orthogonal, device, dtype):
        global _FEATURE_BYTES
        key = (int(r), int(d), int(seed), bool(orthogonal))
        hit = _FEATURE_CACHE.get(key)
        if hit is None:
            hit = original(r, d, seed, orthogonal, torch.device("cpu"), torch.float64)
            nbytes = hit.numel() * hit.element_size()
            if _FEATURE_BYTES + nbytes > max_bytes:
                _FEATURE_CACHE.clear()
                _FEATURE_BYTES = 0
            _FEATURE_CACHE[key] = hit
            _FEATURE_BYTES += nbytes
        return hit.to(device=device, dtype=dtype)

    cached._attnbank_original = original
    _perf.gaussian_features = cached
    _trig.gaussian_features = cached      # trigrff imported the name into its module
    _perf._attnbank_feature_cache_installed = True


def save_fig(fig, name: str):
    from ..plotting import style
    return style.save(fig, name, out_dir=FIGURES)
