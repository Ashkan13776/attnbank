"""ZipCache: bit width allocated per token by a **normalised** attention score.

``sec:compress-quant``, and the place where this family meets ``sec:compress-evict``.
KIVI, KVQuant and Atom all spend the same number of bits on every token.  ZipCache does
not: it ranks tokens by how much attention they actually receive and gives the salient
ones ``bits_salient`` while the rest get ``bits_regular``.  Eviction is the degenerate
case of this -- zero bits for the unsalient -- and ``prop:compress-noeviction`` applies
to the ranking either way.

Two ideas, and the second is the one worth implementing carefully.

**Channel-separable token-wise quantization.**  Each channel is first divided by its
own maximum absolute value over tokens, then whole tokens are quantized with a single
scale, then the channel normalizer is multiplied back::

    a_c = max_j |x_jc|,     x_hat_j = a * Q_b(x_j / a).

What this buys is *relative* precision, channel by channel.  A plain per-token scale is
set by whichever channel is largest in that token, so every channel is given the same
**absolute** step and the small channels -- the ones a query may well load on -- are
obliterated.  Dividing by ``a_c`` first makes the step proportional to each channel's
own scale, so the relative error is near-uniform.  It is not a smaller Frobenius error,
and the test file says so.  The side information is ``d + 2n`` numbers rather than the
``2 n d / g`` of a group-wise scheme.

**The normalised attention score.**  The usual saliency, ``sum_i A_ij`` accumulated down
a column, is biased twice over by the causal mask: an early token appears in more rows,
and the rows it appears in are shorter, so their softmax puts more mass on each entry.
Both biases reward *position* rather than content.  ZipCache divides by the number of
queries that could see the token::

    p_j = sum_i A_ij / nnz(A_{.,j}),     nnz(A_{.,j}) = #{i : j <= i + (n - m)}.

The saliency is always measured on the **causally masked** prefill attention, whatever
``causal`` is for the forward pass itself, because that is the quantity a prefill
actually has; ``nnz`` is then a genuine correction rather than the constant ``m`` a
non-causal matrix would give.

``probe`` enables the paper's cheap approximation: estimate the column sums from a
sample of query rows (half the most recent, half drawn uniformly) instead of the whole
matrix, which is what makes the metric compatible with a kernel that never materialises
``A``.  ``probe = None`` uses every row and is the exact metric.

**Not causal.**  The channel normalizers pool over tokens, and the salient set is chosen
after seeing the whole prompt and then applied to every query -- the prefill-then-decode
pattern of ``evict_snapkv``, and like it, something that cannot be written as one
causally masked pass.

Quantize--dequantize is **simulated** in the input dtype; nothing is actually stored
smaller, and ``cost()`` states what would be.

``target`` restricts the quantizer to the keys or to the values.  The paper quantizes
both with the same allocation; the split exists because the error decomposition of
``sec:output_error`` -- the distributional term against the value term -- can only be
run on a scheme whose two halves can be switched off independently, and this family is
the one that decomposition was written for.

Exact limit: ``bits_salient = bits_regular = None``.

Reference: He et al. (2024) ZipCache, NeurIPS.
"""
from __future__ import annotations

import torch

from ...components.quantization import affine_quantize
from ...core import (TINY, AttentionMethod, AttentionOutput, Category, Objective,
                     causal_mask, make_generator, masked_softmax, scores)
from ...registry import register


def channel_separable_quantize(x, bits):
    """Normalise each channel by its max absolute value, quantize per token, rescale."""
    if bits is None:
        return x
    a = x.abs().amax(-2, keepdim=True).clamp_min(TINY)
    return affine_quantize(x / a, bits, "token", None) * a


def normalized_attention_score(pi, visible):
    """``p_j = sum_i pi_ij / nnz_j`` -- column mass per *eligible* query, not per query.

    ``visible`` is the ``[m, n]`` causal mask; dividing by its column sums removes the
    two positional biases of the raw accumulated score.
    """
    return pi.sum(-2) / visible.sum(-2).clamp_min(1).to(pi.dtype)


def probe_rows(m, probe, seed, device):
    """Indices of the probe queries: half the most recent, half drawn uniformly."""
    want = max(1, int(round(probe * m)))
    recent = max(1, want // 2)
    keep = torch.zeros(m, dtype=torch.bool, device=device)
    keep[m - recent:] = True
    pool = (~keep).nonzero(as_tuple=True)[0]
    extra = want - recent
    if extra > 0 and pool.numel():
        take = torch.randperm(pool.numel(), generator=make_generator(seed))[:extra]
        keep[pool[take.to(pool.device)]] = True
    return keep.nonzero(as_tuple=True)[0]


@register
class ZipCacheAttention(AttentionMethod):
    """Mixed-precision KV quantization with salient tokens chosen by normalised score."""

    name = "zipcache"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False          # channel normalizers and a prompt-wide salient set
    expresses_weights = True
    row_stochastic = True
    references = ("he2024zipcache",)
    paper_section = "sec:compress"

    def __init__(self, bits_salient: int | None = 4, bits_regular: int | None = 2,
                 salient: float = 0.5, probe: float | None = None,
                 target: str = "kv", seed: int = 0, **kw):
        super().__init__(bits_salient=bits_salient, bits_regular=bits_regular,
                         salient=salient, probe=probe, target=target, seed=seed, **kw)
        if not 0.0 <= salient <= 1.0:
            raise ValueError(f"zipcache: salient must be in [0, 1]; got {salient}")
        if probe is not None and not 0.0 < probe <= 1.0:
            raise ValueError(f"zipcache: probe must be in (0, 1] or None; got {probe}")
        if target not in ("kv", "k", "v"):
            raise ValueError(f"zipcache: target must be 'kv', 'k' or 'v'; got {target!r}")
        self.bits_salient, self.bits_regular = bits_salient, bits_regular
        self.salient, self.probe, self.target, self.seed = salient, probe, target, seed

    def _saliency(self, q, k, beta):
        """Normalised attention score from the causal prefill view, optionally probed."""
        m, n = q.shape[-2], k.shape[-2]
        visible = causal_mask(m, n, device=q.device)
        pi = masked_softmax(scores(q, k, beta, causal=True))
        if self.probe is not None:
            rows = probe_rows(m, self.probe, self.seed, q.device)
            pi = pi.index_select(-2, rows)
            visible = visible.index_select(0, rows)
        return normalized_attention_score(pi, visible), pi.sum(-2)

    def _mixed(self, x, keep):
        """Salient tokens at ``bits_salient``, the rest at ``bits_regular``."""
        if self.bits_salient is None and self.bits_regular is None:
            return x
        hi = channel_separable_quantize(x, self.bits_salient)
        lo = channel_separable_quantize(x, self.bits_regular)
        return torch.where(keep.unsqueeze(-1), hi, lo)

    def _forward(self, q, k, v, causal, return_weights):
        beta = self._scale(q.shape[-1])
        n = k.shape[-2]
        score, accumulated = self._saliency(q, k, beta)
        budget = min(n, max(1, int(round(self.salient * n)))) if self.salient > 0 else 0
        keep = torch.zeros_like(score, dtype=torch.bool)
        if budget:
            keep.scatter_(-1, score.topk(budget, dim=-1).indices, True)

        k_hat = self._mixed(k, keep) if "k" in self.target else k
        v_hat = self._mixed(v, keep) if "v" in self.target else v
        s, s_hat = scores(q, k, beta, causal), scores(q, k_hat, beta, causal)
        pi = masked_softmax(s_hat)
        return AttentionOutput(
            y=pi @ v_hat, weights=pi if return_weights else None, v_hat=v_hat,
            extras={"s": s, "s_hat": s_hat, "delta_k": (k - k_hat).norm(dim=-1),
                    "delta_v": (v - v_hat).norm(dim=-1),
                    "saliency": score, "accumulated": accumulated, "salient": keep})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        """Two bit widths mixed by ``salient``, one scale per token, one per channel."""
        bs = (self.bits_salient or 8 * bytes_per_el) / 8
        br = (self.bits_regular or 8 * bytes_per_el) / 8
        bits = self.salient * bs + (1.0 - self.salient) * br
        side = (d + dv) * bytes_per_el + 4 * n * bytes_per_el     # a_c, plus 2 per token
        probe = m if self.probe is None else max(1, int(round(self.probe * m)))
        return {"flops": 2 * m * n * (d + dv) + 2 * probe * n * d,
                "state_bytes": n * (d + dv) * bits + side,
                "bytes_moved": m * (d + dv) * bytes_per_el + n * (d + dv) * bits + side}

    @classmethod
    def limit_config(cls):
        return {"bits_salient": None, "bits_regular": None}
