"""ORACLE: Low-rank logits: truncate the *logits* to rank r, then softmax what is left.

``sec:kernel-matrix``.  Everything else in the bank that says "low rank" is low rank in
the wrong space.  ``linformer`` projects the sequence axis of ``K`` and ``V``,
``svd_oracle`` truncates ``P`` itself, ``nystrom`` factors ``P`` through landmarks --
all three produce a **signed** measure, because a rank-``r`` approximation of a
nonnegative matrix is not nonnegative.  This file truncates one step earlier:

    S   = beta Q K^T,      S_r = U_r Sigma_r V_r^T   (truncated SVD, float64),
    g_hat(q, x) = exp((S_r)_{qx}),   pi_hat = softmax(S_r).

The exponential is applied *after* the truncation, so ``pi_hat`` is a genuine
probability measure for every ``r``: nonnegative, unit mass, output inside the convex
hull of the values.  The whole contract survives, and the error is pure ``hl`` -- the
measure is wrong, but it is still a measure.  That is the comparison this file exists
to make: at equal rank, is it better to be optimal in ``P`` and signed, or suboptimal
in ``P`` and stochastic?  ``svd_oracle`` is the other arm.

The rank that matters.  ``S = beta Q K^T`` already has rank at most ``d``, so the
method is *exactly* softmax for any ``r >= min(m, n, d)`` and the truncation only
bites below the head dimension -- which is exactly Bhojanapalli et al.'s low-rank
bottleneck, stated as an operator rather than as an architectural constraint.  Note
also that ``rank(S_r) = r`` says nothing about ``rank(exp(S_r))``: the entrywise
exponential of a rank-one matrix is generically full rank, so a low-rank *interaction*
is not a low-rank *attention matrix*, and this method is not a factorization in the
sense of ``sec:factor``.

This is an **oracle**: it needs the full ``m x n`` logit matrix and an SVD of it, so it
buys nothing at inference.  It exists to bound what a rank-``r`` surrogate of the
interaction can achieve, and to separate that bound from the rank of ``P``.

Exact limit: ``r = None`` keeps the full spectrum.  Not causal: the truncation does not
preserve the mask's ``-inf`` entries, and an SVD of a matrix containing ``-inf`` is not
defined.

Reference: Bhojanapalli et al. (2020); Wang et al. (2020) Linformer.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, masked_softmax, scores
from ...registry import register


@register
class LowRankLogitAttention(AttentionMethod):
    """Truncated SVD of the logit matrix; the softmax of a rank-r interaction."""

    name = "lowrank_logit"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = False          # the mask's -inf entries do not survive an SVD
    expresses_weights = True
    row_stochastic = True            # exp comes after the truncation: still a measure
    references = ("bhojanapalli2020lowrank", "wang2020linformer")
    paper_section = "sec:kernel"

    def __init__(self, r: int | None = 4, **kw):
        """``r``: rank kept in the logit matrix; ``None`` keeps all of it (the limit)."""
        super().__init__(r=r, **kw)
        self.r = r

    def _forward(self, q, k, v, causal, return_weights):
        s = scores(q, k, self._scale(q.shape[-1]), causal=False).double()
        rmax = min(s.shape[-1], s.shape[-2])
        r = min(self.r or rmax, rmax)
        u, sv, vh = torch.linalg.svd(s, full_matrices=False)
        s_r = (u[..., :r] * sv[..., :r].unsqueeze(-2)) @ vh[..., :r, :]
        pi = masked_softmax(s_r).to(q.dtype)
        kept = sv[..., :r].pow(2).sum(-1) / sv.pow(2).sum(-1).clamp_min(1e-30)
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras={"r": r, "singular_values": sv.to(q.dtype),
                                       "spectral_energy": kept.to(q.dtype),
                                       "logit_error": (s_r - s).abs().amax(-1).to(q.dtype)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        rmax = min(m, n)
        r = min(self.r or rmax, rmax)
        return {"flops": 2 * m * n * d + 2 * m * n * rmax + 2 * m * n * dv,
                "state_bytes": r * (m + n) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"r": None}
