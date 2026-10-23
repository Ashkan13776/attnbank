"""ORACLE: best rank-r *signed* factorization of the attention matrix.

``sec:factor-rank``.  The truncated SVD ``P_r = U_r Sigma_r V_r^T`` is the optimal
rank-``r`` approximation of ``P`` in every unitarily invariant norm (Eckart--Young), so
it is the lower bound that no stochastic factorization of the same rank can beat:
``rank(P) <= rank_+(P) = rank_s(P)``, and the inequality is usually strict.  The gap
between this file and ``nmf_oracle`` is the empirical nonnegative-rank penalty of
study S5.

The price of optimality is the whole contract: ``P_r`` is **signed**, its rows do not
sum to one, and the output can leave the convex hull of the values -- exactly the
failure mode ``prop:factor-closure`` was stated to rule out for Markov compositions.
``row_stochastic = False`` records it; the row sums and the most negative weight are
returned in ``extras`` so the violation can be measured instead of assumed.

This is an **oracle**: no one computes a per-head SVD of ``P`` at inference.  It exists
to bound what any rank-``r`` method could achieve.  The decomposition is taken in
float64.

Exact limit: ``r = None`` keeps the full spectrum.  Not causal: the truncation does not
preserve the mask's zeros.

Reference: Bhojanapalli et al. (2020) (the rank bottleneck of attention);
Wang et al. (2020) Linformer; Gillis (2020).
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, masked_softmax, scores
from ...registry import register


@register
class SVDOracleAttention(AttentionMethod):
    """Truncated SVD of P: the best signed rank-r factorization. Oracle."""

    name = "svd_oracle"
    category = Category.FACTORIZE
    objective = Objective.APPROXIMATE
    supports_causal = False
    expresses_weights = True
    row_stochastic = False           # signed; rows do not sum to one
    references = ("bhojanapalli2020lowrank", "wang2020linformer", "gillis2020nonnegative")
    paper_section = "sec:factor"

    def __init__(self, r: int | None = 4, **kw):
        super().__init__(r=r, **kw)
        self.r = r

    def _forward(self, q, k, v, causal, return_weights):
        p = masked_softmax(scores(q, k, self._scale(q.shape[-1]), causal)).double()
        rmax = min(p.shape[-1], p.shape[-2])
        r = min(self.r or rmax, rmax)
        u, s, vh = torch.linalg.svd(p, full_matrices=False)
        pi = (u[..., :r] * s[..., :r].unsqueeze(-2)) @ vh[..., :r, :]
        pi = pi.to(q.dtype)
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras={"singular_values": s.to(q.dtype), "r": r,
                                       "row_sum": pi.sum(-1), "min_weight": pi.amin(-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        rmax = min(m, n)
        r = min(self.r or rmax, rmax)
        return {"flops": 2 * m * n * d + 2 * (m + n) * r * dv,
                "state_bytes": r * n * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"r": None}
