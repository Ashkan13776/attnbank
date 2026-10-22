"""Oracle top-k restriction on the true scores, renormalized.

``sec:sparse-operator``, ``sec:sparse-query``.  ``S_q`` is the set of the ``k`` largest
exact attention weights, which by ``prop:error-topk`` minimizes the omitted mass
``eps_q`` -- and hence the bound ``eps_q osc(v)`` -- over all supports of size ``k``.
It is the **oracle** against which every proxy selector (Quest, SparQ, Loki, H2O) is
measured: computing all scores saves only the value aggregation.

The proposition's second half matters more for this library: minimizing ``eps_q`` is
*not* minimizing ``||y - y_hat||``.  With ``pi = (0.45, 0.35, 0.20)`` and scalar values
``(0, 0, 1)``, top-2 retains ``{1,2}`` and errs by ``0.2`` while ``{1,3}`` omits more
mass and errs by ``0.11``.  ``value_topk`` is the output-aware alternative; the gap
between them is the quantity study S2 reports.

Reference: Gupta et al. (2021).
"""
from __future__ import annotations

from ...core import AttentionMethod, Category, Objective
from ...registry import register
from ._restriction import reference_pi, restrict, topk_support, visible


@register
class TopKAttention(AttentionMethod):
    """Exact top-k selection on the true scores (an oracle for proxy selectors)."""

    name = "topk"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("gupta2021memory", "tang2024quest")
    paper_section = "sec:sparse"

    def __init__(self, k: int | None = 8, **kw):
        """``k``: budget; ``None`` keeps every visible key (the exact limit)."""
        super().__init__(k=k, **kw)
        self.k = k

    def _forward(self, q, k, v, causal, return_weights):
        pi, s = reference_pi(self, q, k, causal)
        sup = topk_support(s, visible(q, k, causal), self.k)
        return restrict(pi, sup, v, return_weights, {"scores": s})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        kk = self.k or n
        return {"flops": 2 * m * n * d + 2 * m * kk * dv + m * n,
                "state_bytes": n * (d + dv) * bytes_per_el,     # full context stays resident
                "bytes_moved": (m * d + n * d + m * kk * dv + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"k": None}
