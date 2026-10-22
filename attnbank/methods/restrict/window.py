"""Sliding-window (banded) attention: a query-position-dependent, content-free support.

``sec:sparse-fixed``.  Query at absolute position ``p`` attends to
``{p - w + 1, ..., p}``, intersected with what the mask allows.  The diagonal is always
included, so no row is ever empty.

Applied to a dense-trained model this is an approximation whose error is exactly
``eps_q (m_{S^c} - m_S)`` (``prop:sparse-identity``); it is the cleanest demonstration
of the sink effect of ``ex:sparse-sink``, because once the window has scrolled past
position 0 it omits a large mass whose conditional mean is the sink value, and
renormalization inflates the remaining weights by ``1/(1 - eps_q)``.  Compare with
``sink_window``, which differs only by the first ``s`` positions.

Reference: Beltagy et al. (2020); Jiang et al. (2023); Child et al. (2019).
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, query_positions
from ...registry import register
from ._restriction import reference_pi, restrict, visible


def window_support(m, n, w, device):
    """Boolean [m, n]: key ``j`` is in the left window of width ``w`` of query ``i``."""
    p = query_positions(m, n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    if w is None:
        return torch.ones(m, n, dtype=torch.bool, device=device)
    return (j <= p) & (j > p - w)


@register
class WindowAttention(AttentionMethod):
    """Left sliding window of width ``w``, always including the diagonal."""

    name = "window"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("beltagy2020longformer", "jiang2023mistral", "child2019generating")
    paper_section = "sec:sparse"

    def __init__(self, w: int | None = 8, **kw):
        """``w``: window width; ``None`` is the full context (the exact limit)."""
        super().__init__(w=w, **kw)
        self.w = w

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        pi, s = reference_pi(self, q, k, causal)
        sup = visible(q, k, causal) & window_support(m, n, self.w, q.device)
        return restrict(pi, sup, v, return_weights)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        w = self.w or n
        return {"flops": 2 * m * w * (d + dv),
                "state_bytes": w * (d + dv) * bytes_per_el,     # rolling buffer
                "bytes_moved": (m * d + m * w * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"w": None}
