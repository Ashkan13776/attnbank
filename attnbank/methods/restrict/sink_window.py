"""StreamingLLM: the first ``s`` "attention sink" positions plus a sliding window.

``sec:sparse-fixed``, ``sec:compress-evict``, ``ex:sparse-sink``.  The support is
``{0, ..., s-1} union {p-w+1, ..., p}``: content-independent, and therefore an eviction
policy as much as a sparsity pattern.

The reason it works is ``prop:sparse-identity`` read backwards.  A head that routes
mass ``eps`` to a sink with a near-null value computes ``y = (1-eps) m_S``; a pure
window outputs ``m_S``, an amplification by ``1/(1-eps)``.  Retaining the sink restores
the normalizer, not information: the sink carries *normalization*.  Comparing
``sink_window(s>0)`` with ``window`` at the same total budget isolates that effect,
which is what makes the pair worth having in the library.

Reference: Xiao et al. (2024) StreamingLLM; Han et al. (2024) LM-Infinite.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective
from ...registry import register
from ._restriction import reference_pi, restrict, visible
from .window import window_support


@register
class SinkWindowAttention(AttentionMethod):
    """``s`` initial sinks plus a left window of width ``w``."""

    name = "sink_window"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("xiao2024efficient", "han2024lminfinite")
    paper_section = "sec:sparse"

    def __init__(self, sinks: int = 2, w: int | None = 8, **kw):
        super().__init__(sinks=sinks, w=w, **kw)
        self.sinks, self.w = sinks, w

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        sup = window_support(m, n, self.w, q.device)
        if self.sinks > 0:
            j = torch.arange(n, device=q.device)[None, :]
            sup = sup | (j < self.sinks)
        sup = visible(q, k, causal) & sup
        pi, s = reference_pi(self, q, k, causal)
        return restrict(pi, sup, v, return_weights, {"sinks": self.sinks})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        kept = (self.w or n) + self.sinks
        return {"flops": 2 * m * kept * (d + dv),
                "state_bytes": kept * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"sinks": 0, "w": None}
