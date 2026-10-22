"""cosFormer: ReLU features with a cosine re-weighting.

``sec:kernel-det``.  Qin et al. (2022) use

    g_hat(q_i, k_j) = relu(q_i)^T relu(k_j) * cos(pi (i - j) / (2 M)),   M >= n,

separable with ``2r`` features via ``cos(a - a') = cos a cos a' + sin a sin a'``.  The
cosine factor injects a *locality prior*: the re-weighting decays monotonically with
``|i - j|`` and stays nonnegative because ``|i - j| <= n - 1 < M`` keeps the argument
inside ``[-pi/2, pi/2]``, so ``pi_hat_q`` remains a probability measure.

Like ``linear_elu`` this is a fixed feature map with no resource parameter: an
operator modification (``tab:kernel-methods``: "cosFormer ... no"), and
``rem:kernel-homogeneous`` applies exactly, since ``relu`` is positively homogeneous --
scaling the query cannot change ``pi_hat_q`` at all.

Positions follow the library's end-alignment convention: query ``i`` sits at absolute
context position ``i + (n - m)``.

Reference: Qin et al. (2022) cosFormer; Qin et al. (2022) TransNormer.
"""
from __future__ import annotations

import math

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective, causal_mask,
                     normalize_nonnegative, query_positions)
from ...registry import register


@register
class CosFormerAttention(AttentionMethod):
    """``relu`` features times ``cos(pi (i - j) / 2M)``."""

    name = "cosformer"
    category = Category.INTERACTION
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("qin2022cosformer", "qin2022devil")
    paper_section = "sec:kernel"

    def __init__(self, length_constant: int | None = None, **kw):
        """``length_constant`` is ``M``; defaults to ``n`` (the smallest legal value)."""
        super().__init__(length_constant=length_constant, **kw)
        self.length_constant = length_constant

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        big_m = max(self.length_constant or n, n)
        pi_q = query_positions(m, n, device=q.device).to(q.dtype) * (math.pi / (2 * big_m))
        pi_k = torch.arange(n, device=q.device, dtype=q.dtype) * (math.pi / (2 * big_m))

        rq, rk = torch.relu(q), torch.relu(k)
        content = rq @ rk.transpose(-1, -2)
        decay = torch.cos(pi_q[:, None] - pi_k[None, :])        # [m, n], >= 0
        g_hat = content * decay
        support = causal_mask(m, n, device=q.device) if causal else None
        pi, z = normalize_nonnegative(g_hat, support)
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras={"g_hat": g_hat, "decay": decay, "Z_hat": z})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = 2 * d
        return {"flops": 2 * n * r * dv + 2 * m * r * dv + (m + n) * r,
                "state_bytes": r * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
