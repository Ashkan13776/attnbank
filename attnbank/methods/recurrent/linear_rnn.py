"""Linear attention as a recurrence: H_t = H_{t-1} + psi(k_t) v_t^T.

``sec:memory-state``, ``prop:memory-suff``, ``prop:memory-rank``.  A separable
interaction of rank ``r`` admits a sufficient statistic of size ``r x d_v``: the prefix
moments ``H_t = sum_{i<=t} psi(k_i) v_i^T`` (and ``z_t = sum psi(k_i)``) are all a
future query needs.  ``prop:memory-rank`` is the converse: an *additive* finite-state
memory forces the interaction to have finite rank, so recall is bounded by the state.

With the normalizer kept this is ``linear_elu``.  The deployed form, and the one here,
drops it (``readout = H_t^T phi(q_t)``) because the denominator can approach zero and
produce unbounded gradients.  Three properties of ``eq:attention_expectation`` go with
it (``sec:memory-state``): normalization invariance under ``g -> c(q) g``, constant
preservation (``y_hat = c <phi(q), z>`` now scales with context length), and the
convex-hull bound.  Set ``normalize=True`` to put the denominator back and watch all
three return.

The weights on the original positions are ``w_{tj} = <phi(q_t), psi(k_j)>`` for
``j <= t``: nonnegative for a positive feature map, but *not* summing to one.  Invariant
W still holds exactly, which is why this category is in the library at all.

Reference: Katharopoulos et al. (2020); Schlag et al. (2021);
Mongaras and Larson (2025).
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, causal_mask
from ...registry import register
from ._state import feature_map


@register
class LinearRNNAttention(AttentionMethod):
    """Additive linear recurrence with an unnormalized readout."""

    name = "linear_rnn"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = False           # the normalizer is dropped: nm, cp, hl all lost
    references = ("katharopoulos2020transformers", "schlag2021linear", "sun2024learning")
    paper_section = "sec:memory-state"

    def __init__(self, feature: str = "elu", normalize: bool = False, **kw):
        super().__init__(feature=feature, normalize=normalize, **kw)
        self.feature, self.normalize = feature, normalize

    def _weights(self, q, k, causal):
        phi = feature_map(self.feature)
        w = phi(q) @ phi(k).transpose(-1, -2)
        if causal:
            w = w * causal_mask(q.shape[-2], k.shape[-2], device=q.device)
        return w

    def _forward(self, q, k, v, causal, return_weights):
        w = self._weights(q, k, causal)
        z = w.sum(-1, keepdim=True)
        if self.normalize:
            w = w / z.clamp_min(torch.finfo(w.dtype).tiny)
        return AttentionOutput(y=w @ v, weights=w if return_weights else None,
                               extras={"mass": z.squeeze(-1), "state_rank": q.shape[-1]})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = d
        return {"flops": 2 * n * r * dv + 2 * m * r * dv,
                "state_bytes": r * dv * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
