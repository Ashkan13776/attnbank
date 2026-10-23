"""ReBased: a learnable quadratic feature map with a movable root.

``sec:kernel-det``.  Aksenov et al. (2024) start from the observation that Based's
surrogate ``1 + z + z^2/2`` has minimum ``1/2`` at ``z = -1``: however negative the
logit, the weight it assigns is bounded below, so the measure can never concentrate
the way softmax does.  Their fix keeps the quadratic but makes its root learnable --
normalize each head's queries and keys, then apply a per-channel affine before
squaring,

    phi(q) = (gamma_Q . norm(q) + beta_Q)^2,   phi(k) = (gamma_K . norm(k) + beta_K)^2,
    g_hat(q, x) = <phi(q), phi(k_x)>,   y_i = phi(q_i)^T sum_j phi(k_j) v_j^T
                                             / phi(q_i)^T sum_j phi(k_j),

with ``norm`` a LayerNorm over the head dimension (no affine of its own: ``gamma``
and ``beta`` are that affine, learned separately for queries and keys) and ``.`` the
elementwise product.  ``phi >= 0`` is a square, so ``pi_hat`` is a probability
measure, and ``phi`` now *reaches* zero -- at ``norm(x)_c = -beta_c / gamma_c`` -- which
is the whole point.

This is an **operator modification**: ``r = d`` features, no resource parameter, and
no configuration under which it becomes softmax, so both correctness anchors are
``None`` by design (``tab:kernel-methods``).  Two structural consequences follow from
the normalization and are worth separating from the learnability.  First
``rem:kernel-homogeneous`` applies *exactly*: LayerNorm discards ``||q||``, so unlike
``based`` -- and unlike softmax -- scaling the query cannot sharpen ``pi_hat_q`` at
all.  Second, ``g_hat`` is a quadratic form in ``norm(q)``, hence bounded on the
sphere: the dynamic range of the surrogate is bounded uniformly in the input, where
``exp`` is not.  Both are visible in ``extras``.

``gamma`` and ``beta`` are the learned parameters, so ``trainable = True``.  The
defaults are their initialization, ``gamma = 1`` and ``beta = 0``, at which
``phi(x) = norm(x)^2``; pass floats to move them (the same value in every channel) and
see the root move with them.

Reference: Aksenov et al. (2024) ReBased; Arora et al. (2024) for the Based kernel it
modifies.
"""
from __future__ import annotations

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    causal_mask,
    normalize_nonnegative,
)
from ...registry import register

#: LayerNorm epsilon; PyTorch's default, small enough to be invisible at fp32 tolerance
EPS = 1e-5


def rebased_feature(x, gamma: float, beta: float):
    """``(gamma . norm(x) + beta)^2`` with ``norm`` a LayerNorm over the last axis."""
    nx = torch.nn.functional.layer_norm(x, x.shape[-1:], eps=EPS)
    return (gamma * nx + beta) ** 2


@register
class ReBasedAttention(AttentionMethod):
    """``phi(x) = (gamma . LayerNorm(x) + beta)^2``, normalized separable attention."""

    name = "rebased"
    category = Category.INTERACTION
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = True            # phi is a square, so g_hat >= 0
    trainable = True                 # gamma and beta are learned; these are their init
    references = ("aksenov2024rebased", "arora2024based")
    paper_section = "sec:kernel"

    def __init__(self, gamma_q: float = 1.0, beta_q: float = 0.0,
                 gamma_k: float = 1.0, beta_k: float = 0.0, **kw):
        super().__init__(gamma_q=gamma_q, beta_q=beta_q, gamma_k=gamma_k,
                         beta_k=beta_k, **kw)
        self.gamma_q, self.beta_q = gamma_q, beta_q
        self.gamma_k, self.beta_k = gamma_k, beta_k

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        pq = rebased_feature(q, self.gamma_q, self.beta_q)
        pk = rebased_feature(k, self.gamma_k, self.beta_k)
        g_hat = pq @ pk.transpose(-1, -2)                       # >= 0 entrywise
        support = causal_mask(m, n, device=q.device) if causal else None
        pi, norm = normalize_nonnegative(g_hat, support)
        extras = {"g_hat": g_hat, "Z_hat": norm, "phi_q": pq, "phi_k": pk,
                  "dynamic_range": g_hat.amax(-1) / g_hat.amin(-1).clamp_min(1e-30)}
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = d
        return {"flops": 2 * n * r * dv + 2 * m * r * dv + 4 * (m + n) * r,
                "state_bytes": r * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
