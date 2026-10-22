"""Linear transformer: the fixed feature map ``phi(u) = elu(u) + 1``.

``sec:kernel-det``.  Katharopoulos et al. (2020) replace the interaction by

    g_hat(q, x) = <phi(q), phi(k_x)>,   phi(u) = elu(u) + 1 > 0,   r = d,

with no resource parameter, so ``<phi(q), phi(k)> -> exp(beta <q,k>)`` holds *in no
sense*: this is an **operator modification**, not an approximation
(``def:kernel-regimes``, ``tab:kernel-methods`` row "Linear transformer ... no").  The
library records that by ``objective = modify`` and by returning ``None`` from both
correctness anchors: there is no configuration under which it becomes softmax, and
nothing it converges to.

Because ``phi > 0`` the induced measure is a genuine probability measure, so this is
one of the normalized members of the family; ``lem:kernel-mixture`` then says
``pi_hat_q`` is a mixture of ``r = d`` *query-independent* context distributions, and
``prop:kernel-retrieval`` bounds what it can retrieve.  ``rem:kernel-homogeneous``
applies approximately: for large positive coordinates ``phi`` is linear, so the query
norm cannot sharpen ``pi_hat_q``.

The normalized form of ``eq:kernel-moments`` is used (divide by ``phi(q)^T z``).  Many
deployed descendants drop the denominator and normalize the output instead
(``sec:memory-state``); that variant lives in ``recurrent/linear_rnn.py``, where the
loss of normalization invariance is the point.

Reference: Katharopoulos et al. (2020); Schlag et al. (2021).
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective, causal_mask,
                     normalize_nonnegative)
from ...registry import register


def elu_feature(x):
    return torch.nn.functional.elu(x) + 1.0


@register
class LinearELUAttention(AttentionMethod):
    """``phi(u) = elu(u) + 1``, normalized separable attention."""

    name = "linear_elu"
    category = Category.INTERACTION
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("katharopoulos2020transformers", "schlag2021linear")
    paper_section = "sec:kernel"

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        pq, pk = elu_feature(q), elu_feature(k)
        g_hat = pq @ pk.transpose(-1, -2)                       # > 0 entrywise
        support = causal_mask(m, n, device=q.device) if causal else None
        pi, z = normalize_nonnegative(g_hat, support)
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras={"g_hat": g_hat, "Z_hat": z})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = d
        return {"flops": 2 * n * r * dv + 2 * m * r * dv + (m + n) * r,
                "state_bytes": r * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
