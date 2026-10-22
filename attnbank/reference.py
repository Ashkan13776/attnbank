"""The reference operator: dense softmax attention, y(q) = sum_j pi_q(j) v_j.

    pi_q(j) = exp(beta <q, k_j>) / sum_i exp(beta <q, k_i>),   beta = 1/sqrt(d)

Everything in the library is compared against this.  Section ``sec:found``.
"""
from __future__ import annotations

import torch

from .core import (AttentionMethod, AttentionOutput, Category, Objective, causal_mask,
                   masked_softmax, scores)
from .registry import register


def softmax_scores(q, k, scale, causal):
    """Scaled logits with -inf on masked entries; kept as a public name."""
    return scores(q, k, scale, causal)


@register
class SoftmaxAttention(AttentionMethod):
    """Dense softmax attention (Vaswani et al., 2017); the reference operator."""

    name = "softmax"
    category = Category.EXACT
    objective = Objective.ACCELERATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("vaswani2017attention",)
    paper_section = "sec:found"

    def _forward(self, q, k, v, causal, return_weights):
        pi = masked_softmax(softmax_scores(q, k, self._scale(q.shape[-1]), causal))
        y = pi @ v
        return AttentionOutput(y=y, weights=pi if return_weights else None)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        return {"flops": 2 * m * n * (d + dv), "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + 2 * m * n) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {}


def reference_weights(q, k, causal=False, scale=None):
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    return masked_softmax(softmax_scores(q, k, scale, causal))
