"""Shared machinery for ``restrict/``: the renormalized restriction of prop:sparse-identity.

Every method in this category differs only in how it chooses ``S_q``.  Given the
boolean support, the operator is the same:

    pi_hat_q = pi_q |_{S_q} / pi_q(S_q),     y_hat(q) = m_{S_q}(q),
    y - y_hat = eps_q (m_{S_q^c} - m_{S_q}),   eps_q = pi_q(S_q^c).

``eps_q``, ``TV = eps_q`` and ``chi2 = eps_q/(1-eps_q)`` are returned in ``extras`` so
that studies can check the identity instead of trusting it.
"""
from __future__ import annotations

import torch

from ...core import AttentionOutput, causal_mask, masked_softmax, normalize_nonnegative, scores


def reference_pi(method, q, k, causal):
    """The exact softmax measure and the scaled logits, both masked."""
    s = scores(q, k, method._scale(q.shape[-1]), causal)
    return masked_softmax(s), s


def visible(q, k, causal):
    m, n = q.shape[-2], k.shape[-2]
    if causal:
        return causal_mask(m, n, device=q.device).expand(*q.shape[:-2], m, n)
    return torch.ones(*q.shape[:-2], m, n, dtype=torch.bool, device=q.device)


def restrict(pi, support, v, return_weights, extras=None):
    """Build the ``AttentionOutput`` of a renormalized restriction to ``support``."""
    pi_hat, mass = normalize_nonnegative(pi, support)
    eps = 1.0 - mass
    ex = {"support": support, "eps": eps, "k_realized": support.sum(-1),
          "chi2": eps / (1 - eps).clamp_min(1e-30)}
    if extras:
        ex.update(extras)
    return AttentionOutput(y=pi_hat @ v, weights=pi_hat if return_weights else None, extras=ex)


def topk_support(score, visible_mask, budget):
    """Boolean support of the ``budget`` largest ``score`` among the visible keys.

    Invisible keys get ``-inf`` so they can never be selected; ``budget = None`` or a
    budget at least as large as the row selects every visible key, which is the exact
    limit of every selector in this package.
    """
    n = score.shape[-1]
    if budget is None or budget >= n:
        return visible_mask.clone()
    s = score.masked_fill(~visible_mask, float("-inf"))
    idx = s.topk(budget, dim=-1).indices
    sup = torch.zeros_like(visible_mask).scatter_(-1, idx, True)
    return sup & visible_mask
