"""The renormalized restriction of ``prop:sparse-identity``.

Every method in this category differs only in how it chooses ``S_q``.  Given the
boolean support, the operator is the same:

    pi_hat_q = pi_q |_{S_q} / pi_q(S_q),     y_hat(q) = m_{S_q}(q),
    y - y_hat = eps_q (m_{S_q^c} - m_{S_q}),   eps_q = pi_q(S_q^c).

``eps_q``, ``TV = eps_q`` and ``chi2 = eps_q/(1-eps_q)`` are returned in ``extras`` so
that studies can check the identity instead of trusting it.
"""
from __future__ import annotations

import torch

from ..core import AttentionOutput, causal_mask, masked_softmax, normalize_nonnegative, scores


def reference_pi(method, q, k, causal):
    """The exact softmax measure and the scaled logits, both masked.

    Examples
    --------
    >>> import torch, attnbank as ab
    >>> from attnbank.components import reference_pi
    >>> pi, s = reference_pi(ab.build("topk"), torch.randn(1, 1, 4, 8),
    ...                      torch.randn(1, 1, 6, 8), causal=False)
    >>> pi.shape, bool(torch.allclose(pi.sum(-1), torch.ones(1)))
    (torch.Size([1, 1, 4, 6]), True)
    """
    s = scores(q, k, method._scale(q.shape[-1]), causal)
    return masked_softmax(s), s


def visible(q, k, causal):
    """Boolean ``[..., m, n]``: which keys a query is allowed to see at all.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import visible
    >>> visible(torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 3, 4), causal=True)[0, 0].int()
    tensor([[1, 1, 0],
            [1, 1, 1]], dtype=torch.int32)
    """
    m, n = q.shape[-2], k.shape[-2]
    if causal:
        return causal_mask(m, n, device=q.device).expand(*q.shape[:-2], m, n)
    return torch.ones(*q.shape[:-2], m, n, dtype=torch.bool, device=q.device)


def restrict(pi, support, v, return_weights, extras=None):
    """Build the ``AttentionOutput`` of a renormalized restriction to ``support``.

    This is the operator every selector in ``restrict/`` -- and the evictions in
    ``compress/`` -- share; they differ only in the support they pass in.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import restrict
    >>> pi = torch.tensor([[[0.5, 0.3, 0.2]]])
    >>> v = torch.tensor([[0.0], [1.0], [2.0]])
    >>> out = restrict(pi, torch.tensor([[[True, True, False]]]), v, True)
    >>> out.weights
    tensor([[[0.6250, 0.3750, 0.0000]]])
    >>> round(float(out.extras["eps"]), 4)               # the omitted mass
    0.2
    """
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

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import topk_support
    >>> score = torch.tensor([[0.1, 0.9, 0.5]])
    >>> topk_support(score, torch.ones(1, 3, dtype=torch.bool), 2).int()
    tensor([[0, 1, 1]], dtype=torch.int32)
    """
    n = score.shape[-1]
    if budget is None or budget >= n:
        return visible_mask.clone()
    s = score.masked_fill(~visible_mask, float("-inf"))
    idx = s.topk(budget, dim=-1).indices
    sup = torch.zeros_like(visible_mask).scatter_(-1, idx, True)
    return sup & visible_mask


__all__ = ["reference_pi", "visible", "restrict", "topk_support"]
