"""The online-softmax monoid of ``prop:exact-monoid`` (``sec:exact-tiling``).

For a block ``J`` of key indices the partial state is

    sigma_J = (a_J, l_J, u_J),
    a_J = max_{j in J} s_j,
    l_J = sum_{j in J} e^{s_j - a_J},
    u_J = sum_{j in J} e^{s_j - a_J} v_j,

with merge (``eq:exact-merge``)

    (a1,l1,u1) + (a2,l2,u2) = (a, e^{a1-a} l1 + e^{a2-a} l2, e^{a1-a} u1 + e^{a2-a} u2),
    a = max(a1, a2),

readout ``rd(a,l,u) = u/l`` and identity ``(-inf, 0, 0)``.  ``(Sigma, +, 0)`` is a
*commutative* monoid, so the fold over any partition, in any bracketing, gives ``y(q)``.

``flash`` folds left-to-right over key blocks in one pass; ``splitk`` folds per chunk
and merges once (or as a tree). They share this file precisely because the proposition
says the two must agree.
"""
from __future__ import annotations

import torch


def online_state(s_block, v_block):
    """State ``sigma_J`` of one key block from its scaled logits and values.

    A fully masked block (every logit ``-inf``) yields ``a = 0, l = 0, u = 0``, which is
    the monoid identity up to the shift and keeps the merge finite -- the documented
    no-``nan`` rule of DESIGN.md Sec. 4.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import online_state, readout
    >>> s = torch.tensor([[0.0, 1.0]])
    >>> v = torch.tensor([[1.0], [3.0]])
    >>> y = readout(online_state(s, v))
    >>> torch.allclose(y, torch.softmax(s, -1) @ v)
    True
    """
    a = s_block.amax(dim=-1)                                    # [..., m]
    a = torch.where(torch.isneginf(a), torch.zeros_like(a), a)  # fully masked block
    p = torch.exp(s_block - a.unsqueeze(-1))
    return a, p.sum(-1), p @ v_block


def merge_states(s1, s2):
    """The monoid operation ``+`` of ``eq:exact-merge``.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import merge_states, online_state, readout
    >>> s = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    >>> v = torch.arange(4.0).reshape(4, 1)
    >>> left = online_state(s[..., :1], v[:1])
    >>> right = online_state(s[..., 1:], v[1:])
    >>> torch.allclose(readout(merge_states(left, right)), torch.softmax(s, -1) @ v)
    True
    """
    (a1, l1, u1), (a2, l2, u2) = s1, s2
    a = torch.maximum(a1, a2)
    w1, w2 = torch.exp(a1 - a), torch.exp(a2 - a)
    return a, w1 * l1 + w2 * l2, w1.unsqueeze(-1) * u1 + w2.unsqueeze(-1) * u2


def readout(state):
    """``rd(a, l, u) = u / l``, with the denominator floored so a dead row is finite.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import online_state, readout
    >>> readout(online_state(torch.zeros(1, 2), torch.tensor([[0.0], [2.0]])))
    tensor([[1.]])
    """
    _, ell, u = state
    return u / ell.unsqueeze(-1).clamp_min(torch.finfo(u.dtype).tiny)


def log_normalizer(state):
    """``lse = a + log l``, the log of the softmax normalizer.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import online_state
    >>> from attnbank.components.online_softmax import log_normalizer
    >>> s = torch.tensor([[0.0, 1.0, 2.0]])
    >>> torch.allclose(log_normalizer(online_state(s, torch.zeros(3, 1))),
    ...                torch.logsumexp(s, -1))
    True
    """
    a, ell, _ = state
    return a + torch.log(ell)
