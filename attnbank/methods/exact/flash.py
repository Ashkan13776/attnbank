"""FlashAttention: tiled online softmax (``sec:exact``, ``sec:exact-tiling``).

The operator is unchanged.  For a block ``J`` of key indices the partial state is

    sigma_J = (a_J, l_J, u_J),
    a_J = max_{j in J} s_j,  l_J = sum_{j in J} e^{s_j - a_J},  u_J = sum_{j in J} e^{s_j - a_J} v_j,

with merge (``eq:exact-merge``)

    (a1,l1,u1) + (a2,l2,u2) = (a, e^{a1-a} l1 + e^{a2-a} l2, e^{a1-a} u1 + e^{a2-a} u2),
    a = max(a1, a2),

readout ``rd(a,l,u) = u/l`` and identity ``(-inf, 0, 0)``.  ``prop:exact-monoid``
states that ``(Sigma, +, 0)`` is a commutative monoid and that the readout of the
fold over *any* partition is ``y(q)``.  This file folds left-to-right over key
blocks in one pass, which is what a FlashAttention kernel does.

Reference: Dao et al. (2022, 2024); Milakov and Gimelshein (2018);
Rabe and Staats (2021).
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, causal_mask
from ...registry import register


def online_state(s_block, v_block):
    """State ``sigma_J`` of one key block from its scaled logits and values."""
    a = s_block.amax(dim=-1)                                    # [..., m]
    a = torch.where(torch.isneginf(a), torch.zeros_like(a), a)  # fully masked block
    p = torch.exp(s_block - a.unsqueeze(-1))
    return a, p.sum(-1), p @ v_block


def merge_states(s1, s2):
    """The monoid operation ``+`` of ``eq:exact-merge``."""
    (a1, l1, u1), (a2, l2, u2) = s1, s2
    a = torch.maximum(a1, a2)
    w1, w2 = torch.exp(a1 - a), torch.exp(a2 - a)
    return a, w1 * l1 + w2 * l2, w1.unsqueeze(-1) * u1 + w2.unsqueeze(-1) * u2


@register
class FlashAttention(AttentionMethod):
    """Single-pass left fold of the online-softmax monoid over key blocks."""

    name = "flash"
    category = Category.EXACT
    objective = Objective.ACCELERATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("dao2022flashattention", "dao2024flashattention2", "milakov2018online",
                  "rabe2021selfattention")
    paper_section = "sec:exact"

    def __init__(self, block_n: int = 16, **kw):
        super().__init__(block_n=block_n, **kw)
        self.block_n = block_n

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        beta = self._scale(d)
        mask = causal_mask(m, n, device=q.device) if causal else None

        state = None
        for start in range(0, n, self.block_n):
            stop = min(start + self.block_n, n)
            s = torch.einsum("...md,...nd->...mn", q, k[..., start:stop, :]) * beta
            if mask is not None:
                s = s.masked_fill(~mask[:, start:stop], float("-inf"))
            blk = online_state(s, v[..., start:stop, :])
            state = blk if state is None else merge_states(state, blk)

        a, ell, u = state
        y = u / ell.unsqueeze(-1).clamp_min(torch.finfo(u.dtype).tiny)
        extras = {"lse": a + torch.log(ell), "n_blocks": (n + self.block_n - 1) // self.block_n}
        pi = None
        if return_weights:
            s = torch.einsum("...md,...nd->...mn", q, k) * beta
            if mask is not None:
                s = s.masked_fill(~mask, float("-inf"))
            pi = torch.exp(s - extras["lse"].unsqueeze(-1))
        return AttentionOutput(y=y, weights=pi, extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        # Same arithmetic as dense; intermediate storage is one state per query.
        return {"flops": 2 * m * n * (d + dv),
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {}   # exact for every block size
