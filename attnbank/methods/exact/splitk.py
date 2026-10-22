"""Split-K / Flash-Decoding: partition the keys, attend independently, merge by LSE.

``sec:exact-tiling`` (decode-specialized kernels).  The context is split into ``p``
chunks, each chunk produces a state ``sigma_{J_t}`` of ``def:exact-state``, and the
states are combined by the monoid of ``prop:exact-monoid``.  Parts (i)-(iii) of that
proposition say the result is independent of the partition *and* of the bracketing,
which is exactly what ``tests/test_exact.py`` checks: every partition, contiguous or
not, gives the identical answer.

Reference: Dao (2023) Flash-Decoding; Sanovar et al. (2024) Lean Attention;
Shyam et al. (2024) Tree Attention.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, causal_mask
from ...registry import register
from .flash import merge_states, online_state


@register
class SplitKAttention(AttentionMethod):
    """Two-level reduction: per-chunk states, then one merge."""

    name = "splitk"
    category = Category.EXACT
    objective = Objective.ACCELERATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("dao2023flash", "sanovar2024lean", "shyam2024tree")
    paper_section = "sec:exact"

    def __init__(self, splits: int = 4, partition=None, tree: bool = False, **kw):
        """``partition``: an explicit list of index tensors/lists; overrides ``splits``.
        ``tree``: merge pairwise (a balanced reduction tree) instead of left-to-right."""
        super().__init__(splits=splits, partition=partition, tree=tree, **kw)
        self.splits = splits
        self.partition = partition
        self.tree = tree

    def _chunks(self, n, device):
        if self.partition is not None:
            return [torch.as_tensor(p, dtype=torch.long, device=device)
                    for p in self.partition if len(p) > 0]
        size = max(1, (n + self.splits - 1) // self.splits)
        idx = torch.arange(n, device=device)
        return [idx[s:s + size] for s in range(0, n, size)]

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        beta = self._scale(d)
        mask = causal_mask(m, n, device=q.device) if causal else None
        s_full = torch.einsum("...md,...nd->...mn", q, k) * beta
        if mask is not None:
            s_full = s_full.masked_fill(~mask, float("-inf"))

        parts = self._chunks(n, q.device)
        states = [online_state(s_full[..., idx], v[..., idx, :]) for idx in parts]
        if self.tree:
            while len(states) > 1:
                nxt = [merge_states(states[i], states[i + 1])
                       for i in range(0, len(states) - 1, 2)]
                if len(states) % 2:
                    nxt.append(states[-1])
                states = nxt
            state = states[0]
        else:
            state = states[0]
            for st in states[1:]:
                state = merge_states(state, st)

        a, ell, u = state
        y = u / ell.unsqueeze(-1).clamp_min(torch.finfo(u.dtype).tiny)
        lse = a + torch.log(ell)
        pi = torch.exp(s_full - lse.unsqueeze(-1)) if return_weights else None
        return AttentionOutput(y=y, weights=pi,
                               extras={"lse": lse, "n_splits": len(parts)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        p = self.splits
        return {"flops": 2 * m * n * (d + dv) + 2 * m * p * dv,
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + 2 * m * p * (dv + 2))
                               * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {}
