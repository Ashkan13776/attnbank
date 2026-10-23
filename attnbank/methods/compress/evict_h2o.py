"""H2O: accumulated-attention heavy-hitter eviction plus a recent window.

``sec:compress-evict``, ``prop:compress-noeviction``.  Eviction is a restriction whose
set is chosen *before* the queries that will use it exist, and the sets are nested:
``S_{t+1} subset S_t union {t+1}``.  A deleted token is gone for every future query, so
``prop:compress-noeviction`` says no query-oblivious rule has a uniform guarantee --
for any retained ``S`` and any evicted ``j``, the query ``q = tau k_j`` drives
``eps_q -> 1`` while ``y_hat`` stays in ``conv{v_i : i in S}``.

H2O estimates the future-query distribution ``nu`` by the empirical distribution of
*past* queries: token ``j`` is scored by ``sum_{t' <= t} pi_{q_{t'}}(j)``, and the
lowest-scoring non-recent token is dropped whenever the cache exceeds its budget.  This
file runs that process literally, one query at a time, which is why it is causal: the
state at step ``t`` depends on keys and queries up to ``t`` only.

In non-causal mode the whole context is admitted at the first step and the same
accumulate-and-evict loop runs over the queries in order; the semantics are then "a
streaming cache served by a fixed context".

Exact limit: ``budget = None`` (no eviction).

Reference: Zhang et al. (2023) H2O; Liu et al. (2023) Scissorhands;
Oren et al. (2024) TOVA.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, masked_softmax, query_positions, scores
from ...registry import register
from ..restrict._restriction import restrict


@register
class H2OAttention(AttentionMethod):
    """Online heavy-hitter eviction with a protected recent window."""

    name = "evict_h2o"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("zhang2023h2o", "liu2023scissorhands", "oren2024transformers")
    paper_section = "sec:compress"

    def __init__(self, budget: int | None = 8, recent: int = 2, **kw):
        """``budget``: cache size; ``recent``: most recent positions never evicted."""
        super().__init__(budget=budget, recent=recent, **kw)
        self.budget, self.recent = budget, recent

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        s = scores(q, k, self._scale(d), causal)
        pi = masked_softmax(s)
        if self.budget is None or self.budget >= n:
            vis = torch.isfinite(s) if causal else torch.ones_like(s, dtype=torch.bool)
            return restrict(pi, vis, v, return_weights, {"budget": self.budget})

        pos = query_positions(m, n, device=q.device)
        idx = torch.arange(n, device=q.device)
        cache = torch.zeros(*q.shape[:-2], n, dtype=torch.bool, device=q.device)
        acc = torch.zeros(*q.shape[:-2], n, dtype=pi.dtype, device=q.device)
        supports = []
        big = torch.finfo(pi.dtype).max

        for i in range(m):
            p = int(pos[i])
            cache = cache | (idx <= p) if causal else cache | torch.ones_like(cache)
            excess = int(cache.sum(-1).amax()) - self.budget
            if excess > 0:                                   # evict the coldest non-recent
                protect = ((idx > p - self.recent) | ~cache) if causal else (~cache)
                score = torch.where(protect, torch.full_like(acc, big), acc)
                # stable sort so ties break by position (oldest first), identically on
                # every device -- otherwise the policy would be device-dependent
                drop = score.sort(dim=-1, stable=True).indices[..., :excess]
                cache = cache.scatter(-1, drop, False)
            sup = cache & torch.isfinite(s[..., i, :]) if causal else cache.clone()
            supports.append(sup)
            row = pi[..., i, :] * sup
            acc = acc + row / row.sum(-1, keepdim=True).clamp_min(1e-30)

        support = torch.stack(supports, dim=-2)
        return restrict(pi, support, v, return_weights,
                        {"budget": self.budget, "recent": self.recent, "accumulated": acc})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        b = min(self.budget or n, n)
        return {"flops": 2 * m * b * (d + dv) + m * b,
                "state_bytes": b * (d + dv) * bytes_per_el + b * 4,   # +accumulated scores
                "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"budget": None}
