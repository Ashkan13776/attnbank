"""TOVA: drop the token the *current* query cares least about, one per step.

``sec:compress-evict``, ``prop:compress-noeviction``.  Reading a decoder as a
multi-state RNN, the KV cache is the state and a cache policy is a rule for keeping it
bounded.  TOVA is the simplest non-trivial such rule: at every step the new token is
admitted, and if the cache is over budget the state with the **smallest weight under
the query that just arrived** is deleted.  No accumulation over history (that is
``evict_h2o``), no protected recent window (that is ``sink_window``), no observation
window (that is ``evict_snapkv``): one query decides, once, and the decision is
permanent.

    S_0 = {0},   S_t = topkeep_{b}( S_{t-1} u {t}; pi_{q_t} ),   S_t \\ {t} subset S_{t-1}

**Semantics: eviction, and the file is built so that this is visible.**  The supports
are produced by a loop over queries in order, so the set serving query ``t`` was fixed
by queries ``<= t`` and nothing later can restore what they threw away.  Two
consequences are tested in ``tests/test_compress_evict.py`` rather than asserted here:

* the supports are nested -- ``S_{t+1} subset S_t u {t+1}``.  ``window`` and
  ``sink_window`` are nested too, which is exactly why they are listed in both
  ``restrict/`` and ``sec:compress-evict``; the *content-dependent* selectors are not,
  because a restriction recomputes its set per query and is free to come back to a
  position it ignored a step ago.  ``topk`` at the same budget is the comparison;
* at equal budget the error against dense softmax is larger than ``topk``'s, and the
  gap *grows along the query axis*: early queries are served by a cache that still
  holds almost everything, late ones by a cache pruned long before their question was
  asked.  That gap is the whole content of ``prop:compress-noeviction`` -- restriction
  hides context, eviction destroys it.

The current position is the one protection the policy keeps, so that no row is ever
left with an empty support (DESIGN.md Sec. 4, rule 10); TOVA keeps it in practice
anyway, since a token's own key usually dominates its row.  ``extras['evicted_at']``
records the step at which each position was deleted, or ``-1`` if it survived, which is
the permanence made inspectable.

In non-causal mode the whole context is admitted at the first step and the same
evict-per-query loop runs over the queries in order: "a streaming cache served by a
fixed context", the convention ``evict_h2o`` also uses.

Exact limit: ``budget = None`` (nothing is ever evicted).

Reference: Oren et al. (2024) Transformers are Multi-State RNNs; Zhang et al. (2023)
H2O.
"""
from __future__ import annotations

import torch

from ...components.restriction import restrict
from ...core import AttentionMethod, Category, Objective, masked_softmax, query_positions, scores
from ...registry import register


@register
class TOVAAttention(AttentionMethod):
    """Greedy one-token-per-step eviction scored by the current query alone."""

    name = "evict_tova"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("oren2024transformers", "zhang2023h2o")
    paper_section = "sec:compress"

    def __init__(self, budget: int | None = 8, **kw):
        """``budget``: the multi-state size, i.e. how many (k, v) pairs survive."""
        super().__init__(budget=budget, **kw)
        self.budget = budget

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
        evicted_at = torch.full(cache.shape, -1, dtype=torch.long, device=q.device)
        supports = []
        big = torch.finfo(pi.dtype).max

        # admit only what has just arrived: re-admitting the whole prefix at every step
        # would silently turn the policy into a per-query restriction, which is the one
        # thing this file exists to be distinguishable from
        prev = -1
        for i in range(m):
            p = int(pos[i])
            if causal:
                cache = cache | ((idx > prev) & (idx <= p))      # only the new tokens
            elif i == 0:
                cache = torch.ones_like(cache)                   # the fixed context
            prev = p
            excess = int(cache.sum(-1).amax()) - self.budget
            if excess > 0:
                # score by THIS query's attention; protect only the current position
                protect = ((idx == p) | ~cache) if causal else (~cache)
                row = pi[..., i, :]
                score = torch.where(protect, torch.full_like(row, big), row)
                # stable sort so ties break by position (oldest first), identically on
                # every device -- otherwise the policy would be device-dependent
                drop = score.sort(dim=-1, stable=True).indices[..., :excess]
                cache = cache.scatter(-1, drop, False)
                evicted_at = evicted_at.scatter(-1, drop, i)
            sup = cache & torch.isfinite(s[..., i, :]) if causal else cache.clone()
            supports.append(sup)

        support = torch.stack(supports, dim=-2)
        return restrict(pi, support, v, return_weights,
                        {"budget": self.budget, "evicted_at": evicted_at})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        b = max(1, min(self.budget or n, n))
        return {"flops": 2 * m * b * (d + dv) + m * b,
                "state_bytes": b * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"budget": None}
