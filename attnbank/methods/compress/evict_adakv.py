"""Ada-KV: one eviction budget for the whole layer, allocated across heads.

``sec:compress-evict``, ``prop:compress-noeviction``.  SnapKV, H2O and PyramidKV give
every head of a layer the same cache size.  Ada-KV observes that heads differ enormously
in how concentrated their attention is -- a head whose mass sits on three tokens wastes a
budget of sixty, while a diffuse head is crippled by it -- and that the top-``k``
eviction rule is itself the minimiser of an upper bound on the output perturbation.
Allocating the layer's *total* budget ``H b`` so as to minimise that bound jointly over
heads turns out to be a single global selection:

    take the ``H b`` largest observation scores over the concatenation of all heads,
    let ``f_h`` be how many of them belong to head ``h``, and give head ``h`` a cache
    of size ``f_h``.

The released code then blends that with the uniform allocation through a safeguard
``alpha``, ``b_h = round((1 - alpha) f_h + alpha b)``, which stops a head from being
starved to the point where the bound's assumptions stop holding.  ``alpha = 1`` is
uniform allocation and recovers SnapKV head for head; ``alpha = 0`` is the raw
argmin.  Either way ``sum_h b_h = H b`` up to rounding: the budget is *moved* between
heads, never increased, and ``tests/test_compress_evict.py`` checks it.

**Heads are coupled, and that is the mechanism.**  The method declares
``head_independent = False``: perturbing head 1's keys moves head 2's cache size,
because they are drawing on one pool.  This is the same kind of honest declaration
``gqa`` and ``mla`` make, for a different reason -- they share the stored K/V, Ada-KV
shares the *budget*.

**Semantics: eviction, not restriction.**  As in SnapKV the support is fixed once from
the observation window and then serves every query, so a query outside that window is
answered from a cache that was pruned without knowing what it would ask.  Hence
``supports_causal = False``, and hence ``prop:compress-noeviction``: adaptive allocation
buys a better *average*, not a uniform guarantee.

Exact limit: ``budget = None`` (nothing is evicted).

Reference: Feng et al. (2025) Ada-KV; Li et al. (2024) SnapKV.
"""
from __future__ import annotations

import torch

from ...components.restriction import restrict
from ...core import AttentionMethod, Category, Objective, masked_softmax, scores
from ...errors import ConfigError
from ...registry import register
from .evict_pyramidkv import observation_scores


def head_budgets(score, base: int, floor_alpha: float) -> torch.Tensor:
    """Ada-KV's allocation: ``[B, H]`` cache sizes from ``[B, H, n]`` scores.

    ``score`` must already have the always-resident positions set to ``-inf`` so that
    they cannot consume the adaptive pool.  The global top-``H base`` over the flattened
    ``[H, n]`` scores is taken per batch element -- never across the batch, which would
    make one sequence's allocation depend on another's.
    """
    b, h, n = score.shape
    usable = int((score > float("-inf")).sum(-1).amin())
    total = min(h * base, h * usable)
    budgets = torch.full((b, h), float(base), device=score.device, dtype=torch.float64)
    if total > 0:
        # stable sort, not topk: pooled scores tie on plateaus and the tie-break must
        # not depend on the device (see evict_pyramidkv)
        idx = score.reshape(b, h * n).sort(
            dim=-1, descending=True, stable=True).indices[:, :total]        # [B, total]
        owner = torch.div(idx, n, rounding_mode="floor")                   # which head
        counts = torch.zeros(b, h, device=score.device, dtype=torch.float64)
        counts.scatter_add_(1, owner, torch.ones_like(owner, dtype=torch.float64))
        budgets = counts
    blended = (1.0 - floor_alpha) * budgets + floor_alpha * float(base)
    return blended.round().long().clamp_(0, usable)


@register
class AdaKVAttention(AttentionMethod):
    """Head-wise adaptive budget allocation over SnapKV-style observation scores."""

    name = "evict_adakv"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False          # one support per head, chosen at the end of the prompt
    expresses_weights = True
    row_stochastic = True
    head_independent = False         # heads draw on one budget pool -- the mechanism
    references = ("feng2025adakv", "li2024snapkv")
    paper_section = "sec:compress"

    def __init__(self, budget: int | None = 8, floor_alpha: float = 0.2, obs: int = 4,
                 pool: int = 3, pooling: str = "max", recent: int = 2, **kw):
        """``budget`` is the per-head *average*; ``floor_alpha = 1`` recovers SnapKV."""
        super().__init__(budget=budget, floor_alpha=floor_alpha, obs=obs, pool=pool,
                         pooling=pooling, recent=recent, **kw)
        if not 0.0 <= floor_alpha <= 1.0:
            raise ConfigError(f"floor_alpha must be in [0, 1]; got {floor_alpha}")
        self.budget, self.floor_alpha = budget, floor_alpha
        self.obs, self.pool, self.pooling, self.recent = obs, pool, pooling, recent

    def _forward(self, q, k, v, causal, return_weights):
        b, h, m, d = q.shape
        n = k.shape[-2]
        pi = masked_softmax(scores(q, k, self._scale(d), causal))
        full = torch.ones(b, h, m, n, dtype=torch.bool, device=q.device)
        if self.budget is None or self.budget >= n:
            return restrict(pi, full, v, return_weights, {"budget": self.budget})

        pooled = observation_scores(pi, self.obs, self.pool, self.pooling)     # [B,H,n]
        resident = torch.arange(n, device=q.device) >= n - self.recent
        sel = pooled.masked_fill(resident, float("-inf")).double()
        base = max(0, self.budget - int(resident.sum()))
        per_head = head_budgets(sel, base, self.floor_alpha)                   # [B,H]

        order = sel.argsort(dim=-1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(-1, order, torch.arange(n, device=q.device).expand_as(order))
        keep = (rank < per_head.unsqueeze(-1)) | resident                      # [B,H,n]
        support = keep.unsqueeze(-2).expand(b, h, m, n)
        return restrict(pi, support, v, return_weights,
                        {"budget": self.budget, "head_budget": per_head + int(resident.sum()),
                         "adaptive_budget": per_head, "pooled_scores": pooled,
                         "obs": min(self.obs, m)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        b = max(1, min(self.budget or n, n))
        return {"flops": 2 * m * b * (d + dv) + self.obs * n * d,
                "state_bytes": b * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"budget": None}
