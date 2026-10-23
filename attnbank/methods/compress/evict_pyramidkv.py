"""PyramidKV: the eviction budget is allocated across layers, not uniformly.

``sec:compress-evict``, ``prop:compress-noeviction``.  SnapKV and H2O give every layer
the same cache size.  PyramidKV observes that attention concentrates as depth grows --
lower layers spread mass broadly, upper layers funnel it onto a few tokens -- and so
spends the same *total* budget unevenly: the per-layer budgets form a decreasing
arithmetic sequence, wide at the bottom and narrow at the top.

    base   = budget - recent                      # what a uniform scheme would give
    lo     = base // beta,   hi = 2 base - lo     # the ends of the sequence
    b^(l)  = hi - l (hi - lo) / (L - 1) + recent

The mean of the sequence is ``base``, so the allocation *moves* cache between layers
rather than buying more of it; ``pyramid_budgets`` returns the whole schedule and
``tests/test_compress_evict.py`` checks the conservation.  The official implementation
uses integer floor division for the step, which loses up to one token per layer; the
exact arithmetic sequence is used here so that the conservation is a statement and not
an approximation.

A single attention call has no notion of depth, so the layer is a config key:
``ab.build("evict_pyramidkv", layer=l, n_layers=L)`` is layer ``l`` of the stack, and a
sweep over ``l`` is what exhibits the pyramid.  Within a layer the selection is
SnapKV's -- the last ``obs`` queries score the prompt tokens, a 1-D pooling of width
``pool`` keeps neighbourhoods rather than isolated tokens, and the recent window is
always resident.  ``pooling='max'`` is the default here, which is what PyramidKV and
SnapKV's released code use; ``evict_snapkv`` in this package uses average pooling, so
the pair isolates the effect of the pooling operator.

**Semantics: eviction, not restriction.**  The support is chosen once, from the
observation window, and the *same* support then serves every query -- including the
queries that are not in that window and whose own top-``b`` is a different set.  The
context those queries would have needed is gone, which is why the method declares
``supports_causal = False``: query 0's support depends on query ``m-1``, and no causal
mask can express that.  ``tests/test_compress_evict.py`` measures the gap against the
per-query restriction ``topk`` at the same budget; it is the price of deleting rather
than hiding, and ``prop:compress-noeviction`` says no budget makes it uniformly small.

Exact limit: ``budget = None`` (every layer keeps everything).

Reference: Cai et al. (2024) PyramidKV; Li et al. (2024) SnapKV.
"""
from __future__ import annotations

import torch

from ...components.restriction import restrict
from ...core import AttentionMethod, Category, Objective, masked_softmax, scores
from ...errors import ConfigError
from ...registry import register


def observation_scores(pi, obs: int, pool: int, pooling: str = "max"):
    """SnapKV's prompt-token score: the last ``obs`` queries' attention, pooled.

    ``pi`` is ``[..., m, n]``.  The score of key ``j`` is ``sum_{i >= m - obs} pi_i(j)``,
    then smoothed along the key axis by a width-``pool`` max (SnapKV's and PyramidKV's
    released default) or average pooling.  The pooling is what keeps a *neighbourhood*
    of a heavy hitter rather than the single token that happened to peak, and is the
    only difference between ``evict_snapkv`` and the SnapKV in this file.
    """
    n = pi.shape[-1]
    obs = max(1, min(obs, pi.shape[-2]))
    raw = pi[..., -obs:, :].sum(-2)                                   # [..., n]
    if pool is None or pool <= 1:
        return raw
    flat = raw.reshape(-1, 1, n)
    if pooling == "max":
        out = torch.nn.functional.max_pool1d(flat, kernel_size=pool, stride=1,
                                             padding=pool // 2)
    elif pooling == "avg":
        out = torch.nn.functional.avg_pool1d(flat, kernel_size=pool, stride=1,
                                             padding=pool // 2, count_include_pad=False)
    else:
        raise ConfigError(f"pooling must be 'max' or 'avg'; got {pooling!r}")
    return out.reshape(*raw.shape[:-1], -1)[..., :n]


def pyramid_budgets(budget: int | None, n_layers: int, recent: int = 0,
                    beta: int = 20, n: int | None = None) -> list[int]:
    """The per-layer cache sizes of ``eq`` above: a decreasing arithmetic sequence.

    ``budget`` is what a *uniform* scheme would give every layer, so the mean of the
    returned schedule is ``budget`` (up to rounding) whatever ``beta`` is.  ``beta``
    controls the steepness: ``beta = 1`` is uniform, large ``beta`` starves the top.
    ``n``, when given, caps the bottom layer at the context length and compensates at
    the top, exactly as the released code does.
    """
    if budget is None:
        return [None] * n_layers
    base = max(0, budget - recent)
    lo = base // max(1, beta)
    hi = 2 * base - lo
    if n is not None and hi > n - recent:
        hi = max(1, n - recent)
        lo = min(hi, max(0, 2 * base - hi))
    if n_layers <= 1:
        return [base + recent]
    step = (hi - lo) / (n_layers - 1)
    return [int(round(hi - layer * step)) + recent for layer in range(n_layers)]


@register
class PyramidKVAttention(AttentionMethod):
    """Layer-wise budget allocation on top of SnapKV's pooled observation-window scores."""

    name = "evict_pyramidkv"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False          # one support, chosen at the end of the prompt
    expresses_weights = True
    row_stochastic = True
    references = ("cai2024pyramidkv", "li2024snapkv")
    paper_section = "sec:compress"

    def __init__(self, budget: int | None = 8, n_layers: int = 4, layer: int = 0,
                 beta: int = 4, obs: int = 4, pool: int = 3, pooling: str = "max",
                 recent: int = 2, **kw):
        """``budget`` is the *uniform* budget the pyramid redistributes; ``layer`` of
        ``n_layers`` picks this call's share of it."""
        super().__init__(budget=budget, n_layers=n_layers, layer=layer, beta=beta,
                         obs=obs, pool=pool, pooling=pooling, recent=recent, **kw)
        if not 0 <= layer < n_layers:
            raise ConfigError(f"layer must be in [0, n_layers); got layer={layer}, "
                              f"n_layers={n_layers}")
        self.budget, self.n_layers, self.layer, self.beta = budget, n_layers, layer, beta
        self.obs, self.pool, self.pooling, self.recent = obs, pool, pooling, recent

    def _layer_budgets(self, n):
        return pyramid_budgets(self.budget, self.n_layers, self.recent, self.beta, n)

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        pi = masked_softmax(scores(q, k, self._scale(d), causal))
        schedule = self._layer_budgets(n)
        budget = schedule[self.layer]
        full = torch.ones(*q.shape[:-2], m, n, dtype=torch.bool, device=q.device)
        if budget is None or budget >= n:
            return restrict(pi, full, v, return_weights,
                            {"budget": budget, "layer_budgets": schedule})

        pooled = observation_scores(pi, self.obs, self.pool, self.pooling)
        recent = torch.arange(n, device=q.device) >= n - self.recent
        take = max(0, budget - int(recent.sum()))
        sel = torch.where(recent, torch.full_like(pooled, float("-inf")), pooled)
        keep = recent.expand_as(sel).clone()
        if take > 0:
            # a stable descending sort, not topk: max pooling creates plateaus of
            # exactly equal scores and topk breaks those ties differently on CPU and
            # CUDA, which would make the policy device-dependent
            chosen = sel.sort(dim=-1, descending=True, stable=True).indices[..., :take]
            keep = keep | torch.zeros_like(sel, dtype=torch.bool).scatter_(
                -1, chosen, True)
        support = keep.unsqueeze(-2).expand(*q.shape[:-2], m, n)
        return restrict(pi, support, v, return_weights,
                        {"budget": budget, "layer_budgets": schedule,
                         "pooled_scores": pooled, "obs": min(self.obs, m)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        b = self._layer_budgets(n)[self.layer]
        b = max(1, min(b if b is not None else n, n))
        return {"flops": 2 * m * b * (d + dv) + self.obs * n * d,
                "state_bytes": b * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"budget": None}
