"""SnapKV: prompt-token scores pooled over an observation window.

``sec:compress-evict`` ("Estimating importance from past queries").  SnapKV scores
prompt tokens with the attention of the *last* ``obs`` prompt queries, applies a 1-D
pooling of width ``pool`` along the key axis so that whole local neighbourhoods are
kept rather than isolated tokens, keeps the top ``budget - recent`` of them, and adds
the recent window.  The compressed cache then serves all subsequent decoding queries.

This works when the question ends the prompt -- the observation window is then a good
sample of ``nu`` in ``eq:compress-evict-objective`` -- and degrades when it arrives
earlier.

**Not causal, deliberately.**  The support is chosen after seeing the whole prompt and
is then applied to every query, including earlier ones.  That is what SnapKV does in a
prefill-then-decode pipeline, and it cannot be expressed as a causally masked single
forward pass: query 0's support would depend on query ``m-1``.  Declaring
``supports_causal = False`` records the fact rather than hiding it behind a mask.

Exact limit: ``budget = None``.

Reference: Li et al. (2024) SnapKV; Cai et al. (2024) PyramidKV.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, masked_softmax, scores
from ...registry import register
from ..restrict._restriction import restrict


@register
class SnapKVAttention(AttentionMethod):
    """Observation-window scoring with pooled, budgeted selection."""

    name = "evict_snapkv"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False
    expresses_weights = True
    row_stochastic = True
    references = ("li2024snapkv", "cai2024pyramidkv")
    paper_section = "sec:compress"

    def __init__(self, budget: int | None = 8, obs: int = 4, pool: int = 3,
                 recent: int = 2, **kw):
        super().__init__(budget=budget, obs=obs, pool=pool, recent=recent, **kw)
        self.budget, self.obs, self.pool, self.recent = budget, obs, pool, recent

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        pi = masked_softmax(scores(q, k, self._scale(d), causal))
        full = torch.ones(*q.shape[:-2], m, n, dtype=torch.bool, device=q.device)
        if self.budget is None or self.budget >= n:
            return restrict(pi, full, v, return_weights, {"budget": self.budget})

        obs = min(self.obs, m)
        raw = pi[..., -obs:, :].mean(-2)                                  # [..., n]
        pooled = torch.nn.functional.avg_pool1d(
            raw.reshape(-1, 1, n), kernel_size=self.pool, stride=1,
            padding=self.pool // 2, count_include_pad=False).reshape(raw.shape)[..., :n]

        recent = torch.arange(n, device=q.device) >= n - self.recent
        take = max(0, min(self.budget, n) - int(recent.sum()))
        sel = torch.where(recent, torch.full_like(pooled, float("-inf")), pooled)
        keep = recent.expand_as(sel).clone()
        if take > 0:
            keep = keep | torch.zeros_like(sel, dtype=torch.bool).scatter_(
                -1, sel.topk(take, dim=-1).indices, True)
        support = keep.unsqueeze(-2).expand(*q.shape[:-2], m, n)
        return restrict(pi, support, v, return_weights,
                        {"budget": self.budget, "pooled_scores": pooled, "obs": obs})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        b = min(self.budget or n, n)
        return {"flops": 2 * m * b * (d + dv) + self.obs * n * d,
                "state_bytes": b * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"budget": None}
