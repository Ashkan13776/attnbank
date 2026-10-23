"""Grouped-query attention: K and V shared across groups of query heads.

``sec:compress-arch``, ``sec:exact-roofline``.  Query heads are partitioned into groups
of ``group_size``; the heads of a group share one key--value pair, here obtained by
mean-pooling the group's ``K`` and ``V`` (the conversion recipe of Ainslie et al.,
2023).  ``group_size = 1`` is multi-head attention, ``group_size = h`` is multi-query.

This is an **operator modification** (``tab:comparative``: "MQA / GQA / MLA ... mod."),
trained as such; measuring it against a multi-head reference answers a conversion
question, not an approximation question.  Its purpose is the roofline: by
``eq:exact-decode-intensity`` the decode arithmetic intensity is ``(16/b)(h/h_kv)``, so
sharing KV is one of the very few interventions that *raises* it.  ``cost()`` divides
``state_bytes`` and the KV traffic by ``group_size`` accordingly.

**Heads are deliberately coupled**, so ``head_independent = False``: perturbing head
``h`` changes every head in its group and nothing outside it.  ``tests/test_compress.py``
checks exactly that boundary.

Exact limit: ``group_size = 1``.

Reference: Ainslie et al. (2023) GQA; Shazeer (2019) MQA.
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     masked_softmax, scores)
from ...components.sdpa import sdpa_attend
from ...registry import register


@register
class GQAAttention(AttentionMethod):
    """Mean-pool K and V within groups of query heads."""

    name = "gqa"
    category = Category.COMPRESS
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    head_independent = False
    references = ("ainslie2023gqa", "shazeer2019fast")
    paper_section = "sec:compress"

    backends = ("reference", "sdpa")

    def __init__(self, group_size: int = 2, sdpa_kernel: str | None = None, **kw):
        super().__init__(group_size=group_size, sdpa_kernel=sdpa_kernel, **kw)
        self.group_size = group_size
        self.sdpa_kernel = sdpa_kernel

    def _pool(self, k, v):
        """Mean-pool K and V within groups of query heads; the whole of the method."""
        h = k.shape[-3]
        g = max(1, min(self.group_size, h))
        if h % g:
            raise ValueError(f"gqa: group_size {g} must divide the head count {h}")
        if g == 1:
            return k, v, g
        pool = lambda t: t.unflatten(-3, (h // g, g)).mean(-3, keepdim=True).expand(
            *t.shape[:-3], h // g, g, *t.shape[-2:]).flatten(-4, -3)
        return pool(k), pool(v), g

    def _forward(self, q, k, v, causal, return_weights):
        h = q.shape[-3]
        k_bar, v_bar, g = self._pool(k, v)
        pi = masked_softmax(scores(q, k_bar, self._scale(q.shape[-1]), causal))
        return AttentionOutput(y=pi @ v_bar, weights=pi if return_weights else None,
                               v_hat=v_bar, extras={"group_size": g, "n_kv_heads": h // g})

    def _forward_sdpa(self, q, k, v, causal, return_weights):
        """Pool first, then the fused kernel: the pooling *is* the method, not the kernel."""
        if return_weights:
            return self._fallback_to_reference(
                "materialise the attention weights or v_hat")(q, k, v, causal,
                                                              return_weights)
        h = q.shape[-3]
        k_bar, v_bar, g = self._pool(k, v)
        y = sdpa_attend(q, k_bar, v_bar, scale=self._scale(q.shape[-1]), causal=causal,
                        kernel=self.sdpa_kernel)
        return AttentionOutput(y=y, extras={"group_size": g, "n_kv_heads": h // g,
                                            "backend": "sdpa"})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        g = max(1, self.group_size)
        return {"flops": 2 * m * n * (d + dv),
                "state_bytes": n * (d + dv) * bytes_per_el / g,
                "bytes_moved": m * (d + dv) * bytes_per_el + n * (d + dv) * bytes_per_el / g}

    @classmethod
    def limit_config(cls):
        return {"group_size": 1}
