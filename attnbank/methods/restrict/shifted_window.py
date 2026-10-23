"""Swin-style shifted local windows: a *block-diagonal* support, optionally displaced.

``sec:sparse-fixed``.  Unlike the banded support of ``window``, Swin partitions the
sequence into non-overlapping windows of ``w`` positions and lets each query see only
its own window (Liu et al., 2021).  Consecutive layers alternate between the regular
partition and one "displaced from the regularly partitioned windows by
``(floor(w/2), floor(w/2))`` pixels", which is what connects windows that would
otherwise never communicate.  In one dimension this is

    S_p = { j : floor((j + shift)/w) = floor((p + shift)/w) },

a block-diagonal matrix whose blocks are offset by ``shift``.  ``shift = 0`` is the
regular partition -- plain chunked / block-local attention, the pattern of BlockBERT
(Qiu et al., 2020) -- and ``shift = floor(w/2)`` is Swin's displaced partition; the
default follows Swin.  Under ``shift > 0`` the first window is a short one of
``w - shift`` positions, exactly as in the paper's "efficient batch computation".

The difference from ``window`` is the point of having both: at an equal budget ``w``
the banded support is *query-centred* and always keeps the ``w`` nearest keys, while
the block support keeps between ``1`` and ``w`` of them depending on where in its
block the query sits.  The shift is what repairs the queries that land at a block
boundary, and ``extras['window_offset']`` reports the realised distance to the start
of the block so that the effect can be measured rather than argued.

A query is always in its own window, so the diagonal is kept and no row is empty.
Intersecting with the causal mask turns this into *block-causal* attention: query ``p``
sees the earlier positions of its own block only. Values are untouched, so the error
is ``eps_q (m_{S^c} - m_S)`` (``prop:sparse-identity``); ``w = None`` is a single
window over the whole sequence and recovers dense softmax exactly.

Reference: Liu et al. (2021) Swin Transformer; Qiu et al. (2020) BlockBERT.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, query_positions
from ...registry import register
from ._restriction import reference_pi, restrict, visible


@register
class ShiftedWindowAttention(AttentionMethod):
    """Non-overlapping local windows of w positions, displaced by shift (Swin)."""

    name = "shifted_window"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("liu2021swin", "qiu2020blockwise")
    paper_section = "sec:sparse"

    def __init__(self, w: int | None = 4, shift: int | None = None, **kw):
        """``w``: window size (``None`` = one window, the exact limit).

        ``shift``: displacement of the partition; ``None`` means Swin's
        ``floor(w/2)``, ``0`` is the regular (block-diagonal) partition.
        """
        super().__init__(w=w, shift=shift, **kw)
        if w is not None and w < 1:
            raise ValueError(f"w must be >= 1 or None; got {w}")
        self.w, self.shift = w, shift

    def resolved_shift(self) -> int:
        """The displacement actually used: ``floor(w/2)`` when ``shift`` is ``None``."""
        if self.w is None:
            return 0
        return self.w // 2 if self.shift is None else self.shift % self.w

    def window_index(self, pos: torch.Tensor) -> torch.Tensor:
        """Which window each position belongs to; constant when ``w`` is ``None``."""
        if self.w is None:
            return torch.zeros_like(pos)
        return (pos + self.resolved_shift()) // self.w

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        pi, s = reference_pi(self, q, k, causal)
        p = query_positions(m, n, device=q.device)
        j = torch.arange(n, device=q.device)
        qi, ki = self.window_index(p)[:, None], self.window_index(j)[None, :]
        sup = visible(q, k, causal) & (qi == ki)
        offset = p + self.resolved_shift() - qi[:, 0] * (self.w or max(n, 1))
        return restrict(pi, sup, v, return_weights,
                        {"window_size": self.w, "shift": self.resolved_shift(),
                         "n_windows": int(ki.max()) + 1, "window_offset": offset})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        kept = min(n, self.w or n)
        return {"flops": 2 * m * kept * (d + dv),
                "state_bytes": kept * (d + dv) * bytes_per_el,   # one resident window
                "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        # a single window covering the sequence: the support is the whole visible row
        return {"w": None}
