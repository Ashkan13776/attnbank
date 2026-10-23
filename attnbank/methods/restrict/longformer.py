"""Longformer: a *dilated* sliding window plus task-global tokens.

``sec:sparse-fixed``.  Longformer's support is the union of three query-independent
pieces (Beltagy et al., 2020, Sec. 3):

* **sliding window** -- "given a fixed window size ``w``, each token attends to
  ``w/2`` tokens on each side";
* **dilated sliding window** -- the window is given "gaps of size dilation ``d``", so
  ``l`` stacked layers reach ``l * d * w``.  The paper uses *different dilations on
  different heads*, "allowing some heads without dilation to focus on local context,
  while others with dilation focus on longer context"; ``head_dilation=True``
  reproduces that;
* **global attention** -- "a token with a global attention attends to all tokens
  across the sequence, and all tokens in the sequence attend to it".  The pattern is
  therefore *not* symmetric in the two directions of the restriction: a global
  position widens every row, and its own row is dense.

Written in the paper's notation, query ``p`` keeps

    S_p = { p + i*d : -w/2 <= i <= w/2 }  u  { 0, ..., n_global - 1 },

and ``S_p`` is the whole visible row when ``p < n_global``.  Offset ``i = 0`` is always
present, so no row is ever empty; the support is then intersected with the causal
mask, which both removes the right half of the window (Longformer's own
autoregressive configuration) and makes the global prefix behave like the sinks of
``sink_window`` -- except that the global rows stay dense, which is the whole
difference between a *sink* and a *global token*.

The error is purely distributional: ``y - y_hat = eps_p (m_{S^c} - m_S)``
(``prop:sparse-identity``), with ``eps_p`` returned in ``extras``.

Dilation alone (``w=None, dilation=d``) is the pure "strided column" pattern that
LongNet later scales up with exponentially growing dilation rates.

Reference: Beltagy et al. (2020) Longformer; Child et al. (2019); Ding et al. (2023)
LongNet.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, query_positions
from ...registry import register
from ._restriction import reference_pi, restrict, visible


def dilated_support(m, n, w, dilation, device):
    """Boolean ``[m, n]``: ``w//2`` attended positions per side, spaced ``dilation`` apart.

    ``w`` is Longformer's window size, so the query itself plus ``w//2`` dilated
    neighbours on each side; the reach is ``(w//2) * dilation`` positions.  ``w=None``
    drops the width limit, leaving only the stride condition; with ``dilation=1`` that
    is every position, which is the exact limit of the method.
    """
    p = query_positions(m, n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    off = (p - j).abs()
    sup = off % dilation == 0
    if w is not None:
        sup = sup & (off <= (w // 2) * dilation)
    return sup


@register
class LongformerAttention(AttentionMethod):
    """Dilated sliding window of size w (w//2 positions per side), plus global tokens."""

    name = "longformer"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("beltagy2020longformer", "child2019generating", "ding2023longnet")
    paper_section = "sec:sparse"

    def __init__(self, w: int | None = 8, dilation: int = 1, n_global: int = 0,
                 head_dilation: bool = False, **kw):
        """``w``: Longformer's window size -- ``w//2`` per side (``None`` = unbounded).

        ``dilation``: gap between attended positions.  ``head_dilation``: give head
        ``h`` the dilation ``1 + (h % dilation)``, so head 0 stays local while later
        heads reach further -- Longformer's per-head configuration.  ``n_global``: the
        number of leading positions that are globally attended *and* attend globally.
        """
        super().__init__(w=w, dilation=dilation, n_global=n_global,
                         head_dilation=head_dilation, **kw)
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1; got {dilation}")
        if w is not None and w < 1:
            raise ValueError(f"w must be >= 1 or None; got {w}")
        self.w, self.dilation = w, dilation
        self.n_global, self.head_dilation = n_global, head_dilation

    def head_dilations(self, h: int) -> list[int]:
        """The dilation used by each of the ``h`` heads."""
        if not self.head_dilation:
            return [self.dilation] * h
        return [1 + (i % self.dilation) for i in range(h)]

    def _pattern(self, q, k):
        """Boolean ``[m, n]`` or ``[h, m, n]`` support before masking."""
        m, n, h = q.shape[-2], k.shape[-2], q.shape[-3]
        dils = self.head_dilations(h)
        if len(set(dils)) == 1:
            sup = dilated_support(m, n, self.w, dils[0], q.device)
        else:
            sup = torch.stack([dilated_support(m, n, self.w, d, q.device) for d in dils])
        if self.n_global > 0:
            j = torch.arange(n, device=q.device)
            p = query_positions(m, n, device=q.device)
            sup = sup | (j < self.n_global)                       # attended by everyone
            sup = sup | (p < self.n_global)[:, None]              # global rows are dense
        return sup

    def _forward(self, q, k, v, causal, return_weights):
        pi, s = reference_pi(self, q, k, causal)
        sup = visible(q, k, causal) & self._pattern(q, k)
        return restrict(pi, sup, v, return_weights,
                        {"dilation": self.dilation, "n_global": self.n_global,
                         "head_dilations": self.head_dilations(q.shape[-3])})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        kept = min(n, (self.w + 1 if self.w is not None else n) + self.n_global)
        return {"flops": 2 * m * kept * (d + dv),
                "state_bytes": kept * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        # an unbounded window with no dilation keeps every visible key
        return {"w": None, "dilation": 1}
