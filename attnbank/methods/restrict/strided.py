"""Sparse Transformer: the *strided* and *fixed* factorized patterns.

``sec:sparse-fixed``.  Child et al. (2019, Sec. 4.3) factor the attention matrix into
two heads whose index sets are, for query ``i`` and stride ``l``,

    strided:   A_i^(1) = {t, ..., i},  t = max(0, i - l)
               A_i^(2) = {j : (i - j) mod l = 0}

    fixed:     A_i^(1) = {j : floor(j/l) = floor(i/l)}
               A_i^(2) = {j : j mod l in {l - c, ..., l - 1}}

with ``l ~ sqrt(n)`` and ``c`` a small integer (8, 16 or 32 in the paper, for
``l = 128`` or ``256``).  The strided pattern suits data with a natural period -- it
is the one-dimensional reading of attending to the previous row and the same column
of an image -- and the fixed pattern suits data without one, where "specific cells
summarize previous locations and propagate that information to all future cells".

This method takes the **union** ``A_i^(1) u A_i^(2)``, which is what a single
restricted layer can see; the paper's own alternative of giving one head each is a
statement about *depth*, and the bank's unit of measurement is one operator.  Both
components are stated relative to the query position and then intersected with the
mask, so the two-sided reading used in a bidirectional model collapses to Child et
al.'s autoregressive sets under ``causal=True`` and nothing in the future is ever
reachable.

Offset ``0`` is in both patterns (``j = i`` satisfies ``(i - j) mod l = 0`` and lies in
the query's own block), so every row keeps at least the diagonal.  ``stride = 1``
makes ``A^(2)`` the whole row under either pattern, which is the exact limit.

Values are untouched: the error is ``eps_q (m_{S^c} - m_S)`` (``prop:sparse-identity``).
Against ``longformer`` this isolates what a *periodic* support buys over a contiguous
one at the same budget -- the two keep ``O(l + n/l)`` and ``O(w)`` keys respectively.

Reference: Child et al. (2019) Generating Long Sequences with Sparse Transformers.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, query_positions
from ...registry import register
from ._restriction import reference_pi, restrict, visible

PATTERNS = ("strided", "fixed")


@register
class StridedAttention(AttentionMethod):
    """The union of the two factorized index sets of a Sparse Transformer head."""

    name = "strided"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("child2019generating",)
    paper_section = "sec:sparse"

    def __init__(self, stride: int = 4, pattern: str = "strided", c: int = 1, **kw):
        """``stride``: the paper's ``l``.  ``pattern``: ``'strided'`` or ``'fixed'``.

        ``c``: the width of the summarising column of the ``'fixed'`` pattern; ignored
        by ``'strided'``.
        """
        super().__init__(stride=stride, pattern=pattern, c=c, **kw)
        if stride < 1:
            raise ValueError(f"stride must be >= 1; got {stride}")
        if pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}; got {pattern!r}")
        if c < 0:
            raise ValueError(f"c must be >= 0; got {c}")
        self.stride, self.pattern, self.c = stride, pattern, c

    def pattern_support(self, m, n, device=None):
        """The two index sets of the factorization, as boolean ``[m, n]``: ``(A1, A2)``."""
        l = self.stride
        p = query_positions(m, n, device=device)[:, None]
        j = torch.arange(n, device=device)[None, :]
        if self.pattern == "strided":
            local = (p - j).abs() <= l                      # A^(1): the l nearest
            periodic = (p - j) % l == 0                     # A^(2): same residue class
        else:
            local = (j // l) == (p // l)                    # A^(1): the query's own block
            periodic = (j % l) >= (l - self.c)              # A^(2): the summarising columns
        return local, periodic

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        pi, s = reference_pi(self, q, k, causal)
        local, periodic = self.pattern_support(m, n, q.device)
        sup = visible(q, k, causal) & (local | periodic)
        return restrict(pi, sup, v, return_weights,
                        {"pattern": self.pattern, "stride": self.stride, "c": self.c,
                         "n_local": local.sum(-1), "n_periodic": periodic.sum(-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        l, blocks = self.stride, (n + self.stride - 1) // self.stride
        # strided: l + 1 local and one key per block; fixed: one block and c per block
        kept = min(n, l + 1 + blocks if self.pattern == "strided"
                   else l + max(self.c, 1) * blocks)
        return {"flops": 2 * m * kept * (d + dv),
                "state_bytes": n * (d + dv) * bytes_per_el,   # the column heads reach back
                "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        # stride 1 makes the residue class (or the summarising column) the whole row
        return {"stride": 1}
