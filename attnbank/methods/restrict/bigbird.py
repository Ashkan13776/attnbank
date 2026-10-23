"""BigBird: window blocks + global blocks + *random* blocks, at block granularity.

``sec:sparse-fixed``.  Zaheer et al. (2020) build the support as a graph on blocks of
``block`` consecutive positions and take the union of three families:

* **window** -- the ``window_blocks`` blocks centred on the query's own block
  (``w/2`` neighbouring blocks on each side);
* **global** -- the first ``global_blocks`` blocks, in the *internal* (ITC)
  construction: a global token has ``A(i,:) = 1`` and ``A(:,i) = 1``, so it is
  attended by every query **and** its own row is dense;
* **random** -- ``random_blocks`` blocks drawn uniformly, without replacement, from
  the blocks that are neither window nor global.  This is the piece that makes the
  pattern a (sparse) expander and gives the paper its reachability argument; it is
  also the only randomness in the family, so it is drawn through a seeded
  ``torch.Generator`` and is reproducible across builds, devices and dtypes.

The draw depends on the query *block index*, the sequence length and ``causal`` --
nothing else.  It is shared across batch and head, which keeps the pattern a property
of the geometry rather than of the data; per-head draws would change nothing that the
library measures.

**Causality.**  Under a causal mask the candidate set for query block ``pb`` is
restricted to blocks strictly before the window, so no random block can ever sit in
the future; the support is intersected with the causal mask afterwards in any case.
The query's own block is always in the window (``window_blocks`` down to 1 still keeps
it), so the diagonal is retained and no row is empty.

BigBird is an encoder pattern; as everywhere in ``restrict/`` the values are untouched,
so the error is exactly ``eps_q (m_{S^c} - m_S)`` of ``prop:sparse-identity``.  Put
beside ``sink_window`` it isolates what the *random* blocks buy: the same budget spent
on uniformly scattered mass instead of on the prefix.

Reference: Zaheer et al. (2020) Big Bird.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, make_generator, query_positions
from ...registry import register
from ._restriction import reference_pi, restrict, visible


@register
class BigBirdAttention(AttentionMethod):
    """Block-sparse window + global + random support, with a seeded random draw."""

    name = "bigbird"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("zaheer2020bigbird",)
    paper_section = "sec:sparse"

    def __init__(self, block: int = 4, window_blocks: int | None = 3,
                 random_blocks: int = 2, global_blocks: int = 1, seed: int = 0, **kw):
        """``window_blocks``: blocks centred on the query's own (``None`` = all, the limit).

        ``random_blocks`` blocks are drawn per query block; ``global_blocks`` leading
        blocks are global in the ITC sense.  ``block`` is the block size (64 in the
        paper; small here because the bank's test shapes are small).
        """
        super().__init__(block=block, window_blocks=window_blocks,
                         random_blocks=random_blocks, global_blocks=global_blocks,
                         seed=seed, **kw)
        if block < 1:
            raise ValueError(f"block must be >= 1; got {block}")
        self.block, self.window_blocks = block, window_blocks
        self.random_blocks, self.global_blocks, self.seed = random_blocks, global_blocks, seed

    def block_pattern(self, n: int, causal: bool, device=None) -> torch.Tensor:
        """Boolean ``[nb, nb]``: which key blocks query block ``pb`` keeps.

        The three families of the paper, in order: window, global column, global row,
        random.  Returned separately from ``_forward`` so the geometry can be inspected
        and tested without running attention.
        """
        nb = (n + self.block - 1) // self.block
        pb = torch.arange(nb, device=device)[:, None]
        kb = torch.arange(nb, device=device)[None, :]

        if self.window_blocks is None:
            return torch.ones(nb, nb, dtype=torch.bool, device=device)

        half = max(self.window_blocks, 1) // 2
        sel = (kb - pb).abs() <= half                       # window
        sel = sel | (kb < self.global_blocks)               # global columns
        sel = sel | (pb < self.global_blocks)               # global rows (ITC)

        if self.random_blocks > 0:
            cand = ((kb - pb).abs() > half) & (kb >= self.global_blocks)
            if causal:
                cand = cand & (kb < pb)
            u = torch.rand(nb, nb, generator=make_generator(self.seed)).to(device)
            u = u.masked_fill(~cand, float("-inf"))
            take = min(self.random_blocks, nb)
            drawn = torch.zeros_like(cand).scatter_(
                -1, u.topk(take, dim=-1).indices, True) & cand
            sel = sel | drawn
        return sel

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        pi, s = reference_pi(self, q, k, causal)
        vis = visible(q, k, causal)

        sel = self.block_pattern(n, causal, q.device)                   # [nb, nb]
        rows = sel.repeat_interleave(self.block, dim=-1)[:, :n]         # [nb, n]
        pb = query_positions(m, n, device=q.device) // self.block       # [m]
        sup = vis & rows[pb]                                            # [..., m, n]
        return restrict(pi, sup, v, return_weights,
                        {"block_size": self.block, "n_blocks": sel.shape[0],
                         "blocks_selected": sel[pb]})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        nb = (n + self.block - 1) // self.block
        blocks = nb if self.window_blocks is None else min(
            nb, self.window_blocks + self.random_blocks + self.global_blocks)
        kept = min(n, blocks * self.block)
        return {"flops": 2 * m * kept * (d + dv),
                "state_bytes": n * (d + dv) * bytes_per_el,   # random blocks reach anywhere
                "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        # every block is a window block, so the support is the whole visible row
        return {"window_blocks": None}
