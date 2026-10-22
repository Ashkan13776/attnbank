"""Quest-style page / block selection: restriction at block granularity.

``sec:sparse-query`` ("Blocks and pages") and ``prop:sparse-certificate``.  Keys are
partitioned into contiguous pages of ``block`` positions.  A page is ranked by an upper
bound on the scores it can contain,

    tau(B) = beta * sum_c max( q_c * kmin_{B,c},  q_c * kmax_{B,c} ),

from the elementwise minima and maxima of the keys in the page (Tang et al., 2024).
The top ``n_blocks`` pages are fetched and attended *exactly*; everything else is
omitted, so the error is again ``eps_q (m_{S^c} - m_S)``.

The point of having it alongside ``topk`` is granularity: a page-level selector can
only ever omit or retain whole pages, so its realized ``eps_q`` exceeds the top-k
minimum by a selection error.  ``score="max"`` and ``score="mean"`` are provided to
separate the cost of the *bound* from the cost of the *granularity*: the paper notes
that an upper bound never misses a page containing a heavy key but may waste budget,
whereas a pooled representative can average a heavy key away.

Causality.  Page statistics are computed over whole pages, so only *complete* past
pages may be ranked.  The page containing the query's own position is always selected
(as in Quest, where the current page is resident anyway) and later pages are excluded.
This makes the support depend on nothing after the query.

Reference: Tang et al. (2024) Quest; Xiao et al. (2024) InfLLM.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, query_positions
from ...registry import register
from ._restriction import reference_pi, restrict, visible


@register
class BlockTopKAttention(AttentionMethod):
    """Page-level selection by an upper bound on the page's scores."""

    name = "blocktopk"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("tang2024quest", "xiao2024infllm")
    paper_section = "sec:sparse"

    def __init__(self, block: int = 4, n_blocks: int | None = 2, score: str = "bound", **kw):
        """``score``: ``'bound'`` (Quest), ``'max'`` (exact page maximum) or ``'mean'``."""
        super().__init__(block=block, n_blocks=n_blocks, score=score, **kw)
        self.block, self.n_blocks, self.score = block, n_blocks, score

    def _block_scores(self, q, k, beta):
        """Per-page score [..., m, nb] from whole-page statistics."""
        n = k.shape[-2]
        nb = (n + self.block - 1) // self.block
        pad = nb * self.block - n
        kk = torch.nn.functional.pad(k, (0, 0, 0, pad), value=0.0) if pad else k
        kb = kk.unflatten(-2, (nb, self.block))                      # [..., nb, block, d]
        if pad:                                                      # ignore padded slots
            keep = torch.ones(nb * self.block, dtype=torch.bool, device=k.device)
            keep[n:] = False
            keep = keep.view(nb, self.block)
            kb_min = kb.masked_fill(~keep[..., None], float("inf")).amin(-2)
            kb_max = kb.masked_fill(~keep[..., None], float("-inf")).amax(-2)
        else:
            kb_min, kb_max = kb.amin(-2), kb.amax(-2)
        if self.score == "bound":
            qe = q.unsqueeze(-2)                                     # [..., m, 1, d]
            lo, hi = qe * kb_min.unsqueeze(-3), qe * kb_max.unsqueeze(-3)
            return beta * torch.maximum(lo, hi).sum(-1)              # [..., m, nb]
        s = torch.einsum("...md,...nd->...mn", q, k) * beta
        s = torch.nn.functional.pad(s, (0, pad), value=float("-inf") if self.score == "max" else 0.0)
        sb = s.unflatten(-1, (nb, self.block))
        return sb.amax(-1) if self.score == "max" else sb.mean(-1)

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        beta = self._scale(d)
        nb = (n + self.block - 1) // self.block
        pi, s = reference_pi(self, q, k, causal)
        vis = visible(q, k, causal)

        if self.n_blocks is None or self.n_blocks >= nb:
            return restrict(pi, vis, v, return_weights, {"n_pages": nb, "page_size": self.block})

        bs = self._block_scores(q, k, beta)                          # [..., m, nb]
        bidx = torch.arange(nb, device=q.device)
        if causal:
            own = (query_positions(m, n, device=q.device) // self.block)[:, None]   # [m, 1]
            complete = bidx[None, :] < own                           # rankable pages
        else:
            own = torch.full((m, 1), -1, dtype=torch.long, device=q.device)
            complete = torch.ones(m, nb, dtype=torch.bool, device=q.device)
        sel = bs.masked_fill(~complete, float("-inf"))
        take = min(self.n_blocks, nb)
        chosen = torch.zeros_like(sel, dtype=torch.bool).scatter_(
            -1, sel.topk(take, dim=-1).indices, True) & complete
        chosen = chosen | (bidx[None, :] == own)                     # always keep the own page

        sup = chosen.repeat_interleave(self.block, dim=-1)[..., :n] & vis
        return restrict(pi, sup, v, return_weights,
                        {"n_pages": nb, "page_size": self.block, "page_scores": bs,
                         "pages_selected": chosen})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        nb = (n + self.block - 1) // self.block
        kept = min(self.n_blocks or nb, nb) * self.block
        return {"flops": 2 * m * nb * d + 2 * m * kept * (d + dv),
                "state_bytes": n * (d + dv) * bytes_per_el + 2 * nb * d * bytes_per_el,
                "bytes_moved": (m * d + nb * 2 * d + m * kept * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"n_blocks": None}
