"""ORACLE: best rank-r *nonnegative* factorization of the attention matrix.

``sec:factor-rank``, ``prop:factor-rank``.  For a row-stochastic ``P`` the hierarchy is

    rank(P) <= rank_+(P) = rank_s(P),

so the smallest rank of a *stochastic* factorization equals the nonnegative rank, and
stochastic factorization is NMF with a free renormalization.  This file is the empirical
side of that statement: it fits ``P ~ W H`` with ``W, H >= 0`` of inner dimension ``r``
by Lee--Seung multiplicative updates on the Frobenius loss,

    W <- W * (P H^T) / (W H H^T),     H <- H * (W^T P) / (W^T W H),

from ``restarts`` seeded initializations, keeping the best, and then renormalizes the
rows of ``W H`` so the result is a genuine row-stochastic ``P_hat`` -- legitimate by
the rescaling argument in the proof of ``prop:factor-rank``.  The *unnormalized*
product and its Frobenius residual are returned in ``extras`` so that the fit quality
can be separated from the renormalization.

This is an **oracle**, not a method anyone deploys: it gives the best a nonnegative
rank-``r`` factorization *could* do, so that the gap against ``svd_oracle`` -- which
has no nonnegativity constraint and is optimal by Eckart--Young -- measures the
nonnegative-rank penalty that every stochastic latent attention pays.  Multiplicative
updates converge to a stationary point, not a global optimum, so the gap it reports is
an upper bound on the true penalty.

Exact limit: ``r = None`` returns ``W = P``, ``H = I`` -- the exact nonnegative
factorization of rank ``n``.

Not causal: a nonnegative factorization of a masked ``P`` only approximates the zeros.

Reference: Lee and Seung (1999); Cohen and Rothblum (1993); Gillis (2020).
"""
from __future__ import annotations

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    make_generator,
    masked_softmax,
    renormalize,
    scores,
)
from ...registry import register

EPS = 1e-12


def nmf(p, r, seed, iters, restarts):
    """Lee--Seung multiplicative updates; returns the best (W, H, residual) over restarts."""
    best = None
    for s in range(restarts):
        g = make_generator(seed + s)
        w = torch.rand(*p.shape[:-1], r, generator=g, dtype=torch.float64).to(p.device) + EPS
        h = torch.rand(*p.shape[:-2], r, p.shape[-1], generator=g,
                       dtype=torch.float64).to(p.device) + EPS
        for _ in range(iters):
            w = w * (p @ h.transpose(-1, -2)) / ((w @ (h @ h.transpose(-1, -2))) + EPS)
            h = h * (w.transpose(-1, -2) @ p) / (((w.transpose(-1, -2) @ w) @ h) + EPS)
        res = (p - w @ h).flatten(-2).norm(dim=-1)          # [...]  one per (batch, head)
        if best is None:
            best = (w, h, res)
        else:
            take = (res < best[2])[..., None, None]         # broadcast over (m, r) / (r, n)
            best = (torch.where(take, w, best[0]),
                    torch.where(take, h, best[1]),
                    torch.minimum(res, best[2]))
    return best


@register
class NMFOracleAttention(AttentionMethod):
    """Best nonnegative rank-r factorization of P, renormalized. Oracle."""

    name = "nmf_oracle"
    category = Category.FACTORIZE
    objective = Objective.APPROXIMATE
    supports_causal = False
    expresses_weights = True
    row_stochastic = True
    references = ("lee1999learning", "cohen1993nonnegative", "gillis2020nonnegative")
    paper_section = "sec:factor"

    def __init__(self, r: int | None = 4, iters: int = 200, restarts: int = 3,
                 seed: int = 0, **kw):
        super().__init__(r=r, iters=iters, restarts=restarts, seed=seed, **kw)
        self.r, self.iters, self.restarts, self.seed = r, iters, restarts, seed

    def _forward(self, q, k, v, causal, return_weights):
        p = masked_softmax(scores(q, k, self._scale(q.shape[-1]), causal)).double()
        n = p.shape[-1]
        if self.r is None or self.r >= n:
            w = p
            h = torch.eye(n, dtype=p.dtype, device=p.device).expand(*p.shape[:-2], n, n)
        else:
            w, h, _ = nmf(p, self.r, self.seed, self.iters, self.restarts)
        raw = (w @ h)
        pi = renormalize(raw).to(q.dtype)
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras={"W": w.to(q.dtype), "H": h.to(q.dtype),
                                       "raw": raw.to(q.dtype),
                                       "fro_residual": (p - raw).flatten(-2).norm(dim=-1),
                                       "r": min(self.r or n, n)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = min(self.r or n, n)
        return {"flops": 2 * m * n * d + 2 * (m + n) * r * dv,
                "state_bytes": r * n * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"r": None}
