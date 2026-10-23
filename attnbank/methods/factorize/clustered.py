"""Clustered attention: route queries through r query centroids.

``sec:factor-methods``.  The queries are clustered; one attention row is computed per
centroid and shared by every query in the cluster:

    pi_hat_{q_i} = softmax_x( beta <c_{g(i)}, k_x> ).

In the language of ``def:factor-stochastic`` this is ``P_hat = A B`` with ``A`` the
(0/1) cluster-assignment matrix -- row-stochastic -- and ``B`` the ``r`` centroid rows,
also row-stochastic: a latent stochastic factorization whose latent space is a
partition of the *queries* rather than of the context.  It therefore inherits all of
``prop:factor-closure``.

Unlike ``latent_stochastic`` it is an **approximation** with a resource parameter:
``r = None`` gives every query its own cluster and reproduces softmax exactly, which is
the correctness anchor.  Its error is controlled by how far a query sits from its
centroid, via ``prop:error-score`` with ``delta = beta ||q - c|| max_j ||k_j||``.

Not causal: queries in one cluster have different visibility, so a shared row would
leak.  Clustering is Lloyd's algorithm with a seeded initialization.

Reference: Vyas et al. (2020) Clustered Attention.
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
)
from ...registry import register


def lloyd(x, r, seed, iters=25):
    """Batched Lloyd's algorithm on [..., t, d]; returns (assignment [..., t], centroids)."""
    g = make_generator(seed)
    t = x.shape[-2]
    init = torch.randperm(t, generator=g)[:r].to(x.device)
    cent = x.index_select(-2, init).clone()
    for _ in range(iters):
        a = torch.cdist(x, cent).argmin(-1)                       # [..., t]
        onehot = torch.nn.functional.one_hot(a, r).to(x.dtype)    # [..., t, r]
        cnt = onehot.sum(-2).clamp_min(1).unsqueeze(-1)
        new = torch.einsum("...tr,...td->...rd", onehot, x) / cnt
        empty = (onehot.sum(-2) == 0).unsqueeze(-1)
        cent = torch.where(empty, cent, new)
    return torch.cdist(x, cent).argmin(-1), cent


@register
class ClusteredAttention(AttentionMethod):
    """One attention row per query centroid, shared within the cluster."""

    name = "clustered"
    category = Category.FACTORIZE
    objective = Objective.APPROXIMATE
    supports_causal = False
    expresses_weights = True
    row_stochastic = True
    references = ("vyas2020fast",)
    paper_section = "sec:factor"

    def __init__(self, r: int | None = 4, seed: int = 0, **kw):
        """``r``: number of query clusters; ``None`` is one cluster per query (exact)."""
        super().__init__(r=r, seed=seed, **kw)
        self.r, self.seed = r, seed

    def _forward(self, q, k, v, causal, return_weights):
        m, d = q.shape[-2], q.shape[-1]
        beta = self._scale(d)
        if self.r is None or self.r >= m:
            pi = masked_softmax(torch.einsum("...md,...nd->...mn", q, k) * beta)
            assign = torch.arange(m, device=q.device).expand(*q.shape[:-1])
            return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                                   extras={"assign": assign, "r": m})
        assign, cent = lloyd(q.double(), self.r, self.seed)
        cent = cent.to(q.dtype)
        b = masked_softmax(torch.einsum("...rd,...nd->...rn", cent, k) * beta)  # [..., r, n]
        pi = b.gather(-2, assign.unsqueeze(-1).expand(*assign.shape, k.shape[-2]))
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras={"assign": assign, "centroids": cent, "b": b,
                                       "r": self.r,
                                       "centroid_dist": (q - cent.gather(
                                           -2, assign.unsqueeze(-1).expand(
                                               *assign.shape, d))).norm(dim=-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = min(self.r or m, m)
        return {"flops": 2 * r * n * (d + dv) + m * r * d,
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"r": None}
