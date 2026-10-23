"""Routing Transformer: the support is a learned cluster.

``sec:sparse-content`` ("Content-based supports").  Where ``reformer`` partitions the
sphere at random and ``topk`` reads the true scores, routing places ``c`` centroids in
the query/key space and lets a query attend to the keys routed to *its* centroid:

    r(x) = argmax_i  <x / ||x||, mu_i>,    S_q = { j : r(k_j) = r(q) }  (and the
                                                   query's own position),

so the support costs ``O((m + n) c d)`` to compute and the partition is shared across
queries rather than recomputed per query.  Given ``S_q`` the operator is again the
renormalized restriction of ``prop:sparse-identity``.  Optionally ``w > 0`` adds a
sliding window, because the published model combines content routing with local
attention and attributes part of its result to the combination.

**The centroids are trained.** In the paper they are parameters updated by online
k-means: an exponential moving average of the queries and keys assigned to each
centroid, carried across minibatches, with the centroids re-normalized to the sphere.
They are never fitted to the sequence being attended to, and they could not be --
doing so inside a causal decode would let the support of an early query depend on
later keys.  This class therefore declares ``trainable = True``, and its **untrained
behaviour is the paper's initialization**: ``c`` directions drawn from a seeded
standard Gaussian and normalized.  That is a random partition of the sphere into
Voronoi cones, which is a *lower* bound on what routing buys -- it is roughly
``reformer`` with one round and cones that are not axis-paired, and it should be read
as the floor the training is supposed to lift, not as the method's accuracy.  Pass
``centroids=`` a ``[c, d]`` tensor to supply trained ones; the forward pass is the
same either way.

Load balance.  The published implementation takes, per cluster, the ``n/c`` keys with
the largest centroid dot product, so that every cluster has the same size and the
attention can be batched as a dense ``c x (n/c)`` tensor.  Nearest-centroid membership
-- what is implemented here -- is the routing rule that device approximates; it leaves
clusters uneven, which is visible in ``extras["k_realized"]`` and is part of what the
selector costs.

Exact limit: ``n_clusters = 1`` routes everything to one centroid, so ``S_q`` is the
whole visible context and ``pi_hat = pi``.

Reference: Roy, Saffar, Vaswani & Grangier (2021) Routing Transformers;
Vyas, Katharopoulos & Fleuret (2020) clustered attention.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, make_generator, query_positions
from ...registry import register
from ._restriction import reference_pi, restrict, visible
from .reformer import diagonal_support


def route(x, centroids):
    """Nearest centroid in cosine similarity: ``[..., t]`` of cluster indices."""
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    return (xn @ centroids.transpose(-1, -2)).argmax(-1)


@register
class RoutingAttention(AttentionMethod):
    """Support = the keys routed to the query's centroid. Centroids are trained."""

    name = "routing"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    trainable = True              # the centroids are EMA k-means parameters
    references = ("roy2021routing", "vyas2020fast")
    paper_section = "sec:sparse"

    def __init__(self, n_clusters: int = 4, w: int = 0, seed: int = 0,
                 centroids: torch.Tensor | None = None, **kw):
        """``w`` adds a sliding window of radius ``w``; ``centroids`` is ``[c, d]``."""
        super().__init__(n_clusters=n_clusters, w=w, seed=seed, centroids=centroids, **kw)
        self.n_clusters, self.w, self.seed = n_clusters, w, seed
        self.given_centroids = centroids

    def _centroids(self, d, like):
        """The untrained initialization: seeded Gaussian directions on the sphere."""
        if self.given_centroids is not None:
            c = self.given_centroids.to(device=like.device, dtype=like.dtype)
        else:
            g = make_generator(self.seed)
            c = torch.randn(self.n_clusters, d, generator=g, dtype=torch.float64)
            c = c.to(device=like.device, dtype=like.dtype)
        return c / c.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(c.dtype).tiny)

    def _window(self, m, n, like):
        pos = query_positions(m, n, device=like.device)[:, None]
        j = torch.arange(n, device=like.device)[None, :]
        return (j - pos).abs() <= self.w

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        pi, _ = reference_pi(self, q, k, causal)
        vis = visible(q, k, causal)
        if self.n_clusters <= 1 and self.w <= 0:
            return restrict(pi, vis, v, return_weights, {"n_clusters": 1})

        mu = self._centroids(d, q)
        cq, ck = route(q, mu), route(k, mu)
        same = cq.unsqueeze(-1) == ck.unsqueeze(-2)                  # [..., m, n]
        sup = (same | diagonal_support(m, n, q)) & vis
        if self.w > 0:
            sup = sup | (self._window(m, n, q) & vis)
        return restrict(pi, sup, v, return_weights,
                        {"n_clusters": self.n_clusters, "w": self.w,
                         "cluster_q": cq, "cluster_k": ck,
                         "trained_centroids": self.given_centroids is not None})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        c = max(1, self.n_clusters)
        kept = max(1.0, n / c) + 2 * self.w + 1
        return {"flops": 2 * (m + n) * d * c + 2 * m * kept * (d + dv),
                "state_bytes": (n * (d + dv) + c * d) * bytes_per_el,
                "bytes_moved": (m * d + n * d + c * d + m * kept * (d + dv) + m * dv)
                               * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"n_clusters": 1}
