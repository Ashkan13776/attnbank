"""Token merging: a pushforward compression of the context measure.

``sec:compress-merge``, ``prop:compress-merge``, ``prop:error-merge``.  A partition
``C_1, ..., C_r`` of the positions and the map ``T: x -> cluster(x)`` define

    mu_hat = T_# mu,    mu_hat({i}) = mu(C_i),
    y_hat(q) = sum_i mu(C_i) e^{beta <q, k_hat_i>} v_hat_i / sum_i mu(C_i) e^{beta <q, k_hat_i>},

i.e. ToMe's *proportional attention*: the merged token carries the ``mu``-mass of its
preimage, which in log form is the ``log |C_i|`` bias.  Without it the pushforward is
replaced by counting measure on the compressed domain and large clusters are
systematically under-weighted.

Expressing it on the original positions.  Lift the representatives back,
``k~_j = k_hat_{T(j)}`` and ``v~_j = v_hat_{T(j)}``; then
``pi~_j ∝ e^{beta <q, k~_j>}`` on the *original* ``n`` positions satisfies
``pi~ @ v~ = y_hat`` and ``sum_j pi~_j = 1``.  The multiplicity appears automatically,
because a cluster of size ``|C_i|`` contributes ``|C_i|`` identical atoms.  That is the
sense in which the pushforward is mass-preserving, and it is what
``tests/test_compress.py`` checks.

Representatives.  ``k_hat_i`` and ``v_hat_i`` are the within-cluster means.  The paper
notes this is *not* the right key representative even for one query: the exact
contribution is ``mu(C_i) exp(Lambda_i(q))`` with ``Lambda_i`` the log-MGF of the
cluster's keys, and by Jensen mean-key merging under-weights every cluster, to leading
order by ``exp(-beta^2 q^T Sigma_i q / 2)``.  The within-cluster covariance is returned
in ``extras`` so that correction can be studied.

Not causal: clusters span positions on both sides of a query.

Exact limit: ``n_clusters = None`` (singleton clusters).

Reference: Bolya and Hoffman (2023) ToMe; Wang et al. (2024) KVMerger;
Nawrot et al. (2024) DMC.
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     make_generator, masked_softmax, scores)
from ...registry import register


def contiguous_clusters(n, r, device):
    """Contiguous equal-size clusters: the grouping of consecutive similar tokens."""
    bounds = torch.linspace(0, n, r + 1).round().long()
    assign = torch.zeros(n, dtype=torch.long, device=device)
    for i in range(r):
        assign[bounds[i]:bounds[i + 1]] = i
    return assign


def kmeans_clusters(k, r, seed, iters=25):
    """Lloyd's algorithm on the keys of the first (batch, head), seeded and deterministic."""
    x = k.reshape(-1, k.shape[-1]).double()
    g = make_generator(seed)
    cent = x[torch.randperm(x.shape[0], generator=g)[:r]]
    for _ in range(iters):
        a = torch.cdist(x, cent).argmin(-1)
        for i in range(r):
            sel = a == i
            if bool(sel.any()):
                cent[i] = x[sel].mean(0)
    return torch.cdist(x, cent).argmin(-1).reshape(k.shape[:-1])


@register
class MergeKVAttention(AttentionMethod):
    """Cluster keys and values, merge with proportional (pushforward) mass."""

    name = "merge_kv"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False          # clusters span positions on both sides of a query
    expresses_weights = True
    row_stochastic = True
    references = ("bolya2023token", "wang2024model", "nawrot2024dynamic")
    paper_section = "sec:compress"

    def __init__(self, n_clusters: int | None = 6, mode: str = "contiguous",
                 seed: int = 0, **kw):
        """``mode``: ``'contiguous'`` (consecutive tokens) or ``'kmeans'`` (on keys)."""
        super().__init__(n_clusters=n_clusters, mode=mode, seed=seed, **kw)
        self.n_clusters, self.mode, self.seed = n_clusters, mode, seed

    def _forward(self, q, k, v, causal, return_weights):
        n, d = k.shape[-2], k.shape[-1]
        beta = self._scale(d)
        r = min(self.n_clusters or n, n)
        if r >= n:
            pi = masked_softmax(scores(q, k, beta, causal))
            assign = torch.arange(n, device=q.device)
            return AttentionOutput(y=pi @ v, weights=pi if return_weights else None, v_hat=v,
                                   extras={"assign": assign, "sizes": torch.ones_like(assign)})

        if self.mode == "kmeans":
            assign = kmeans_clusters(k, r, self.seed)[(0,) * (k.dim() - 2)]
        else:
            assign = contiguous_clusters(n, r, q.device)
        onehot = torch.nn.functional.one_hot(assign, r).to(k.dtype)      # [n, r]
        sizes = onehot.sum(0)                                            # [r]

        k_bar = torch.einsum("nr,...nd->...rd", onehot, k) / sizes[:, None]
        v_bar = torch.einsum("nr,...nd->...rd", onehot, v) / sizes[:, None]

        # lift the representatives back to the original positions
        k_lift = k_bar.index_select(-2, assign)                          # [..., n, d]
        v_lift = v_bar.index_select(-2, assign)
        pi = masked_softmax(scores(q, k_lift, beta, causal))             # pi~, mass-preserving

        # R_i = max_{j in C_i} ||k_j - k_hat_i||, the key radius of prop:compress-merge
        dev = (k - k_lift).norm(dim=-1)                                  # [..., n]
        radius = torch.zeros(*k.shape[:-2], r, dtype=k.dtype, device=k.device).scatter_reduce(
            -1, assign.expand(*k.shape[:-1]), dev, "amax", include_self=False)
        extras = {"assign": assign, "sizes": sizes, "k_bar": k_bar, "v_bar": v_bar,
                  "cluster_radius": radius}
        return AttentionOutput(y=pi @ v_lift, weights=pi if return_weights else None,
                               v_hat=v_lift, extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = min(self.n_clusters or n, n)
        return {"flops": 2 * m * r * (d + dv) + 2 * n * (d + dv),
                "state_bytes": r * (d + dv) * bytes_per_el + r * 4,      # + multiplicities
                "bytes_moved": (m * d + r * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"n_clusters": None}
