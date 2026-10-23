"""ToMe: bipartite soft matching -- the pushforward whose partition is *matched*.

``sec:compress-merge``, ``prop:compress-merge``, ``prop:error-merge``.  ``merge_kv``
already implements the pushforward ``mu_hat = T_# mu`` and ToMe's proportional
attention; what it does not implement is ToMe's actual choice of ``T``.  Contiguous
blocks ignore similarity and Lloyd's algorithm is iterative and order-dependent; ToMe's
point was that a *single* round of bipartite matching is almost as good and costs one
matrix product:

1. split the current tokens alternately into ``A`` (even slots) and ``B`` (odd slots);
2. draw one edge from each ``a in A`` to its most cosine-similar ``b in B``, using the
   attention **keys** as the metric;
3. keep the ``r`` heaviest edges;
4. merge each matched ``a`` into its ``b`` by averaging, and concatenate back.

A round can remove at most half the tokens, so reaching a target of ``n_clusters`` takes
``ceil(log2(n / n_clusters))`` rounds; ToMe spends them one per transformer layer, this
file spends them inside one call, which is the same map.  The matching is greedy and
deterministic -- no seed, unlike ``merge_kv(mode='kmeans')``.

Mass.  Exactly as in ``merge_kv``: the representatives ``k_hat_i, v_hat_i`` are the
within-cluster means and are *lifted back* to the original ``n`` positions, so

    pi~_j ∝ e^{beta <q, k_hat_{T(j)}>},   sum_j pi~_j = 1,   pi~ @ v~ = y_hat,

and a cluster of size ``|C_i|`` contributes ``|C_i|`` identical atoms -- ToMe's
``+ log s`` proportional-attention term, appearing automatically rather than by hand.
Dropping it would replace ``T_# mu`` by counting measure on the compressed domain.
``extras`` carries ``assign``, ``sizes`` and ``cluster_radius`` so that
``metrics.merge_bound`` applies, and ``v_bar`` so it can be called directly.

One deviation, stated rather than hidden.  ToMe merges tokens of the residual stream,
so a single partition is shared by every head.  Here each ``(batch, head)`` matches on
its own keys, which keeps the head independence the conformance suite checks (DESIGN.md
Sec. 4, check 5) and makes the method comparable with the rest of ``compress/``;
``assign`` is therefore ``[B, H, n]`` and ``merge_bound`` is applied per slice.

Not causal: a cluster spans positions on both sides of a query.

Exact limit: ``n_clusters = None`` (no edge is kept; every token is its own cluster).

Reference: Bolya and Hoffman (2023) ToMe; Wang et al. (2024) KVMerger.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, masked_softmax, scores
from ...registry import register


def bipartite_soft_match(metric: torch.Tensor, target: int) -> torch.Tensor:
    """ToMe's matching for one ``[n, d]`` metric; returns ``assign`` ``[n]``.

    Rounds of alternate-slot bipartite matching are repeated until exactly ``target``
    clusters remain.  After each round the ids are compacted, keeping the surviving
    representative's slot, so the next round splits the shortened list -- the analogue
    of ToMe concatenating ``[unmerged A, B]`` before the next layer.  A merged cluster
    therefore sits where its ``B`` member sat, not where its first token sat; ToMe's
    output order is likewise not the input order.
    """
    n = int(metric.shape[-2])
    assign = torch.arange(n, device=metric.device)
    c = n
    while c > target:
        onehot = torch.nn.functional.one_hot(assign, c).to(metric.dtype)      # [n, c]
        rep = (onehot.t() @ metric) / onehot.sum(0)[:, None]                  # [c, d]
        a_idx = torch.arange(0, c, 2, device=metric.device)
        b_idx = torch.arange(1, c, 2, device=metric.device)
        if b_idx.numel() == 0:
            break
        unit = torch.nn.functional.normalize(rep, dim=-1)
        sim = unit[a_idx] @ unit[b_idx].t()                                   # [|A|, |B|]
        best, target_of_a = sim.max(-1)
        keep = min(c - target, int(a_idx.numel()))
        edges = best.argsort(descending=True, stable=True)[:keep]
        relabel = torch.arange(c, device=metric.device)
        relabel[a_idx[edges]] = b_idx[target_of_a[edges]]
        assign = torch.unique(relabel[assign], sorted=True, return_inverse=True)[1]
        c = int(assign.max()) + 1
    return assign


@register
class ToMeAttention(AttentionMethod):
    """Token merging by bipartite soft matching on the keys, with proportional mass."""

    name = "merge_tome"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False          # clusters span positions on both sides of a query
    expresses_weights = True
    row_stochastic = True
    references = ("bolya2023token", "wang2024model")
    paper_section = "sec:compress"

    def __init__(self, n_clusters: int | None = 6, **kw):
        """``n_clusters``: how many tokens survive; ``None`` merges nothing."""
        super().__init__(n_clusters=n_clusters, **kw)
        self.n_clusters = n_clusters

    def _forward(self, q, k, v, causal, return_weights):
        n, d = k.shape[-2], k.shape[-1]
        beta = self._scale(d)
        r = min(self.n_clusters or n, n)
        if r >= n:
            pi = masked_softmax(scores(q, k, beta, causal))
            assign = torch.arange(n, device=q.device).expand(*k.shape[:-1])
            return AttentionOutput(
                y=pi @ v, weights=pi if return_weights else None, v_hat=v,
                extras={"assign": assign, "sizes": torch.ones_like(assign),
                        "cluster_radius": torch.zeros_like(k[..., 0]), "v_bar": v})

        flat = k.reshape(-1, n, d)
        assign = torch.stack([bipartite_soft_match(flat[i], r)
                              for i in range(flat.shape[0])]).reshape(*k.shape[:-1])

        onehot = torch.nn.functional.one_hot(assign, r).to(k.dtype)           # [..., n, r]
        sizes = onehot.sum(-2)                                                # [..., r]
        k_bar = torch.einsum("...nr,...nd->...rd", onehot, k) / sizes[..., None]
        v_bar = torch.einsum("...nr,...nd->...rd", onehot, v) / sizes[..., None]

        # lift the representatives back to the original positions: the multiplicity of
        # a cluster becomes |C_i| identical atoms, i.e. ToMe's + log s, for free
        k_lift = torch.gather(k_bar, -2, assign.unsqueeze(-1).expand(*assign.shape, d))
        v_lift = torch.gather(v_bar, -2, assign.unsqueeze(-1).expand(*assign.shape,
                                                                     v.shape[-1]))
        pi = masked_softmax(scores(q, k_lift, beta, causal))

        dev = (k - k_lift).norm(dim=-1)                                       # [..., n]
        radius = torch.zeros(*k.shape[:-2], r, dtype=k.dtype,
                             device=k.device).scatter_reduce(
            -1, assign, dev, "amax", include_self=False)
        extras = {"assign": assign, "sizes": sizes, "k_bar": k_bar, "v_bar": v_bar,
                  "cluster_radius": radius}
        return AttentionOutput(y=pi @ v_lift, weights=pi if return_weights else None,
                               v_hat=v_lift, extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = max(1, min(self.n_clusters or n, n))
        return {"flops": 2 * m * r * (d + dv) + n * n * d // 2,   # + the matching
                "state_bytes": r * (d + dv) * bytes_per_el + r * 4,
                "bytes_moved": (m * d + r * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"n_clusters": None}
