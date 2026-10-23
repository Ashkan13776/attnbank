"""HyperAttention: a hashed heavy block plus an unbiased uniform estimate of the rest.

``sec:sparse-content``, and the boundary ``sec:sparse-content`` draws between
restricting and *estimating*.  HyperAttention splits the row of ``pi_q`` in two and
treats the halves differently:

1. **Heavy part.** ``sortLSH`` hashes queries and keys, sorts by bucket, and keeps a
   contiguous block of ``block`` keys per query -- a mask ``H_q`` that, after the
   permutation, is block diagonal and so can be run by a dense kernel.  These entries
   are kept exactly.
2. **Light part.** The remaining visible keys are sampled uniformly at inclusion
   probability ``u = min(1, n_sampled / |S_q^c|)`` and reweighted by ``1/u``.

The implied measure is therefore neither a restriction nor a pure importance sample:

    w_j = pi_j                   j in H_q,
    w_j = pi_j / u               j sampled outside H_q,
    pi_hat = w / sum_i w_i.

Numerator and denominator of the *residual* are unbiased, so the only systematic error
is the ``O(1/n_sampled)`` self-normalization bias of ``prop:error-ratio-est``, and the
hashed block is what keeps the variance of the residual small: the keys whose
``pi_j / u`` is large are the ones the hash was meant to catch.  The contrast with
``topk`` -- which simply discards the tail and absorbs ``eps_q`` into the head -- and
with ``lsh_sample`` -- which samples everything -- is the point of having all three.

Scope of the paper's guarantee.  Han et al. prove a spectral-norm bound on the
approximate attention under two conditions on the matrix (a bounded ratio of column
norms, ``kappa``, and a bounded max column ``l_1`` mass), together with their specific
sampling scheme.  Nothing here re-derives or checks that bound; this file implements
the estimator, and the only error statement it makes is the one ``extras`` lets you
verify numerically.

Deviations from the paper, stated rather than hidden:

* The paper sorts queries and keys *independently* by bucket and takes the block
  diagonal of the permuted ``m x n`` matrix, which pairs a query with keys at the same
  *rank quantile*.  This file places the query at its own bucket's position in the
  sorted key order and takes the ``block`` keys around it.  The two agree when queries
  and keys have the same bucket distribution; pairing by bucket value is what remains
  well defined for ``m != n`` and under causal masking, where the sorted key order is
  the order of the *visible prefix* and therefore differs per query.  That restriction
  to the prefix is also what makes the support causal: a key after the query can
  neither enter the block nor move another key into it.
* The paper handles causal masking by a recursive decomposition into two halves plus a
  diagonal block, for which it can reuse the same subroutine.  The prefix formulation
  above is a different route to the same requirement and is not that algorithm.
* Uniform column sampling is done here as independent per-row Bernoulli draws rather
  than as one shared column subset, so each row's estimator is unbiased on its own.

Exact limit: ``n_sampled = None`` sets ``u = 1`` on the whole complement, so every
visible key carries its exact weight and ``pi_hat = pi`` -- independent of the hash,
which is what makes it a test of the reweighting rather than of the mask.

Reference: Han et al. (2024) HyperAttention; Kitaev et al. (2020) Reformer;
Owen (2013).
"""
from __future__ import annotations

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    make_generator,
    normalize_nonnegative,
    query_positions,
)
from ...registry import register
from ._restriction import reference_pi, visible
from .reformer import angular_buckets


def sort_lsh_mask(bq, bk, pos, block):
    """The sortLSH block: ``block`` keys around the query's own position in bucket order.

    ``bq`` is ``[..., m]`` and ``bk`` ``[..., n]`` of bucket ids; ``pos`` is ``[m]``,
    the last context position each query may see.  Keys are ordered by
    ``(bucket, position)`` *within the visible prefix*, so the returned ranks -- and
    hence the mask -- depend on nothing after the query.
    """
    n = bk.shape[-1]
    idx = torch.arange(n, device=bk.device)
    before = ((bk.unsqueeze(-2) < bk.unsqueeze(-1))
              | ((bk.unsqueeze(-2) == bk.unsqueeze(-1)) & (idx[None, :] < idx[:, None])))
    rank = before.to(torch.int32).cumsum(-1).index_select(-1, pos).transpose(-1, -2)

    lt = (bk.unsqueeze(-2) < bq.unsqueeze(-1)).to(torch.int32).cumsum(-1)
    start = lt.gather(-1, pos.expand(lt.shape[:-1]).unsqueeze(-1)).squeeze(-1)

    n_vis = (pos + 1).to(torch.int32)
    lo = torch.minimum((start - block // 2).clamp_min(0), (n_vis - block).clamp_min(0))
    lo = lo.unsqueeze(-1)
    return (rank >= lo) & (rank < lo + block)


@register
class HyperAttention(AttentionMethod):
    """Exact on a hashed block, unbiased-with-reweighting on a uniform sample of the rest."""

    name = "hyperattention"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True            # pi_j / u >= 0, renormalized
    references = ("han2024hyperattention", "kitaev2020reformer", "owen2013monte")
    paper_section = "sec:sparse"

    def __init__(self, block: int = 4, n_sampled: int | None = 4, n_buckets: int = 8,
                 seed: int = 0, hash_seed: int = 0, **kw):
        """``hash_seed`` draws the sortLSH rotation and ``seed`` the uniform sample.

        They are separate generators so that either randomization can be held fixed
        while the other varies -- which is what makes the residual estimator's bias
        and variance measurable at a fixed mask.
        """
        super().__init__(block=block, n_sampled=n_sampled, n_buckets=n_buckets,
                         seed=seed, hash_seed=hash_seed, **kw)
        self.block, self.n_sampled, self.n_buckets = block, n_sampled, n_buckets
        self.seed, self.hash_seed = seed, hash_seed

    def _mask(self, q, k, causal):
        """The sortLSH heavy block, ``[..., m, n]``."""
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        half = max(1, self.n_buckets // 2)
        g = make_generator(self.hash_seed)
        rot = torch.randn(1, d, half, generator=g, dtype=torch.float64).to(q.device, q.dtype)
        bq, bk = angular_buckets(q, rot)[0], angular_buckets(k, rot)[0]
        pos = (query_positions(m, n, device=q.device) if causal
               else torch.full((m,), n - 1, dtype=torch.long, device=q.device))
        return sort_lsh_mask(bq, bk, pos, min(self.block, n))

    def _forward(self, q, k, v, causal, return_weights):
        pi, _ = reference_pi(self, q, k, causal)
        vis = visible(q, k, causal)
        heavy = self._mask(q, k, causal) & vis
        rest = vis & ~heavy

        if self.n_sampled is None:
            u = rest.to(pi.dtype)
            take = rest.clone()
        else:
            n_c = rest.sum(-1, keepdim=True).to(pi.dtype)
            u = (self.n_sampled / n_c.clamp_min(1.0)).clamp(max=1.0) * rest
            g = make_generator(self.seed)
            draw = torch.rand(u.shape, generator=g, dtype=torch.float64).to(u.device, u.dtype)
            take = (draw < u) & rest

        tiny = torch.finfo(pi.dtype).tiny
        w = pi * heavy + torch.where(take, pi / u.clamp_min(tiny), torch.zeros_like(pi))
        support = heavy | take
        pi_hat, _ = normalize_nonnegative(w, support)
        extras = {"support": support, "heavy": heavy, "u": u,
                  "eps": 1.0 - (pi * support).sum(-1),
                  "heavy_mass": (pi * heavy).sum(-1),
                  "k_realized": support.sum(-1),
                  "n_sampled_realized": take.sum(-1)}
        return AttentionOutput(y=pi_hat @ v, weights=pi_hat if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        half = max(1, self.n_buckets // 2)
        kept = self.block + (self.n_sampled if self.n_sampled is not None else n)
        return {"flops": 2 * (m + n) * d * half + 2 * m * kept * (d + dv),
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * d + m * kept * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"n_sampled": None}

    @classmethod
    def convergence_configs(cls):
        return [{"n_sampled": 1}, {"n_sampled": 2}, {"n_sampled": 4}, {"n_sampled": 8}]
