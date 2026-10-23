"""Reformer LSH attention: the support is a hash bucket.

``sec:sparse-content`` ("Content-based supports").  Angular locality-sensitive hashing
gives a *query-dependent, content-based* support that costs ``O(n d b)`` to build
rather than the ``O(n d)`` of the full score row it replaces.  With ``R`` a random
``d x (b/2)`` Gaussian matrix, Reformer's hash is

    h(x) = argmax( [x R ; -x R] )  in  {0, ..., b-1},

a random partition of the sphere into ``b`` cones whose collision probability
``P[h(q) = h(k)] = 1 - theta(q, k) / pi`` decreases with the angle, so the bucket of a
query is biased towards its large-score keys.  ``n_rounds`` independent hashes are
drawn and the supports unioned, which raises recall at linear cost -- the standard
amplification, and the knob Reformer actually reports.

Given the support the operator is the renormalized restriction of
``prop:sparse-identity``: the error is ``eps_q (m_{S^c} - m_S)`` with
``eps_q = pi_q(S^c)``, exactly as for ``topk``.  What LSH buys is that ``S_q`` is
found without computing ``pi_q``; what it costs is that ``S_q`` is not the top-k --
collisions are probabilistic, so a heavy key is missed with probability
``(theta/pi)^n_rounds`` and light keys are retained.  Contrast ``lsh_sample``, which
uses the same hashing to build a *proposal* and reweights by ``1/u`` instead of
restricting: that estimator is consistent, this one is biased, and the bias is the
omitted mass.

Deviations from the paper, which are about the kernel rather than the operator:

* Reformer *ties* queries and keys (``k_j = q_j / ||q_j||``) so that one hash serves
  both and a position always collides with itself.  The bank's queries and keys are
  independent, so both are hashed with the same ``R`` and the query's own position is
  added to its support unconditionally -- which is what tying makes automatic, and
  what guarantees a nonempty row.  With shared QK the two agree.
* Reformer then *sorts* by bucket and attends within fixed-size chunks (plus the
  previous chunk), to bound memory and to handle uneven buckets.  That is a batching
  device: the set it approximates is the one implemented here, "keys sharing a bucket
  with the query in at least one round".  ``hyperattention`` does implement a sorted,
  block-structured version of the same idea.
* Reformer masks a position from attending to itself when it has other options, an
  artifact of shared QK that does not apply here.

Exact limit: ``n_buckets = 1`` puts every key in the query's bucket, so the support is
the whole visible context and ``pi_hat = pi``.

Reference: Kitaev, Kaiser & Levskaya (2020) Reformer; Andoni et al. (2015).
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, make_generator, query_positions
from ...registry import register
from ._restriction import reference_pi, restrict, visible


def angular_buckets(x, rot):
    """Reformer's hash: ``argmax([x R ; -x R])`` per round. ``[rounds, ..., t]`` of ints.

    ``rot`` is ``[rounds, d, b/2]``.  The sign-doubling is what makes the partition a
    set of ``b`` cones rather than ``b/2`` half spaces.
    """
    proj = torch.einsum("...td,rdh->r...th", x, rot)
    return torch.cat([proj, -proj], dim=-1).argmax(-1)


def diagonal_support(m, n, like):
    """Boolean ``[m, n]`` marking each query's own context position, ``i + (n - m)``."""
    sup = torch.zeros(m, n, dtype=torch.bool, device=like.device)
    sup[torch.arange(m, device=like.device), query_positions(m, n, device=like.device)] = True
    return sup


@register
class ReformerAttention(AttentionMethod):
    """Support = the keys that share an LSH bucket with the query in some round."""

    name = "reformer"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("kitaev2020reformer", "andoni2015practical")
    paper_section = "sec:sparse"

    def __init__(self, n_buckets: int = 8, n_rounds: int = 2, seed: int = 0, **kw):
        super().__init__(n_buckets=n_buckets, n_rounds=n_rounds, seed=seed, **kw)
        self.n_buckets, self.n_rounds, self.seed = n_buckets, n_rounds, seed

    def _rotations(self, d, like):
        """``[n_rounds, d, n_buckets // 2]``, drawn in fp64 so fp32 hashes the same way."""
        half = max(1, self.n_buckets // 2)
        g = make_generator(self.seed)
        rot = torch.randn(self.n_rounds, d, half, generator=g, dtype=torch.float64)
        return rot.to(device=like.device, dtype=like.dtype)

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        pi, _ = reference_pi(self, q, k, causal)
        vis = visible(q, k, causal)
        if self.n_buckets <= 1:
            return restrict(pi, vis, v, return_weights,
                            {"n_buckets": 1, "n_rounds": self.n_rounds})

        rot = self._rotations(d, q)
        bq = angular_buckets(q, rot)                     # [rounds, ..., m]
        bk = angular_buckets(k, rot)                     # [rounds, ..., n]
        collide = (bq.unsqueeze(-1) == bk.unsqueeze(-2)).any(0)       # [..., m, n]
        sup = (collide | diagonal_support(m, n, q)) & vis
        return restrict(pi, sup, v, return_weights,
                        {"n_buckets": 2 * max(1, self.n_buckets // 2),
                         "n_rounds": self.n_rounds,
                         "bucket_q": bq, "bucket_k": bk,
                         "collision_rate": collide.to(pi.dtype).mean(-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        half = max(1, self.n_buckets // 2)
        kept = max(1.0, n / max(1, 2 * half)) * self.n_rounds
        return {"flops": 2 * (m + n) * d * half * self.n_rounds + 2 * m * kept * (d + dv),
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * d + m * kept * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"n_buckets": 1}
