"""Performer / FAVOR+: positive random features for the exponential kernel.

``sec:kernel-rf``.  With ``qt = q / d^{1/4}`` and ``kt = k / d^{1/4}`` the softmax
interaction is ``g(q,x) = exp(<qt, kt_x>)`` and FAVOR+ uses the positive, unbiased
estimator

    phi_omega(u) = exp(omega^T u - ||u||^2 / 2),
    g_hat(q,x)   = (1/r) sum_{l=1}^r phi_{omega_l}(qt) phi_{omega_l}(kt_x),
    omega_l ~ N(0, I_d)   i.i.d. or exactly orthogonal in blocks of d.

Because ``phi >= 0`` the normalizer ``Z_hat > 0`` and ``pi_hat`` is a genuine
probability measure (``prop:error-additive``: positivity is essential).  The relative
variance at a pair is ``(exp(||qt + kt||^2) - 1) / r`` (``eq:kernel-pos-var``), so the
cost of a fixed accuracy is exponential in the squared logit scale.

The output is a *ratio* of two unbiased estimators, so it is itself biased:
``prop:error-ratio-est`` gives bias ``O(1/r)`` and standard deviation ``O(r^{-1/2})``.
``convergence_configs`` exists for exactly this reason -- there is no finite ``r`` at
which the method is exact.

Implementation note.  The sum over features is evaluated as a log-sum-exp rather than
as the ``O(nr)`` moment product, because the library needs ``pi_hat`` on the original
positions and because ``phi`` is bounded above by ``exp(||omega||^2/2)`` but can
underflow to zero in the tail.  The value computed is the same estimator.

Reference: Choromanski et al. (2021); Yu et al. (2016) (orthogonal features);
Likhosherstov et al. (2022).
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective, causal_mask,
                     make_generator, weights_from_log_magnitude)
from ...registry import register


@lru_cache(maxsize=128)
def _gaussian_features_f64(r: int, d: int, seed: int, orthogonal: bool):
    """The expensive part of ``gaussian_features``: the float64 CPU draw.

    A pure function of ``(r, d, seed, orthogonal)``, so it is memoised. Without this,
    the orthogonal branch redoes ``r/d`` float64 QR factorisations on every forward
    pass -- 7h48m of CPU in 28min wall during study S3. Caching is bit-identical: the
    generator is seeded from ``seed`` alone, so every call produced the same tensor.
    """
    g = make_generator(seed)
    if not orthogonal:
        return torch.randn(r, d, generator=g, dtype=torch.float64)
    blocks = []
    for b in range((r + d - 1) // d):
        a = torch.randn(d, d, generator=g, dtype=torch.float64)
        qmat, rmat = torch.linalg.qr(a)
        qmat = qmat * torch.sign(torch.diagonal(rmat)).unsqueeze(0)   # make QR unique
        blocks.append(qmat.T)
    om = torch.cat(blocks, 0)[:r]
    norms = torch.randn(r, d, generator=g, dtype=torch.float64).norm(dim=-1, keepdim=True)
    return om * norms


def gaussian_features(r: int, d: int, seed: int, orthogonal: bool, device, dtype):
    """``Omega`` of shape [r, d]: i.i.d. N(0, I_d) rows, or exactly orthogonal blocks
    with chi_d-distributed norms (the FAVOR+ variance reduction of Yu et al., 2016)."""
    cached = _gaussian_features_f64(r, d, seed, orthogonal)
    out = cached.to(device=device, dtype=dtype)
    # ``.to`` is a no-op when the request is already cpu/float64; never hand back the
    # cached tensor itself, or a caller could mutate every later draw.
    return out.clone() if out is cached else out


@register
class PerformerAttention(AttentionMethod):
    """FAVOR+ positive random features, optionally orthogonal."""

    name = "performer"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True            # phi >= 0, so pi_hat is a probability measure
    references = ("choromanski2021rethinking", "yu2016orthogonal", "likhosherstov2022chefs")
    paper_section = "sec:kernel"

    def __init__(self, r: int = 64, orthogonal: bool = True, seed: int = 0, **kw):
        super().__init__(r=r, orthogonal=orthogonal, seed=seed, **kw)
        self.r, self.orthogonal, self.seed = r, orthogonal, seed

    def _log_features(self, u, d):
        """``log phi_omega(ut)`` for every feature: [..., t, r]."""
        ut = u / d ** 0.25
        om = gaussian_features(self.r, d, self.seed, self.orthogonal, u.device, u.dtype)
        return ut @ om.T - 0.5 * (ut ** 2).sum(-1, keepdim=True)

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        lq = self._log_features(q, d)                       # [..., m, r]
        lk = self._log_features(k, d)                       # [..., n, r]
        # log g_hat(q_i, x_j) = logsumexp_l (lq_il + lk_jl) - log r
        log_g = torch.logsumexp(lq.unsqueeze(-2) + lk.unsqueeze(-3), dim=-1) - math.log(self.r)
        support = causal_mask(m, n, device=q.device) if causal else None
        pi, z = weights_from_log_magnitude(log_g, support=support)
        y = pi @ v
        extras = {"log_g_hat": log_g, "Z_rel": z, "r": self.r}
        return AttentionOutput(y=y, weights=pi if return_weights else None, extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.r
        return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv,
                "state_bytes": r * (dv + 1) * bytes_per_el,       # the moments (H, z)
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return None       # no finite r is exact; see convergence_configs

    @classmethod
    def convergence_configs(cls):
        return [{"r": 8}, {"r": 32}, {"r": 128}, {"r": 512}]
