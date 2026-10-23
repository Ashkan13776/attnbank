"""FAVOR#: dense-exponential random features, the anisotropic optimum.

``sec:kernel-rf``.  ``favorpp`` optimises one scalar ``A`` over the family
``D exp(A ||omega||^2 + B omega^T u + C ||u||^2)``.  Likhosherstov et al. (2023) let
the quadratic form be a full matrix -- *dense-exponential random features* --

    f(omega, u) = D exp(omega^T A omega + omega^T B u + u^T C u),   omega ~ N(0, I_d),

and their Thm. 4.1 gives unbiasedness for ``g(q,x) = exp(<qt, kt_x>)`` iff

    8A < I,   B^T (I - 4A)^{-1} B = I,   C = -1/2 B^T (I - 4A)^{-1} B,
    D = det(I - 4A)^{1/4},

which for the symmetric choice ``B^(1) = B^(2) = B`` collapses to ``C = -I/2``.  The
optimum of the shifted log-variance over the pair set (their Thm. 4.3) is the
eigendecomposition of the **pair covariance**

    Sigma = mean_{i,j} (qt_i + kt_j)(qt_i + kt_j)^T
          = M^(1) + mu^(4) (mu^(5))^T + mu^(5) (mu^(4))^T + M^(2) = Q Lam Q^T,

after which, with :func:`attnbank.methods.interaction.favorpp.optimal_a` applied to
each eigenvalue,

    A = diag(A*(Lam_1), ..., A*(Lam_d)),   B = (I - 4A)^{1/2} Q^T,
    C = -I/2,   D = det(I - 4A)^{1/4} = prod_l (1 - 4 A_ll)^{1/4}.

So FAVOR# is FAVOR++ run independently in each eigendirection of ``Sigma``, and
FAVOR++ is FAVOR# with every eigenvalue replaced by their mean ``tr(Sigma)/d``: the
two closed forms are the *same* scalar map ``A*``, which is why ``optimal_a`` lives in
one place and is used by both.  At ``Sigma = 0`` it collapses further still: ``A = 0``,
``B = Q^T`` is an orthogonal rotation, and ``Q omega`` is again ``N(0, I_d)``, so the
estimator is FAVOR+ in distribution.

Because ``Sigma`` is a mean of outer products it is positive semidefinite, so every
``Lam_l >= 0``, every ``A*(Lam_l) <= 0`` and ``1 - 4A > 0``: the feature map is real
and strictly positive, and ``pi_hat`` is a probability measure.

**Causality.**  As in ``favorpp``, the statistic is conditioned on each query's
visible support -- ``Sigma_i = qt_i qt_i^T + qt_i ybar_i^T + ybar_i qt_i^T + M2_i``
with ``ybar_i``, ``M2_i`` the prefix means of ``kt_j`` and ``kt_j kt_j^T`` -- so the
operator is exactly causal and head-independent, and unbiasedness is untouched because
``Sigma`` does not depend on ``omega``.  The statistic and the eigendecomposition are
done in float64 so that the choice of basis, which a sign flip of an eigenvector would
otherwise make dtype-dependent, is identical in every precision; eigenvector signs are
additionally canonicalised (largest-magnitude entry positive), which the estimator's
distribution is invariant to.

The per-query ``O(d^3)`` eigendecomposition is the price of exact causality and is a
property of this reference implementation; ``cost`` reports the deployed form, in which
``Sigma`` is a layer constant fitted once (``O(L d^2 + d^3)``) and the features are
separable.

Reference: Likhosherstov et al. (2023), *FAVOR# / Dense-Exponential Random Features*;
Likhosherstov et al. (2022) for the isotropic case; Yu et al. (2016) for the
orthogonal block construction.
"""
from __future__ import annotations

import math

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    causal_mask,
    weights_from_log_magnitude,
)
from ...registry import register
from .favorpp import optimal_a, visible_prefix_mean
from .performer import gaussian_features


def pair_covariance(qt, kt, causal: bool):
    """``Sigma_i``, the covariance of ``qt_i + kt_j`` over visible ``j``: ``[..., m, d, d]``.

    Expanded as ``qt qt^T + qt ybar^T + ybar qt^T + M2`` so that the key moments are
    prefix sums and nothing a query cannot see enters its statistic.
    """
    m = qt.shape[-2]
    qt64, kt64 = qt.double(), kt.double()
    ybar = visible_prefix_mean(kt64, m, causal, extra_dims=1)            # [..., m, d]
    outer = kt64.unsqueeze(-1) * kt64.unsqueeze(-2)                      # [..., n, d, d]
    m2 = visible_prefix_mean(outer, m, causal, extra_dims=2)             # [..., m, d, d]
    cross = qt64.unsqueeze(-1) * ybar.unsqueeze(-2)                      # qt ybar^T
    return qt64.unsqueeze(-1) * qt64.unsqueeze(-2) + cross + cross.transpose(-1, -2) + m2


def canonical_eigh(sigma):
    """``(Lam, Q)`` of a symmetric PSD batch, with the eigenvector sign pinned.

    LAPACK fixes an eigenvector only up to sign; pinning the largest-magnitude entry
    positive makes the returned basis a function of ``sigma`` alone.  The estimator is
    invariant in distribution to the choice, since ``omega`` is spherically symmetric.
    """
    lam, qmat = torch.linalg.eigh(sigma)
    pivot = qmat.abs().argmax(dim=-2, keepdim=True)                      # [..., 1, d]
    flip = torch.sign(torch.gather(qmat, -2, pivot))
    return lam.clamp_min(0.0), qmat * torch.where(flip == 0, torch.ones_like(flip), flip)


@register
class FAVORSharpAttention(AttentionMethod):
    """Dense-exponential random features: FAVOR++ per eigendirection of the pair covariance."""

    name = "favorsharp"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("likhosherstov2023favorsharp", "likhosherstov2022chefs", "yu2016orthogonal")
    paper_section = "sec:kernel"

    def __init__(self, r: int = 64, orthogonal: bool = True, seed: int = 0, **kw):
        super().__init__(r=r, orthogonal=orthogonal, seed=seed, **kw)
        self.r, self.orthogonal, self.seed = r, orthogonal, seed

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        qt, kt = q / d ** 0.25, k / d ** 0.25

        sigma = pair_covariance(qt, kt, causal)              # [..., m, d, d], float64
        lam, qmat = canonical_eigh(sigma)                    # [..., m, d], [..., m, d, d]
        a = optimal_a(lam)                                   # <= 0, one per eigendirection
        gain = torch.sqrt(1.0 - 4.0 * a)                     # diag of (I - 4A)^{1/2}
        log_d = 0.25 * torch.log(1.0 - 4.0 * a).sum(-1, keepdim=True)        # [..., m, 1]
        lam, a, gain, log_d, qmat = (t.to(q.dtype)
                                     for t in (lam, a, gain, log_d, qmat))

        om = gaussian_features(self.r, d, self.seed, self.orthogonal, q.device, q.dtype)
        # rows of ``proj`` are the per-query feature directions B_i^T omega_l
        proj = (om * gain.unsqueeze(-2)) @ qmat.transpose(-1, -2)        # [..., m, r, d]
        quad = ((om ** 2) @ a.unsqueeze(-1)).squeeze(-1)     # omega^T A_i omega, [..., m, r]
        qw = torch.einsum("...mrd,...md->...mr", proj, qt)
        kw = torch.einsum("...mrd,...nd->...mnr", proj, kt)

        base = 2.0 * log_d + 2.0 * quad + qw - 0.5 * (qt ** 2).sum(-1, keepdim=True)
        log_phi_k = kw - 0.5 * (kt ** 2).sum(-1)[..., None, :, None]
        log_g = torch.logsumexp(base.unsqueeze(-2) + log_phi_k, dim=-1) - math.log(self.r)

        support = causal_mask(m, n, device=q.device) if causal else None
        pi, z = weights_from_log_magnitude(log_g, support=support)
        extras = {"log_g_hat": log_g, "Z_rel": z, "r": self.r,
                  "eigenvalues": lam, "A_diag": a}
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.r
        return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv
                         + (m + n) * d * d + d ** 3,     # the pair covariance and its eigh
                "state_bytes": r * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return None

    @classmethod
    def convergence_configs(cls):
        return [{"r": 8}, {"r": 32}, {"r": 128}, {"r": 512}]
