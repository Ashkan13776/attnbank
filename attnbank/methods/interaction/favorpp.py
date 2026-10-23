"""FAVOR++: optimal positive random features for the softmax kernel.

``sec:kernel-rf``.  FAVOR+ (``performer``) is one point of a much larger family.  With
``qt = q / d^{1/4}``, ``kt = k / d^{1/4}`` and ``g(q,x) = exp(<qt, kt_x>)``,
Likhosherstov et al. (2022) consider the *generalized exponential* random features

    f_A(omega, u) = D exp(A ||omega||^2 + B omega^T u + C ||u||^2),   omega ~ N(0, I_d),

and show (their Thm. 3.1) that ``E[f_A(omega,qt) f_A(omega,kt)] = g`` exactly when

    B = sqrt(1 - 4A),   C = -1/2,   D = (1 - 4A)^{d/4},   A < 1/4.

``A = 0`` gives ``B = D = 1`` and recovers FAVOR+ verbatim, so this file contains
``performer`` as the special case ``a = 0``.  Every other ``A`` is an equally unbiased
estimator with a different variance, and Thm. 3.3 minimises that variance in closed
form: for a pair ``(qt, kt)`` with ``S = ||qt + kt||^2`` the optimum is

    A* = (1 - 1/rho*) / 8,
    rho* = (sqrt((2S + d)^2 + 8 d S) - 2S - d) / (4 S),

equivalently, writing ``lam = S / d`` for the mean eigenvalue of the pair covariance,

    A* = (1 - 2 lam - sqrt((2 lam + 1)^2 + 8 lam)) / 16

-- the two expressions are identical after rationalising ``1/rho*``.  The second is
the form :func:`optimal_a` uses and the form ``favorsharp`` generalises one
eigendirection at a time.  ``A* <= 0`` always, and ``A* -> 0`` as ``S -> 0``: FAVOR+ is
optimal only in the small-logit limit, and the gain grows without bound with the logit
scale -- the regime in which FAVOR+'s relative variance ``exp(S) - 1`` explodes.

**Where ``A`` comes from here.**  ``A*`` depends on the pair.  The paper fixes one
``A`` per layer from batch averages of ``||x + y||^2`` (their Eq. 8), which is not a
causal function of the context and would make the operator leak -- the normalizer of
an early query would move when a later key changed.  This file instead conditions the
statistic on each query's *visible* support,

    S_i = ||qt_i||^2 + mean_{j visible to i} ||kt_j||^2 + 2 <qt_i, mean_{j} kt_j>,

which is the same average restricted to the pairs that query actually estimates: the
paper averages over ``i`` as well, so the two agree exactly at ``m = 1`` (decoding) and
otherwise differ only by that conditioning.  It is exactly causal and exactly
head-independent.  Unbiasedness is untouched: ``A`` is a deterministic function of the
data and independent of ``omega``, so ``E_omega[g_hat] = g`` for whatever ``A`` is
chosen.  Pass ``a=<float>`` to pin ``A`` instead and recover the separable,
layer-constant form (``a = 0`` is ``performer``, bit for bit).

``cost`` reports that deployed separable form, as ``performer`` does; the
query-conditioned statistic used here costs an extra ``O(m n r)`` and is a property of
this reference implementation, not of the method.

Like FAVOR+ the estimator is positive, so ``pi_hat`` is a genuine probability measure,
and like FAVOR+ it is a ratio of two unbiased estimators and therefore biased at
``O(1/r)``: there is no finite ``r`` at which it is exact, hence
``convergence_configs`` rather than a ``limit_config``.

Reference: Likhosherstov et al. (2022), *Chefs' Random Tables*; Choromanski et al.
(2021) for FAVOR+; Yu et al. (2016) for the orthogonal block construction.
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
    query_positions,
    weights_from_log_magnitude,
)
from ...errors import ConfigError
from ...registry import register
from .performer import gaussian_features


def visible_prefix_mean(t, m: int, causal: bool, extra_dims: int = 0):
    """Mean of ``t`` over the keys each query can see: ``[..., n, *] -> [..., m, *]``.

    ``t`` carries the key axis at position ``-1 - extra_dims``, so the same routine
    averages ``||kt_j||^2`` (``extra_dims = 0``), ``kt_j`` (``1``) and the outer
    products ``kt_j kt_j^T`` (``2``, used by ``favorsharp``).  Under ``causal`` the
    prefix sums follow the library's end alignment: query ``i`` sees ``0 .. i+(n-m)``.
    """
    axis = t.dim() - 1 - extra_dims
    n = t.shape[axis]
    if not causal:
        mean = t.sum(axis, keepdim=True) / n
        shape = list(t.shape)
        shape[axis] = m
        return mean.expand(shape)
    idx = query_positions(m, n, device=t.device)                 # [m]
    count = (idx + 1).to(t.dtype).reshape([-1] + [1] * extra_dims)
    return t.cumsum(axis).index_select(axis, idx) / count


def optimal_a(lam):
    """``A*`` for one eigendirection of the pair covariance with eigenvalue ``lam``.

    ``A*(lam) = (1 - 2 lam - sqrt((2 lam + 1)^2 + 8 lam)) / 16``, the closed form of
    Thm. 3.3 (FAVOR++, isotropic) and of Thm. 4.3 (FAVOR#, per eigendirection) after
    the two are written in the same variable.  ``A*(0) = 0`` (FAVOR+) and
    ``A*`` decreases without bound as ``lam`` grows.
    """
    return (1.0 - 2.0 * lam - torch.sqrt((2.0 * lam + 1.0) ** 2 + 8.0 * lam)) / 16.0


def pair_scale(qt, kt, m: int, causal: bool):
    """``S_i = mean_{j visible to i} ||qt_i + kt_j||^2``, in float64: ``[..., m]``."""
    qt64, kt64 = qt.double(), kt.double()
    sq_k = visible_prefix_mean((kt64 ** 2).sum(-1), m, causal)              # [..., m]
    mean_k = visible_prefix_mean(kt64, m, causal, extra_dims=1)             # [..., m, d]
    return (qt64 ** 2).sum(-1) + sq_k + 2.0 * (qt64 * mean_k).sum(-1)


@register
class FAVORPlusPlusAttention(AttentionMethod):
    """Optimal positive random features (OPRF); FAVOR+ is the case ``a = 0``."""

    name = "favorpp"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True            # f_A > 0, so pi_hat is a probability measure
    references = ("likhosherstov2022chefs", "choromanski2021rethinking", "yu2016orthogonal")
    paper_section = "sec:kernel"

    def __init__(self, r: int = 64, a: float | None = None, orthogonal: bool = True,
                 seed: int = 0, **kw):
        """``a = None`` fits ``A*`` from the visible support; a float pins ``A``."""
        super().__init__(r=r, a=a, orthogonal=orthogonal, seed=seed, **kw)
        if a is not None and a >= 0.25:
            raise ConfigError(
                f"{self.name}: a = {a} but the feature map is only unbiased for "
                "a < 1/4 (Thm. 3.1 of Likhosherstov et al., 2022); B = sqrt(1 - 4a) "
                "is imaginary beyond it. a = 0 is FAVOR+, a < 0 reduces variance.")
        self.r, self.a, self.orthogonal, self.seed = r, a, orthogonal, seed

    def _coefficients(self, qt, kt, causal):
        """``(A, B, log D)`` per query, each ``[..., m, 1]`` in the working dtype."""
        m = qt.shape[-2]
        d = qt.shape[-1]
        if self.a is None:
            lam = (pair_scale(qt, kt, m, causal) / d).clamp_min(0.0)
            a = optimal_a(lam)                                              # [..., m], <= 0
        else:
            a = torch.full(qt.shape[:-1], float(self.a), dtype=torch.float64,
                           device=qt.device)
        b = torch.sqrt(1.0 - 4.0 * a)
        log_d = 0.25 * d * torch.log(1.0 - 4.0 * a)
        to = lambda t: t.unsqueeze(-1).to(qt.dtype)
        return to(a), to(b), to(log_d)

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        qt, kt = q / d ** 0.25, k / d ** 0.25
        a, b, log_d = self._coefficients(qt, kt, causal)                    # [..., m, 1]

        om = gaussian_features(self.r, d, self.seed, self.orthogonal, q.device, q.dtype)
        wn2 = (om ** 2).sum(-1)                                             # [r]
        qw, kw = qt @ om.T, kt @ om.T                       # [..., m, r], [..., n, r]

        # log f_A(omega_l, qt_i) + log f_A(omega_l, kt_j), split into the part that
        # depends on the query only and the part that depends on the key.
        base = 2.0 * log_d + 2.0 * a * wn2 + b * qw - 0.5 * (qt ** 2).sum(-1, keepdim=True)
        log_phi_k = (b.unsqueeze(-1) * kw.unsqueeze(-3)
                     - 0.5 * (kt ** 2).sum(-1)[..., None, :, None])         # [..., m, n, r]
        log_g = torch.logsumexp(base.unsqueeze(-2) + log_phi_k, dim=-1) - math.log(self.r)

        support = causal_mask(m, n, device=q.device) if causal else None
        pi, z = weights_from_log_magnitude(log_g, support=support)
        extras = {"log_g_hat": log_g, "Z_rel": z, "r": self.r,
                  "A": a.squeeze(-1), "B": b.squeeze(-1)}
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.r
        return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv
                         + 2 * (m + n) * d,            # the O(Ld) statistic, Eq. 8
                "state_bytes": r * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return None       # a ratio of unbiased estimators: biased at every finite r

    @classmethod
    def convergence_configs(cls):
        return [{"r": 8}, {"r": 32}, {"r": 128}, {"r": 512}]
