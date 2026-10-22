"""Random Fourier features for attention: the trigonometric (signed) estimator.

``sec:kernel-rf``.  Bochner's theorem for the Gaussian kernel plus
``eq:kernel-softmax-gauss`` gives the unbiased estimator

    xi_omega(q,x) = exp((||qt||^2 + ||kt_x||^2)/2) * cos(omega^T (qt - kt_x)),
    omega ~ N(0, I_d),   qt = q/d^{1/4},  kt = k/d^{1/4},

separable with ``2r`` features through ``cos(a-b) = cos a cos b + sin a sin b``.  This
is the basis of RFA (Peng et al., 2021) and of the trigonometric variant of
Choromanski et al. (2021).

Two things go wrong, and this file exists so that they can be measured.
(i) The relative variance is ``2 sinh^2(||qt - kt_x||^2 / 2)`` (``eq:kernel-trig-var``),
largest for keys *far* from the query -- the many small entries whose accumulated noise
makes ``Z_hat`` unreliable.
(ii) ``xi`` takes both signs, so ``Z_hat`` can be zero or negative; then ``pi_hat`` is
not a probability measure and no bound of the form ``prop:error-additive`` holds.  The
fraction of queries with ``Z_hat <= 0`` is reported in ``extras["Z_rel"]``.

Numerics.  The factor ``exp(||kt_j||^2/2)`` does not cancel in ``pi_hat`` and overflows
fp32 for large inputs, so the estimator is carried as a (log-magnitude, sign) pair and
normalized by :func:`attnbank.weights_from_log_magnitude`, which subtracts the maximum
over the *visible* keys.  When ``|Z_hat|`` is below the floor the result is finite but
large; this is a reported failure, not a silent repair.

Reference: Rahimi and Recht (2007); Peng et al. (2021); Choromanski et al. (2021).
"""
from __future__ import annotations

import math

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective, causal_mask,
                     weights_from_log_magnitude)
from ...registry import register
from .performer import gaussian_features


@register
class TrigRFFAttention(AttentionMethod):
    """Rahimi--Recht trigonometric random features; signed, unnormalized measure."""

    name = "trigrff"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = False           # xi is signed: Z_hat may be <= 0 (tab:comparative, nm)
    references = ("rahimi2007random", "peng2021random", "choromanski2021rethinking")
    paper_section = "sec:kernel"

    def __init__(self, r: int = 64, orthogonal: bool = True, seed: int = 0, **kw):
        super().__init__(r=r, orthogonal=orthogonal, seed=seed, **kw)
        self.r, self.orthogonal, self.seed = r, orthogonal, seed

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        qt, kt = q / d ** 0.25, k / d ** 0.25
        om = gaussian_features(self.r, d, self.seed, self.orthogonal, q.device, q.dtype)
        a, b = qt @ om.T, kt @ om.T                       # [..., m, r], [..., n, r]
        # (1/r) sum_l cos(a_il - b_jl), computed from the separable identity
        cq, sq = torch.cos(a), torch.sin(a)
        ck, sk = torch.cos(b), torch.sin(b)
        cossum = (cq @ ck.transpose(-1, -2) + sq @ sk.transpose(-1, -2)) / self.r

        # g_hat = exp(||qt||^2/2) * exp(||kt_j||^2/2) * cossum ; the query factor cancels
        log_pref = 0.5 * (kt ** 2).sum(-1).unsqueeze(-2)   # [..., 1, n]
        tiny = torch.finfo(q.dtype).tiny
        log_g = log_pref + torch.log(cossum.abs().clamp_min(tiny))
        sign = torch.sign(cossum)

        support = causal_mask(m, n, device=q.device) if causal else None
        pi, z = weights_from_log_magnitude(log_g, sign=sign, support=support)
        y = pi @ v
        extras = {"log_g_hat": log_g, "sign": sign, "Z_rel": z,
                  "negative_Z": (z <= 0), "r": self.r}
        return AttentionOutput(y=y, weights=pi if return_weights else None, extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = 2 * self.r                 # cos and sin halves
        return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv,
                "state_bytes": r * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return None

    @classmethod
    def convergence_configs(cls):
        return [{"r": 8}, {"r": 32}, {"r": 128}, {"r": 512}]
