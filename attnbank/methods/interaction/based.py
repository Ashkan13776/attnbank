"""Based: the truncated-Taylor feature map for the exponential interaction.

``sec:kernel-det``, ``sec:kernel-rf``.  Arora et al. (2024) replace
``g(q,x) = exp(<qt, kt_x>)`` -- with ``qt = q / d^{1/4}``, so that ``<qt, kt_x>`` is
the scaled logit -- by its second-order Taylor polynomial

    sim(q, k) = 1 + q^T k + (q^T k)^2 / 2,

which is *exactly* separable: ``phi(u) = [1, u, u (x) u / sqrt 2]`` gives
``<phi(q), phi(k)> = sim(q,k)`` with ``r = 1 + d + d^2`` features, and the usual
normalized readout ``y_i = phi(q_i)^T S / phi(q_i)^T z``.  This file implements the
interaction and the readout; Based the *architecture* interleaves it with exact
softmax over a short sliding window (``w = 64``), which in this taxonomy is a separate
``restrict/`` operator and not part of ``g_hat``.

Unlike every other fixed feature map in this directory, this one has a resource
parameter that actually converges.  ``degree`` sweeps the truncation order,

    g_hat(q, x) = sum_{p=0}^{P} <qt, kt_x>^p / p!   ->   exp(<qt, kt_x>) = g(q, x),

uniformly on bounded logits, so Based is an *approximation* with a genuine anchor
rather than an operator modification -- the first entry in the family for which that
is true without randomness.  ``convergence_configs`` sweeps ``P``; there is no finite
``P`` that is exact, so ``limit_config`` is ``None``.

Only **even** ``P`` is accepted, and this is not a convenience: the truncated
exponential series has no real root exactly when ``P`` is even, which is what makes
``g_hat > 0``, ``pi_hat`` a probability measure and ``row_stochastic`` an honest
declaration.  At odd ``P`` the surrogate goes negative for sufficiently negative
logits and the measure is lost.  ``P = 2`` is the paper's choice and the default; its
minimum ``1/2`` at logit ``-1`` is the floor that ``rebased`` exists to remove --
Based cannot assign a near-zero weight however negative the logit.

Reference: Arora et al. (2024) Based; Katharopoulos et al. (2020) for the normalized
separable readout this shares with every other fixed feature map here.
"""
from __future__ import annotations

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    causal_mask,
    normalize_nonnegative,
)
from ...errors import ConfigError
from ...registry import register


def taylor_exp(z, degree: int):
    """``sum_{p=0}^{degree} z^p / p!``, accumulated term by term.

    Evaluated as ``t_p = t_{p-1} z / p`` rather than through ``z ** p / p!`` so that
    the factorial never appears on its own; for ``degree = 2`` this is literally
    Based's ``1 + z + z^2 / 2``.
    """
    term = torch.ones_like(z)
    total = term
    for p in range(1, degree + 1):
        term = term * z / p
        total = total + term
    return total


@register
class BasedAttention(AttentionMethod):
    """Second-order (by default) Taylor polynomial of ``exp``, normalized."""

    name = "based"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True            # the even-order truncation of exp is positive
    references = ("arora2024based", "katharopoulos2020transformers")
    paper_section = "sec:kernel"

    def __init__(self, degree: int = 2, seed: int = 0, **kw):
        """``degree`` is the (even) Taylor order ``P``; ``seed`` is accepted and unused.

        The estimator is deterministic -- there is nothing to seed. The keyword exists
        because the conformance suite's convergence harness sweeps ``seed`` alongside
        ``convergence_configs`` for every method that declares one.
        """
        super().__init__(degree=degree, seed=seed, **kw)
        if degree < 2 or degree % 2:
            raise ConfigError(
                f"{self.name}: degree must be a positive even integer, got {degree!r}. "
                "Odd truncations of the exponential series take negative values, so "
                "pi_hat would stop being a probability measure.")
        self.degree, self.seed = degree, seed

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        z = torch.einsum("...md,...nd->...mn", q, k) * self._scale(d)
        g_hat = taylor_exp(z, self.degree)                      # > 0 entrywise
        support = causal_mask(m, n, device=q.device) if causal else None
        pi, norm = normalize_nonnegative(g_hat, support)
        extras = {"g_hat": g_hat, "Z_hat": norm, "degree": self.degree,
                  "r": self.feature_dim(d)}
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras=extras)

    def feature_dim(self, d: int) -> int:
        """``sum_{p<=P} d^p``: the width of the tensor-power feature map Based builds."""
        return sum(d ** p for p in range(self.degree + 1))

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.feature_dim(d)
        return {"flops": 2 * n * r * dv + 2 * m * r * dv + (m + n) * r,
                "state_bytes": r * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return None        # no finite truncation reproduces exp

    @classmethod
    def convergence_configs(cls):
        return [{"degree": 2}, {"degree": 4}, {"degree": 6}, {"degree": 8}]
