"""MagicPIG-style importance sampling of keys with unbiased reweighting.

``sec:sparse-content`` ("Sampling is not restriction") and ``prop:error-ratio-est``.
A key set ``S`` is *sampled* with inclusion probabilities ``u_j`` monotone in the
attention score, and the output is the self-normalized importance-sampling estimator
(``eq:sparse-snis``)

    y_hat(q) = sum_{j in S} (g_j / u_j) v_j  /  sum_{j in S} (g_j / u_j),
    pi_hat_j = 1{j in S} (pi_j / u_j) / sum_i 1{i in S} (pi_i / u_i).

Numerator and normalizer are unbiased for ``N(q)`` and ``Z(q)``; the ratio is only
consistent, with ``O(1/N_s)`` bias (Owen, 2013).  Unlike deterministic restriction this
does **not** systematically reallocate tail mass to the head -- it trades the
value-dependent bias of ``prop:sparse-identity`` for variance, whose size is
``(1 + chi2(pi || lambda)) / N_s`` times ``sup_x ||v(x) - y||^2``
(``prop:error-ratio-est`` (b)).

Proposal.  MagicPIG derives ``u_j`` from SimHash collision probabilities; this
reference implementation abstracts that to a tempered proposal
``lambda_j ∝ exp(alpha * s_j)`` on the visible keys, with ``alpha = 0`` uniform and
``alpha = 1`` tracking ``pi`` exactly, and Poisson (independent Bernoulli) sampling at
``u_j = min(1, N_s * lambda_j)``.  Sweeping ``alpha`` is how ``chi2(pi || lambda)``,
the quantity the bound is stated in, is varied.

Exact limit.  ``n_samples = None`` sets ``u_j = 1`` for every visible key, so ``S`` is
the whole visible context and ``pi_hat = pi`` exactly.

Degenerate draw.  If a row samples nothing, the single largest-score visible key is
inserted so the estimator is defined; the event is counted in
``extras["empty_draws"]``.

Reference: Chen et al. (2025) MagicPIG; Owen (2013); Kitaev et al. (2020) Reformer.
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, Category, Objective, make_generator,
                     masked_softmax, normalize_nonnegative, AttentionOutput)
from ...registry import register
from ._restriction import reference_pi, visible


@register
class LSHSampleAttention(AttentionMethod):
    """Self-normalized importance sampling over a score-monotone proposal."""

    name = "lsh_sample"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True            # pi_j / u_j >= 0, renormalized
    references = ("chen2025magicpig", "owen2013monte", "kitaev2020reformer")
    paper_section = "sec:sparse"

    def __init__(self, n_samples: int | None = 8, alpha: float = 1.0, seed: int = 0, **kw):
        super().__init__(n_samples=n_samples, alpha=alpha, seed=seed, **kw)
        self.n_samples, self.alpha, self.seed = n_samples, alpha, seed

    def _forward(self, q, k, v, causal, return_weights):
        pi, s = reference_pi(self, q, k, causal)
        vis = visible(q, k, causal)

        if self.n_samples is None:
            u = vis.to(pi.dtype)
            take = vis.clone()
            empty = torch.zeros(pi.shape[:-1], dtype=torch.bool, device=pi.device)
        else:
            lam = masked_softmax(s * self.alpha, support=vis)       # proposal
            u = (self.n_samples * lam).clamp(max=1.0) * vis
            g = make_generator(self.seed, device=None)
            draw = torch.rand(u.shape, generator=g, dtype=torch.float64).to(u.device, u.dtype)
            take = (draw < u) & vis
            empty = ~take.any(-1)
            if bool(empty.any()):                                   # documented fallback
                top = s.masked_fill(~vis, float("-inf")).argmax(-1, keepdim=True)
                take = take | (torch.zeros_like(take).scatter_(-1, top, True)
                               & empty.unsqueeze(-1))
                u = torch.where(take & (u <= 0), torch.ones_like(u), u)

        w = torch.where(take, pi / u.clamp_min(torch.finfo(pi.dtype).tiny),
                        torch.zeros_like(pi))
        pi_hat, mass = normalize_nonnegative(w, take)
        chi2_prop = ((pi / u.clamp_min(1e-30)) * vis * pi).sum(-1) - 1.0
        extras = {"support": take, "u": u, "n_sampled": take.sum(-1),
                  "eps": 1.0 - (pi * take).sum(-1), "empty_draws": empty,
                  "chi2_pi_lambda": chi2_prop}
        return AttentionOutput(y=pi_hat @ v, weights=pi_hat if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        ns = self.n_samples or n
        return {"flops": 2 * m * n * d + 2 * m * ns * dv,
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * d + m * ns * dv + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"n_samples": None}

    @classmethod
    def convergence_configs(cls):
        return [{"n_samples": 2}, {"n_samples": 4}, {"n_samples": 8}, {"n_samples": 14}]
