"""Latent stochastic factorization: two Markov kernels through r latents.

``sec:factor-def``, ``def:factor-stochastic``, ``prop:factor-closure``.  A rank-``r``
latent stochastic factorization is a pair of Markov kernels ``a: Q -> Z`` (read) and
``b: Z -> X`` (write) with

    pi_hat_q(dx) = sum_z a(q, z) b(z, dx),    P_hat = A B,
    A in R_{>=0}^{m x r} with A 1 = 1,   B in R_{>=0}^{r x n} with B 1 = 1.

Both kernels are attentions (``eq:factor-write``, ``eq:factor-read``):

    b(z, .) = softmax_x( beta <w_z, k_x> ),     w_z a latent query,
    a(q, z) = softmax_z( beta <q, kbar_z> ),    kbar_z = int k_x b(z, dx),

the ABC form in which the latent keys summarize the context rather than being free
(Peng et al., 2022).  The latent queries ``w_z`` are the learned parameters; here they
are a seeded Gaussian, so ``trainable = True`` records that the method is only
meaningful after training.

``prop:factor-closure`` is why this category exists separately from low-rank
approximation: a composition of Markov kernels is automatically a probability measure,
so normalization, constant preservation and convex-hull confinement all survive -- none
of which holds for a generic signed rank-``r`` factorization.  The output has the
two-level form ``y_hat(q) = sum_z a(q,z) u(z)`` with ``u(z) = int v db(z,.)``, so the
context is seen only through ``r`` latent values.

This is an **operator modification** (``tab:comparative``): there is no ``r`` at which
it becomes softmax, and nothing it converges to, so both correctness anchors are
``None`` by design.  It loses ``fs`` (full support in the exact sense) and ``ql``.

Not causal: the write kernel pools over the whole context.

Reference: Ma et al. (2021) Luna; Dolga et al. (2024) Latte; Peng et al. (2022) ABC;
Lee et al. (2019) Set Transformer.
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     make_generator, masked_softmax)
from ...registry import register


@register
class LatentStochasticAttention(AttentionMethod):
    """pi_hat = A B with both factors row-stochastic (two-hop attention)."""

    name = "latent_stochastic"
    category = Category.FACTORIZE
    objective = Objective.MODIFY
    supports_causal = False
    expresses_weights = True
    row_stochastic = True
    trainable = True
    references = ("ma2021luna", "dolga2024latte", "peng2022abc", "lee2019set")
    paper_section = "sec:factor"

    def __init__(self, r: int = 8, seed: int = 0, **kw):
        super().__init__(r=r, seed=seed, **kw)
        self.r, self.seed = r, seed

    def _forward(self, q, k, v, causal, return_weights):
        d = k.shape[-1]
        beta = self._scale(d)
        g = make_generator(self.seed)
        w = torch.randn(self.r, d, generator=g, dtype=torch.float64).to(k.device, k.dtype)

        b = masked_softmax(torch.einsum("rd,...nd->...rn", w, k) * beta)   # write kernel
        k_bar = b @ k                                                      # latent keys
        a = masked_softmax(torch.einsum("...md,...rd->...mr", q, k_bar) * beta)  # read kernel
        pi = a @ b
        u = b @ v                                                          # latent values
        return AttentionOutput(y=a @ u, weights=pi if return_weights else None,
                               extras={"a": a, "b": b, "latent_values": u,
                                       "latent_keys": k_bar, "r": self.r})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.r
        return {"flops": 2 * n * r * (d + dv) + 2 * m * r * (d + dv),
                "state_bytes": r * (d + dv) * bytes_per_el,
                "bytes_moved": m * (d + dv) * bytes_per_el + r * (d + dv) * bytes_per_el}
