"""Linformer: a fixed linear projection of K and V along the *sequence* axis.

``sec:compress-lowrank``, ``sec:factor-methods``.  With ``E, F in R^{r x n}``,

    P_hat = softmax(beta Q (E K)^T) in R^{m x r},    y_hat = P_hat (F V),

so ``E K`` and ``F V`` are ``r`` learned pseudo-tokens: the method compresses keys and
values separately, changes both ``mu`` and ``v``, fixes ``n``, and is not causal.

It *can* be written on the original positions, and that is how the library stores it:
``y_hat = (P_hat F) V``, so ``weights = P_hat F in R^{m x n}`` with the **original**
values.  Those weights are generally signed and their rows do not sum to one unless
``F`` is row-stochastic, which is the paper's point -- ``tab:comparative`` lists
Linformer as losing ``nm`` and ``hl``.  ``row_stochastic = False`` records it, and the
row sums are returned in ``extras`` so the deviation can be measured.

Note also that the latent keys ``(EK)_z`` enter the softmax *linearly*, and
``exp(beta <q, sum_x E_zx k_x>) != sum_x E_zx exp(beta <q, k_x>)``, so the entries of
``P`` are not approximated at all; the Johnson--Lindenstrauss argument is an existence
statement about the reference operator with random ``E, F``, not an error bound for the
trained operator.

Exact limit: ``r = None`` sets ``E = F = I_n``, which gives dense softmax.

Reference: Wang et al. (2020) Linformer.
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     make_generator, masked_softmax)
from ...registry import register


@register
class LinformerAttention(AttentionMethod):
    """Random (or identity) sequence-axis projections of K and V."""

    name = "linformer"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False
    expresses_weights = True
    row_stochastic = False           # P_hat F is signed and does not sum to one
    references = ("wang2020linformer",)
    paper_section = "sec:compress"

    def __init__(self, r: int | None = 8, seed: int = 0, **kw):
        super().__init__(r=r, seed=seed, **kw)
        self.r, self.seed = r, seed

    def _forward(self, q, k, v, causal, return_weights):
        n, d = k.shape[-2], k.shape[-1]
        if self.r is None or self.r >= n:
            e = f = torch.eye(n, device=q.device, dtype=q.dtype)
        else:
            g = make_generator(self.seed)
            e = (torch.randn(self.r, n, generator=g, dtype=torch.float64)
                 / self.r ** 0.5).to(q.device, q.dtype)
            f = (torch.randn(self.r, n, generator=g, dtype=torch.float64)
                 / self.r ** 0.5).to(q.device, q.dtype)
        k_lat = e @ k                                            # [..., r, d]
        p_hat = masked_softmax(torch.einsum("...md,...rd->...mr", q, k_lat)
                               * self._scale(d))                 # row-stochastic over latents
        pi = p_hat @ f                                           # [..., m, n], signed
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras={"r": k_lat.shape[-2], "row_sum": pi.sum(-1),
                                       "latent_weights": p_hat})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = min(self.r or n, n)
        return {"flops": 2 * n * r * (d + dv) + 2 * m * r * (d + dv),
                "state_bytes": r * (d + dv) * bytes_per_el,
                "bytes_moved": m * (d + dv) * bytes_per_el + r * (d + dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"r": None}
