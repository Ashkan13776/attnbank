"""LOTFormer-style factored coupling: transport through r pivots.

``sec:ot`` ("Factorization through a pivot measure"), ``eq:ot-glued``, and
``eq:factor-pivot``.  Let ``rho`` be a probability measure on ``r`` pivots.  Given
``gamma_1 in Pi(nu, rho)`` and ``gamma_2 in Pi(rho, mubar)``, the glued coupling

    gamma(dq, dx) = sum_z rho_z gamma_1(dq | z) gamma_2(dx | z)
                  = Gamma_1 diag(rho)^{-1} Gamma_2

lies in ``Pi(nu, mubar)``, has rank at most ``r``, and has query conditional
``pi^piv_q = sum_z gamma_1(z | q) gamma_2(. | z)``.  Both halves are solved by
Sinkhorn on dot-product costs (queries-to-pivots, pivots-to-keys), so the whole thing
costs ``O((m + n) r)`` and never forms the ``m x n`` matrix.

Two readings of the same object.  As a *coupling* it is doubly stochastic and so
balances participation of context positions -- no key absorbs the mass of all queries.
As a *factorization* it is exactly ``def:factor-stochastic``: ``A_{iz} = gamma_1(z|q_i)``
and ``B_{zj} = gamma_2(j|z)`` are both row-stochastic, so ``prop:factor-closure``
applies and ``pi_hat`` is a probability measure by construction.  It is filed under
``coupling`` because the column constraint, absent from the row-stochastic
factorizations of ``sec:factor``, is the thing it adds.

It is a **new operator**: a doubly stochastic coupling does not approximate softmax,
whose column sums are unconstrained, and it is not claimed to approximate the full
plan of ``prop:ot-conditional`` either.  Hence no exact limit and no convergence
sequence -- the pivots are learned end to end, so ``trainable = True``.

Not causal, for the reason in ``prop:ot-causal``.

Reference: Shahbazi et al. (2025) LOTFormer; Scetbon et al. (2021) low-rank Sinkhorn;
Forrow et al. (2019); Mialon et al. (2021) OTKE.
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     make_generator)
from ...registry import register
from .sinkhorn import sinkhorn_log


@register
class LowRankOTAttention(AttentionMethod):
    """Doubly stochastic, rank-<= r attention through a pivot measure."""

    name = "lowrank_ot"
    category = Category.COUPLING
    objective = Objective.MODIFY
    supports_causal = False
    expresses_weights = True
    row_stochastic = True
    trainable = True
    references = ("shahbazi2025lotformer", "scetbon2021lowrank", "forrow2019statistical",
                  "mialon2021trainable")
    paper_section = "sec:ot"

    def __init__(self, r: int = 4, n_iter: int = 10, eps: float = 1.0, seed: int = 0, **kw):
        super().__init__(r=r, n_iter=n_iter, eps=eps, seed=seed, **kw)
        self.r, self.n_iter, self.eps, self.seed = r, n_iter, eps, seed

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        beta = self._scale(d)
        r = min(self.r, m, n)
        g = make_generator(self.seed)
        pivots = torch.randn(r, d, generator=g, dtype=torch.float64).to(q.device, q.dtype)

        log_nu = torch.full(q.shape[:-1], -float(torch.tensor(float(m)).log()),
                            dtype=q.dtype, device=q.device)
        log_rho = torch.full(q.shape[:-2] + (r,), -float(torch.tensor(float(r)).log()),
                             dtype=q.dtype, device=q.device)
        log_mu = torch.full(k.shape[:-1], -float(torch.tensor(float(n)).log()),
                            dtype=q.dtype, device=q.device)

        s1 = torch.einsum("...md,rd->...mr", q, pivots) * beta / self.eps
        s2 = torch.einsum("rd,...nd->...rn", pivots, k) * beta / self.eps
        f1, g1 = sinkhorn_log(s1, log_nu, log_rho, self.n_iter)
        f2, g2 = sinkhorn_log(s2, log_rho, log_mu, self.n_iter)

        a = torch.softmax(s1 + g1.unsqueeze(-2), dim=-1)      # gamma_1(z | q): [..., m, r]
        b = torch.softmax(s2 + g2.unsqueeze(-2), dim=-1)      # gamma_2(x | z): [..., r, n]
        pi = a @ b
        return AttentionOutput(y=a @ (b @ v), weights=pi if return_weights else None,
                               extras={"a": a, "b": b, "pivots": pivots, "r": r,
                                       "col_marginal": pi.mean(-2)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.r
        return {"flops": 2 * (m + n) * r * d + 2 * (m + n) * r * dv
                        + 2 * self.n_iter * (m + n) * r,
                "state_bytes": r * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
