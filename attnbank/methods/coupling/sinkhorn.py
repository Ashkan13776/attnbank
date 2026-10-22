"""Sinkhorn / entropic-OT attention: prescribe the column marginal too.

``sec:ot-eot``.  Softmax attention is *semi-relaxed* entropic OT: it minimizes
``int c dgamma + eps KL(gamma || nu x mubar)`` subject to the **row** marginal only
(``prop:ot-gibbs``).  Adding the column constraint gives entropic OT, and the central
structural fact (``prop:ot-conditional``) is that its conditionals are still softmax --
tilted by a *query-independent key potential*:

    pi^OT_q(dx) ∝ exp( beta ( <q, k_x> + psi(x) ) ) mubar(dx).

So a doubly stochastic attention operator is characterised, completely, by one vector
``psi`` per context: an economic price that is negative on keys over-subscribed under
softmax and positive on under-subscribed ones.  ``extras["potential"]`` returns the
quantity ``psi`` in the form in which the identity can be checked directly,
``pi_hat = softmax_j(s_ij / eps + potential_j)``, which is what study S7 does.

Sinkhorn's algorithm alternates the two identities of ``prop:ot-conditional`` (b) in
log domain (``eq:ot-sinkhorn``): a row log-sum-exp (a softmax normalizer) followed by a
column log-sum-exp.  **One step from ``psi = 0``, stopped after the row update, is
exactly softmax attention** -- so ``n_iter = 0`` is the exact limit, and the whole
operator family is a continuation of softmax in the number of iterations.

Not causal, and not by omission: ``prop:ot-causal`` shows that a lower-triangular
matrix with equal row and column sums *is diagonal*.  Causal attention with uniform
marginals is the identity, every token attending only to itself.  A non-uniform column
marginal satisfying ``sum_{j>=t} b_j <= sum_{i>=t} a_i`` is the way out, and is not
implemented here.

Reference: Cuturi (2013); Sinkhorn (1964, 1967); Sander et al. (2022) Sinkformers;
Peyre and Cuturi (2019).
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, scores
from ...registry import register


def sinkhorn_log(logk, log_row, log_col, n_iter):
    """Log-domain IPFP for ``Gamma = diag(a) exp(logk) diag(b)`` with prescribed marginals.

    ``logk`` is ``-c/eps`` (shape [..., m, n]), ``log_row`` and ``log_col`` are the log
    target marginals.  Returns the potentials ``(f, g)`` with
    ``log Gamma = logk + f[..., :, None] + g[..., None, :]``.
    """
    f = torch.zeros(logk.shape[:-1], dtype=logk.dtype, device=logk.device)
    g = torch.zeros(logk.shape[:-2] + logk.shape[-1:], dtype=logk.dtype, device=logk.device)
    for _ in range(n_iter):
        f = log_row - torch.logsumexp(logk + g.unsqueeze(-2), dim=-1)
        g = log_col - torch.logsumexp(logk + f.unsqueeze(-1), dim=-2)
    return f, g


@register
class SinkhornAttention(AttentionMethod):
    """Entropic-OT attention: Sinkhorn normalization of exp(s) to prescribed marginals."""

    name = "sinkhorn"
    category = Category.COUPLING
    objective = Objective.MODIFY
    supports_causal = False          # prop:ot-causal: causal + uniform marginals = identity
    expresses_weights = True
    row_stochastic = True            # the conditionals are probability measures
    references = ("cuturi2013sinkhorn", "sinkhorn1964relationship", "sander2022sinkformers",
                  "peyre2019computational")
    paper_section = "sec:ot"

    def __init__(self, n_iter: int = 5, eps: float = 1.0, **kw):
        """``n_iter``: Sinkhorn iterations (0 = softmax exactly).
        ``eps``: entropic regularization, applied as a temperature on the scaled logits."""
        super().__init__(n_iter=n_iter, eps=eps, **kw)
        self.n_iter, self.eps = n_iter, eps

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        s = scores(q, k, self._scale(q.shape[-1]), causal) / self.eps     # = -c/eps
        log_row = torch.full(s.shape[:-1], -torch.log(torch.tensor(float(m))).item(),
                             dtype=s.dtype, device=s.device)
        log_col = torch.full(s.shape[:-2] + s.shape[-1:],
                             -torch.log(torch.tensor(float(n))).item(),
                             dtype=s.dtype, device=s.device)
        f, g = sinkhorn_log(s, log_row, log_col, self.n_iter)

        # prop:ot-conditional (b): the conditional is softmax of the tilted logits
        potential = g + log_col                      # psi/eps + log mubar
        pi = torch.softmax(s + potential.unsqueeze(-2), dim=-1)
        gamma = torch.exp(s + f.unsqueeze(-1) + g.unsqueeze(-2))
        extras = {"potential": potential, "f": f, "g": g, "gamma": gamma,
                  "col_marginal": gamma.sum(-2), "row_marginal": gamma.sum(-1),
                  "target_col": torch.exp(log_col), "target_row": torch.exp(log_row),
                  "marginal_error": (gamma.sum(-1) - torch.exp(log_row)).abs().amax(-1)
                                    + (gamma.sum(-2) - torch.exp(log_col)).abs().amax(-1),
                  "n_iter": self.n_iter}
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None, extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        return {"flops": 2 * m * n * (d + dv) + 2 * self.n_iter * m * n,
                "state_bytes": n * (d + dv) * bytes_per_el + n * 4,   # + the key potential
                "bytes_moved": (m * d + n * (d + dv) + m * dv
                                + 2 * self.n_iter * m * n) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        # one step from psi = 0, stopped after the row update, is exactly softmax
        return {"n_iter": 0, "eps": 1.0}
