"""OTKE: transport the context onto a trainable reference, then read it out.

``sec:ot``, "The same constraint elsewhere", and ``sec:ot-implications``.  The optimal
transport kernel embedding of Mialon et al. (2021) is not an attention layer -- it is a
*pooling* layer, with no queries at all.  It embeds a set by transporting it onto a
trainable reference ``Z`` of ``p`` points and aggregating along the plan
(their Definition 3.1, eq. 5):

    P = argmin_{P in Pi(1_n/n, 1_p/p)}  -sum_ij <z_s, x_j> P_js - eps H(P),
    Phi_Z(x)_s = sqrt(p) * sum_j P_js phi(x_j).

The plan is computed by Sinkhorn, so each *reference slot* ends up with a normalized
weighting of the whole set, ``W_sj = p P_js``, and ``Phi`` is the stack of the ``p``
weighted means.  The paper's own reading of the construction is the one that places it
here: elements are pooled together when they align with *the same part of the
reference*, and the column constraint is what stops one slot from swallowing the set.

**What this file does, and what it adds.**  To fit the bank's ``q, k, v`` interface the
layer needs a query, and OTKE does not have one -- downstream it flattens ``Phi`` into a
classifier.  The transport half is OTKE verbatim: keys to a trainable reference, uniform
marginals, Sinkhorn, aggregation along the plan.  The readout half, ``a_is =
softmax_s(beta <q_i, z_s>)``, is *supplied here* and is not part of the paper.  The
operator is therefore

    pi_hat = A W ,     A in R^{m x p} row-stochastic,  W in R^{p x n} row-stochastic,

a pivot factorization (``def:factor-stochastic``) of rank at most ``p``, exactly as
``sec:ot`` classifies OTKE.  ``lowrank_ot`` is the neighbouring object and the contrast
is the point: it Sinkhorns *both* halves, so the full coupling is doubly stochastic;
OTKE Sinkhorns only the key half, so the constraint binds between keys and slots and
each key is guaranteed its share of the *reference's* attention, not of the queries'.
``extras["slot_col_marginal"]`` is that guarantee, equal to ``1/n`` on every key.

``phi`` is a kernel embedding, Nystrom-approximated in the paper when the kernel is
infinite dimensional; a linear kernel needs no approximation and is the case
implemented, so ``phi(x_j) = v_j`` and the aggregation is a plain weighted mean of
values.  The ``sqrt(p)`` factor of Definition 3.1 is a scaling of the embedding, not of
the conditional, and drops out of a row-stochastic operator.

``extras`` exposes the transport diagnostics in the ``sinkhorn`` spelling -- ``f``,
``g``, ``potential``, ``gamma``, ``row_marginal``, ``col_marginal``,
``marginal_error`` -- for the key-to-reference coupling, which is the coupling the
method computes.  ``prop:ot-conditional`` applies to it unchanged: a slot's weighting of
the keys is softmax of ``beta <z_s, k_j>`` tilted by one potential per key.

``trainable = True``, and not as a formality: ``Z`` carries the method.  The paper
initializes it by k-means on the training features and learns it end to end through the
differentiable Sinkhorn; here it is drawn from a seeded ``torch.Generator`` so that the
library reproduces, which makes the untrained operator arbitrary in the same way
``lowrank_ot``'s pivots are.

Not causal: the plan is computed over the whole key set at once, and the column
constraint is the reason ``prop:ot-causal`` applies to this family.

Reference: Mialon et al. (2021) OTKE; Cuturi (2013); Peyre and Cuturi (2019);
Naderializadeh et al. (2021) sliced-Wasserstein pooling.
"""
from __future__ import annotations

import math

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, make_generator
from ...registry import register
from .sinkhorn import sinkhorn_log


@register
class OTKEAttention(AttentionMethod):
    """Sinkhorn transport of the keys onto a trainable reference of p points."""

    name = "otke"
    category = Category.COUPLING
    objective = Objective.MODIFY
    supports_causal = False
    expresses_weights = True
    row_stochastic = True
    trainable = True                 # the reference IS the method
    references = ("mialon2021trainable", "cuturi2013sinkhorn", "peyre2019computational",
                  "naderializadeh2021pooling")
    paper_section = "sec:ot"

    def __init__(self, p: int = 4, n_iter: int = 10, eps: float = 1.0, seed: int = 0, **kw):
        """``p``: size of the trainable reference. ``n_iter``: Sinkhorn iterations
        (the paper reports ten is enough). ``eps``: entropic regularization.
        ``seed``: the generator the untrained reference is drawn from."""
        super().__init__(p=p, n_iter=n_iter, eps=eps, seed=seed, **kw)
        self.p, self.n_iter, self.eps, self.seed = p, n_iter, eps, seed

    def _forward(self, q, k, v, causal, return_weights):
        n, d = k.shape[-2], q.shape[-1]
        beta = self._scale(d)
        p = self.p
        gen = make_generator(self.seed)
        ref = torch.randn(p, d, generator=gen, dtype=torch.float64).to(q.device, q.dtype)

        # the OTKE plan: keys to reference, uniform marginals, cost -<z_s, k_j>
        s_ref = torch.einsum("pd,...nd->...pn", ref, k) * beta / self.eps
        log_rho = torch.full(s_ref.shape[:-1], -math.log(float(p)),
                             dtype=q.dtype, device=q.device)
        log_mu = torch.full(s_ref.shape[:-2] + (n,), -math.log(float(n)),
                            dtype=q.dtype, device=q.device)
        f, g = sinkhorn_log(s_ref, log_rho, log_mu, self.n_iter)

        potential = g + log_mu                               # psi/eps + log mubar
        w = torch.softmax(s_ref + potential.unsqueeze(-2), dim=-1)   # W: [..., p, n]
        a = torch.softmax(torch.einsum("...md,pd->...mp", q, ref) * beta, dim=-1)
        pi = a @ w
        gamma = torch.exp(s_ref + f.unsqueeze(-1) + g.unsqueeze(-2))

        extras = {"reference": ref, "a": a, "w": w, "p": p,
                  "potential": potential, "f": f, "g": g, "gamma": gamma,
                  "row_marginal": gamma.sum(-1), "col_marginal": gamma.sum(-2),
                  "target_row": torch.exp(log_rho), "target_col": torch.exp(log_mu),
                  "marginal_error": (gamma.sum(-1) - torch.exp(log_rho)).abs().amax(-1)
                                    + (gamma.sum(-2) - torch.exp(log_mu)).abs().amax(-1),
                  "slot_col_marginal": w.mean(-2), "n_iter": self.n_iter}
        return AttentionOutput(y=a @ (w @ v), weights=pi if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        p = self.p
        return {"flops": 2 * (m + n) * p * d + 2 * (n + m) * p * dv
                         + 2 * self.n_iter * n * p,
                # the plan needs every key, so the reference buys rank, not capacity
                "state_bytes": n * (d + dv) * bytes_per_el + p * d * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + p * d) * bytes_per_el}
