"""Partial-assignment attention: Sinkhorn on a score matrix with a dustbin.

``sec:ot``, "Unbalanced relaxation".  A doubly stochastic plan forces every query to
spend all of its mass on real keys and every key to receive exactly its share.  The
discrete way to buy slack is the one SuperGlue uses for unmatchable keypoints
(Sarlin et al. 2020, eq. 8-9): augment the scores with one extra row and one extra
column -- a *dustbin* -- filled with a single scalar ``z``,

    Sbar[i, j]   = beta <q_i, k_j> / eps      i <= m, j <= n
    Sbar[i, n+1] = Sbar[m+1, j] = Sbar[m+1, n+1] = z / eps,

and prescribe marginals that give the bin exactly the surplus capacity,

    a = [1_m, n] / (m + n),      b = [1_n, m] / (m + n),

so that both sides carry unit total mass.  Sinkhorn on ``exp(Sbar)`` then returns a
*partial* assignment: the ``m x n`` block is a coupling of what was matched, and the
dustbin entries hold what was not.

As an attention operator the conditional is read off the augmented row and
renormalized over the real keys.  Because the plan is still of Gibbs form, this is
exact rather than a projection:

    pi_hat_i(j) = softmax_j ( s_ij + g_j ),        j = 1 .. n,

with ``g`` the column potential restricted to the real keys -- ``prop:ot-conditional``
again, with the tilt now computed in the presence of the bin.  The mass the query sent
to the bin, ``extras["dustbin_mass"]``, is the quantity renormalization removes; it is
the *abstention* the operator permits, and it is reported rather than hidden.

Two limits bracket the method.  As ``z -> -infinity`` the bin becomes unattractive, the
surplus is absorbed by the bin-to-bin cell, and the real block approaches the doubly
stochastic plan of ``sinkhorn``.  As ``z`` grows, queries divert mass to the bin and
``dustbin_mass -> 1``; the renormalized conditional stays finite and tends to the
conditional induced by the bin row's own tilt.  The default ``dustbin = 0.0`` sits
between them; in SuperGlue ``z`` is a learned parameter, which is the only sense in
which this operator has one -- it is a scalar, and the operator is meaningful at any
value of it, so ``trainable`` stays ``False``.

One cost of the deep-bin limit is worth recording, because it is not visible in the
construction.  As ``z`` falls, the bin's entries vanish relative to the real block and
the augmented matrix loses *total support* in the sense of Sinkhorn (1967); the scaling
iterations then give up their geometric rate.  Measured at ``z = -40`` the distance to
the balanced plan decays like ``1/n_iter``, so recovering ``sinkhorn`` this way costs
thousands of iterations where ``sinkhorn`` itself needs tens.  At the useful values of
``z`` -- those where abstention is actually happening -- convergence is geometric and
the default ``n_iter = 20`` is ample.

This is the discrete counterpart of ``unbalanced_sinkhorn``: a hard constraint with an
escape hatch, rather than a soft penalty.  ``sec:ot`` lists both, alongside explicit
register tokens, as ways to keep abstention under a column constraint.

Not causal: ``prop:ot-causal``.  The dustbin does relax the marginals -- the
compatibility inequality ``eq:ot-hall`` is no longer tight -- but the slack it provides
is a single lump, not the position-decreasing column marginal that the proposition says
would be needed, and the plan of a prefix still changes when a token arrives.

Reference: Sarlin et al. (2020) SuperGlue; Cuturi (2013); Sinkhorn (1964);
Peyre and Cuturi (2019).
"""
from __future__ import annotations

import math

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, scores
from ...registry import register
from .sinkhorn import sinkhorn_log


@register
class DustbinSinkhornAttention(AttentionMethod):
    """Sinkhorn with a dustbin row and column: a coupling that allows abstention."""

    name = "dustbin_sinkhorn"
    category = Category.COUPLING
    objective = Objective.MODIFY
    supports_causal = False          # prop:ot-causal
    expresses_weights = True
    row_stochastic = True            # after the documented renormalization over real keys
    references = ("sarlin2020superglue", "cuturi2013sinkhorn",
                  "sinkhorn1964relationship", "peyre2019computational")
    paper_section = "sec:ot"

    def __init__(self, n_iter: int = 20, eps: float = 1.0, dustbin: float = 0.0, **kw):
        """``dustbin``: the scalar ``z`` filling the extra row and column (learned in
        SuperGlue).  Very negative recovers ``sinkhorn``; larger values let queries
        abstain.  ``n_iter``, ``eps``: as for ``sinkhorn``."""
        super().__init__(n_iter=n_iter, eps=eps, dustbin=dustbin, **kw)
        self.n_iter, self.eps, self.dustbin = n_iter, eps, dustbin

    def _augment(self, s, m, n):
        """``Sbar``: ``s`` bordered by the dustbin score on the last row and column."""
        z = self.dustbin / self.eps
        row = torch.full(s.shape[:-2] + (1, n), z, dtype=s.dtype, device=s.device)
        col = torch.full(s.shape[:-2] + (m + 1, 1), z, dtype=s.dtype, device=s.device)
        return torch.cat([torch.cat([s, row], dim=-2), col], dim=-1)

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        s = scores(q, k, self._scale(q.shape[-1]), causal) / self.eps
        sbar = self._augment(s, m, n)                        # [..., m+1, n+1]

        total = float(m + n)
        a = torch.cat([torch.ones(m, dtype=s.dtype, device=s.device),
                       torch.tensor([float(n)], dtype=s.dtype, device=s.device)]) / total
        b = torch.cat([torch.ones(n, dtype=s.dtype, device=s.device),
                       torch.tensor([float(m)], dtype=s.dtype, device=s.device)]) / total
        log_a = a.log().expand(sbar.shape[:-1])
        log_b = b.log().expand(sbar.shape[:-2] + (n + 1,))
        f, g = sinkhorn_log(sbar, log_a, log_b, self.n_iter)

        # the augmented conditional of a real query, in log space, then split
        logits = sbar[..., :m, :] + g.unsqueeze(-2)          # [..., m, n+1]
        log_den = torch.logsumexp(logits, dim=-1)
        dustbin_mass = torch.exp(logits[..., n] - log_den)   # abstention per query
        potential = g[..., :n] + math.log(1.0 / n)           # psi/eps + log mubar
        pi = torch.softmax(s + potential.unsqueeze(-2), dim=-1)

        gamma_bar = torch.exp(sbar + f.unsqueeze(-1) + g.unsqueeze(-2))
        gamma = gamma_bar[..., :m, :n]
        extras = {"potential": potential, "f": f[..., :m], "g": g[..., :n],
                  "f_bin": f[..., m], "g_bin": g[..., n],
                  "gamma": gamma, "gamma_bar": gamma_bar,
                  # the marginals of the REAL block: each falls short of the prescribed
                  # 1/(m+n) by exactly the mass the corresponding dustbin cell holds, so
                  # these two are inequalities where sinkhorn's are equalities.  The
                  # equalities live on the augmented plan, and that is what
                  # marginal_error measures.
                  "col_marginal": gamma.sum(-2), "row_marginal": gamma.sum(-1),
                  "target_col": torch.full_like(gamma[..., 0, :], 1.0 / total),
                  "target_row": torch.full_like(gamma[..., 0], 1.0 / total),
                  "marginal_error": (gamma_bar.sum(-1) - a).abs().amax(-1)
                                    + (gamma_bar.sum(-2) - b).abs().amax(-1),
                  "dustbin_mass": dustbin_mass,
                  "bin_to_keys": gamma_bar[..., m, :n].sum(-1),
                  "queries_to_bin": gamma_bar[..., :m, n].sum(-1),
                  "dustbin": self.dustbin, "n_iter": self.n_iter}
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        mb, nb = m + 1, n + 1
        return {"flops": 2 * m * n * (d + dv) + 2 * self.n_iter * mb * nb,
                "state_bytes": n * (d + dv) * bytes_per_el + (n + 1) * 4,
                "bytes_moved": (m * d + n * (d + dv) + m * dv
                                + 2 * self.n_iter * mb * nb) * bytes_per_el}
