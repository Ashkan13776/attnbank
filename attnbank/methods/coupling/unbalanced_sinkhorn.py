"""Unbalanced entropic OT attention: the column constraint as a knob, not a law.

``sec:ot``, ``eq:ot-unbalanced``.  ``sinkhorn`` imposes the column marginal exactly;
softmax does not impose it at all.  Replacing the hard constraints by Kullback--Leibler
penalties puts the two at the ends of one continuum (Frogner et al. 2015; Chizat et al.
2016; Liero et al. 2018; Sejourne et al. 2022):

    min_{gamma >= 0}  int c dgamma + eps KL(gamma || nu x mubar)
                      + lam_row KL(gamma_Q || nu) + lam_col KL(gamma_X || mubar).

Chizat et al. show the scaling iterations keep the form of ``eq:ot-sinkhorn`` with each
potential update *damped* by ``rho_i = lam_i / (lam_i + eps)``:

    f <- rho_row ( log a - LSE_j ( s + g ) ),
    g <- rho_col ( log b - LSE_i ( s + f ) ),     s = -c/eps.

``rho = 1`` is the hard constraint of ``sinkhorn``; ``rho = 0`` drops the constraint and
leaves the potential at zero.  The default here keeps the row constraint hard
(``lam_row = None``, meaning ``lam_row = infinity``), which is what makes the operator a
probability measure per query, and treats ``lam_col`` as the single dial the paper
describes: **at ``lam_col = 0`` this is exactly softmax** -- the semi-relaxed problem of
``prop:ot-gibbs`` with ``psi = 0`` -- and as ``lam_col -> infinity`` it is the doubly
stochastic coupling of ``prop:ot-conditional``.

That is why this file, alone among the ``coupling`` methods, declares a ``limit_config``
even though its ``objective`` is ``MODIFY``: the family *contains* softmax at an interior
value of its own parameter.  The operator it defines away from that value is still a
different operator, not an approximation of softmax, so no convergence sequence is
claimed and fidelity is not the goal.

The practical reading is the one ``sec:ot`` gives: ``lam_col`` interpolates between
"sinks allowed" and "sinks forbidden".  A finite ``lam_col`` lets a key still absorb more
than its share -- attention sinks survive, in proportion -- while bounding how far it
can go, and it does so without the abstention mass that a dustbin introduces
(``dustbin_sinkhorn`` is the discrete alternative).

``extras`` matches ``sinkhorn`` key for key -- ``potential``, ``f``, ``g``, ``gamma``,
``row_marginal``, ``col_marginal``, ``marginal_error`` -- because the plan is still of
Gibbs form ``exp(s + f + g)`` and ``prop:ot-conditional`` still applies verbatim: the
conditional is softmax tilted by one number per key.  Only the number is different, and
``col_kl``, the divergence the penalty charges, is added.

Not causal, and the relaxation does not rescue it.  ``prop:ot-causal`` is the limiting
obstruction -- at ``lam_col = infinity`` a triangular doubly stochastic plan is the
identity -- but at *any* ``lam_col > 0`` there is a second, more basic one: the column
potential ``g_j`` is a reduction over all queries that can see key ``j``, including
queries later than the one being answered, so the conditional of query ``i`` moves when
a key after ``i`` changes.  That is leakage, not degeneracy, and the conformance suite
measures it directly.  Only at ``lam_col = 0``, where the operator is softmax, does it
vanish; ``supports_causal`` is one declaration for the whole class, so ``False`` is the
one that is true of it.

Reference: Chizat et al. (2016) scaling algorithms for unbalanced transport;
Frogner et al. (2015); Liero et al. (2018); Sejourne et al. (2022);
Sander et al. (2022) Sinkformers; Peyre and Cuturi (2019).
"""
from __future__ import annotations

import math

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, scores
from ...errors import ConfigError
from ...registry import register


def unbalanced_sinkhorn_log(logk, log_row, log_col, n_iter, rho_row, rho_col):
    """Damped IPFP of Chizat et al. (2016), in log domain.

    ``logk = -c/eps``; ``rho_i = lam_i / (lam_i + eps)`` in ``[0, 1]`` damps the
    corresponding potential update.  ``rho_row = rho_col = 1`` is
    :func:`attnbank.methods.coupling.sinkhorn.sinkhorn_log`; ``rho_col = 0`` leaves
    ``g`` at zero and the row update is a single softmax normalization.  Returns
    ``(f, g)`` with ``log Gamma = logk + f[..., :, None] + g[..., None, :]``.
    """
    f = torch.zeros(logk.shape[:-1], dtype=logk.dtype, device=logk.device)
    g = torch.zeros(logk.shape[:-2] + logk.shape[-1:], dtype=logk.dtype, device=logk.device)
    for _ in range(n_iter):
        f = rho_row * (log_row - torch.logsumexp(logk + g.unsqueeze(-2), dim=-1))
        g = rho_col * (log_col - torch.logsumexp(logk + f.unsqueeze(-1), dim=-2))
    return f, g


@register
class UnbalancedSinkhornAttention(AttentionMethod):
    """Entropic OT attention with a KL-relaxed column marginal: softmax at ``lam_col=0``."""

    name = "unbalanced_sinkhorn"
    category = Category.COUPLING
    objective = Objective.MODIFY
    supports_causal = False          # prop:ot-causal; the relaxation softens, not removes
    expresses_weights = True
    row_stochastic = True
    references = ("chizat2016scaling", "frogner2015learning", "liero2018optimal",
                  "sejourne2022unbalanced", "sander2022sinkformers",
                  "peyre2019computational")
    paper_section = "sec:ot"

    def __init__(self, n_iter: int = 10, eps: float = 1.0, lam_col: float = 1.0,
                 lam_row: float | None = None, **kw):
        """``lam_col``: KL penalty on the column marginal. ``0`` is softmax exactly,
        ``inf`` (or a large value) is the doubly stochastic plan of ``sinkhorn``.
        ``lam_row``: penalty on the row marginal; ``None`` means ``infinity``, the hard
        constraint that keeps every query's conditional a probability measure.
        ``eps``: entropic regularization, a temperature on the scaled logits.
        ``n_iter``: scaling iterations."""
        if eps <= 0:
            raise ConfigError(f"unbalanced_sinkhorn: eps must be > 0; got {eps}")
        for nm, lam in (("lam_col", lam_col), ("lam_row", lam_row)):
            if lam is not None and lam < 0:
                raise ConfigError(f"unbalanced_sinkhorn: {nm} is a KL penalty and must be "
                                  f">= 0 (0 drops the constraint, None means infinity); "
                                  f"got {lam}")
        super().__init__(n_iter=n_iter, eps=eps, lam_col=lam_col, lam_row=lam_row, **kw)
        self.n_iter, self.eps = n_iter, eps
        self.lam_col, self.lam_row = lam_col, lam_row

    def _rho(self, lam):
        """``lam / (lam + eps)``; ``None`` or ``inf`` is the hard constraint, ``rho = 1``."""
        if lam is None or math.isinf(lam):
            return 1.0
        return float(lam) / (float(lam) + self.eps)

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        s = scores(q, k, self._scale(q.shape[-1]), causal) / self.eps     # = -c/eps
        log_row = torch.full(s.shape[:-1], -math.log(float(m)),
                             dtype=s.dtype, device=s.device)
        log_col = torch.full(s.shape[:-2] + s.shape[-1:], -math.log(float(n)),
                             dtype=s.dtype, device=s.device)
        rho_row, rho_col = self._rho(self.lam_row), self._rho(self.lam_col)
        f, g = unbalanced_sinkhorn_log(s, log_row, log_col, self.n_iter, rho_row, rho_col)

        # prop:ot-conditional (b) survives the relaxation: the tilt is still one per key
        potential = g + log_col                      # psi/eps + log mubar
        pi = torch.softmax(s + potential.unsqueeze(-2), dim=-1)
        gamma = torch.exp(s + f.unsqueeze(-1) + g.unsqueeze(-2))
        col, row = gamma.sum(-2), gamma.sum(-1)
        target_col, target_row = torch.exp(log_col), torch.exp(log_row)
        col_kl = (col * ((col / target_col.clamp_min(1e-300)).clamp_min(1e-300).log())
                  - col + target_col).sum(-1)
        extras = {"potential": potential, "f": f, "g": g, "gamma": gamma,
                  "col_marginal": col, "row_marginal": row,
                  "target_col": target_col, "target_row": target_row,
                  "marginal_error": (row - target_row).abs().amax(-1)
                                    + (col - target_col).abs().amax(-1),
                  "col_kl": col_kl, "rho_col": rho_col, "rho_row": rho_row,
                  "lam_col": self.lam_col, "n_iter": self.n_iter}
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        return {"flops": 2 * m * n * (d + dv) + 2 * self.n_iter * m * n,
                "state_bytes": n * (d + dv) * bytes_per_el + n * 4,   # + the key potential
                "bytes_moved": (m * d + n * (d + dv) + m * dv
                                + 2 * self.n_iter * m * n) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        # eq:ot-unbalanced with lam_col = 0 is the semi-relaxed problem of
        # prop:ot-gibbs: psi = 0, and the conditional is softmax exactly
        return {"lam_col": 0.0, "eps": 1.0}
