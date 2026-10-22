"""DeltaNet: the delta-rule update on a fast-weight memory.

``sec:memory-state``, ``eq:memory-delta``, ``tab:memory-recurrences`` (row "DeltaNet").
The transition is a rank-one correction rather than a decay:

    H_t = (I - beta_t psi_t psi_t^T) H_{t-1} + beta_t psi_t v_t^T,    psi_t = psi(k_t),
    y_hat_t = H_t^T phi(q_t),

which is exactly one gradient step of size ``beta_t`` on the per-token regression loss
``L_t(H) = 1/2 ||H^T psi_t - v_t||^2`` (Widrow--Hoff).  Plain linear attention is a
gradient step on the *linear* loss ``-<H^T psi_t, v_t>``, which never saturates and so
never corrects interference between non-orthogonal keys; the delta rule does.  That is
the whole capacity argument: at a fixed state of ``r x d_v``, the delta rule stores more
distinguishable associations than the additive recurrence, which is what study S6
measures.

The regression view also identifies the operator.  Softmax attention is a
Nadaraya--Watson estimator that never leaves the convex hull of the stored values; the
delta rule fits a *global parametric* regression of ``v`` on ``psi(k)`` online and
returns arbitrary linear combinations.  It is a new operator and is judged by downstream
quality, not by fidelity to ``y(q)``.

Weights.  ``H_t`` is linear in ``V``, so ``H_t = M_t V`` for an ``r x n`` coefficient
matrix obeying the same recurrence driven by the basis vectors,
``M_t = (I - beta psi_t psi_t^T) M_{t-1} + beta psi_t e_t^T``.  Running it gives the
exact weights ``w_{tj} = <phi(q_t), M_t[:, j]>`` at a cost of ``O(n^2 r)``, which is the
price of exposing ``pi_hat`` on the original positions.  ``psi`` is l2-normalized, as in
DeltaNet, so that ``I - beta psi psi^T`` is a contraction for ``beta in (0, 2)``.

Note on notation: DESIGN.md writes the update as ``H_{t-1}(I - beta phi phi^T)``, which
is the same map with ``H`` transposed (``d_v x r``).  This file follows the paper's
``eq:memory-delta`` with ``H in R^{r x d_v}``.

Reference: Schlag et al. (2021); Yang et al. (2024) parallel DeltaNet;
Widrow and Hoff (1960); Yang et al. (2025) Gated DeltaNet.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective
from ...registry import register
from ._state import feature_map, readout_positions


@register
class DeltaRNNAttention(AttentionMethod):
    """Delta-rule (Widrow--Hoff) fast-weight memory."""

    name = "delta_rnn"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = False
    references = ("schlag2021linear", "yang2024parallelizing", "widrow1960adaptive",
                  "yang2025gated")
    paper_section = "sec:memory-state"

    def __init__(self, beta: float = 1.0, feature: str = "elu", **kw):
        """``beta``: write strength (not an inverse temperature; see sec:memory-state)."""
        super().__init__(beta=beta, feature=feature, **kw)
        self.beta, self.feature = beta, feature

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        phi = feature_map(self.feature)
        psi = phi(k)
        psi = psi / psi.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(psi.dtype).tiny)
        r = psi.shape[-1]

        # M_t in R^{r x n}: the coefficients of H_t in the basis of the values, so that
        # H_t = M_t V exactly.  Only the states actually read out are kept, which bounds
        # the memory by O(m r n) instead of O(n^2 r).
        idx = readout_positions(m, n, causal, q.device)
        wanted = {int(t): [] for t in idx.tolist()}
        for i, t in enumerate(idx.tolist()):
            wanted[int(t)].append(i)

        mt = torch.zeros(*k.shape[:-2], r, n, dtype=k.dtype, device=k.device)
        eye_cols = torch.eye(n, dtype=k.dtype, device=k.device)
        sel = torch.zeros(*k.shape[:-2], m, r, n, dtype=k.dtype, device=k.device)
        for t in range(n):
            p = psi[..., t, :].unsqueeze(-1)                       # [..., r, 1]
            mt = mt - self.beta * p * (p.transpose(-1, -2) @ mt)
            mt = mt + self.beta * p * eye_cols[t]
            for i in wanted.get(t, ()):
                sel[..., i, :, :] = mt
        w = torch.einsum("...mr,...mrn->...mn", phi(q), sel)
        return AttentionOutput(y=w @ v, weights=w if return_weights else None,
                               extras={"state_rank": r, "beta": self.beta,
                                       "mass": w.sum(-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = d
        return {"flops": 4 * n * r * dv + 2 * m * r * dv,
                "state_bytes": r * dv * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
