"""RWKV-5/6 time mixing: a matrix-valued state that the current token bypasses.

``sec:memory-state``, ``tab:memory-recurrences``.  Eagle (RWKV-5) and Finch (RWKV-6)
write the time-mixing step as

    wkv_t = S_{t-1} + Diag(u) k_t v_t^T,       S_t = Diag(w_t) S_{t-1} + k_t v_t^T,
    y_hat_t = wkv_t^T q_t,                     w_t = exp(-exp(d_t)) in (0, 1)^r,

which is the family's recurrence with two deliberate departures.

*The state is matrix-valued.*  RWKV-4 carried a per-channel numerator and denominator
-- ``2d`` numbers, a diagonal operator, no mixing between channels.  Eagle replaces the
element-wise product by the outer product ``k_t v_t^T`` and the state becomes
``r x d_v``.  That is the same move HGRN2 makes, and for the same reason: at a fixed
parameter count ``prop:memory-rank`` ties recall to the *number of state variables*,
and ``r d_v`` is ``r`` times more of them than ``d``.

*The current token is not in the state.*  The readout uses ``S_{t-1}`` and adds the
token's own write back with a separate bonus gate ``u``.  So the decay a key at ``j``
suffers is ``prod_{u = j+1}^{t-1} w_u`` -- lag ``t - 1 - j``, not ``t - j`` -- and the
diagonal weight is ``u <q_t, k_t>`` with no decay at all.  RWKV is therefore not a pure
exponential-recency operator: it has a recency *prior* plus an always-available
present, which is what the bonus term buys.  ``bonus=0`` removes it and the method
becomes ordinary gated linear attention one step out of phase.

    w_{tj} = <q_t, exp(c_{t-1} - c_j) * k_j>     j < t,     c = cumsum(log w),
    w_{tt} = u <q_t, k_t>.

Finch versus Eagle.  Eagle's decay ``w = exp(-exp(omega))`` is a learned per-channel
constant; Finch makes ``omega`` a function of the token, ``w_t = exp(-exp(d_t))``.
``data_dependent=False`` selects the Eagle form and is the controlled comparison the
family exists to make: same state, same bonus, gate input-dependent or not.  With no
training there is no projection to produce ``d_t``, so this reference uses
``d_t = decay_base + k_t`` -- stated plainly, and all that is needed of it is that the
gate be per channel, in ``(0, 1)``, and a function of the token.  The double
exponential is the paper's and is what keeps ``log w_t = -exp(d_t)`` strictly negative
and finite however large the logits get.

No denominator survives -- RWKV-4's was dropped precisely when the state became a
matrix -- so the rows are signed and do not sum to one.  The LayerNorm, the SiLU gate
and the token shift belong to the RWKV *block*, not to this operator.

Reference: Peng et al. (2024), Eagle and Finch; Peng et al. (2023), RWKV.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective
from ...registry import register
from ._gated import gated_weights, normalized, recurrent_cost, state_indices, truncate


@register
class RWKVAttention(AttentionMethod):
    """RWKV-5/6 time mixing: diagonal decay, matrix state, bonus on the current token."""

    name = "rwkv"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = False
    references = ("peng2024eagle", "peng2023rwkv")
    paper_section = "sec:memory-state"

    def __init__(self, decay_base: float = -2.0, bonus: float = 1.0,
                 data_dependent: bool = True, normalize: bool = False,
                 state_rank: int | None = None, **kw):
        """``decay_base``: the ``omega`` of ``w = exp(-exp(omega + k_t))``.

        ``bonus`` is RWKV's ``u``, the weight the current token gets *outside* the
        state.  ``data_dependent=False`` drops the ``k_t`` term and gives Eagle
        (RWKV-5); ``True`` gives Finch (RWKV-6).  ``state_rank`` truncates the
        key/query dimension and so *is* the size of the state, ``state_rank x d_v``.
        """
        super().__init__(decay_base=decay_base, bonus=bonus,
                         data_dependent=data_dependent, normalize=normalize,
                         state_rank=state_rank, **kw)
        self.decay_base, self.bonus = decay_base, bonus
        self.data_dependent, self.normalize = data_dependent, normalize
        self.state_rank = state_rank

    def log_decay(self, k: torch.Tensor) -> torch.Tensor:
        """``log w_t = -exp(decay_base [+ k_t])``, ``[..., n, r]``, strictly negative."""
        omega = k + self.decay_base if self.data_dependent else \
            torch.full_like(k, float(self.decay_base))
        return -torch.exp(omega)

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        q, k = truncate(self.state_rank, q, k)
        log_w = self.log_decay(k)
        pos = state_indices(m, n, causal, q.device)                 # p_t
        # the state the readout sees is S_{p_t - 1}: the current token is not in it.
        w = gated_weights(q, k, log_w, pos - 1)
        # ... and is added back with the bonus gate u, with no decay.  One column of
        # each row -- the query's own position -- bypasses the recurrence entirely.
        diag = (q * k.index_select(-2, pos)).sum(-1) * self.bonus   # [B, H, m]
        here = torch.nn.functional.one_hot(pos, n).to(w.dtype)      # [m, n]
        w = w + diag.unsqueeze(-1) * here
        if self.scale is not None:            # the paper normalises with LayerNorm
            w = w * self.scale
        z = w.sum(-1, keepdim=True)
        w = normalized(w, self.normalize)
        return AttentionOutput(
            y=w @ v, weights=w if return_weights else None,
            extras={"mass": z.squeeze(-1), "decay": log_w.exp(), "bonus": self.bonus,
                    "data_dependent": self.data_dependent,
                    "state_rank": q.shape[-1]})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.state_rank or d
        c = recurrent_cost(m, n, r, dv, bytes_per_el, gate_flops=4)
        c["flops"] += 2 * m * r                      # the bonus term
        return c
