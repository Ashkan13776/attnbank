"""Decayed linear recurrence: H_t = gamma H_{t-1} + psi(k_t) v_t^T.

``sec:memory-state``, ``tab:memory-recurrences`` (rows "Retention", "GLA / Mamba-2").
``A_t = gamma Id`` with a fixed ``gamma < 1`` is retention (RetNet); a data-dependent
scalar or diagonal gate gives Mamba-2, GLA, HGRN2, RWKV-5/6 and the mLSTM.  The weights
on the original positions are

    w_{tj} = gamma^{(t - j)} <phi(q_t), psi(k_j)>,    j <= t,

an exponential recency prior multiplying a rank-``r`` content term.

Decay restores *boundedness* without restoring normalization: the effective mass
``sum_i prod_{u>i} gamma_u`` is bounded by ``1/(1 - gamma)`` rather than growing with
``n``, so the output cannot blow up with context length, but it is still not an
expectation under a probability measure (set ``normalize=True`` to recover one).  This
is the distinction ``sec:memory-state`` draws between "restoring boundedness" and
"restoring the contract".

State size is ``r x d_v`` regardless of ``gamma``, so sweeping ``gamma`` at fixed state
separates *recency* from *capacity* -- which is what study S6 needs.

Reference: Sun et al. (2023) RetNet; Yang et al. (2024) GLA; Dao and Gu (2024) SSD;
Peng et al. (2023) RWKV; Beck et al. (2024) xLSTM.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, query_positions
from ...registry import register
from ._state import feature_map


@register
class DecayRNNAttention(AttentionMethod):
    """Fixed exponential decay on the recurrent state."""

    name = "decay_rnn"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = False
    references = ("sun2023retentive", "yang2024gated", "dao2024transformers",
                  "peng2023rwkv", "beck2024xlstm")
    paper_section = "sec:memory-state"

    def __init__(self, decay: float = 0.9, feature: str = "elu", normalize: bool = False, **kw):
        super().__init__(decay=decay, feature=feature, normalize=normalize, **kw)
        self.decay, self.feature, self.normalize = decay, feature, normalize

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        phi = feature_map(self.feature)
        content = phi(q) @ phi(k).transpose(-1, -2)

        p = query_positions(m, n, device=q.device) if causal else \
            torch.full((m,), n - 1, device=q.device)
        j = torch.arange(n, device=q.device)
        lag = p[:, None].to(q.dtype) - j[None, :].to(q.dtype)
        gate = torch.where(lag >= 0, torch.pow(torch.tensor(self.decay, dtype=q.dtype,
                                                            device=q.device), lag),
                           torch.zeros_like(lag))
        w = content * gate
        z = w.sum(-1, keepdim=True)
        if self.normalize:
            w = w / z.clamp_min(torch.finfo(w.dtype).tiny)
        return AttentionOutput(y=w @ v, weights=w if return_weights else None,
                               extras={"mass": z.squeeze(-1), "decay": self.decay,
                                       "gate": gate})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = d
        return {"flops": 2 * n * r * dv + 2 * m * r * dv + m * n,
                "state_bytes": r * dv * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
