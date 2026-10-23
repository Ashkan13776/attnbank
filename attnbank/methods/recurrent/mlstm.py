"""mLSTM (xLSTM): exponential gating, a matrix memory, and a normaliser state.

``sec:memory-state``, ``tab:memory-recurrences``.  The mLSTM is the one member of this
family that keeps a denominator, and keeping it is the whole reason it is here:

    C_t = f_t C_{t-1} + i_t v_t k_t^T,            C_t in R^{d_v x r}   (matrix memory)
    n_t = f_t n_{t-1} + i_t k_t,                  n_t in R^r           (normaliser)
    y_hat_t = C_t q_t / max(|n_t^T q_t|, 1),

with ``i_t = exp(i~_t)`` an *exponential* input gate and ``f_t = sigmoid(f~_t)`` a
scalar forget gate -- so ``A_t = f_t Id``, data-dependent but not diagonal: one rate
for the whole head, unlike GLA.  The interest is entirely in the second and third
lines.  ``n_t`` is the same recurrence driven by ``k_t`` instead of ``k_t v_t^T``, so
``n_t^T q_t`` is exactly the row sum of the unnormalised weights: the mLSTM *carries
the denominator of* ``eq:attention_expectation`` *as part of its state*, which linear
attention threw away and which ``linear_rnn`` does not have.

It still does not get the contract back, and the ``max(., 1)`` is why.  Dividing by
``max(|n_t^T q_t|, 1)`` makes the readout bounded -- that was the goal, since an
exponential input gate can make the true denominator astronomically small -- but it is
not division by the mass.  When the mass exceeds one the operator *is* normalised and
the row sums to one; when it does not, the row is left alone and sums to less.  Row
sums are therefore in ``[0, 1]`` and the weights are signed (``q_t . k_t`` is a raw
inner product), so normalization invariance and constant preservation fail exactly in
the small-mass regime, which is reported in ``extras["clipped"]``.  That partial
restoration is a sharper illustration of ``sec:memory-state`` than dropping the
normaliser outright: the denominator is present and the invariant still breaks.

Exponential gating needs the paper's stabiliser or it overflows at the first large
logit.  With ``c = cumsum(log f)``,

    m_t = max_{j <= t} ( log i_j + c_t - c_j ) = c_t + cummax( log i - c )_t,

and every stabilised coefficient ``a_{tj} = exp(log i_j + c_t - c_j - m_t)`` lies in
``(0, 1]`` by construction.  The denominator floor becomes ``exp(-m_t)`` in the same
units.  This is the recurrence ``m_t = max(log f_t + m_{t-1}, log i_t)`` of the paper,
written as the prefix scan it is.

The gates.  A trained mLSTM reads ``i~_t`` and ``f~_t`` off ``x_t`` through learned
projections.  With no training this reference takes two fixed orthogonal directions of
the key -- the mean ``<k_t, 1>/sqrt(d)`` for forgetting and the alternating-sign
direction ``<k_t, (-1)^c>/sqrt(d)`` for writing -- plus the biases below.  They are
stand-ins, stated rather than hidden; what matters downstream is that both gates are
scalar per head and functions of the token, and both are.

Reference: Beck et al. (2024), xLSTM.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective
from ...registry import register
from ._gated import recurrent_cost, state_indices, truncate


@register
class MLSTMAttention(AttentionMethod):
    """mLSTM: exponential input gate, scalar forget gate, matrix memory + normaliser."""

    name = "mlstm"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = False           # signed; rows sum to <= 1, and to 1 only sometimes
    references = ("beck2024xlstm",)
    paper_section = "sec:memory-state"

    def __init__(self, forget_bias: float = 3.0, input_bias: float = 0.0,
                 state_rank: int | None = None, **kw):
        """``forget_bias`` shifts ``f~`` (positive = remember longer, the LSTM default).

        ``input_bias`` shifts ``i~``.  ``state_rank`` truncates the key/query dimension
        and so *is* the size of the matrix memory, ``state_rank x d_v`` plus the
        ``state_rank`` of the normaliser; ``None`` keeps all ``d`` channels.
        """
        super().__init__(forget_bias=forget_bias, input_bias=input_bias,
                         state_rank=state_rank, **kw)
        self.forget_bias, self.input_bias = forget_bias, input_bias
        self.state_rank = state_rank

    def gates(self, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(log f_t, log i_t)``, both ``[..., n]``: one scalar per head per step."""
        r = k.shape[-1]
        ones = torch.ones(r, dtype=k.dtype, device=k.device)
        alt = torch.where(torch.arange(r, device=k.device) % 2 == 0, 1.0, -1.0).to(k.dtype)
        root = r ** 0.5
        log_f = torch.nn.functional.logsigmoid((k @ ones) / root + self.forget_bias)
        log_i = (k @ alt) / root + self.input_bias       # i_t = exp(i~_t): log i IS i~
        return log_f, log_i

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        q, k = truncate(self.state_rank, q, k)
        log_f, log_i = self.gates(k)
        c = log_f.cumsum(-1)                                   # [B, H, n]
        mm = c + torch.cummax(log_i - c, dim=-1).values        # m_t, the stabiliser

        read = state_indices(m, n, causal, q.device)
        cr, mr = c.index_select(-1, read), mm.index_select(-1, read)      # [B, H, m]
        j = torch.arange(n, device=q.device)
        vis = j.unsqueeze(0) <= read.unsqueeze(1)                          # [m, n]
        # a_{tj} = exp(log i_j + c_t - c_j - m_t) in (0, 1] for every visible j.
        logit = (log_i - c).unsqueeze(-2) + (cr - mr).unsqueeze(-1)        # [B, H, m, n]
        a = logit.masked_fill(~vis, float("-inf")).exp()

        s = a * (q @ k.transpose(-1, -2)) * self._scale(q.shape[-1])
        mass = s.sum(-1, keepdim=True)
        floor = torch.exp(-mr).unsqueeze(-1)                   # max(., 1) in stable units
        clipped = mass.abs() < floor
        w = s / torch.where(clipped, floor, mass.abs())
        return AttentionOutput(
            y=w @ v, weights=w if return_weights else None,
            extras={"mass": mass.squeeze(-1), "stabilizer": mm,
                    "forget_gate": log_f.exp(), "input_gate": log_i,
                    "clipped": clipped.squeeze(-1), "state_rank": q.shape[-1]})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.state_rank or d
        c = recurrent_cost(m, n, r, dv, bytes_per_el, gate_flops=4)
        c["state_bytes"] += r * bytes_per_el         # the normaliser state n_t
        c["flops"] += 2 * n * r + 2 * m * r          # and its update and readout
        return c
