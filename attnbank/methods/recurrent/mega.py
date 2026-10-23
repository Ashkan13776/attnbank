"""Mega: a multi-dimensional damped EMA in front of chunked gated attention.

``sec:memory-state``, ``sec:sparse``.  Mega keeps softmax attention but puts a
recurrence in front of it.  A multi-dimensional damped exponential moving average
smooths the sequence before the queries and keys are read off it,

    h_t^{(j)} = alpha_j * (beta_j x_{t,j}) + (1 - alpha_j * delta_j) * h_{t-1}^{(j)},
    EMA(x)_{t,j} = <eta_j, h_t^{(j)}>,

one ``ndim``-dimensional state per feature ``j``, and the attention that follows is
restricted to chunks of ``c`` positions so that the cost is linear in the sequence
length.  In the bank's signature that reads

    q' = EMA(q),   k' = EMA(k),   pi_t = softmax_{j in chunk(t), j <= t} (<q'_t, k'_j> / sqrt d),
    y_t = sum_j pi_t(j) v_j,

with ``v`` left alone -- in Mega the values come from the input directly while ``Q``
and ``K`` are read off the smoothed sequence, which is the whole mechanism: the EMA
supplies the local, position-based inductive bias that lets a *chunk-local* attention
still see something about the rest of the sequence.

**Input-independent dynamics, with a real caveat.**  The EMA's coefficients
``alpha, delta, beta, eta`` are parameters; the recurrence ``h_t = A h_{t-1} + B x_t``
has ``A`` independent of ``x_t``, so the EMA belongs to this family exactly as S4D and
the LRU do.  The attention branch does not: ``pi_t`` is computed from the content of
``q'`` and ``k'`` and is as input-dependent as any softmax.  Mega is thus *not* an
instance of the family's defining restriction taken as a whole -- only its recurrent
component is -- and the operator is a hybrid rather than a pure state-space layer.
``sec:memory-state`` records the same caveat about the headline model's cost: full
gated attention plus an EMA is quadratic, and only the chunked variant implemented here
(``chunk`` not ``None``) is linear.

**Weights.**  Unlike the rest of this family, Mega *does* express a measure on the
original ``n`` positions, and a row-stochastic one: the chunk-restricted softmax.  The
EMA changes which positions get the mass, not whether the mass is a probability.  So
``expresses_weights = True``, ``row_stochastic = True``, and Mega is the member of the
state-space lineage that the decomposition studies can actually score.

**Omitted.**  Mega's output gates (``sigmoid``-weighted residual mixing) and its
single-head shared-representation projections are not modelled: they act elementwise on
``y`` after the aggregation, so including them would break Invariant W without changing
anything about the measure.  The EMA parameters and the ``q``/``k`` smoothing are what
the bank models.  The EMA is kept unidirectional (causal) even when ``causal=False``,
so that the chunk boundary is the only thing the flag moves.

**No softmax anchor is declared.**  ``objective = MODIFY``: Mega was never an
approximation of dense attention.  It happens to contain one -- switch the EMA off and
remove the chunking and the operator *is* softmax -- and ``tests/test_recurrent_ssm.py``
uses that as a correctness check, but declaring it as ``limit_config`` would claim
fidelity to softmax as Mega's goal, which is not what the paper proposes.

Reference: Ma et al. (2023) Mega.
"""
from __future__ import annotations

import math

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    make_generator,
    masked_softmax,
    query_positions,
)
from ...registry import register
from ._ssm_kernels import channel_conv, lag_and_visibility


@register
class MegaAttention(AttentionMethod):
    """Mega: multi-dimensional damped EMA on q and k, then chunk-local softmax attention."""

    name = "mega"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    trainable = True                 # the EMA coefficients are learned in the real model
    references = ("ma2023mega",)
    paper_section = "sec:memory-state"

    def __init__(self, chunk: int | None = 8, ndim: int = 2, ema: bool = True,
                 seed: int = 0, **kw):
        """``chunk``: attention span in positions (``None`` = one chunk, i.e. no chunking).

        ``ndim`` is the EMA's expansion -- the dimension of the state carried per
        feature, Mega's ``h`` -- and ``ema=False`` removes the moving average, leaving
        the chunked attention alone.
        """
        if chunk is not None and chunk < 1:
            raise ValueError(f"chunk must be a positive integer or None, got {chunk}")
        super().__init__(chunk=chunk, ndim=ndim, ema=ema, seed=seed, **kw)
        self.chunk, self.ndim, self.ema, self.seed = chunk, ndim, ema, seed

    # ---- the moving average -------------------------------------------------
    def _ema_kernel(self, d: int, length: int, device, dtype):
        """``[d, length]``: ``K_j[l] = sum_e eta_je beta_je alpha_je (1 - alpha_je delta_je)^l``.

        Unrolling the damped EMA gives a per-feature geometric kernel; ``ndim`` of them
        are summed, which is what "multi-dimensional" buys -- a mixture of decay rates
        per feature rather than a single one.
        """
        g = make_generator(self.seed)
        rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
        alpha = torch.sigmoid(rnd(d, self.ndim))
        delta = torch.sigmoid(rnd(d, self.ndim))
        beta, eta = rnd(d, self.ndim), rnd(d, self.ndim) / math.sqrt(max(self.ndim, 1))
        phi = (1.0 - alpha * delta).clamp(0.0, 1.0 - 1e-12)          # strictly damped
        lags = torch.arange(length, dtype=torch.float64)
        kernel = ((eta * beta * alpha).unsqueeze(-1)
                  * phi.unsqueeze(-1) ** lags).sum(-2)               # [d, length]
        return kernel.to(device=device, dtype=dtype)

    def _smooth(self, x):
        """Causal EMA along ``x``'s own time axis, or ``x`` unchanged when ``ema=False``."""
        if not self.ema or self.ndim < 1:
            return x
        t, c = x.shape[-2], x.shape[-1]
        lag, visible = lag_and_visibility(t, t, causal=True, device=x.device)
        return channel_conv(self._ema_kernel(c, t, x.device, x.dtype), x, lag, visible)

    # ---- operator -----------------------------------------------------------
    def _chunk_support(self, m, n, causal, device):
        """``[m, n]``: key ``j`` is visible to query ``t`` iff same chunk (and ``j <= t``)."""
        p = query_positions(m, n, device=device)
        j = torch.arange(n, device=device)
        c = n if self.chunk is None else self.chunk
        same = (p[:, None] // c) == (j[None, :] // c)
        return same & (j[None, :] <= p[:, None]) if causal else same

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        q_s, k_s = self._smooth(q), self._smooth(k)
        s = torch.einsum("...md,...nd->...mn", q_s, k_s) * self._scale(d)
        support = self._chunk_support(m, n, causal, q.device)
        pi = masked_softmax(s, support)
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras={"chunk": n if self.chunk is None else self.chunk,
                                       "ndim": self.ndim, "ema": self.ema,
                                       "support_size": support.sum(-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        c = min(n if self.chunk is None else self.chunk, n)
        return {"flops": 4 * (m + n) * d * max(self.ndim, 1) + 2 * m * c * (d + dv),
                # one ndim-dimensional EMA state per feature, plus one chunk of KV
                "state_bytes": (2 * max(self.ndim, 1) * d + c * (d + dv)) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
