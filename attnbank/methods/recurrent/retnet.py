"""Retention (RetNet): a *multi-scale* fixed decay, one time constant per head.

``sec:memory-state``, ``tab:memory-recurrences`` (row "Retention").  The recurrence is

    S_n = gamma S_{n-1} + k_n v_n^T,      y_hat_n = S_n^T q_n,

and its parallel form is the one the paper writes,

    Retention(X) = (Q K^T * D) V,    D_{nm} = gamma^{n-m} for n >= m, else 0,

so ``A_t = gamma Id`` is a scalar multiple of the identity and does **not** depend on
the input.  That is the whole content of the row: retention is the input-independent
member of this family, and everything after it (GLA, HGRN2, RWKV-6, the mLSTM) is the
same recurrence with ``A_t`` made a function of ``x_t``.

What ``decay_rnn`` does not already cover is the *multi-scale* part.  RetNet runs
``h`` heads at ``h`` different time constants,

    gamma = 1 - 2^{-5 - arange(0, h)},

so head 0 keeps a half-life of about 22 tokens and head ``h-1`` is almost lossless.
A single scalar ``gamma`` -- ``decay_rnn`` -- has to choose one point on that curve;
the head schedule is how RetNet gets a range of recency priors out of one operator at
one state size, which is exactly the recency-versus-capacity separation study S6 is
after.  Pass ``decay=`` to collapse the schedule back to one rate and recover
``decay_rnn`` with the identity feature map.

Two further differences from ``decay_rnn``, both from the paper.  The interaction is
the raw scaled inner product ``<q_t, k_j> / sqrt(d)``, with no positive feature map, so
the weights are **signed**; and the readout carries no denominator, so the rows do not
sum to one.  The effective mass is still bounded by ``1 / (1 - gamma)`` rather than
growing with ``n`` -- boundedness without normalization, the distinction
``sec:memory-state`` draws.  ``normalize=True`` puts the denominator back.

xPos relative position embedding, the GroupNorm on the readout and the swish gate are
part of the RetNet *block*, not of the retention operator, and are left out: this file
is the operator of ``tab:memory-recurrences``.

Reference: Sun et al. (2023), Retentive Network.
"""
from __future__ import annotations

import math

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective
from ...registry import register
from ._gated import normalized, recurrent_cost, scalar_gate_matrix, state_indices, truncate


def _half_life(gamma: float) -> float:
    """Tokens until a write has decayed to half its mass; ``inf`` when gamma == 1."""
    return math.inf if gamma >= 1.0 else math.log(0.5) / math.log(gamma)


@register
class RetNetAttention(AttentionMethod):
    """Multi-scale retention: a fixed per-head exponential decay on an r x d_v state."""

    name = "retnet"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = False           # signed, unnormalized: nm, cp and hl are all gone
    references = ("sun2023retentive",)
    paper_section = "sec:memory-state"

    def __init__(self, decay: float | None = None, base_exponent: int = 5,
                 normalize: bool = False, state_rank: int | None = None, **kw):
        """``decay``: one rate for every head; ``None`` uses the paper's head schedule.

        ``base_exponent`` is the ``5`` in ``gamma_h = 1 - 2^{-5-h}``.  ``state_rank``
        truncates the key/query dimension, which *is* the size of the recurrent state
        (``state_rank x d_v``); ``None`` keeps all ``d`` channels.
        """
        super().__init__(decay=decay, base_exponent=base_exponent,
                         normalize=normalize, state_rank=state_rank, **kw)
        self.decay, self.base_exponent = decay, base_exponent
        self.normalize, self.state_rank = normalize, state_rank

    def gammas(self, h: int, dtype=torch.float32, device=None) -> torch.Tensor:
        """The per-head decay rates ``[h]``: ``1 - 2^{-base_exponent - head}``."""
        if self.decay is not None:
            return torch.full((h,), float(self.decay), dtype=dtype, device=device)
        head = torch.arange(h, dtype=dtype, device=device)
        return 1.0 - torch.pow(2.0, -float(self.base_exponent) - head)

    def _forward(self, q, k, v, causal, return_weights):
        m, n, h = q.shape[-2], k.shape[-2], q.shape[1]
        q, k = truncate(self.state_rank, q, k)
        gamma = self.gammas(h, dtype=q.dtype, device=q.device)
        if float(gamma.min()) <= 0.0 or float(gamma.max()) > 1.0:
            raise ValueError(f"decay must lie in (0, 1], got {float(gamma.min())}")

        # D_{nm} = gamma^{n-m}, built from the shared log-space scan so that the
        # multi-scale schedule and the data-dependent gates share one code path.
        log_gate = gamma.log()[None, :, None].expand(q.shape[0], h, n)
        read = state_indices(m, n, causal, q.device)
        decay = scalar_gate_matrix(log_gate, read)                  # [B, h, m, n]

        w = (q @ k.transpose(-1, -2)) * self._scale(q.shape[-1]) * decay
        z = w.sum(-1, keepdim=True)
        w = normalized(w, self.normalize)
        return AttentionOutput(
            y=w @ v, weights=w if return_weights else None,
            extras={"mass": z.squeeze(-1), "gamma": gamma, "decay": decay,
                    "state_rank": q.shape[-1],
                    "half_life": _half_life(float(gamma.min()))})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.state_rank or d
        return recurrent_cost(m, n, r, dv, bytes_per_el)
