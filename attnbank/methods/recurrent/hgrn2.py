"""HGRN2: a gated *vector* recurrence expanded to a matrix state by an outer product.

``sec:memory-state``, ``tab:memory-recurrences``.  HGRN is an element-wise gated RNN,

    h_t = f_t * h_{t-1} + (1 - f_t) * i_t,                     h_t in R^d,

whose state is a *vector*: ``prop:memory-rank`` caps what it can recall at ``d``
numbers however wide the layer is.  HGRN2 changes one symbol -- the element-wise
product becomes an outer product --

    H_t = Diag(f_t) H_{t-1} + (1 - f_t) v_t^T,                 H_t in R^{d x d_v},
    y_hat_t = H_t^T q_t,

and the state becomes a matrix of ``d x d_v`` numbers at exactly the same parameter
count.  In the vocabulary of this category that is GLA with the key *tied to the gate*,
``k_t = 1 - f_t`` and ``alpha_t = f_t``: HGRN2 has no separate key projection at all,
which is the point -- the expansion is free.

Two consequences worth stating, because they are what distinguishes this row from GLA.

*Per channel the write is a convex combination.*  Summing the unrolled recurrence
channel by channel, ``sum_{j <= t} (1 - f_j) prod_{u > j} f_u = 1 - prod_{u <= t} f_u
<= 1``.  So the state cannot grow without bound in ``t``, and the weights

    w_{tj} = sum_r q_{t,r} (1 - f_{j,r}) prod_{u = j+1}^{t} f_{u,r}

have row sum ``<q_t, 1 - prod_{u <= t} f_u>`` -- bounded, readable, and equal to one
only by accident.  Boundedness without normalization again (``sec:memory-state``); the
rows are not a probability measure and the operator declares
``row_stochastic = False``.

*The lower bound is where "hierarchical" lives.*  HGRN sets
``f_t = gamma + (1 - gamma) sigmoid(.)`` and makes ``gamma`` increase monotonically
with depth (via ``cumax`` over layers), so lower layers forget fast and model local
structure while upper layers approach ``f = 1`` and carry long-range state.  A single
operator cannot be hierarchical on its own; ``lower_bound=`` is that layer's ``gamma``,
and sweeping it from 0 to 1 traces the depth axis of the real model.  At
``lower_bound=1`` the gate is stuck open, nothing is ever written (``1 - f = 0``) and
the output is exactly zero -- the degenerate end of the schedule, reported rather than
clamped away.

Reference: Qin et al. (2024), HGRN2; Qin et al. (2023), HGRN.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective
from ...registry import register
from ._gated import gated_weights, normalized, recurrent_cost, state_indices, truncate


@register
class HGRN2Attention(AttentionMethod):
    """HGRN2: outer-product state expansion with the input gate tied to ``1 - f_t``."""

    name = "hgrn2"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = False
    references = ("qin2024hgrn2", "qin2023hgrn")
    paper_section = "sec:memory-state"

    def __init__(self, lower_bound: float = 0.9, gate_bias: float = 0.0,
                 normalize: bool = False, state_rank: int | None = None, **kw):
        """``lower_bound``: this layer's ``gamma`` in ``f = gamma + (1 - gamma) sigmoid``.

        It is the hierarchical schedule's one free number -- 0 at the bottom of the
        stack, approaching 1 at the top; the default sits mid-stack, because at 0 the
        untrained gate averages one half and the state keeps barely a token.
        ``state_rank`` truncates the gate/key dimension and so *is* the size of the
        recurrent state, ``state_rank x d_v``.
        """
        super().__init__(lower_bound=lower_bound, gate_bias=gate_bias,
                         normalize=normalize, state_rank=state_rank, **kw)
        self.lower_bound, self.gate_bias = lower_bound, gate_bias
        self.normalize, self.state_rank = normalize, state_rank

    def forget_gate(self, k: torch.Tensor) -> torch.Tensor:
        """``f_t = gamma + (1 - gamma) sigmoid(k_t + bias)``, ``[..., n, r]`` in (0, 1]."""
        if not 0.0 <= self.lower_bound <= 1.0:
            raise ValueError(f"lower_bound must lie in [0, 1], got {self.lower_bound}")
        g = self.lower_bound
        return g + (1.0 - g) * torch.sigmoid(k + self.gate_bias)

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        q, k = truncate(self.state_rank, q, k)
        f = self.forget_gate(k)
        write = 1.0 - f                       # the tied input gate: HGRN2's "key"
        log_f = f.clamp_min(torch.finfo(f.dtype).tiny).log()
        read = state_indices(m, n, causal, q.device)
        w = gated_weights(q, write, log_f, read)
        if self.scale is not None:            # the paper has no 1/sqrt(d) here
            w = w * self.scale
        z = w.sum(-1, keepdim=True)
        w = normalized(w, self.normalize)
        return AttentionOutput(
            y=w @ v, weights=w if return_weights else None,
            extras={"mass": z.squeeze(-1), "forget_gate": f, "input_gate": write,
                    "lower_bound": self.lower_bound, "state_rank": q.shape[-1]})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.state_rank or d
        # no key projection: the gate is the key, so the per-step gate work is the
        # only extra over the additive recurrence.
        return recurrent_cost(m, n, r, dv, bytes_per_el, gate_flops=3)
