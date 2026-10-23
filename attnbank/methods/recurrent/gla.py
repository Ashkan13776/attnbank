"""Gated linear attention: a *data-dependent diagonal* gate on the recurrent state.

``sec:memory-state``, ``tab:memory-recurrences`` (row "GLA / Mamba-2").  The operator is

    S_t = Diag(alpha_t) S_{t-1} + k_t v_t^T,      y_hat_t = S_t^T q_t,
    alpha_t = sigmoid(W_alpha x_t)^{1 / tau} in (0, 1)^r,     tau = 16,

the first method in the family whose transition ``A_t = Diag(alpha_t)`` depends on the
input.  Retention can only trade recency against nothing -- one rate, chosen in
advance, applied to every channel and every token.  A data-dependent diagonal gate lets
the operator *decide per token and per channel* what to keep, which is how a fixed
``r x d_v`` state stops being a pure recency window and starts being selective.  This
is the same mechanism Mamba-2's SSD writes as a scalar ``a_t``; GLA keeps it diagonal.

What it does not do is restore the contract.  ``y_hat_t = S_t^T q_t`` has no
denominator -- the GroupNorm in the GLA *block* is not one -- and ``q, k`` are raw
projections with no positive feature map, so the weights

    w_{tj} = < q_t , (prod_{u = j+1}^{t} alpha_u) * k_j > / sqrt(d),    j <= t,

are **signed** and their row sums are whatever the gates and the content make them.
That is three of the properties of ``eq:attention_expectation`` gone at once
(``sec:memory-state``): normalization invariance, constant preservation and
convex-hull confinement.  ``normalize=True`` restores a measure and is not GLA.

The gate.  A trained GLA computes ``alpha_t`` from a low-rank projection of the token;
with no training there is no ``W_alpha``, so this reference reads the gate off the key
itself, ``alpha_t = sigmoid(k_t)^{1 / tau}``.  It is a stand-in for the learned map and
is stated here rather than hidden: what the library needs from it is that the gate be
in ``(0, 1)``, data-dependent, and per channel, and all three survive.  The
``1 / tau`` exponent is the paper's, and it is load-bearing -- it keeps
``log alpha = logsigmoid(k) / tau`` close to zero, which is what stops the cumulative
product from underflowing over a long prefix.  ``gate_temp=1`` turns it off and the
state's memory collapses to a few tokens.

The cumulative products are taken in log space by ``_gated.gated_weights``, which is
the numerically stable half of the chunked GLA kernel; see that module.

Reference: Yang et al. (2024), Gated Linear Attention; Dao and Gu (2024), SSD.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective
from ...registry import register
from ._gated import gated_weights, normalized, recurrent_cost, state_indices, truncate


@register
class GatedLinearAttention(AttentionMethod):
    """GLA: ``S_t = Diag(alpha_t) S_{t-1} + k_t v_t^T`` with a data-dependent gate."""

    name = "gla"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = False           # signed and unnormalized, by construction
    references = ("yang2024gated", "dao2024transformers")
    paper_section = "sec:memory-state"

    def __init__(self, gate_temp: float = 16.0, gate_bias: float = 0.0,
                 normalize: bool = False, state_rank: int | None = None, **kw):
        """``gate_temp`` is the paper's ``tau`` in ``alpha = sigmoid(.)^{1/tau}``.

        ``gate_bias`` shifts the gate logits (positive = remember longer).
        ``state_rank`` truncates the key/query dimension and so *is* the size of the
        recurrent state, ``state_rank x d_v``; ``None`` keeps all ``d`` channels.
        """
        super().__init__(gate_temp=gate_temp, gate_bias=gate_bias,
                         normalize=normalize, state_rank=state_rank, **kw)
        self.gate_temp, self.gate_bias = gate_temp, gate_bias
        self.normalize, self.state_rank = normalize, state_rank

    def log_gate(self, k: torch.Tensor) -> torch.Tensor:
        """``log alpha_t = logsigmoid(k_t + bias) / tau``, ``[..., n, r]`` and ``<= 0``."""
        if self.gate_temp <= 0:
            raise ValueError(f"gate_temp must be > 0, got {self.gate_temp}")
        return torch.nn.functional.logsigmoid(k + self.gate_bias) / self.gate_temp

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        q, k = truncate(self.state_rank, q, k)
        log_alpha = self.log_gate(k)
        read = state_indices(m, n, causal, q.device)
        w = gated_weights(q, k, log_alpha, read) * self._scale(q.shape[-1])
        z = w.sum(-1, keepdim=True)
        w = normalized(w, self.normalize)
        return AttentionOutput(
            y=w @ v, weights=w if return_weights else None,
            extras={"mass": z.squeeze(-1), "log_gate": log_alpha,
                    "gate": log_alpha.exp(), "state_rank": q.shape[-1]})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.state_rank or d
        return recurrent_cost(m, n, r, dv, bytes_per_el, gate_flops=3)
