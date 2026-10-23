"""Hyena: implicitly parametrised long convolutions interleaved with gating.

``sec:memory-state``.  The Hyena operator of order ``N`` alternates a long causal
convolution with an elementwise ("data-controlled") gate:

    z^1 = x^1,     z^{i+1} = x^{i+1} * (h^i conv z^i),     y = z^{N+1},

where ``x^1 ... x^{N+1}`` are projections of the input and ``h^1 ... h^N`` are filters
as long as the sequence.  The filters are not stored as ``n`` free numbers per channel:
they are *implicit*, the output of a small network applied to a positional encoding of
the lag, multiplied by an exponential decay window and normalised.  That is what makes
a filter of unbounded length a bounded number of parameters, and it is the idea Hyena
inherits from continuous-kernel convolutions (Romero et al., 2021).

**The claim this file is here to test, and the caveat.**  ``h^i`` depends on the lag
alone -- it is fixed before any token is seen -- so Hyena's *convolutional* branch has
input-independent dynamics exactly like S4, S5 and the LRU.  Its *gating* does not:
``x^{i+1}`` is a projection of the input, and multiplying by it makes the composite
operator depend on the content at the output position.  Hyena is therefore a genuine
member of this family in its filters and a partial exception in its gates, which is
precisely the distinction ``sec:memory-state`` draws when it says the step that
produced the current generation of models was to make the *state dynamics* themselves
input-dependent.  Hyena moves the data dependence into a pointwise product sitting
outside the recurrence; Mamba and the gated linear RNNs move it inside.

**No finite state.**  Unlike every other method in ``recurrent/``, Hyena's filter is
not realised by a finite-dimensional recurrence: an arbitrary length-``n`` FIR filter
needs ``O(n)`` memory to run causally, which is why ``cost()['state_bytes']`` grows
with ``n`` here and does not for ``s4d``, ``s5`` or ``lru``.  ``prop:memory-rank`` has
nothing to bound.  The *parameters* are few; the *state* is not.

**It does not read the queries or the keys.**  As in the rest of the family, the three
projections are taken from the value field -- fixed seeded ``d_v x d_v`` matrices, the
stand-ins for the learned ones -- because a long convolution has nowhere for ``q`` and
``k`` to enter.  ``extras['reads_qk']`` is ``False``.

**Weights.**  ``expresses_weights = False``, twice over: each channel is convolved with
its own filter, and the gate multiplies channel by channel, so the output is not
``sum_j w_{tj} v_j`` for any scalar ``w`` -- it is not even linear in ``v``.

**Simplifications.**  The implicit filter here is a two-hidden-layer MLP with sine
activations on the positional features ``[t, sin(2 pi f t), cos(2 pi f t)]``, windowed
by ``exp(-alpha_c t)`` with ``alpha_c`` log-spaced across channels and L1-normalised,
which is the published recipe's shape but not its exact schedule; the short depthwise
convolution Hyena applies to the projections before the recurrence is omitted.  As
everywhere in ``recurrent/`` the convolution is evaluated directly rather than by FFT,
so that causality is bit-exact (see ``s4d``).

Reference: Poli et al. (2023) Hyena Hierarchy; Romero et al. (2021) CKConv;
Gu, Goel and Re (2022) S4.
"""
from __future__ import annotations

import math

import torch

from ...components.recurrence import readout_positions
from ...core import AttentionMethod, AttentionOutput, Category, Objective, make_generator
from ...registry import register
from ._ssm_kernels import channel_conv, lag_and_visibility


@register
class HyenaAttention(AttentionMethod):
    """Hyena: order-N implicit long convolutions interleaved with elementwise gating."""

    name = "hyena"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = False        # per-channel filters and an elementwise gate
    row_stochastic = False
    trainable = True
    references = ("poli2023hyena", "romero2022ckconv", "gu2022efficiently")
    paper_section = "sec:memory-state"

    def __init__(self, order: int = 2, width: int = 16, n_freqs: int = 4,
                 decay_min: float = 0.5, decay_max: float = 8.0, seed: int = 0, **kw):
        """``order``: how many (convolution, gate) stages; ``2`` is the paper's default.

        ``width`` and ``n_freqs`` size the implicit filter network -- the hidden width
        and the number of Fourier features in the positional encoding of the lag --
        and ``decay_min`` / ``decay_max`` bound the log-spaced exponential window, which
        is what sets the effective filter length per channel.
        """
        if order < 1:
            raise ValueError(f"order must be at least 1, got {order}")
        super().__init__(order=order, width=width, n_freqs=n_freqs,
                         decay_min=decay_min, decay_max=decay_max, seed=seed, **kw)
        self.order, self.width, self.n_freqs = order, width, n_freqs
        self.decay_min, self.decay_max, self.seed = decay_min, decay_max, seed

    # ---- the implicit filter ------------------------------------------------
    def _filters(self, dv: int, length: int, device, dtype):
        """``[order, dv, length]``: the windowed, L1-normalised implicit filters.

        The network is evaluated once over all lags, which is the point of an implicit
        parametrisation -- the filter is a function of the lag, so its length is a
        property of the call and not of the parameter count.
        """
        g = make_generator(self.seed)
        t = (torch.arange(length, dtype=torch.float64) / max(length - 1, 1))   # [L]
        freqs = torch.arange(1, self.n_freqs + 1, dtype=torch.float64)
        ang = 2.0 * math.pi * t[:, None] * freqs[None, :]
        emb = torch.cat([t[:, None], torch.sin(ang), torch.cos(ang)], dim=-1)  # [L, E]

        def layer(x, out_dim):
            w = torch.randn(x.shape[-1], out_dim, generator=g,
                            dtype=torch.float64) / math.sqrt(x.shape[-1])
            return x @ w

        h = torch.sin(layer(emb, self.width))
        h = torch.sin(layer(h, self.width))
        h = layer(h, self.order * dv).reshape(length, self.order, dv)

        alpha = torch.exp(torch.linspace(math.log(self.decay_min),
                                         math.log(self.decay_max), dv,
                                         dtype=torch.float64))                 # [dv]
        window = torch.exp(-alpha[None, :] * t[:, None])                       # [L, dv]
        h = (h * window[:, None, :]).permute(1, 2, 0)                          # [order, dv, L]
        h = h / h.abs().sum(-1, keepdim=True).clamp_min(1e-12)   # Hyena normalises filters
        return h.to(device=device, dtype=dtype)

    def _projections(self, dv: int, device, dtype):
        """``order + 1`` seeded ``[dv, dv]`` projections: Hyena's ``x^1 ... x^{N+1}``."""
        g = make_generator(self.seed + 1)
        return [(torch.randn(dv, dv, generator=g, dtype=torch.float64)
                 / math.sqrt(dv)).to(device=device, dtype=dtype)
                for _ in range(self.order + 1)]

    # ---- operator -----------------------------------------------------------
    def _forward(self, q, k, v, causal, return_weights):
        m, n, dv = q.shape[-2], v.shape[-2], v.shape[-1]
        filters = self._filters(dv, n, v.device, v.dtype)
        proj = self._projections(dv, v.device, v.dtype)
        # Every stage needs the convolution at *every* position, because the gate at
        # position t feeds the next stage's input there; the readout happens once, at
        # the end, through the category's usual convention.
        lag, visible = lag_and_visibility(n, n, causal=True, device=v.device)

        z = v @ proj[0]
        for i in range(self.order):
            z = (v @ proj[i + 1]) * channel_conv(filters[i], z, lag, visible)
        y = z.index_select(-2, readout_positions(m, n, causal, v.device))
        return AttentionOutput(y=y, weights=None, extras={
            "filters": filters, "order": self.order,
            "filter_length": n, "reads_qk": False})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        """``state_bytes`` grows with ``n``: a long convolution has no finite state."""
        logn = max(1.0, math.log2(max(n, 2)))
        return {"flops": self.order * (15 * dv * n * logn + 2 * n * dv * dv)
                + 2 * n * dv * dv,
                "state_bytes": (self.order + 1) * n * dv * bytes_per_el,
                "bytes_moved": (n * dv + m * dv
                                + self.order * n * dv) * bytes_per_el}
