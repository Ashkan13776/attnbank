"""S5: one diagonal MIMO state-space model, run as a parallel associative scan.

``sec:memory-state``, ``tab:memory-recurrences``.  S4D gives every channel its own
single-input state-space model and evaluates each as a convolution.  S5 keeps the
diagonal state matrix and makes two simplifications that turn out to matter more than
the machinery they replace: one state ``h_t in C^N`` is **shared by all ``d_v``
channels** (multi-input multi-output), and the recurrence is evaluated by a parallel
scan rather than by an FFT convolution.

    h_t = lambdabar * h_{t-1} + Bbar v_t,      y_t = 2 Re(C h_t) + D * v_t,

with ``lambdabar = exp(dt lambda)``, ``Bbar = (lambdabar - 1) lambda^{-1} B``,
``B in C^{N x d_v}`` and ``C in C^{d_v x N}``.  Composing two steps of a *diagonal*
recurrence is an elementwise product, so the affine-map monoid underneath it is cheap
and the sequence can be swept in ``ceil(log2 n)`` parallel rounds
(:func:`~attnbank.methods.recurrent._ssm_kernels.diagonal_scan`).  That is the whole S5
contribution, and it is also what made the data-dependent successors possible: a scan
does not care whether ``lambdabar`` is a parameter or a function of the token, whereas
the FFT convolution S4 relies on does.

Here, it is a parameter.  ``lambdabar``, ``Bbar``, ``C`` and ``D`` do not depend on the
input, so the operator is still a (now matrix-valued) convolution ``K[l] = C A^l B`` in
the lag -- the family's defining restriction, and the one Mamba lifted.

**State.**  ``N`` complex numbers *in total*, against S4D's ``N`` per channel.  That is
the contrast ``prop:memory-rank`` is about and the reason ``n_state`` is a plain
configuration key: the capacity sweeps vary it.

**It does not read the queries or the keys** -- see ``s4d`` for why that is the point
rather than an omission.  ``extras['reads_qk']`` is ``False``.

**Weights.**  ``expresses_weights = False``.  The convolution kernel ``C A^l B`` is a
``d_v x d_v`` *matrix* per lag, so the output is not ``sum_j w_{tj} v_j`` for any scalar
``w``: the operator mixes channels as well as positions.  No measure over the original
positions exists to report, and the library says so instead of inventing one.

**Simplifications.**  The poles are placed by the S4D-Lin (or S4D-Inv) closed form,
which approximates the spectrum of the normal part of HiPPO-N; S5 obtains them by
numerically diagonalising HiPPO-N itself.  The two spectra agree in their leading term
``-1/2 + i pi n``, and a closed form keeps the layer deterministic across LAPACK
versions.  ``B``, ``C``, ``D`` and the per-mode timestep are seeded random stand-ins for
learned parameters, so ``trainable = True``.

Reference: Smith, Warrington and Linderman (2023) S5; Gu, Gupta, Goel and Re (2022)
S4D; Gu et al. (2020) HiPPO; Blelloch (1990) parallel prefix sums.
"""
from __future__ import annotations

import math

import torch

from ...components.recurrence import readout_positions
from ...core import AttentionMethod, AttentionOutput, Category, Objective, make_generator
from ...registry import register
from ._ssm_kernels import complex_dtype, diagonal_scan, s4d_poles, zoh


@register
class S5Attention(AttentionMethod):
    """Simplified state-space layer (S5): one diagonal MIMO SSM, swept by a parallel scan."""

    name = "s5"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = False        # the kernel C A^l B is matrix-valued, not scalar
    row_stochastic = False
    trainable = True
    references = ("smith2023simplified", "gu2022parameterization", "gu2020hippo",
                  "blelloch1990prefix")
    paper_section = "sec:memory-state"

    def __init__(self, n_state: int = 16, init: str = "lin", dt_min: float = 1e-3,
                 dt_max: float = 1e-1, feedthrough: bool = True, seed: int = 0, **kw):
        """``n_state``: the size ``N`` of the single state shared by all channels.

        ``feedthrough`` keeps S5's ``D * v_t`` skip term; ``init``, ``dt_min`` and
        ``dt_max`` place the poles and the per-mode timestep exactly as in ``s4d``.
        """
        super().__init__(n_state=n_state, init=init, dt_min=dt_min, dt_max=dt_max,
                         feedthrough=feedthrough, seed=seed, **kw)
        self.n_state, self.init = n_state, init
        self.dt_min, self.dt_max = dt_min, dt_max
        self.feedthrough, self.seed = feedthrough, seed

    def _params(self, dv: int, device, dtype):
        """``(lambdabar [N], Bbar [N, dv], C_eff [dv, N], D [dv])`` in the working dtype.

        ``C_eff`` folds S5's factor of two (one pole per conjugate pair) into ``C``, so
        the readout is ``Re(C_eff h) + D v`` -- the *same* expression ``lru`` uses.  That
        is not a coincidence and it is the point of ``lru``'s docstring.
        """
        cdtype = complex_dtype(dtype)
        g = make_generator(self.seed)
        lam = s4d_poles(self.n_state, self.init).to(device)                   # [N]
        lo, hi = math.log(self.dt_min), math.log(self.dt_max)
        dt = torch.exp(torch.rand(self.n_state, generator=g, dtype=torch.float64)
                       * (hi - lo) + lo).to(device)                           # [N]

        def cplx(*shape, std):
            re = torch.randn(*shape, generator=g, dtype=torch.float64)
            im = torch.randn(*shape, generator=g, dtype=torch.float64)
            return (torch.complex(re, im) * (std / math.sqrt(2.0))).to(device)

        b = cplx(self.n_state, dv, std=1.0 / math.sqrt(dv))
        c = cplx(dv, self.n_state, std=1.0 / math.sqrt(self.n_state))
        d = (torch.randn(dv, generator=g, dtype=torch.float64).to(device)
             if self.feedthrough else torch.zeros(dv, dtype=torch.float64, device=device))
        lam_bar, b_bar = zoh(lam, dt, torch.ones_like(lam))                   # [N]
        return (lam_bar.to(cdtype), (b_bar.unsqueeze(-1) * b).to(cdtype),
                (2.0 * c).to(cdtype), d.to(dtype))

    def _forward(self, q, k, v, causal, return_weights):
        m, n, dv = q.shape[-2], v.shape[-2], v.shape[-1]
        lam, b_bar, c_eff, d_skip = self._params(dv, v.device, v.dtype)
        u = torch.einsum("...nc,kc->...nk", v.to(c_eff.dtype), b_bar)   # [..., n, N]
        h = diagonal_scan(lam, u)                                       # [..., n, N]
        y_full = torch.einsum("...nk,ck->...nc", h, c_eff).real + d_skip * v
        y = y_full.index_select(-2, readout_positions(m, n, causal, v.device))
        return AttentionOutput(y=y, weights=None, extras={
            "lam": lam, "b_bar": b_bar, "c_eff": c_eff, "d_skip": d_skip,
            "n_state": self.n_state, "state_size": self.n_state, "reads_qk": False})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.n_state
        return {"flops": 8 * n * r * dv + 8 * n * r + 8 * m * r * dv,
                # ONE complex state of N poles, shared by every channel
                "state_bytes": 2 * r * bytes_per_el,
                "bytes_moved": (n * dv + m * dv + 4 * r * dv) * bytes_per_el}
