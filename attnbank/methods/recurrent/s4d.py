"""S4D: the diagonal state-space layer, as a long convolution over the values.

``sec:memory-state``, ``tab:memory-recurrences``.  Each value channel carries its own
single-input single-output state-space model

    h_t^{(c)} = A_c h_{t-1}^{(c)} + B_c v_{t,c},     y_{t,c} = C_c h_t^{(c)},

with ``A_c = diag(exp(dt_c lambda))`` diagonal and stable.  Because ``A``, ``B``, ``C``
and the step ``dt`` are **parameters and not functions of the input**, the recurrence
unrolls into a convolution whose kernel depends on the lag alone,

    y_{t,c} = sum_{l >= 0} K_c[l] v_{t-l, c},        K_c[l] = 2 Re sum_n C_{cn} Bbar_{cn} lambdabar_{cn}^l,

and the layer can be evaluated for the whole sequence at once.  That input-independence
is the property this corner of the library exists to exhibit, and the step that produced
the current generation of models -- Mamba, GLA, the gated linear RNNs -- was precisely
to give it up and let ``A_t`` depend on ``x_t``.

**What is implemented, and what is not.**  This is **S4D** (Gu, Gupta, Goel and Re,
2022), the *diagonal* state-space model, not the full S4 of Gu, Goel and Re (2022),
whose state matrix is diagonal-plus-low-rank and whose kernel is evaluated through a
Cauchy kernel and a Woodbury correction on the generating function.  S4D keeps the
diagonal part, drops the low-rank correction -- DSS (Gupta et al., 2022) showed that
costs nothing in accuracy -- and has a closed-form kernel, which is what is coded here.
Calling this file ``s4`` would have named something it is not.

**It does not read the queries or the keys.**  An LTI layer mixes positions with a
filter indexed by the lag; there is nowhere for ``q`` and ``k`` to enter.  The library
keeps the attention signature so that the operator can be put side by side with softmax
on identical inputs, and the resulting gap -- an SSM of this kind cannot do associative
recall at all -- is the finding, not a defect of the wrapper.  ``extras['reads_qk']`` is
``False`` and ``tests/test_recurrent_ssm.py`` asserts it.

**Weights.**  ``expresses_weights = False``.  Every value channel is convolved with a
*different* kernel, so there is no single measure ``pi_hat`` over the ``n`` context
positions that reproduces the output: the operator mixes positions, but with a different
mixture per channel.  Manufacturing a weight matrix here would mean averaging kernels
across channels and reporting something the method does not compute.  The per-channel
kernels are returned in ``extras['kernel']`` instead, which is the honest object.

**Frequency domain.**  ``fft=True`` evaluates the same convolution with three FFTs,
which is how S4 is actually run.  It is not the default because an FFT mixes the whole
sequence and is therefore not bit-exactly causal -- a later value perturbs an earlier
output in the last few bits -- and the conformance suite checks causality exactly.
``tests/test_recurrent_ssm.py`` checks the two paths agree to 1e-10 in fp64.

Reference: Gu, Goel and Re (2022) S4; Gu, Gupta, Goel and Re (2022) S4D;
Gupta, Gu and Berant (2022) DSS; Gu et al. (2020) HiPPO.
"""
from __future__ import annotations

import math

import torch

from ...components.recurrence import readout_positions
from ...core import AttentionMethod, AttentionOutput, Category, Objective, make_generator
from ...registry import register
from ._ssm_kernels import (
    channel_conv,
    fft_channel_conv,
    lag_and_visibility,
    s4d_poles,
    vandermonde,
    zoh,
)


@register
class S4DAttention(AttentionMethod):
    """Diagonal state-space layer (S4D): one SISO SSM per value channel."""

    name = "s4d"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = False        # a different convolution kernel per value channel
    row_stochastic = False
    trainable = True                 # dt, B, C stand in for parameters that are learned
    references = ("gu2022efficiently", "gu2022parameterization", "gupta2022diagonal",
                  "gu2020hippo")
    paper_section = "sec:memory-state"

    def __init__(self, n_state: int = 16, init: str = "lin", dt_min: float = 1e-3,
                 dt_max: float = 1e-1, fft: bool = False, seed: int = 0, **kw):
        """``n_state``: poles per channel (each standing for a conjugate pair).

        ``init`` selects the pole placement, ``'lin'`` (S4D-Lin) or ``'inv'``
        (S4D-Inv); ``dt_min`` / ``dt_max`` bound the log-uniform timestep, which is what
        sets the range of timescales the channels cover; ``fft`` switches to the
        frequency-domain evaluation (see the module docstring).
        """
        super().__init__(n_state=n_state, init=init, dt_min=dt_min, dt_max=dt_max,
                         fft=fft, seed=seed, **kw)
        self.n_state, self.init = n_state, init
        self.dt_min, self.dt_max = dt_min, dt_max
        self.fft, self.seed = fft, seed

    # ---- parameters ---------------------------------------------------------
    def _params(self, dv: int, device):
        """``(lambdabar, Bbar, C)``, each ``[dv, N]`` complex128, on ``device``.

        Built in double precision from a seeded generator and cast by the caller, so
        that an fp32 and an fp64 run are the same operator at two precisions.
        """
        g = make_generator(self.seed)
        lam = s4d_poles(self.n_state, self.init).to(device)              # [N]
        lo, hi = math.log(self.dt_min), math.log(self.dt_max)
        dt = torch.exp(torch.rand(dv, 1, generator=g, dtype=torch.float64) * (hi - lo)
                       + lo).to(device)                                  # [dv, 1]
        c = torch.complex(torch.randn(dv, self.n_state, generator=g, dtype=torch.float64),
                          torch.randn(dv, self.n_state, generator=g, dtype=torch.float64))
        c = (c / math.sqrt(2.0)).to(device)                              # E|C|^2 = 1
        b = torch.ones(dv, self.n_state, dtype=torch.complex128, device=device)
        lam_bar, b_bar = zoh(lam.unsqueeze(0), dt, b)                    # [dv, N]
        return lam_bar, b_bar, c

    def _kernel(self, dv: int, length: int, device, dtype):
        """The real convolution kernel ``[dv, length]``, ``K_c[l] = 2 Re(C A^l B)``."""
        lam_bar, b_bar, c = self._params(dv, device)
        powers = vandermonde(lam_bar, length)                            # [dv, N, L]
        k = 2.0 * (c * b_bar).unsqueeze(-1).mul(powers).sum(-2).real     # [dv, L]
        return k.to(dtype), lam_bar

    # ---- operator -----------------------------------------------------------
    def _forward(self, q, k, v, causal, return_weights):
        m, n, dv = q.shape[-2], v.shape[-2], v.shape[-1]
        kernel, poles = self._kernel(dv, n, v.device, v.dtype)
        if self.fft:
            y = fft_channel_conv(kernel, v).index_select(
                -2, readout_positions(m, n, causal, v.device))
        else:
            lag, visible = lag_and_visibility(m, n, causal, v.device)
            y = channel_conv(kernel, v, lag, visible)
        return AttentionOutput(y=y, weights=None, extras={
            "kernel": kernel, "poles": poles, "n_state": self.n_state,
            "state_size": self.n_state * dv, "reads_qk": False})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        """Costed as the deployed layer runs: build the kernel, then one FFT convolution."""
        logn = max(1.0, math.log2(max(n, 2)))
        return {"flops": 6 * dv * n * self.n_state + 15 * dv * n * logn,
                # a complex state of N poles for each of the dv channels
                "state_bytes": 2 * self.n_state * dv * bytes_per_el,
                "bytes_moved": (n * dv + m * dv + 2 * self.n_state * dv) * bytes_per_el}
