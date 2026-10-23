"""LRU: a diagonal linear RNN with a stable parametrisation, and no state-space theory.

``sec:memory-state``, ``tab:memory-recurrences``.  The recurrence is exactly S5's,

    h_t = lambda * h_{t-1} + gamma * (B v_t),      y_t = Re(C h_t) + D * v_t,

run with the same parallel scan.  What is different is where ``lambda`` comes from.
There is no continuous-time system, no HiPPO matrix and no discretisation step.  The
eigenvalues are parametrised directly in polar form,

    lambda_n = exp(-exp(nu_n) + i exp(theta_n)),

which is stable by construction -- ``|lambda_n| = exp(-exp(nu_n)) < 1`` for every real
``nu_n``, so no projection or clipping is ever needed -- and initialised with
``|lambda_n|`` uniform on a ring ``[r_min, r_max]`` and the phase uniform on
``[0, max_phase]``.  The input is rescaled by ``gamma_n = sqrt(1 - |lambda_n|^2)``, the
normalisation that keeps the state's variance ``O(1)`` instead of ``O(1/(1-|lambda|))``
however long the memory is.

That is the claim the method is here to make, and it is the reason the library carries
``s5`` and ``lru`` side by side rather than one of them: Orvieto et al. show that a
linear diagonal recurrence with a careful parametrisation matches deep state-space
models on Long Range Arena, so what S4 and S5 contribute is the *structure* -- linear,
diagonal, stable, normalised -- and not the state-space derivation that motivated it.
``tests/test_recurrent_ssm.py`` makes this concrete: one sequential loop reproduces
``s5`` and ``lru`` from their own ``extras``, because there is only one recurrence.

**Dynamics are input-independent.**  ``lambda``, ``gamma``, ``B``, ``C`` and ``D`` are
parameters.  Nothing in the transition depends on ``v_t``, and the queries and keys are
not read at all (``extras['reads_qk']`` is ``False``); the operator is a matrix-valued
convolution in the lag.  Letting ``lambda`` depend on the token is exactly the step the
gated linear RNNs took, and it is the boundary of this family.

**Weights.**  ``expresses_weights = False``, for S5's reason: ``C lambda^l B`` is a
``d_v x d_v`` matrix per lag, the operator mixes channels as well as positions, and
there is no scalar measure over the original positions to report.

Reference: Orvieto et al. (2023) LRU; Smith, Warrington and Linderman (2023) S5;
Blelloch (1990) parallel prefix sums.
"""
from __future__ import annotations

import math

import torch

from ...components.recurrence import readout_positions
from ...core import AttentionMethod, AttentionOutput, Category, Objective, make_generator
from ...registry import register
from ._ssm_kernels import complex_dtype, diagonal_scan


@register
class LRUAttention(AttentionMethod):
    """Linear Recurrent Unit: a stably parametrised diagonal linear RNN."""

    name = "lru"
    category = Category.RECURRENT
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = False        # matrix-valued kernel, as in s5
    row_stochastic = False
    trainable = True
    references = ("orvieto2023resurrecting", "smith2023simplified", "blelloch1990prefix")
    paper_section = "sec:memory-state"

    def __init__(self, n_state: int = 16, r_min: float = 0.0, r_max: float = 1.0,
                 max_phase: float = 6.283185307179586, feedthrough: bool = True,
                 seed: int = 0, **kw):
        """``n_state``: the size ``N`` of the (channel-shared) complex state.

        ``r_min`` / ``r_max`` bound ``|lambda|`` at initialisation -- the memory
        horizon, since mode ``n`` decays with time constant ``-1/log|lambda_n|`` -- and
        ``max_phase`` bounds the initial phase.  The defaults are the paper's.
        """
        super().__init__(n_state=n_state, r_min=r_min, r_max=r_max,
                         max_phase=max_phase, feedthrough=feedthrough, seed=seed, **kw)
        self.n_state, self.r_min, self.r_max = n_state, r_min, r_max
        self.max_phase, self.feedthrough, self.seed = max_phase, feedthrough, seed

    def _params(self, dv: int, device, dtype):
        """``(lambda [N], gamma B [N, dv], C [dv, N], D [dv])`` in the working dtype.

        The same four objects ``s5`` returns and in the same roles, which is what lets
        one loop check both.
        """
        cdtype = complex_dtype(dtype)
        g = make_generator(self.seed)
        u1 = torch.rand(self.n_state, generator=g, dtype=torch.float64)
        u2 = torch.rand(self.n_state, generator=g, dtype=torch.float64)
        # |lambda|^2 uniform on [r_min^2, r_max^2]; clamped off 1 so that gamma > 0 and
        # the recurrence is a strict contraction even at r_max = 1.
        mag2 = (u1 * (self.r_max ** 2 - self.r_min ** 2) + self.r_min ** 2).clamp(
            1e-12, 1.0 - 1e-12)
        nu_log = torch.log(-0.5 * torch.log(mag2))
        theta_log = torch.log(torch.full_like(u2, self.max_phase) * u2
                              + torch.finfo(torch.float64).tiny)
        lam = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma = torch.sqrt(1.0 - lam.abs() ** 2)                      # [N], real
        lam, gamma = lam.to(device), gamma.to(device)

        def cplx(*shape, std):
            re = torch.randn(*shape, generator=g, dtype=torch.float64)
            im = torch.randn(*shape, generator=g, dtype=torch.float64)
            return (torch.complex(re, im) * (std / math.sqrt(2.0))).to(device)

        b = cplx(self.n_state, dv, std=1.0 / math.sqrt(dv))
        c = cplx(dv, self.n_state, std=1.0 / math.sqrt(self.n_state))
        d = (torch.randn(dv, generator=g, dtype=torch.float64).to(device)
             if self.feedthrough else torch.zeros(dv, dtype=torch.float64, device=device))
        return (lam.to(cdtype), (gamma.unsqueeze(-1).to(b.dtype) * b).to(cdtype),
                c.to(cdtype), d.to(dtype))

    def _forward(self, q, k, v, causal, return_weights):
        m, n, dv = q.shape[-2], v.shape[-2], v.shape[-1]
        lam, b_bar, c_eff, d_skip = self._params(dv, v.device, v.dtype)
        u = torch.einsum("...nc,kc->...nk", v.to(c_eff.dtype), b_bar)   # [..., n, N]
        h = diagonal_scan(lam, u)
        y_full = torch.einsum("...nk,ck->...nc", h, c_eff).real + d_skip * v
        y = y_full.index_select(-2, readout_positions(m, n, causal, v.device))
        return AttentionOutput(y=y, weights=None, extras={
            "lam": lam, "b_bar": b_bar, "c_eff": c_eff, "d_skip": d_skip,
            "n_state": self.n_state, "state_size": self.n_state,
            "radius": lam.abs(), "reads_qk": False})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.n_state
        return {"flops": 8 * n * r * dv + 8 * n * r + 8 * m * r * dv,
                "state_bytes": 2 * r * bytes_per_el,
                "bytes_moved": (n * dv + m * dv + 4 * r * dv) * bytes_per_el}
