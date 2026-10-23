"""Low-precision attention: simulated FP8 / INT8 logits and values.

``sec:exact-boundary``.  This is the method that marks *where exactness ends*.  An
FP8/INT8 kernel keeps the evaluation order of ``prop:exact-monoid`` but stores the
operands at a precision ``u' >> u``, so it satisfies the first clause of
``def:exact-scheme`` and fails the second.  By the criterion of "which object is
changed" it belongs with context compression: quantizing ``q, k`` replaces ``g`` by
``g_hat`` (a distributional error) and quantizing ``v`` or ``P`` replaces ``v`` by
``v_hat`` (a value-representation error).  Its objective is therefore ``approximate``,
not ``accelerate``, which is the whole point of putting it in ``exact/``.

Simulation is quantize--dequantize in the kernel, per attention head and per tensor:

    Q_b(x) = clamp(round(x / Delta), -2^{b-1}, 2^{b-1} - 1) * Delta,  Delta = max|x| / (2^{b-1} - 1)

for signed INT-b, and round-to-nearest on the FP8 E4M3 / E5M2 grids.  Optional
*incoherent processing* (``hadamard=True``) multiplies ``q`` and ``k`` by a shared
random orthogonal Hadamard matrix before quantizing, which is exact in infinite
precision (``<Rq, Rk> = <q, k>``) and spreads outlier coordinates, as in
FlashAttention-3's FP8 path.  Optional *smoothing* (``smooth_k=True``) subtracts the
token mean of ``k``, which shifts every logit of a query by one constant and so
leaves ``pi_q`` exactly invariant (SageAttention).

The quantizers themselves live in :mod:`attnbank.components.quantization`, shared with
``quantkv``: one implementation, one place where the rounding rule is argued about.

Reference: Shah et al. (2024); Zhang et al. (2025) SageAttention;
Micikevicius et al. (2022); Chen et al. (2024) INT-FlashAttention.
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     make_generator, masked_softmax, scores)
from ...components.quantization import hadamard, quantize_fp8, quantize_int
from ...registry import register

@register
class LowPrecisionAttention(AttentionMethod):
    """Quantize--dequantize ``q, k, v`` (and optionally ``P``) inside the kernel."""

    name = "lowprec"
    category = Category.EXACT
    objective = Objective.APPROXIMATE   # tab:comparative: "Low-precision attention ... appr."
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("shah2024flashattention3", "zhang2025sageattention",
                  "micikevicius2022fp8", "chen2024intflashattention")
    paper_section = "sec:exact-boundary"

    def __init__(self, bits: int | None = 8, fmt: str | None = None, bits_p: int | None = None,
                 hadamard_qk: bool = False, smooth_k: bool = False, seed: int = 0, **kw):
        """``bits``: INT width for q, k, v (``None`` = no quantization, the exact limit).
        ``fmt``: ``'e4m3'`` / ``'e5m2'`` to use FP8 instead of INT.
        ``bits_p``: optional INT width for the attention weights themselves."""
        super().__init__(bits=bits, fmt=fmt, bits_p=bits_p, hadamard_qk=hadamard_qk,
                         smooth_k=smooth_k, seed=seed, **kw)
        self.bits, self.fmt, self.bits_p = bits, fmt, bits_p
        self.hadamard_qk, self.smooth_k, self.seed = hadamard_qk, smooth_k, seed

    def _q(self, x):
        if self.fmt is not None:
            return quantize_fp8(x, self.fmt)
        return quantize_int(x, self.bits)

    def _forward(self, q, k, v, causal, return_weights):
        d = q.shape[-1]
        beta = self._scale(d)
        qq, kk = q, k
        if self.smooth_k:                     # exact: a per-query constant logit shift
            if causal:
                raise ValueError("lowprec(smooth_k=True) subtracts the mean of all keys and "
                                 "is therefore not causal; use it on prefill only")
            kk = kk - kk.mean(dim=-2, keepdim=True)
        if self.hadamard_qk:                  # exact: <Rq, Rk> = <q, k>
            r = hadamard(d, q.device, q.dtype)
            qq, kk = qq @ r, kk @ r
        qq, kk, vv = self._q(qq), self._q(kk), self._q(v)

        s_hat = scores(qq, kk, beta, causal)
        pi = masked_softmax(s_hat)
        if self.bits_p is not None:
            pi = quantize_int(pi, self.bits_p, dim=-1)
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(torch.finfo(pi.dtype).tiny)
        y = pi @ vv
        extras = {"s_hat": s_hat, "delta_v": (v - vv).norm(dim=-1)}
        return AttentionOutput(y=y, weights=pi if return_weights else None,
                               v_hat=vv if return_weights else None, extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        b = bytes_per_el if self.bits is None and self.fmt is None else \
            max(1, (8 if self.fmt else self.bits) // 8)
        return {"flops": 2 * m * n * (d + dv), "state_bytes": n * (d + dv) * b,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * b}

    @classmethod
    def limit_config(cls):
        # 16-bit uniform affine quantization is not bit-exact in fp32, so the honest
        # exact limit is the identity quantizer.
        return {"bits": None, "fmt": None, "bits_p": None}
