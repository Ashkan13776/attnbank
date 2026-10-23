"""Quantise--dequantise, in every flavour the bank needs (``sec:exact-boundary``, ``sec:compress-quant``).

Two methods quantise, for different reasons, and they share this file so that the
*quantiser* is one implementation and the *claim* about it is one test.

* ``lowprec`` quantises ``q, k, v`` (and optionally ``P``) **inside the kernel**: the
  evaluation order of ``prop:exact-monoid`` is kept, the operand precision is not. That
  is where exactness stops.
* ``quantkv`` quantises the **stored cache**: keys and values stay on the same domain,
  only their representation changes. The key error enters as an exponential tilt of
  ``pi_q``, the value error linearly and attention-weighted -- the asymmetry of
  ``prop:compress-sensitivity``.

Everything here is a quantise--dequantise *simulation* in the input dtype: the point is
the error it introduces, not the storage it would save, and simulating keeps the bank
runnable on any device.

All scales are taken over an axis that makes the quantiser **causal and
head-independent** unless the caller explicitly asks otherwise (``axis='channel'``
pools statistics across tokens, which ``quantkv`` refuses under ``causal=True``).
"""
from __future__ import annotations

import torch

__all__ = ["quantize_int", "quantize_fp8", "affine_quantize", "hadamard"]

_FP8 = {"e4m3": (4, 3, 448.0), "e5m2": (5, 2, 57344.0)}


def quantize_int(x, bits, dim=-1):
    """Symmetric round-to-nearest INT-``bits`` with the scale taken over ``dim``.

    ``dim=-1`` is *per-token* (per-vector) scaling: the scale of token ``j`` depends on
    token ``j`` alone, which is what makes the quantizer causal and head-independent.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import quantize_int
    >>> x = torch.tensor([[-1.0, 0.0, 0.5, 1.0]])
    >>> quantize_int(x, bits=2)             # 2 bits: three usable levels, step = 1
    tensor([[-1.,  0.,  0.,  1.]])
    >>> bool((quantize_int(x, bits=None) == x).all())        # None is the identity
    True
    """
    if bits is None:
        return x
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax(dim=dim, keepdim=True)
    step = (amax / qmax).clamp_min(torch.finfo(x.dtype).tiny)
    return torch.round(x / step).clamp(-qmax - 1, qmax) * step


def quantize_fp8(x, fmt="e4m3", dim=-1):
    """Round-to-nearest onto the FP8 grid, simulated in the input dtype, scaled over ``dim``.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import quantize_fp8
    >>> x = torch.randn(1, 8)
    >>> float((quantize_fp8(x, "e4m3") - x).abs().max()) < 0.05 * float(x.abs().max())
    True
    """
    _, mant, big = _FP8[fmt]
    amax = x.abs().amax(dim=dim, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    s = big / amax
    z = x * s
    e = torch.floor(torch.log2(z.abs().clamp_min(torch.finfo(x.dtype).tiny)))
    step = torch.ldexp(torch.ones_like(z), (e - mant).to(torch.int32))
    return (torch.round(z / step) * step).clamp(-big, big) / s


def hadamard(d, device, dtype):
    """A Hadamard-like orthogonal matrix of size ``d`` (Sylvester, padded by identity).

    Incoherent processing: ``<Rq, Rk> = <q, k>`` exactly, so the rotation is free in
    infinite precision and spreads outlier coordinates before quantisation.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import hadamard
    >>> r = hadamard(4, "cpu", torch.float64)
    >>> torch.allclose(r @ r.T, torch.eye(4, dtype=torch.float64), atol=1e-12)
    True
    """
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] * 2 <= d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0) / 2 ** 0.5
    if h.shape[0] < d:
        out = torch.eye(d, device=device, dtype=dtype)
        out[:h.shape[0], :h.shape[0]] = h
        return out
    return h


def affine_quantize(x, bits, axis="token", group=None):
    """Uniform asymmetric quantize--dequantize of ``x`` ``[..., t, c]``.

    ``Q_b(x) = round((x - lo) / Delta) * Delta + lo`` with ``Delta = (hi - lo) / (2^b - 1)``
    and ``lo, hi`` the min and max over a *group*. ``axis='token'`` shares a scale among
    ``group`` channels of one token (KIVI's value grouping); ``axis='channel'`` shares
    it among ``group`` tokens of one channel (KIVI's key grouping). ``bits=None`` is the
    identity, which is the exact limit of ``quantkv``.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import affine_quantize
    >>> x = torch.tensor([[[0.0, 0.25, 0.5, 1.0]]])
    >>> affine_quantize(x, bits=1, axis="token")     # 1 bit: two levels, 0 and 1
    tensor([[[0., 0., 0., 1.]]])
    >>> q = affine_quantize(x, bits=3, axis="token")
    >>> torch.allclose(affine_quantize(q, bits=3, axis="token"), q)   # idempotent
    True
    """
    if bits is None:
        return x
    levels = 2 ** bits - 1
    t, c = x.shape[-2], x.shape[-1]
    if axis == "token":
        g = group or c
        pad = (-c) % g
        xx = torch.nn.functional.pad(x, (0, pad)) if pad else x
        blk = xx.unflatten(-1, (xx.shape[-1] // g, g))          # [..., t, ng, g]
        out = _quant_blocks(blk, levels).flatten(-2)
        return out[..., :c]
    if axis == "channel":
        g = group or t
        pad = (-t) % g
        xx = torch.nn.functional.pad(x, (0, 0, 0, pad)) if pad else x
        blk = xx.unflatten(-2, (xx.shape[-2] // g, g))          # [..., ng, g, c]
        blk = blk.transpose(-2, -1)                             # [..., ng, c, g]
        out = _quant_blocks(blk, levels).transpose(-2, -1).flatten(-3, -2)
        return out[..., :t, :]
    raise ValueError(f"axis must be 'token' or 'channel', got {axis!r}")


def _quant_blocks(blk, levels):
    lo = blk.amin(-1, keepdim=True)
    hi = blk.amax(-1, keepdim=True)
    step = ((hi - lo) / levels).clamp_min(torch.finfo(blk.dtype).tiny)
    return torch.round((blk - lo) / step) * step + lo
