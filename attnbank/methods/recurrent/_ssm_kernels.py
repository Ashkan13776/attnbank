"""Machinery shared by the *time-invariant* state-space lineage (``sec:memory-state``).

Every operator in ``s4d``, ``s5``, ``lru`` and ``hyena`` is a linear time-invariant
(LTI) recurrence

    h_t = A h_{t-1} + B u_t,      y_t = C h_t  (+ D u_t),

in which ``A``, ``B`` and ``C`` are *parameters*: they do not depend on ``u_t``.  The
whole family therefore collapses to a causal convolution with a kernel that depends on
the lag alone,

    y_t = sum_{l >= 0} K[l] u_{t - l},        K[l] = C A^l B,

and the two ways of evaluating it -- materialise ``K`` and convolve, or run the
recurrence with a parallel associative scan -- are the two halves of this module.
``prop:memory-rank`` still applies: the state is ``A``'s dimension, whatever the choice
of ``A``.

Three details are worth stating because the conformance suite depends on them.

*Causality is bit-exact.*  :func:`channel_conv` multiplies by a weight matrix whose
invisible entries are exactly ``0.0``, so a later key cannot perturb an earlier output
even in the last bit.  An FFT convolution does *not* have that property -- it mixes the
whole sequence in the frequency domain -- which is why the methods that offer an FFT
path make it opt-in and check it against this one instead of defaulting to it.
:func:`diagonal_scan` is bit-exactly causal for the same reason: the padded elements
are the monoid identity ``(1, 0)``, and multiplying by one and adding zero are exact.

*Parameters are built in float64 and cast.*  As in ``linformer``, the random draws go
through a seeded ``torch.Generator`` in double precision and are only then moved to the
working dtype, so an fp32 and an fp64 run of the same method are the same operator
evaluated at two precisions rather than two different operators.

*Nothing here reads the queries or the keys.*  That is the defining property of this
family, not an omission; see the module docstrings of ``s4d``, ``s5`` and ``lru``.

This module is private to ``methods/recurrent/``.  If a second family ever needs the
scan it should be promoted to ``attnbank/components/recurrence.py``.
"""
from __future__ import annotations

import math

import torch

from ...components.recurrence import readout_positions

__all__ = ["complex_dtype", "lag_and_visibility", "channel_conv", "diagonal_scan",
           "s4d_poles", "zoh", "vandermonde", "fft_channel_conv"]

#: real dtype -> the complex dtype of the same precision
_COMPLEX = {torch.float32: torch.complex64, torch.float64: torch.complex128,
            torch.float16: torch.complex32, torch.bfloat16: torch.complex64}


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """The complex dtype of the same precision as the real ``dtype``.

    Examples
    --------
    >>> import torch
    >>> complex_dtype(torch.float64)
    torch.complex128
    """
    return _COMPLEX.get(dtype, torch.complex128)


def lag_and_visibility(m: int, n: int, causal: bool, device=None):
    """``(lag, visible)`` for the ``m`` readout positions against the ``n`` inputs.

    ``lag[t, j] = p(t) - j`` clamped at zero, and ``visible[t, j] = (p(t) >= j)``, where
    ``p`` is :func:`attnbank.components.readout_positions`: a query's own position when
    causal, the terminal position otherwise (the category convention -- with no mask the
    whole context is already in the state).

    Examples
    --------
    >>> lag, vis = lag_and_visibility(2, 3, causal=True)
    >>> lag
    tensor([[1, 0, 0],
            [2, 1, 0]])
    >>> vis
    tensor([[ True,  True, False],
            [ True,  True,  True]])
    """
    p = readout_positions(m, n, causal, device)
    j = torch.arange(n, device=device)
    lag = p[:, None] - j[None, :]
    return lag.clamp_min(0), lag >= 0


def channel_conv(kernel: torch.Tensor, x: torch.Tensor, lag: torch.Tensor,
                 visible: torch.Tensor) -> torch.Tensor:
    """Per-channel causal convolution ``y[..., t, c] = sum_j K[c, lag[t,j]] x[..., j, c]``.

    ``kernel`` is ``[C, L]`` with ``L`` at least ``max(lag) + 1``, ``x`` is
    ``[..., n, C]``, ``lag`` and ``visible`` are ``[T, n]``; the result is
    ``[..., T, C]``.  Invisible entries are multiplied by an exact zero, which is what
    makes the causal-leakage check pass bit for bit.

    The loop over channels is deliberate: materialising ``[T, n, C]`` at once is the
    only part of this family that would not fit in memory at long context, and a
    reference implementation should stay readable rather than fused (DESIGN.md Sec. 6).

    Examples
    --------
    >>> import torch
    >>> lag, vis = lag_and_visibility(3, 3, causal=True)
    >>> k = torch.tensor([[1.0, 0.5, 0.25]])                 # one channel, geometric
    >>> x = torch.tensor([[1.0], [0.0], [0.0]])
    >>> channel_conv(k, x, lag, vis).squeeze(-1)
    tensor([1.0000, 0.5000, 0.2500])
    """
    zero = torch.zeros((), dtype=kernel.dtype, device=kernel.device)
    out = []
    for c in range(kernel.shape[0]):
        w = torch.where(visible, kernel[c][lag], zero)       # [T, n], exact 0 off support
        out.append(x[..., c] @ w.transpose(0, 1))            # [..., n] @ [n, T]
    return torch.stack(out, dim=-1)


def fft_channel_conv(kernel: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The same convolution evaluated in the frequency domain, ``[..., n, C]``.

    ``K * x`` by one real FFT of each, a pointwise product and one inverse transform --
    the evaluation that makes a length-``n`` filter cost ``O(n log n)`` instead of
    ``O(n^2)``, and the reason S4 is a practical layer at all.  The result agrees with
    :func:`channel_conv` to floating-point tolerance but **not** bit for bit, and in
    particular not bit-exactly causally, so it is never the default.

    Examples
    --------
    >>> import torch
    >>> k = torch.tensor([[1.0, 0.5, 0.25]])
    >>> x = torch.tensor([[1.0], [1.0], [1.0]])
    >>> torch.allclose(fft_channel_conv(k, x).squeeze(-1),
    ...                torch.tensor([1.0, 1.5, 1.75]), atol=1e-6)
    True
    """
    n = x.shape[-2]
    size = 2 * n
    kf = torch.fft.rfft(kernel, n=size)                      # [C, size//2 + 1]
    xf = torch.fft.rfft(x.transpose(-1, -2), n=size)         # [..., C, size//2 + 1]
    return torch.fft.irfft(kf * xf, n=size)[..., :n].transpose(-1, -2)


def _shift(t: torch.Tensor, d: int, fill) -> torch.Tensor:
    """``t`` shifted ``d`` steps later along the time axis, padded with ``fill``."""
    pad = torch.full_like(t[..., :d, :], fill)
    return torch.cat([pad, t[..., :-d, :]], dim=-2)


def diagonal_scan(lam: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Inclusive parallel scan of ``h_t = lam * h_{t-1} + u_t`` (``h_{-1} = 0``).

    The first-order recurrence is a composition of affine maps, and affine maps form a
    monoid under ``(a1, b1) . (a2, b2) = (a1 a2, a2 b1 + b2)``; a Hillis--Steele sweep
    over it turns the sequential recurrence into ``ceil(log2 n)`` parallel rounds.  This
    is the step S5 took over S4 -- the *diagonal* state matrix is what makes the scan
    cheap, because composing two steps is an elementwise product rather than a matrix
    product (Smith et al., 2023; Blelloch, 1990).

    ``lam`` broadcasts against ``u``'s last axis; ``u`` is ``[..., n, N]``.

    Examples
    --------
    >>> import torch
    >>> lam = torch.tensor([0.5])
    >>> u = torch.ones(4, 1)
    >>> diagonal_scan(lam, u).squeeze(-1)
    tensor([1.0000, 1.5000, 1.7500, 1.8750])
    """
    n = u.shape[-2]
    a = lam.expand_as(u).clone()
    b = u
    d = 1
    while d < n:
        a_prev, b_prev = _shift(a, d, 1), _shift(b, d, 0)
        b = a * b_prev + b
        a = a * a_prev
        d *= 2
    return b


def s4d_poles(n_state: int, init: str = "lin",
              dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The ``N`` continuous-time poles of S4D, ``[N]`` complex, one per conjugate pair.

    ``"lin"``
        ``A_n = -1/2 + i pi n`` -- S4D-Lin, the closed form that approximates the
        spectrum of the normal part of HiPPO-LegS.
    ``"inv"``
        ``A_n = -1/2 + i (N/pi) (N/(2n+1) - 1)`` -- S4D-Inv, the approximation of
        HiPPO-LegS's *inverse* scaling.

    Both have real part ``-1/2``, so every mode is stable and the kernel decays; the
    imaginary parts set the frequencies the layer can represent (Gu et al., 2022b).

    Examples
    --------
    >>> import torch
    >>> s4d_poles(3, "lin").imag
    tensor([0.0000, 3.1416, 6.2832], dtype=torch.float64)
    """
    idx = torch.arange(n_state, dtype=dtype)
    if init == "lin":
        imag = math.pi * idx
    elif init == "inv":
        imag = (n_state / math.pi) * (n_state / (2 * idx + 1) - 1)
    else:
        raise ValueError(f"init must be one of ['inv', 'lin'], got {init!r}")
    return torch.complex(torch.full_like(idx, -0.5), imag)


def zoh(lam: torch.Tensor, dt: torch.Tensor, b: torch.Tensor):
    """Zero-order-hold discretisation: ``(exp(dt A), (exp(dt A) - I) A^{-1} B)``.

    The step ``dt`` is what turns a continuous-time state-space model into a sequence
    layer, and it is a *parameter*, not a function of the input -- the property this
    whole family is in the library to exhibit.

    Examples
    --------
    >>> import torch
    >>> lam = torch.tensor([-1.0 + 0j], dtype=torch.complex128)
    >>> lam_bar, b_bar = zoh(lam, torch.tensor([1.0], dtype=torch.float64),
    ...                      torch.ones(1, dtype=torch.complex128))
    >>> float(lam_bar.real), round(float(b_bar.real), 6)
    (0.36787944117144233, 0.632121)
    """
    lam_bar = torch.exp(dt * lam)
    return lam_bar, (lam_bar - 1.0) / lam * b


def vandermonde(lam: torch.Tensor, length: int) -> torch.Tensor:
    """``[..., length]`` of powers ``lam ** l``, computed as ``exp(l log lam)``.

    Exact for integer ``l`` (the principal branch cancels), and it underflows to zero
    rather than overflowing for the stable poles this family uses.

    Examples
    --------
    >>> import torch
    >>> vandermonde(torch.tensor([0.5 + 0j], dtype=torch.complex128), 3).real
    tensor([[1.0000, 0.5000, 0.2500]], dtype=torch.float64)
    """
    exps = torch.arange(length, dtype=torch.float64, device=lam.device)
    return torch.exp(torch.log(lam).unsqueeze(-1) * exps)
