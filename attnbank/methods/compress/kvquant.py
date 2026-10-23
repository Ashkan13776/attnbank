"""KVQuant: pre-RoPE per-channel keys, a sensitivity-weighted non-uniform datatype, and isolated outliers.

``sec:compress-quant`` and ``prop:compress-sensitivity``.  Where ``kivi`` changes only
the *grouping* axis, KVQuant changes the **coordinate system**, the **datatype** and the
**support** of the quantizer, in four independent moves that this file exposes as four
switches:

1. **Per-channel keys** (``bits_k``).  The scale and zero-point are shared by all tokens
   of one channel, which is where the key outliers live.  Values stay **per token**,
   because ``y`` averages over tokens.
2. **Pre-RoPE** (``pre_rope``).  RoPE rotates channel pairs by a *position-dependent*
   angle, so after it the "outlier channel" structure that per-channel grouping exploits
   is smeared across each pair.  KVQuant stores the key **before** the rotation and
   re-applies RoPE on the fly after dequantization.  The bank's tensors are post-RoPE by
   convention, so this file un-rotates, quantizes, and re-rotates -- the same three
   operations the cache performs, in the same order.
3. **Non-uniform datatype** (``nuq``).  The quantization signposts are the centroids of a
   **sensitivity-weighted k-means** on the entries of each vector, minimising
   ``sum_i F_i (x_i - Q(x_i))^2`` rather than the unweighted error a uniform grid
   minimises.  KVQuant takes ``F`` from the diagonal Fisher information of the language
   model on a calibration set.  No such gradient exists inside a single attention block,
   so the bank uses the *exact* local sensitivity instead, which is the first-order term
   of ``eq:error-joint``::

       F[j, c] = sum_i (beta pi_ij)^2 ||v_j - y_i||^2 q_ic^2

   This is the same objective with the same role for ``F``; it is a different (and, for
   one layer in isolation, sharper) estimate of it.  Under ``pre_rope`` the weight is
   averaged within each RoPE channel pair, which makes it invariant to the rotation it
   has to survive.  For **values** the local sensitivity is ``sum_i pi_ij^2``, constant
   along the channel axis, so sensitivity weighting is provably a no-op for a per-token
   value quantizer and is not applied -- a small result worth stating rather than
   hiding behind a flag.
4. **Dense-and-sparse** (``sparsity``).  A per-vector fraction of numerical outliers is
   held in full precision and *excluded from the range*, so the remaining levels are
   spent on the dense part.  Thresholds are per vector: per channel for the keys, per
   token for the values, as in the paper.  ``keep_first`` additionally pins the first
   token exactly -- the attention sink, which carries disproportionate mass and whose
   value error is therefore not averaged away.

**Not causal.**  A per-channel scale, an outlier threshold and a codebook are all
statistics over tokens.  KVQuant computes them offline on a calibration set, which makes
them constants at serving time; the bank has no calibration set and derives them from
the sequence itself, which is a genuinely non-causal operation.  Declaring
``supports_causal = False`` records that rather than hiding it behind a mask.  ``kivi``
is the member of this family that is causal, and the residual window is what it pays.

Quantize--dequantize is **simulated** in the input dtype; no memory is actually saved.
``cost()`` reports what the scheme would store, including the per-channel scales, the
codebook and the fp16 outliers -- the three overheads that decide whether 2 bits is 2 bits.

Exact limit: ``bits_k = bits_v = None``.  The RoPE round trip is then skipped entirely,
so the limit is exact and not merely accurate.

Reference: Hooper et al. (2024) KVQuant, NeurIPS.
"""
from __future__ import annotations

import torch

from ...components.quantization import affine_quantize
from ...core import (TINY, AttentionMethod, AttentionOutput, Category, Objective,
                     masked_softmax, scores)
from ...registry import register


# ---------------------------------------------------------------------------
# RoPE, as an invertible rotation of channel pairs
# ---------------------------------------------------------------------------
def rope(x, theta=10000.0, inverse=False):
    """Rotary embedding of ``x`` ``[..., n, d]`` at positions ``0 .. n-1``.

    Channel pair ``(2i, 2i+1)`` is rotated by ``p * theta^(-2i/d)`` at position ``p``;
    an odd trailing channel is left alone.  ``inverse=True`` applies the transpose, so
    ``rope(rope(x, inverse=True)) == x`` up to floating point -- that round trip is what
    "quantize the pre-RoPE key" means for a cache that is handed post-RoPE keys.
    """
    d, n = x.shape[-1], x.shape[-2]
    half = d // 2
    if half == 0:
        return x
    pair = x[..., :2 * half].unflatten(-1, (half, 2))
    pos = torch.arange(n, device=x.device, dtype=x.dtype).unsqueeze(-1)
    freq = theta ** (-2.0 * torch.arange(half, device=x.device, dtype=x.dtype) / d)
    ang = pos * freq                                              # [n, half]
    cos, sin = torch.cos(ang), torch.sin(ang)
    if inverse:
        sin = -sin
    a, b = pair[..., 0], pair[..., 1]
    out = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)
    return torch.cat([out, x[..., 2 * half:]], dim=-1)


# ---------------------------------------------------------------------------
# the datatype: sensitivity-weighted k-means on the entries of each vector
# ---------------------------------------------------------------------------
def nuq_levels(z, w, levels, iters=8):
    """Weighted 1-D Lloyd on the last axis of ``z``; returns ``z`` snapped to centroids.

    Zero-weight entries (the isolated outliers) do not move the centroids.  Initialised
    at the mid-points of the weighted range, so the result is deterministic.
    """
    live = w > 0
    big = torch.full_like(z, float("inf"))
    lo = torch.where(live, z, big).amin(-1, keepdim=True)
    hi = torch.where(live, z, -big).amax(-1, keepdim=True)
    dead = ~torch.isfinite(lo) | ~torch.isfinite(hi)
    lo = torch.where(dead, z.amin(-1, keepdim=True), lo)
    hi = torch.where(dead, z.amax(-1, keepdim=True), hi)

    t = (torch.arange(levels, device=z.device, dtype=z.dtype) + 0.5) / levels
    cent = lo + (hi - lo) * t
    for _ in range(iters):
        onehot = torch.nn.functional.one_hot(
            (z.unsqueeze(-1) - cent.unsqueeze(-2)).abs().argmin(-1), levels).to(z.dtype)
        mass = w.unsqueeze(-1) * onehot
        den = mass.sum(-2)
        cent = torch.where(den > 0, (mass * z.unsqueeze(-1)).sum(-2) / den.clamp_min(TINY),
                           cent).sort(-1).values
    idx = (z.unsqueeze(-1) - cent.unsqueeze(-2)).abs().argmin(-1)
    return cent.gather(-1, idx)


def outlier_mask(z, frac):
    """Per-vector dense-and-sparse split: the extreme ``frac`` of each row of ``z``.

    Half the budget is taken from each tail, which is the two-sided per-vector threshold
    of the paper; ``frac > 0`` always isolates at least one entry, so the mechanism is
    visible at bank-sized ``n`` as well as at 4096.
    """
    length = z.shape[-1]
    r = min(length, max(1, int(round(frac * length)))) if frac > 0 else 0
    if r == 0:
        return None
    hi = (r + 1) // 2
    mask = torch.zeros_like(z, dtype=torch.bool)
    if hi:
        mask.scatter_(-1, z.topk(hi, dim=-1).indices, True)
    if r - hi:
        mask.scatter_(-1, (-z).topk(r - hi, dim=-1).indices, True)
    return mask


def quantize_vectors(x, bits, weight=None, frac=0.0, nuq=True, iters=8):
    """Quantize along the **last** axis of ``x``: that axis is one quantization vector."""
    if bits is None:
        return x
    sparse = outlier_mask(x, frac)
    w = torch.ones_like(x) if weight is None else weight.clamp_min(TINY)
    if sparse is not None:
        w = w.masked_fill(sparse, 0.0)
    if nuq:
        xq = nuq_levels(x, w, 2 ** bits, iters)
    else:
        dense = x if sparse is None else torch.where(
            sparse, x.median(-1, keepdim=True).values.expand_as(x), x)
        xq = affine_quantize(dense, bits, "token", None)
    return xq if sparse is None else torch.where(sparse, x, xq)


@register
class KVQuantAttention(AttentionMethod):
    """Pre-RoPE per-channel keys, non-uniform sensitivity-weighted levels, sparse outliers."""

    name = "kvquant"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False          # per-channel scales, thresholds and codebooks
    expresses_weights = True
    row_stochastic = True
    references = ("hooper2024kvquant",)
    paper_section = "sec:compress"

    def __init__(self, bits_k: int | None = 3, bits_v: int | None = 3,
                 pre_rope: bool = True, nuq: bool = True, sensitivity: bool = True,
                 sparsity: float = 0.01, keep_first: int = 1, theta: float = 10000.0,
                 iters: int = 8, **kw):
        super().__init__(bits_k=bits_k, bits_v=bits_v, pre_rope=pre_rope, nuq=nuq,
                         sensitivity=sensitivity, sparsity=sparsity,
                         keep_first=keep_first, theta=theta, iters=iters, **kw)
        if not 0.0 <= sparsity < 0.5:
            raise ValueError(f"kvquant: sparsity must be in [0, 0.5); got {sparsity}")
        self.bits_k, self.bits_v, self.pre_rope = bits_k, bits_v, pre_rope
        self.nuq, self.sensitivity, self.sparsity = nuq, sensitivity, sparsity
        self.keep_first, self.theta, self.iters = keep_first, theta, iters

    # -- the local Fisher diagonal -------------------------------------------
    def _key_sensitivity(self, q, v, pi, beta):
        """``F[j, c] = sum_i (beta pi_ij)^2 ||v_j - y_i||^2 q_ic^2`` -- ``eq:error-joint``."""
        y = pi @ v
        sq = (v * v).sum(-1).unsqueeze(-2) - 2 * (y @ v.transpose(-1, -2)) \
            + (y * y).sum(-1).unsqueeze(-1)                       # [..., m, n]
        f = torch.einsum("...mn,...mc->...nc", (beta * pi) ** 2 * sq.clamp_min(0),
                         q * q)
        if self.pre_rope and f.shape[-1] >= 2:                    # pair-average: RoPE mixes
            half = f.shape[-1] // 2
            pair = f[..., :2 * half].unflatten(-1, (half, 2)).mean(-1, keepdim=True)
            f = torch.cat([pair.expand(*pair.shape[:-1], 2).flatten(-2),
                           f[..., 2 * half:]], dim=-1)
        return f

    def _pin(self, x_hat, x):
        """Keep the first ``keep_first`` tokens exact (the attention sink / pivot token)."""
        if not self.keep_first:
            return x_hat
        pos = torch.arange(x.shape[-2], device=x.device).unsqueeze(-1)
        return torch.where(pos < self.keep_first, x, x_hat)

    def _forward(self, q, k, v, causal, return_weights):
        beta = self._scale(q.shape[-1])
        s = scores(q, k, beta, causal)

        weight, k_hat = None, k
        if self.bits_k is not None:
            if self.nuq and self.sensitivity:
                weight = self._key_sensitivity(q, v, masked_softmax(s), beta) \
                             .transpose(-1, -2)                   # [..., d, n]
            k_pre = rope(k, self.theta, inverse=True) if self.pre_rope else k
            k_hat = quantize_vectors(k_pre.transpose(-1, -2), self.bits_k, weight,
                                     self.sparsity, self.nuq,
                                     self.iters).transpose(-1, -2)
            if self.pre_rope:
                k_hat = rope(k_hat, self.theta)
            k_hat = self._pin(k_hat, k)

        v_hat = v
        if self.bits_v is not None:
            v_hat = self._pin(quantize_vectors(v, self.bits_v, None, self.sparsity,
                                               self.nuq, self.iters), v)

        s_hat = scores(q, k_hat, beta, causal)
        pi = masked_softmax(s_hat)
        return AttentionOutput(
            y=pi @ v_hat, weights=pi if return_weights else None, v_hat=v_hat,
            extras={"s": s, "s_hat": s_hat, "delta_k": (k - k_hat).norm(dim=-1),
                    "delta_v": (v - v_hat).norm(dim=-1),
                    "sensitivity": weight.transpose(-1, -2) if weight is not None else None})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        """Payload, per-channel/per-token scales, the NUQ codebook and the fp16 outliers."""
        bk = (self.bits_k or 8 * bytes_per_el) / 8
        bv = (self.bits_v or 8 * bytes_per_el) / 8
        book = 0.0
        if self.nuq:                                   # one codebook per quantized vector
            book = bytes_per_el * ((2 ** (self.bits_k or 4)) * d
                                   + (2 ** (self.bits_v or 4)) * n)
        scales = 0.0 if self.nuq else 2 * bytes_per_el * (d + n)
        sparse = 2 * self.sparsity * (n * d + n * dv) * (bytes_per_el + 2)
        pinned = self.keep_first * (d + dv) * bytes_per_el
        state = n * (d * bk + dv * bv) + book + scales + sparse + pinned
        return {"flops": 2 * m * n * (d + dv) + self.iters * n * (d + dv),
                "state_bytes": state,
                "bytes_moved": m * (d + dv) * bytes_per_el + state}

    @classmethod
    def limit_config(cls):
        return {"bits_k": None, "bits_v": None}
