"""Atom: mixed-precision quantization with outlier **channel reordering**.

``sec:compress-quant``.  Atom is the third answer in this family to the same fact --
that a few channels of the key/value activations carry far more energy than the rest --
and it is the one that changes neither the axis (``kivi``) nor the datatype
(``kvquant``) but the **bit width per channel**:

* reorder the channels by energy, ``sum_j x_jc^2``, so that the outliers are contiguous
  at the end of the matrix;
* keep the ``outliers`` highest-energy channels at ``bits_outlier`` bits;
* quantize the rest asymmetrically in contiguous groups of ``group`` channels *within
  each token*, which is what the paper calls fine-grained group quantization;
* undo the permutation.

**What the reordering is for, and what it is not for.**  The paper is explicit: Atom
"reorders the scattered outlier channels of activations to the end of the matrix with
the goal of enabling the efficient implementation of mixed-precision within a single
kernel".  The permutation is a *layout* device; the accuracy comes from the mixed
precision it makes implementable.  Numerically the permutation is exactly the identity
when ``group = None`` -- one scale per token either way -- and merely *different*, not
reliably better, when ``group`` is finite: sorting by channel energy can put two
outliers in the same group and widen it.  The test file asserts both of those rather
than the flattering version.  What does hold, and holds strongly, is that giving the
top channels 8 bits beats spending a *larger* uniform budget on every channel.

**What is verbatim and what is transplanted.**  Section 4.4 of the paper quantizes the
KV cache with *asymmetric low-bit quantization within each attention head* and nothing
else -- that is ``outliers = 0, group = None``, and it coincides exactly with
``quantkv(axis='token')``.  The reordered mixed precision of Section 4.2 ("we reorder
the activations and weights based on the square sum of each activation channel.  After
reordering, we select 128 channels with the highest square sum values as outliers and
keep them in INT8") is applied there to the linear layers' activations, not to the
cache.  This file applies it to the cache, because the cache *is* an activation matrix
with exactly the outlier-channel structure KIVI and KVQuant attack by other means, and
because the comparison is the reason to have all three in one bank.  The docstring says
so rather than letting the name imply otherwise.

Note also what Atom argues and what the monograph argues.  Section 4.4: "Due to the
normalization of Softmax, the quantization error of the K cache has less influence on
the output."  ``prop:compress-sensitivity`` says the opposite -- the softmax
*amplifies* a key error by ``e^{osc(delta l)} - 1`` while averaging a value error --
and ``bits_k`` / ``bits_v`` are split here precisely so that study S4 can settle it on
the same scheme the claim is made about.

**Not causal.**  The channel permutation is a statistic over tokens.  Atom computes it
offline from calibration data, so at serving time it is a constant; the bank has no
calibration set and derives it from the sequence, which is not causal.  The grouping
*within* a token is, which is why ``quantkv(axis='token')`` -- Atom's Section 4.4
scheme -- supports ``causal=True`` and this does not.

Quantize--dequantize is **simulated** in the input dtype; no memory is saved here.

Exact limit: ``bits_k = bits_v = None`` -- which switches off the outlier channels too,
so that ``None`` means the same thing here as everywhere else in the family.  The
permutation and its inverse are exact, so the identity quantizer leaves ``k`` and ``v``
bit-for-bit unchanged.

Reference: Zhao et al. (2024) Atom, MLSys.
"""
from __future__ import annotations

import torch

from ...components.quantization import affine_quantize
from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     masked_softmax, scores)
from ...registry import register


def channel_order(x):
    """Ascending order of the channels of ``x`` ``[..., t, c]`` by energy ``sum_t x^2``.

    Outliers land last, which is where Atom puts them so that the mixed-precision
    boundary is a single split point rather than a scatter.
    """
    return (x * x).sum(-2).argsort(dim=-1)


def reorder_mixed(x, bits, bits_outlier, outliers, group):
    """Atom's quantizer: reorder by energy, split off ``outliers`` channels, group the rest.

    ``bits = None`` disables the quantizer entirely, outlier channels included, so that
    ``bits_k``/``bits_v`` keep the family's meaning: ``None`` is the identity.
    """
    if bits is None:
        return x, 0
    order = channel_order(x)
    idx = order.unsqueeze(-2).expand_as(x)
    xs = torch.gather(x, -1, idx)
    n_out = max(0, min(int(outliers), x.shape[-1]))
    n_in = x.shape[-1] - n_out
    parts = []
    if n_in:
        parts.append(affine_quantize(xs[..., :n_in], bits, "token", group))
    if n_out:
        parts.append(affine_quantize(xs[..., n_in:], bits_outlier, "token", None))
    xq = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
    undo = order.argsort(dim=-1).unsqueeze(-2).expand_as(xq)
    return torch.gather(xq, -1, undo), n_out


@register
class AtomAttention(AttentionMethod):
    """Energy-reordered mixed-precision quantization of the key and value caches."""

    name = "atom"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False          # the channel permutation pools over tokens
    expresses_weights = True
    row_stochastic = True
    references = ("zhao2024atom",)
    paper_section = "sec:compress"

    def __init__(self, bits_k: int | None = 4, bits_v: int | None = 4,
                 bits_outlier: int | None = 8, outliers: int = 1,
                 group: int | None = None, **kw):
        super().__init__(bits_k=bits_k, bits_v=bits_v, bits_outlier=bits_outlier,
                         outliers=outliers, group=group, **kw)
        if outliers < 0:
            raise ValueError(f"atom: outliers must be >= 0; got {outliers}")
        if group is not None and group < 1:
            raise ValueError(f"atom: group must be >= 1 or None; got {group}")
        self.bits_k, self.bits_v = bits_k, bits_v
        self.bits_outlier, self.outliers, self.group = bits_outlier, outliers, group

    def _forward(self, q, k, v, causal, return_weights):
        beta = self._scale(q.shape[-1])
        k_hat, n_out_k = reorder_mixed(k, self.bits_k, self.bits_outlier,
                                       self.outliers, self.group)
        v_hat, n_out_v = reorder_mixed(v, self.bits_v, self.bits_outlier,
                                       self.outliers, self.group)
        s, s_hat = scores(q, k, beta, causal), scores(q, k_hat, beta, causal)
        pi = masked_softmax(s_hat)
        return AttentionOutput(
            y=pi @ v_hat, weights=pi if return_weights else None, v_hat=v_hat,
            extras={"s": s, "s_hat": s_hat, "delta_k": (k - k_hat).norm(dim=-1),
                    "delta_v": (v - v_hat).norm(dim=-1),
                    "outlier_channels_k": n_out_k, "outlier_channels_v": n_out_v})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        """Low-bit dense channels, INT8 outlier channels, and one scale pair per group."""
        bo = (self.bits_outlier or 8 * bytes_per_el) / 8
        bk = (self.bits_k or 8 * bytes_per_el) / 8
        bv = (self.bits_v or 8 * bytes_per_el) / 8
        ok, ov = min(self.outliers, d), min(self.outliers, dv)
        g_k, g_v = self.group or max(1, d - ok), self.group or max(1, dv - ov)
        groups = -(-(d - ok) // g_k) + -(-(dv - ov) // g_v) + 2
        state = n * ((d - ok) * bk + ok * bo + (dv - ov) * bv + ov * bo
                     + 2 * groups * bytes_per_el)
        return {"flops": 2 * m * n * (d + dv) + n * (d + dv),
                "state_bytes": state,
                "bytes_moved": m * (d + dv) * bytes_per_el + state}

    @classmethod
    def limit_config(cls):
        return {"bits_k": None, "bits_v": None}
