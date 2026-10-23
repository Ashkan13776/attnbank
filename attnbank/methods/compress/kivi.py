"""KIVI: keys quantized **per channel**, values **per token**, with a full-precision residual.

``sec:compress-quant`` and ``prop:compress-sensitivity``.  KIVI is the scheme that
reads the key/value asymmetry off the operator and groups the two caches along
different axes:

* **Keys, per channel.**  The scale and zero-point are shared by ``group`` consecutive
  *tokens* of one channel, so an outlier channel's coarse step stays inside that
  channel.  What matters for ``pi_q`` is only the token-varying part of
  ``delta l_j = beta <q, delta k_j>`` -- a channel-constant error is absorbed by the
  softmax normalizer -- and per-channel grouping is exactly the choice that makes the
  surviving part small.
* **Values, per token.**  ``y = sum_j pi_j v_j`` is a weighted average over tokens, so
  a per-token scale confines each token's representation error to its own summand;
  no token can poison another.

**The residual window is what makes per-channel grouping causal.**  A per-channel scale
pools statistics over tokens, which is why ``quantkv(axis='channel')`` refuses
``causal=True``.  KIVI avoids this by *streaming*: the last ``residual`` tokens are kept
in full precision, and a group of ``group`` tokens is quantized only once it is complete
and has fallen out of the residual window.  A query at absolute position ``p`` therefore
sees quantized keys only for the groups entirely contained in ``[0, p + 1 - residual)``,
and full-precision keys for everything more recent::

    L(p) = group * max(0, (p + 1 - residual) // group)
    k_hat_j(p) = Q_channel(k)_j   if j < L(p),   else   k_j

The key representation is thus *query-dependent*, which is the honest rendering of a
streaming cache; the score matrix is assembled by selecting between the two score
matrices rather than by materialising ``[m, n, d]`` keys.  Nothing before position ``t``
can depend on a token at or after ``t``, which is what the conformance suite checks.

Values need no residual: per-token quantization is already causal, and Invariant W
requires a single ``v_hat`` per position, so the value residual of Algorithm 1 is
dropped here and recorded as such.

Quantize--dequantize is **simulated** in the input dtype (via
:func:`attnbank.components.quantization.affine_quantize`).  The error is real; the
memory saving is not -- ``cost()`` reports what the scheme *would* store, including the
per-group scale and zero-point, which is the term that stops ``group`` from shrinking
to one.

Exact limit: ``bits_k = bits_v = None`` (the identity quantizer).  Affine quantization
at 16 bits is not bit-exact in fp32, so it would be a dishonest anchor.

Paper defaults are ``group = 32`` and ``residual = 128``; the defaults here are scaled
to the bank's test sizes so that the default configuration actually quantizes something.

Reference: Liu et al. (2024) KIVI, ICML.
"""
from __future__ import annotations

import torch

from ...components.quantization import affine_quantize
from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     masked_softmax, query_positions, scores)
from ...registry import register


@register
class KIVIAttention(AttentionMethod):
    """Per-channel key / per-token value quantization with a full-precision residual."""

    name = "kivi"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = True           # the residual window is what buys this
    expresses_weights = True
    row_stochastic = True
    references = ("liu2024kivi",)
    paper_section = "sec:compress"

    def __init__(self, bits_k: int | None = 2, bits_v: int | None = 2,
                 group: int = 8, residual: int = 8, **kw):
        super().__init__(bits_k=bits_k, bits_v=bits_v, group=group, residual=residual,
                         **kw)
        if group < 1 or residual < 0:
            raise ValueError(f"kivi: group >= 1 and residual >= 0 required; got "
                             f"group={group}, residual={residual}")
        self.bits_k, self.bits_v = bits_k, bits_v
        self.group, self.residual = group, residual

    # -- the two key caches ---------------------------------------------------
    def _sealed_keys(self, k):
        """Per-channel quantization of every *complete* group; the tail is untouched."""
        n, g = k.shape[-2], self.group
        n_sealed = (n // g) * g
        if self.bits_k is None or n_sealed == 0:
            return k, 0
        head = affine_quantize(k[..., :n_sealed, :], self.bits_k, "channel", g)
        return torch.cat([head, k[..., n_sealed:, :]], dim=-2), n_sealed

    def _sealed_mask(self, m, n, n_sealed, causal, device):
        """``[m, n]`` (or ``[1, n]``) boolean: is key ``j`` already quantized for query ``i``?"""
        j = torch.arange(n, device=device)
        if causal:
            pos = query_positions(m, n, device=device)                     # [m]
            lim = ((pos + 1 - self.residual).clamp_min(0) // self.group) * self.group
            lim = lim.clamp(max=n_sealed)
            return j[None, :] < lim[:, None]
        lim = ((max(0, n - self.residual)) // self.group) * self.group
        return (j < min(lim, n_sealed))[None, :]

    def _forward(self, q, k, v, causal, return_weights):
        beta = self._scale(q.shape[-1])
        m, n = q.shape[-2], k.shape[-2]
        k_q, n_sealed = self._sealed_keys(k)
        v_hat = affine_quantize(v, self.bits_v, "token")

        s = scores(q, k, beta, causal)
        s_hat = s
        if self.bits_k is not None and n_sealed:
            sealed = self._sealed_mask(m, n, n_sealed, causal, q.device)
            s_hat = torch.where(sealed, scores(q, k_q, beta, causal), s)
        pi = masked_softmax(s_hat)
        return AttentionOutput(
            y=pi @ v_hat, weights=pi if return_weights else None, v_hat=v_hat,
            extras={"s": s, "s_hat": s_hat,
                    "delta_k": (k - k_q).norm(dim=-1),
                    "delta_v": (v - v_hat).norm(dim=-1),
                    "n_sealed": n_sealed})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        """Quantized groups, the full-precision residual, and the scales both need."""
        res = min(self.residual, n)
        nq = (max(0, n - res) // self.group) * self.group
        nf = n - nq
        bk = (self.bits_k or 8 * bytes_per_el) / 8
        bv = (self.bits_v or 8 * bytes_per_el) / 8
        # per-channel key scales: 2 per (group, channel); per-token value scales: 2 each
        scales = 2 * bytes_per_el * (-(-nq // self.group) * d + nq)
        state = nq * (d * bk + dv * bv) + nf * (d + dv) * bytes_per_el + scales
        return {"flops": 2 * m * n * (d + dv) + n * (d + dv),
                "state_bytes": state,
                "bytes_moved": m * (d + dv) * bytes_per_el + state}

    @classmethod
    def limit_config(cls):
        return {"bits_k": None, "bits_v": None}
