"""KV-cache quantization: uniform affine (asymmetric) quantization of keys and values.

``sec:compress-quant`` and ``prop:compress-sensitivity``.  Keys and values stay on the
same domain; only their representation changes, so

    k_hat_j = Q_{b_k}(k_j),   v_hat_j = Q_{b_v}(v_j),
    Q_b(x)  = round((x - lo) / Delta) * Delta + lo,   Delta = (hi - lo) / (2^b - 1),

with ``lo, hi`` the min and max over a *group*.  The key error enters as a log-weight
perturbation ``delta_l_j = beta <q, delta k_j>`` -- an exponential tilt of ``pi_q`` --
and is amplified by ``e^{osc(delta_l)} - 1`` and by ``||q||``; the value error enters
linearly and is attention-weighted.  That asymmetry is the content of
``prop:compress-sensitivity`` and of ``eq:error-joint``, and is what study S4 measures.

Grouping.  ``axis='token'`` shares a scale among ``group`` channels of one token (KIVI's
value grouping); ``axis='channel'`` shares it among ``group`` tokens of one channel
(KIVI's key grouping, which confines an outlier channel's coarse step to that channel).
Only the *token-varying* part of the key error matters, since softmax cancels a common
shift -- which is exactly why per-channel grouping helps for keys.

Per-channel grouping pools statistics across tokens, so it is **not causal**; the
method raises rather than silently leaking.  ``axis='token'`` is causal and is the
default.

Exact limit: ``bits_k = bits_v = None`` (the identity quantizer).  16-bit affine
quantization is *not* bit-exact in fp32, so it would be a dishonest anchor.

The quantizer itself is :func:`attnbank.components.quantization.affine_quantize`,
shared with ``lowprec``; this file is only the decision to apply it to the *cache*.

Reference: Liu et al. (2024) KIVI; Hooper et al. (2024) KVQuant; Liu et al. (2024) IntactKV.
"""
from __future__ import annotations

from ...components.quantization import affine_quantize
from ...core import AttentionMethod, AttentionOutput, Category, Objective, masked_softmax, scores
from ...registry import register


@register
class QuantKVAttention(AttentionMethod):
    """Uniform affine quantization of the key and value caches."""

    name = "quantkv"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = True           # with axis='token'; axis='channel' raises under causal
    expresses_weights = True
    row_stochastic = True
    references = ("liu2024kivi", "hooper2024kvquant", "liu2024intactkv")
    paper_section = "sec:compress"

    def __init__(self, bits_k: int | None = 4, bits_v: int | None = 4,
                 axis: str = "token", group: int | None = None, **kw):
        super().__init__(bits_k=bits_k, bits_v=bits_v, axis=axis, group=group, **kw)
        self.bits_k, self.bits_v, self.axis, self.group = bits_k, bits_v, axis, group

    def _forward(self, q, k, v, causal, return_weights):
        if causal and self.axis == "channel" and (self.bits_k or self.bits_v):
            raise ValueError("quantkv(axis='channel') shares a scale across tokens and is "
                             "not causal; use axis='token' or causal=False")
        beta = self._scale(q.shape[-1])
        k_hat = affine_quantize(k, self.bits_k, self.axis, self.group)
        v_hat = affine_quantize(v, self.bits_v, self.axis, self.group)
        s = scores(q, k, beta, causal)
        s_hat = scores(q, k_hat, beta, causal)
        pi = masked_softmax(s_hat)
        return AttentionOutput(
            y=pi @ v_hat, weights=pi if return_weights else None, v_hat=v_hat,
            extras={"s": s, "s_hat": s_hat, "delta_k": (k - k_hat).norm(dim=-1),
                    "delta_v": (v - v_hat).norm(dim=-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        bk = (self.bits_k or 8 * bytes_per_el) / 8
        bv = (self.bits_v or 8 * bytes_per_el) / 8
        state = n * (d * bk + dv * bv)
        return {"flops": 2 * m * n * (d + dv), "state_bytes": state,
                "bytes_moved": m * (d + dv) * bytes_per_el + state}

    @classmethod
    def limit_config(cls):
        return {"bits_k": None, "bits_v": None}
