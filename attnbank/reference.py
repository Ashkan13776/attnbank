"""The reference operator: dense softmax attention, y(q) = sum_j pi_q(j) v_j.

    pi_q(j) = exp(beta <q, k_j>) / sum_i exp(beta <q, k_i>),   beta = 1/sqrt(d)

Everything in the library is compared against this.  Section ``sec:found``.
"""
from __future__ import annotations

import torch

from .components.sdpa import sdpa_attend
from .core import (AttentionMethod, AttentionOutput, Category, Objective, causal_mask,
                   masked_softmax, scores)
from .registry import register


def softmax_scores(q, k, scale, causal):
    """Scaled logits with -inf on masked entries; kept as a public name."""
    return scores(q, k, scale, causal)


@register
class SoftmaxAttention(AttentionMethod):
    """Dense softmax attention (Vaswani et al., 2017); the reference operator."""

    name = "softmax"
    category = Category.EXACT
    objective = Objective.ACCELERATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("vaswani2017attention",)
    paper_section = "sec:found"
    backends = ("reference", "sdpa", "flash_attn")

    def __init__(self, sdpa_kernel: str | None = None, **kw):
        """``sdpa_kernel``: pin the SDPA sub-backend (``flash`` / ``mem_efficient`` /
        ``math``) instead of letting torch choose. Ignored by other backends."""
        super().__init__(sdpa_kernel=sdpa_kernel, **kw)
        self.sdpa_kernel = sdpa_kernel

    def _forward(self, q, k, v, causal, return_weights):
        pi = masked_softmax(softmax_scores(q, k, self._scale(q.shape[-1]), causal))
        y = pi @ v
        return AttentionOutput(y=y, weights=pi if return_weights else None)

    # ---- backends ----------------------------------------------------------
    def _forward_sdpa(self, q, k, v, causal, return_weights):
        """PyTorch's fused kernel. Identical operator; it just cannot hand back ``pi``."""
        if return_weights:
            return self._fallback_to_reference("materialise the attention weights")(
                q, k, v, causal, return_weights)
        y = sdpa_attend(q, k, v, scale=self._scale(q.shape[-1]), causal=causal,
                        kernel=self.sdpa_kernel)
        return AttentionOutput(y=y, extras={"backend": "sdpa",
                                            "sdpa_kernel": self.sdpa_kernel})

    def _forward_flash_attn(self, q, k, v, causal, return_weights):
        """The ``flash-attn`` package. CUDA and fp16/bf16 only; anything else falls back."""
        if return_weights:
            return self._fallback_to_reference("materialise the attention weights")(
                q, k, v, causal, return_weights)
        if q.dtype not in (torch.float16, torch.bfloat16) or not q.is_cuda:
            return self._fallback_to_reference(
                "run outside CUDA fp16/bf16")(q, k, v, causal, return_weights)
        from flash_attn import flash_attn_func

        # flash-attn takes [B, S, H, D] and aligns the causal mask bottom-right, which
        # is this library's convention exactly (DESIGN.md Sec. 2).
        y = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                            causal=causal, softmax_scale=self._scale(q.shape[-1]))
        return AttentionOutput(y=y.transpose(1, 2), extras={"backend": "flash_attn"})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        return {"flops": 2 * m * n * (d + dv), "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + 2 * m * n) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {}


def reference_weights(q, k, causal=False, scale=None):
    scale = scale if scale is not None else q.shape[-1] ** -0.5
    return masked_softmax(softmax_scores(q, k, scale, causal))
