"""Multi-head latent attention: one joint low-rank KV latent, per-head up-projection.

``sec:compress-arch``.  Instead of caching ``h`` key--value pairs per position, MLA
caches a single latent ``c_j = W_down [k_j^{(1)}; ...; k_j^{(h)}; v_j^{(1)}; ...]`` of
dimension ``r``, and reconstructs each head's key and value by a per-head
up-projection.  The cached state per position drops from ``h (d + d_v)`` to ``r``.

Like GQA this is an **operator modification** trained as such; it is in the library for
the roofline (it is the other intervention that raises decode arithmetic intensity) and
because it is the cleanest example of a compression that couples heads: the latent is
shared, so ``head_independent = False``.

The projections are *trained* in DeepSeek-V2.  Here ``W_down`` is a fixed seeded
Gaussian and the up-projection is its pseudo-inverse, so the reconstruction is the
orthogonal projection onto ``row(W_down)``.  That choice is data-independent, hence
causal, and makes the exact limit provable: at ``r = h (d + d_v)`` the matrix is square
and invertible, the reconstruction is exact, and the operator is dense softmax.

Reference: DeepSeek-AI (2024) DeepSeek-V2.
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     make_generator, masked_softmax, scores)
from ...components.sdpa import sdpa_attend
from ...registry import register


@register
class MLAAttention(AttentionMethod):
    """Joint low-rank KV latent shared by all heads."""

    name = "mla"
    category = Category.COMPRESS
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    head_independent = False
    references = ("deepseekai2024deepseekv2",)
    paper_section = "sec:compress"
    backends = ("reference", "sdpa")

    def __init__(self, r: int | None = 16, seed: int = 0,
                 sdpa_kernel: str | None = None, **kw):
        """``r``: latent width; ``None`` uses the full width (the exact limit)."""
        super().__init__(r=r, seed=seed, sdpa_kernel=sdpa_kernel, **kw)
        self.r, self.seed = r, seed
        self.sdpa_kernel = sdpa_kernel

    def _reconstruct(self, k, v):
        """Down-project the concatenated KV to ``r`` latents and up-project back."""
        h, d, dv = k.shape[-3], k.shape[-1], v.shape[-1]
        wide = h * (d + dv)
        r = min(self.r or wide, wide)
        # [..., n, h*(d+dv)]: the per-position concatenation over heads
        cat = torch.cat([k, v], dim=-1).transpose(-3, -2).flatten(-2)
        g = make_generator(self.seed)
        w_down = (torch.randn(r, wide, generator=g, dtype=torch.float64) / wide ** 0.5)
        w_up = torch.linalg.pinv(w_down)                              # [wide, r]
        latent = cat.double() @ w_down.T.to(cat.device)
        rec = (latent @ w_up.T.to(cat.device)).to(cat.dtype)
        rec = rec.unflatten(-1, (h, d + dv)).transpose(-3, -2)        # [..., h, n, d+dv]
        return rec[..., :d], rec[..., d:], {"r": r, "latent_width": wide,
                                            "latent": latent}

    def _forward(self, q, k, v, causal, return_weights):
        k_hat, v_hat, extras = self._reconstruct(k, v)
        pi = masked_softmax(scores(q, k_hat, self._scale(k.shape[-1]), causal))
        return AttentionOutput(y=pi @ v_hat, weights=pi if return_weights else None,
                               v_hat=v_hat, extras=extras)

    def _forward_sdpa(self, q, k, v, causal, return_weights):
        """Reconstruct, then the fused kernel: the latent projection is the method."""
        if return_weights:
            return self._fallback_to_reference(
                "materialise the attention weights or v_hat")(q, k, v, causal,
                                                              return_weights)
        k_hat, v_hat, extras = self._reconstruct(k, v)
        y = sdpa_attend(q, k_hat, v_hat, scale=self._scale(k.shape[-1]), causal=causal,
                        kernel=self.sdpa_kernel)
        return AttentionOutput(y=y, extras={**extras, "backend": "sdpa"})

    def cost(self, m, n, d, dv, bytes_per_el=2, heads: int = 1):
        wide = heads * (d + dv)
        r = min(self.r or wide, wide)
        per_head = n * r * bytes_per_el / max(1, heads)
        return {"flops": 2 * m * n * (d + dv) + 2 * n * r * (d + dv),
                "state_bytes": per_head,
                "bytes_moved": m * (d + dv) * bytes_per_el + per_head}

    @classmethod
    def limit_config(cls):
        return {"r": None}
