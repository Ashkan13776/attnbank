"""QJL: a one-bit Johnson--Lindenstrauss sketch of the keys, with an unbiased inner product.

``sec:compress-quant``.  Every other member of this family stores a *rounded copy* of
``k_j`` and accepts the rounding error.  QJL stores no copy at all: it keeps the
``sketch`` sign bits of a random projection,

    H_S(k_j) = sign(S k_j) in {-1, +1}^R,       S in R^{R x d},  S_ab ~ N(0, 1),

together with the scalar ``nu_j = ||k_j||_2`` in full precision, and estimates the
interaction *asymmetrically* -- one side sketched and quantized, the other side merely
sketched::

    g_hat(q, k_j) = sqrt(pi/2) * (nu_j / R) * <S q, sign(S k_j)>.

This is unbiased, ``E_S[g_hat] = <q, k_j>``, because for a Gaussian row ``r``
``E[sign(<r, k>) <r, q>] = sqrt(2/pi) <q, k> / ||k||`` -- the Grothendieck / random
hyperplane identity.  Nothing is grouped, so there is no scale and no zero-point to
store per group; the only side information is one norm per token, and that is the
"zero overhead" of the title.

**It is still a key perturbation.**  Writing the estimator as an inner product with an
explicit vector,

    k_hat_j = sqrt(pi/2) * (nu_j / R) * S^T sign(S k_j),

gives ``g_hat(q, k_j) = <q, k_hat_j>`` identically, so QJL lands in exactly the same
frame as the rest of ``compress/``: ``delta k_j = k_j - k_hat_j`` with
``E[k_hat_j] = k_j``, and ``extras['delta_k']`` is directly comparable with ``kivi``'s.
This file computes ``k_hat`` and then calls the ordinary score path, which is both
clearer and the reason Invariant W holds without special pleading.

**Unbiased logits are not unbiased outputs.**  ``E[s_hat] = s`` does not give
``E[y_hat] = y``: the softmax is convex in the logits and the estimator's variance is
``O(||q||^2 nu_j^2 / R)``, so it is *token dependent* and tilts ``pi_q`` toward
large-norm keys.  That failure is the point of having QJL in the bank, and it is why
the anchor below is a limit rather than only a convergence sequence.

Values are **not** sketched.  The paper is explicit that "standard token-wise
quantization is very effective" for the value cache, so ``bits_v`` here is an ordinary
per-token affine quantizer -- the same asymmetry ``kivi`` reaches by a different route.

Causal and head-independent: the sketch of token ``j`` depends on ``k_j`` alone and the
projection is a seeded constant, so nothing pools over tokens.

Anchors, both declared: ``limit_config`` disables the sketch and the value quantizer and
recovers dense softmax exactly, and ``convergence_configs`` increases ``sketch``, along
which the Monte-Carlo error must fall.

Quantize--dequantize is **simulated**: ``k_hat`` is materialised in the input dtype, so
the error is the one the 1-bit cache would make while the memory is not saved.

Reference: Zandieh, Daliri and Han (2025) QJL, AAAI.
"""
from __future__ import annotations

import math

import torch

from ...components.quantization import affine_quantize
from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     make_generator, masked_softmax, scores)
from ...registry import register


def qjl_keys(k, sketch, seed=0):
    """``k_hat_j = sqrt(pi/2) (||k_j|| / R) S^T sign(S k_j)`` -- the 1-bit sketch, decoded.

    The projection is drawn in float64 from a seeded generator and cast down, so fp32
    and fp64 runs use the same ``S`` and two builds agree bit for bit.
    """
    d = k.shape[-1]
    gen = make_generator(seed)
    s = torch.randn(sketch, d, generator=gen, dtype=torch.float64).to(
        device=k.device, dtype=k.dtype)
    proj = k @ s.transpose(0, 1)                                  # [..., n, R]
    bits = torch.sign(proj)
    nu = k.norm(dim=-1, keepdim=True)                             # [..., n, 1]
    return math.sqrt(math.pi / 2) / sketch * nu * (bits @ s), bits


@register
class QJLAttention(AttentionMethod):
    """One-bit JL sketch of the keys; per-token affine quantization of the values."""

    name = "qjl"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("zandieh2025qjl",)
    paper_section = "sec:compress"

    def __init__(self, sketch: int | None = 64, bits_v: int | None = 4, seed: int = 0,
                 **kw):
        super().__init__(sketch=sketch, bits_v=bits_v, seed=seed, **kw)
        if sketch is not None and sketch < 1:
            raise ValueError(f"qjl: sketch must be >= 1 or None; got {sketch}")
        self.sketch, self.bits_v, self.seed = sketch, bits_v, seed

    def _forward(self, q, k, v, causal, return_weights):
        beta = self._scale(q.shape[-1])
        k_hat = k if self.sketch is None else qjl_keys(k, self.sketch, self.seed)[0]
        v_hat = affine_quantize(v, self.bits_v, "token")
        s, s_hat = scores(q, k, beta, causal), scores(q, k_hat, beta, causal)
        pi = masked_softmax(s_hat)
        return AttentionOutput(
            y=pi @ v_hat, weights=pi if return_weights else None, v_hat=v_hat,
            extras={"s": s, "s_hat": s_hat, "delta_k": (k - k_hat).norm(dim=-1),
                    "delta_v": (v - v_hat).norm(dim=-1),
                    "key_norm": k.norm(dim=-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        """Keys: one bit per sketch coordinate and one norm -- no scale, no zero point.

        The two extra constants per token are the *value* quantizer's; the key side is
        the part with no group overhead, which is what the paper's title is about.
        """
        r = self.sketch if self.sketch is not None else 8 * bytes_per_el * d
        bv = (self.bits_v or 8 * bytes_per_el) / 8
        state = n * (r / 8 + bytes_per_el + dv * bv + 2 * bytes_per_el)
        return {"flops": 2 * m * n * (d + dv) + 2 * n * r * d + 2 * m * r * d,
                "state_bytes": state,
                "bytes_moved": m * (d + dv) * bytes_per_el + state}

    @classmethod
    def limit_config(cls):
        return {"sketch": None, "bits_v": None}

    @classmethod
    def convergence_configs(cls):
        return [{"sketch": r, "bits_v": None} for r in (8, 32, 128, 512)]
