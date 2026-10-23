"""Perceiver / Perceiver IO: a *nonlinear* latent bottleneck -- no induced measure.

``sec:factor-methods``.  A fixed latent array of ``r`` slots cross-attends to the
context, a stack of latent self-attention + MLP blocks processes the latents, and the
real queries cross-attend to the result (Jaegle et al., 2021; Jaegle et al., 2022):

    z_0 = b V,          b(z, x) = softmax_x( beta <w_z, k_x> )          (encode)
    z_{l+1} = Block(z_l)                                                (process)
    y_hat(q) = a(q, .) z_L,   a(q, z) = softmax_z( beta <q, (z_L P)_z> )  (decode)

The two cross-attentions are Markov kernels, and it is tempting to call the product
``a b`` the induced measure.  **It is not, and this method declares
``expresses_weights = False`` for exactly that reason.**  ``Block`` -- pre-norm latent
self-attention with a residual, then a GELU MLP with a residual -- is a *nonlinear*
map of the latents, so ``z_L`` is not ``(something) V`` for any matrix depending only
on ``(q, k)``, and the decoder's keys ``z_L P`` depend on ``V`` as well.  The operator
is not linear in the value field at all: ``y(2V) != 2 y(V)``.  No matrix ``W(q, k)``
with ``y = W V`` exists, let alone a probability measure on the ``n`` positions, so
returning ``a @ b`` would be a fabrication.  This is the sharp line in ``sec:factor``
between a *stochastic factorization*, which composes two kernels and stays a measure
(``latent_stochastic``, ``abc``, ``latte``), and a *latent bottleneck*, which processes
between the hops and leaves the measure behind.  Both kernels are still reported in
``extras`` -- as the two attention maps they are, not as a factorization.

Perceiver was never approximating softmax: it is an architecture for attending to very
large inputs at ``O(r n)`` cost, and there is no ``r`` at which it becomes dense
attention over the context.  ``objective = MODIFY``; both anchors are ``None`` and the
conformance suite asserts the exemption.

Untrained behaviour, which is what a method bank can offer for an architecture whose
content is learned (``trainable = True``): the latent array ``w``, the latent
projection ``P``, the MLP ``W_1, W_2`` and the decoder projection ``P_out`` are seeded
Gaussians scaled by ``1/sqrt(fan_in)``; layer norms are parameter-free (no affine).
Following Perceiver's own weight sharing, one set of block parameters is reused at
every depth.  The map is then a fixed, deterministic, well-defined nonlinear
bottleneck -- the structure is exact, the content is random.

Not causal: the encode kernel pools the whole context into every latent.

Reference: Jaegle et al. (2021) Perceiver; Jaegle et al. (2022) Perceiver IO.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    make_generator,
    masked_softmax,
)
from ...errors import ConfigError
from ...registry import register


@register
class PerceiverAttention(AttentionMethod):
    """Cross-attend into r latents, process them nonlinearly, cross-attend out."""

    name = "perceiver"
    category = Category.FACTORIZE
    objective = Objective.MODIFY
    supports_causal = False
    expresses_weights = False    # the processed latents destroy the kernel composition
    row_stochastic = False       # there are no weights to be stochastic
    trainable = True
    references = ("jaegle2021perceiver", "jaegle2022perceiverio")
    paper_section = "sec:factor"

    def __init__(self, r: int = 8, depth: int = 2, width: int = 2, seed: int = 0, **kw):
        """``r`` latents, ``depth`` latent blocks (>= 1), MLP expansion ``width``."""
        if depth < 1:
            raise ConfigError(
                f"'perceiver' got depth={depth}: at least one latent block is required. "
                "The block is what makes this a nonlinear bottleneck rather than a "
                "stochastic factorization; with depth=0 the method would be "
                "'latent_stochastic', which is in the bank under that name.")
        super().__init__(r=r, depth=depth, width=width, seed=seed, **kw)
        self.r, self.depth, self.width, self.seed = r, depth, width, seed

    def _params(self, d, dv, device, dtype):
        """Seeded latent array and block/decoder projections; one set, shared by depth."""
        g = make_generator(self.seed)
        hid = max(1, self.width * dv)

        def draw(*shape, fan_in):
            t = torch.randn(*shape, generator=g, dtype=torch.float64) / fan_in ** 0.5
            return t.to(device, dtype)

        return (draw(self.r, d, fan_in=1),        # w: the latent array (query side)
                draw(dv, d, fan_in=dv),           # P: latent -> key space (self-attn)
                draw(dv, hid, fan_in=dv),         # W1
                draw(hid, dv, fan_in=hid),        # W2
                draw(dv, d, fan_in=dv))           # P_out: decoder keys

    def _forward(self, q, k, v, causal, return_weights):
        d, dv = k.shape[-1], v.shape[-1]
        beta = self._scale(d)
        w, p, w1, w2, p_out = self._params(d, dv, k.device, k.dtype)
        norm = (dv,)

        # encode: r latents cross-attend to the context
        b = masked_softmax(torch.einsum("rd,...nd->...rn", w, k) * beta)   # [..., r, n]
        z = b @ v                                                          # [..., r, dv]

        # process: latent self-attention + MLP, both residual, both pre-norm
        for _ in range(self.depth):
            zn = F.layer_norm(z, norm)
            zp = zn @ p
            s = masked_softmax(torch.einsum("...rd,...sd->...rs", zp, zp) * beta)
            z = z + s @ zn
            zn = F.layer_norm(z, norm)
            z = z + F.gelu(zn @ w1) @ w2

        # decode: the real queries cross-attend to the processed latents
        z = F.layer_norm(z, norm)
        a = masked_softmax(torch.einsum("...md,...rd->...mr", q, z @ p_out) * beta)
        return AttentionOutput(y=a @ z, weights=None,
                               extras={"a": a, "b": b, "latents": z, "r": self.r,
                                       "depth": self.depth,
                                       "note": "a and b are the two cross-attentions; "
                                               "a @ b is NOT the induced measure -- the "
                                               "latents are processed between them"})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r, hid = self.r, max(1, self.width * dv)
        block = 2 * r * r * (d + dv) + 2 * r * dv * d + 4 * r * dv * hid
        return {"flops": 2 * n * r * (d + dv) + self.depth * block
                         + 2 * m * r * (d + dv) + 2 * r * dv * d,
                "state_bytes": r * dv * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
