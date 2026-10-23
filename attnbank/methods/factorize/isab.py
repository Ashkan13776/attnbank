"""ISAB: the Set Transformer's induced set attention block -- a nonlinear bottleneck.

``sec:factor-methods``.  ISAB routes attention through ``r`` learned *inducing points*
``I`` in two multihead attention blocks (Lee et al., 2019):

    ISAB(X) = MAB(X, H),      H = MAB(I, X) in R^{r x d},
    MAB(X, Y) = LayerNorm( O + rFF(O) ),   O = LayerNorm( X + Attn(X, Y, Y) ).

Written with this library's arguments, the context is ``(k, v)`` and the first block
summarizes it into ``r`` induced rows, the second lets the queries read them:

    b(z, x) = softmax_x( beta <i_z, k_x> ),   H = MAB-wrap( I_v + int v db(z, .) ),
    a(q, z) = softmax_z( beta <q, (H P)_z> ), y_hat(q) = MAB-wrap( q P_q + a(q, .) H ).

The two attention maps are Markov kernels, but ISAB is **not** a stochastic
factorization and this method declares ``expresses_weights = False``.  Three separate
things break the composition, and each one alone is enough:

1. ``MAB`` applies a residual, a layer norm and a row-wise feedforward *between* the
   hops, so ``H`` is a nonlinear function of the values -- there is no matrix ``B``
   with ``H = B V``;
2. the second block's keys are a projection of ``H``, so even the read kernel depends
   on the values;
3. the query-side residual ``q P_q`` puts mass on a term that does not come from the
   value field at all -- with a constant value field the output is not that constant,
   which no probability measure on the ``n`` positions can do.

So ``y`` is not even linear in ``V``: ``y(2V) != 2 y(V)``, and no weight matrix exists
to report.  This is the contrast ``sec:factor`` draws against ``latent_stochastic``,
which cites the same paper for the *bare* two-hop kernel -- strip ``MAB`` down to its
attention and ISAB becomes a stochastic factorization; keep the block that the Set
Transformer actually uses and the measure on the original positions is gone.  Both
kernels are still exposed in ``extras``, as attention maps rather than factors.

``objective = MODIFY``: ISAB is a set-encoder architecture with a parameter-count and
``O(r n)`` cost argument, not an estimator of dense attention, and no ``r`` makes it
softmax.  Both anchors are ``None`` and the suite asserts the exemption.

Untrained behaviour (``trainable = True``): the inducing points ``I_q, I_v``, the two
feedforwards and the two projections are seeded Gaussians scaled by
``1/sqrt(fan_in)``; layer norms are parameter-free.  The projections ``P`` and ``P_q``
exist because this library allows ``d != dv`` while a Set Transformer is uniformly
``d_model`` wide; with ``d == dv`` they are the block's own ``W_K`` and the identity
residual, randomly initialized.

Not causal: the inducing points pool the whole context.

Reference: Lee et al. (2019) Set Transformer.
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
from ...registry import register


@register
class ISABAttention(AttentionMethod):
    """Two MABs through r inducing points; the block between them is nonlinear."""

    name = "isab"
    category = Category.FACTORIZE
    objective = Objective.MODIFY
    supports_causal = False
    expresses_weights = False    # MAB's rFF and residuals break the kernel composition
    row_stochastic = False       # there are no weights to be stochastic
    trainable = True
    references = ("lee2019set",)
    paper_section = "sec:factor"

    def __init__(self, r: int = 8, width: int = 2, seed: int = 0, **kw):
        """``r`` inducing points; ``width`` is the rFF expansion factor."""
        super().__init__(r=r, width=width, seed=seed, **kw)
        self.r, self.width, self.seed = r, width, seed

    def _params(self, d, dv, device, dtype):
        g = make_generator(self.seed)
        hid = max(1, self.width * dv)

        def draw(*shape, fan_in):
            t = torch.randn(*shape, generator=g, dtype=torch.float64) / fan_in ** 0.5
            return t.to(device, dtype)

        return (draw(self.r, d, fan_in=1),      # I_q: inducing points, key space
                draw(self.r, dv, fan_in=1),     # I_v: inducing points, value space
                draw(dv, hid, fan_in=dv),       # rFF_1
                draw(hid, dv, fan_in=hid),
                draw(dv, d, fan_in=dv),         # P:   H -> key space
                draw(d, dv, fan_in=d),          # P_q: query-side residual
                draw(dv, hid, fan_in=dv),       # rFF_2
                draw(hid, dv, fan_in=hid))

    def _forward(self, q, k, v, causal, return_weights):
        d, dv = k.shape[-1], v.shape[-1]
        beta = self._scale(d)
        i_q, i_v, w1, w2, p, p_q, w3, w4 = self._params(d, dv, k.device, k.dtype)
        norm = (dv,)

        def mab_tail(o, wa, wb):
            """LayerNorm(O + rFF(O)) -- the second half of a multihead attention block."""
            return F.layer_norm(o + F.gelu(o @ wa) @ wb, norm)

        # H = MAB(I, X): the inducing points read the context
        b = masked_softmax(torch.einsum("rd,...nd->...rn", i_q, k) * beta)   # [..., r, n]
        h = mab_tail(F.layer_norm(i_v + b @ v, norm), w1, w2)                # [..., r, dv]

        # ISAB(X) = MAB(X, H): the queries read the induced rows
        a = masked_softmax(torch.einsum("...md,...rd->...mr", q, h @ p) * beta)
        y = mab_tail(F.layer_norm(q @ p_q + a @ h, norm), w3, w4)

        return AttentionOutput(y=y, weights=None,
                               extras={"a": a, "b": b, "induced": h, "r": self.r,
                                       "note": "a and b are the two attention maps of "
                                               "the ISAB; a @ b is NOT the induced "
                                               "measure -- MAB processes H between them"})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r, hid = self.r, max(1, self.width * dv)
        return {"flops": 2 * n * r * (d + dv) + 2 * m * r * (d + dv)
                         + 4 * (r + m) * dv * hid + 2 * r * dv * d + 2 * m * d * dv,
                "state_bytes": r * dv * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
