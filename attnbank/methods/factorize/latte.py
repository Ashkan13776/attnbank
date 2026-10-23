"""Latte: latent attention as a probabilistic model -- the *free* latent space.

``sec:factor-def``, ``def:factor-stochastic``.  Latte (Dolga et al., 2024) introduces a
latent variable ``l`` with ``r`` states that renders query and context conditionally
independent, ``x ⊥ q | l``, and reads the attention measure straight off the
factorization

    pi_hat_q(x) = sum_l p(x | l) p(l | q),      Latte(Q, K, V) = softmax_r(Q) softmax_n(K)^T V,

with the two conditionals parameterized by one latent array each:

    p(l | q) = softmax_l( beta <q, w^q_l> ),    normalized over the r LATENTS,
    p(x | l) = softmax_x( beta <k_x, w^k_l> ),  normalized over the n POSITIONS.

Both factors are Markov kernels, so this is a stochastic factorization in the sense of
``def:factor-stochastic``, ``weights`` is their product and ``prop:factor-closure``
applies unchanged.

What separates it from ``latent_stochastic`` / ``abc`` is *where the latent keys come
from*.  There the read kernel is ``softmax_z(beta <q, k_bar_z>)`` with ``k_bar_z =
int k db(z, .)`` **summarizing the context**; here ``w^q_l`` is free, so

    the read kernel a(q, .) does not look at the context at all.

The query's only interaction with the keys is through which of ``r`` latents it
selects: ``pi_hat`` has rank at most ``r``, and perturbing every key and value leaves
``a`` bit-for-bit unchanged.  That is the cleanest instance in the bank of a
factorization whose two hops are genuinely separate -- and the reason it is an
**operator modification**: no setting of ``r`` recovers softmax, because the free
latents never see the keys.  Both anchors are ``None`` by design (``tab:comparative``,
"mod.").

Causal (Latte's autoregressive form): the write kernel is normalized over the prefix,
``p(x | l, t) = softmax_{x <= t}(beta <k_x, w^k_l>)``, which the paper computes with
cumulative sums; the read kernel is untouched, since it never depended on the context.
This reference implementation materializes the prefix-normalized kernel ``[m, r, n]``
and softmaxes each prefix independently -- clarity over speed, and stable at any logit
scale.

The latent arrays ``w^q, w^k`` are the learned parameters; untrained they are seeded
Gaussians, so ``trainable = True``.  Scaling: the paper writes the logits unscaled;
here they carry the bank's ``beta = 1/sqrt(d)`` so that the two kernels are on the same
footing as every other method's.

Reference: Dolga et al. (2024) Latte.
"""
from __future__ import annotations

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    causal_mask,
    make_generator,
    masked_softmax,
)
from ...registry import register


@register
class LatteAttention(AttentionMethod):
    """pi_hat = A B with A = softmax over latents, B = softmax over positions."""

    name = "latte"
    category = Category.FACTORIZE
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    trainable = True
    references = ("dolga2024latte",)
    paper_section = "sec:factor"

    def __init__(self, r: int = 8, seed: int = 0, **kw):
        """``r``: number of latent states. There is no ``r`` at which this is softmax."""
        super().__init__(r=r, seed=seed, **kw)
        self.r, self.seed = r, seed

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], k.shape[-1]
        beta = self._scale(d)
        g = make_generator(self.seed)
        latents = torch.randn(2, self.r, d, generator=g, dtype=torch.float64)
        w_q, w_k = (t.to(k.device, k.dtype) for t in latents)

        a = masked_softmax(torch.einsum("...md,rd->...mr", q, w_q) * beta)   # p(l | q)
        logits = torch.einsum("rd,...nd->...rn", w_k, k) * beta              # p(x | l)

        if causal:
            vis = causal_mask(m, n, device=k.device).unsqueeze(-2)           # [m, 1, n]
            b = masked_softmax(logits.unsqueeze(-3), vis)                    # [..., m, r, n]
            u = torch.einsum("...mrn,...ne->...mre", b, v)
            y = torch.einsum("...mr,...mre->...me", a, u)
            pi = torch.einsum("...mr,...mrn->...mn", a, b) if return_weights else None
        else:
            b = masked_softmax(logits)                                       # [..., r, n]
            u = b @ v
            y = a @ u
            pi = a @ b if return_weights else None

        return AttentionOutput(y=y, weights=pi,
                               extras={"a": a, "b": b, "latent_values": u,
                                       "latent_queries": w_q, "latent_keys": w_k,
                                       "r": self.r})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.r
        return {"flops": 2 * m * r * d + 2 * n * r * (d + dv) + 2 * m * r * dv,
                "state_bytes": r * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
