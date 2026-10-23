"""ABC: attention with bounded-memory control -- a *causal* stochastic factorization.

``sec:factor-def``, ``def:factor-stochastic``, ``prop:factor-closure``.  ABC keeps a
memory of ``r`` slots built by a *control* ``phi_x in R_{>=0}^r`` attached to every
context position (Peng et al., 2022):

    K_tilde = sum_x phi_x ⊗ k_x,    V_tilde = sum_x phi_x ⊗ v_x,
    y_hat(q) = V_tilde^T softmax(K_tilde q).

With the normalized control of the paper's ``AbcMLP`` form, ``phi_{z,x} = alpha_{z,x} /
sum_{x'} alpha_{z,x'}``, the slot rows are probability measures, so the memory *is* a
Markov kernel ``b: Z -> X`` and the read is a Markov kernel ``a: Q -> Z``:

    b(z, x) = softmax_x( beta <w_z, k_x> ),     k_bar_z = int k db(z, .),
    a(q, z) = softmax_z( beta <q, k_bar_z> ),   pi_hat_q = a(q, .) b(., .).

This is a **genuine composition of two Markov kernels** -- nothing happens to the
latents between the two hops -- so ``pi_hat_q`` exists on the original ``n`` positions,
is row-stochastic, and inherits ``prop:factor-closure`` whole.  ``weights`` is that
composition and Invariant W holds exactly.

Two things distinguish this file from ``latent_stochastic``, which is the same two-hop
kernel in the non-causal case:

* **It is causal.**  ABC's memory is a running sum, ``K_tilde_{t+1} = K_tilde_t +
  phi_{t+1} ⊗ k_{t+1}``, so the control normalizer at query ``t`` runs over the prefix
  ``x <= t`` only.  The write kernel therefore depends on the query position,
  ``b_t(z, x) = softmax_{x <= t}( beta <w_z, k_x> )``, and ``pi_hat_{q_t} = a_t b_t``
  is supported on the prefix.  It is the only factorization in the bank that decodes.
* **It has an exact limit.**  "If growing-size memory were allowed (n = N), an ABC with
  ``phi_i = e_i`` would fall back to softmax attention" (Peng et al., 2022, Sec. 3).
  ``r = None`` is that unbounded memory -- one slot per position, identity control --
  and reproduces dense softmax exactly, causal or not.  That is the correctness anchor,
  and it is why ABC is an **approximation** while ``latent_stochastic`` and ``latte``
  are operator modifications: bounded memory is a resource parameter with a limit.

The controls ``w_z`` are the learned parameters.  Untrained they are a seeded Gaussian
in ``R^{r x d}`` (the write kernel is then a random soft partition of the context), so
``trainable = True`` records that the *quality* of the factorization, not its
structure, is what training buys.  Nothing in the declared contract depends on them.

Reference implementation, clarity over speed (DESIGN.md Sec. 6): the causal path
materializes the prefix-normalized kernel ``[m, r, n]`` and softmaxes each prefix
independently, which is stable at any logit scale.  The deployed form is the ``O(r d)``
running memory above.

Reference: Peng et al. (2022) ABC.
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
    scores,
)
from ...registry import register


@register
class ABCAttention(AttentionMethod):
    """Bounded-memory control: pi_hat = a b with b prefix-normalized, so it decodes."""

    name = "abc"
    category = Category.FACTORIZE
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    trainable = True
    references = ("peng2022abc",)
    paper_section = "sec:factor"

    def __init__(self, r: int | None = 8, seed: int = 0, **kw):
        """``r``: memory slots. ``None`` (or ``r >= n``) is unbounded memory: exact."""
        super().__init__(r=r, seed=seed, **kw)
        self.r, self.seed = r, seed

    # -- the unbounded-memory control phi_i = e_i, which is dense softmax -----
    def _unbounded(self, q, k, v, causal, return_weights, beta):
        n = k.shape[-2]
        pi = masked_softmax(scores(q, k, beta, causal))
        eye = torch.eye(n, dtype=q.dtype, device=q.device).expand(*k.shape[:-2], n, n)
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras={"a": pi, "b": eye, "memory_keys": k,
                                       "memory_values": v, "r": n, "unbounded": True})

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], k.shape[-1]
        beta = self._scale(d)
        if self.r is None or self.r >= n:
            return self._unbounded(q, k, v, causal, return_weights, beta)

        g = make_generator(self.seed)
        w = torch.randn(self.r, d, generator=g, dtype=torch.float64).to(k.device, k.dtype)
        logits = torch.einsum("rd,...nd->...rn", w, k) * beta        # control, [..., r, n]

        if causal:
            vis = causal_mask(m, n, device=k.device).unsqueeze(-2)   # [m, 1, n]
            b = masked_softmax(logits.unsqueeze(-3), vis)            # [..., m, r, n]
            k_bar = torch.einsum("...mrn,...nd->...mrd", b, k)       # memory keys
            u = torch.einsum("...mrn,...ne->...mre", b, v)           # memory values
            a = masked_softmax(torch.einsum("...md,...mrd->...mr", q, k_bar) * beta)
            y = torch.einsum("...mr,...mre->...me", a, u)
            pi = torch.einsum("...mr,...mrn->...mn", a, b) if return_weights else None
        else:
            b = masked_softmax(logits)                               # [..., r, n]
            k_bar, u = b @ k, b @ v
            a = masked_softmax(torch.einsum("...md,...rd->...mr", q, k_bar) * beta)
            y = a @ u
            pi = a @ b if return_weights else None

        return AttentionOutput(y=y, weights=pi,
                               extras={"a": a, "b": b, "memory_keys": k_bar,
                                       "memory_values": u, "control": w, "r": self.r,
                                       "unbounded": False})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = min(self.r or n, n)
        return {"flops": 2 * n * r * (d + dv) + 2 * m * r * (d + dv),
                "state_bytes": r * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"r": None}
