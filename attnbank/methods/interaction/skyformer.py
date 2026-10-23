"""Skyformer: kernelize first, then Nystrom on a matrix that is actually PSD.

``sec:kernel-matrix``.  Nystromformer applies the Nystrom method to ``P``, which is
neither symmetric nor PSD, so none of the Nystrom theory applies to it.  Skyformer
fixes the premise rather than the conclusion, in two steps.

*Kernelize.*  With ``beta = 1/sqrt(d)`` the softmax kernel factors through a Gaussian
one,

    SM(q, x)   = exp(beta <q, k_x>)
               = exp(beta ||q||^2 / 2) * C(q, x) * exp(beta ||k_x||^2 / 2),
    C(q, x)    = exp(-beta ||q - k_x||^2 / 2),

i.e. ``A = D_Q^{1/2} C D_K^{1/2}`` with ``(D_Q)_ii = exp(beta ||q_i||^2)`` (Eq. 1 of
the paper).  ``C`` is the *kernelized attention score matrix*: the paper's operator is
``C V``, which is already "automatically normalized" and is therefore a **different**
operator -- the key factor ``D_K^{1/2}`` does not cancel.  The bank targets softmax, so
this file keeps ``C`` as the object that is approximated and then undoes the diagonal:
the query factor cancels under row normalization and the key factor is multiplied back
in.  That is the use the paper states for the same machinery in Sec. 4.2 -- "a low-rank
approximation to A also implies an approximation to D".

*Lift.*  ``C`` is a rectangular block, but it is a block of the Gram matrix of the
*union* of the two point sets, and that matrix is PSD:

    Cbar = kappa([Q; K] / d^{1/4}, [Q; K] / d^{1/4}) in R^{(m+n) x (m+n)},
    Cbar_tilde = Cbar S (S^T Cbar S)^+ S^T Cbar,
    C_tilde    = (I, 0) Cbar_tilde (0, I)^T,

with ``S`` a uniform sub-sampling matrix choosing ``r`` of the ``m + n`` positions.
Since ``||C - C_tilde|| <= ||Cbar - Cbar_tilde||``, approximating the block reduces to
approximating a PSD matrix, which is what Nystrom has guarantees for (Theorem 2 of the
paper, after Musco and Musco).  This is the whole difference from ``nystrom``, and it
is a difference in the premise, not in the formula.

What survives and what does not.  The core pseudo-inverse makes ``C_tilde`` **signed**,
so ``pi_hat`` is a signed measure: ``row_stochastic = False`` and the output can leave
the convex hull of the values.  Rows do sum to one here because the library normalizes
explicitly, which the paper's own operator does not.

Exact limit.  ``r = None`` samples every position, ``S = I``, and
``Cbar Cbar^+ Cbar = Cbar``, so ``C_tilde = C`` and ``pi_hat`` is softmax -- for any
``m`` and ``n``, since the lift is square by construction.  That is the anchor.

Numerics.  The whole factorization is float64, and the Gram matrix is scaled by a
single positive constant chosen to centre its log-range (Nystrom is homogeneous of
degree one, and the scalar cancels in the row normalization).  Without it the
off-diagonal block underflows to zero in the large-logit regime while the diagonal
stays at one -- the Gaussian Gram matrix tends to the identity when the points are far
apart, which is the regime where this kernelization has nothing left to approximate.

Not causal: the sub-sampling draws landmarks from the whole union, future included.

Reference: Chen et al. (2021); Williams and Seeger (2001); Drineas and Mahoney (2005).
"""
from __future__ import annotations

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    make_generator,
    weights_from_log_magnitude,
)
from ...registry import register


def uniform_landmarks(t: int, r: int, seed: int) -> torch.Tensor:
    """``r`` distinct indices drawn uniformly from ``range(t)``, in increasing order.

    Definition 1 of the paper draws the columns of ``S`` i.i.d., i.e. with replacement;
    a repeated column adds nothing and makes ``S^T Cbar S`` singular, so the draw here
    is without replacement -- the standard practical reading of uniform sub-sampling.
    """
    g = make_generator(seed)
    return torch.randperm(t, generator=g)[:r].sort().values


@register
class SkyformerAttention(AttentionMethod):
    """Gaussian-kernel attention, lifted to a PSD Gram matrix, then Nystrom."""

    name = "skyformer"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = False          # landmarks are sampled from the whole union
    expresses_weights = True
    row_stochastic = False           # pseudo-inverse core => signed measure
    references = ("chen2021skyformer", "williams2001using", "drineas2005nystrom")
    paper_section = "sec:kernel"

    def __init__(self, r: int | None = 16, seed: int = 0, **kw):
        """``r``: sub-sample size out of ``m + n``; ``None`` keeps all of them (the limit)."""
        super().__init__(r=r, seed=seed, **kw)
        self.r, self.seed = r, seed

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        beta = self._scale(d)
        t = m + n
        # ---- the lifted Gaussian Gram matrix of the union [Q; K], in float64 -------
        z = torch.cat([q.double(), k.double()], dim=-2)                    # [..., t, d]
        sq = (z ** 2).sum(-1)                                              # [..., t]
        d2 = (sq.unsqueeze(-1) + sq.unsqueeze(-2)
              - 2 * torch.einsum("...ad,...bd->...ab", z, z)).clamp_min(0)
        log_c = -0.5 * beta * d2
        shift = 0.5 * (log_c.amax((-2, -1), keepdim=True)
                       + log_c.amin((-2, -1), keepdim=True))
        gram = torch.exp(log_c - shift)        # exp(-shift) * Cbar; PSD, and in range

        # ---- Nystrom on the PSD lift, then read off the (Q, K) block ---------------
        r = t if self.r is None else min(self.r, t)
        idx = (torch.arange(t) if r == t else uniform_landmarks(t, r, self.seed))
        idx = idx.to(gram.device)
        cs = gram.index_select(-1, idx)                                    # [..., t, r]
        css = cs.index_select(-2, idx)                                     # [..., r, r]
        c_tilde = (cs[..., :m, :] @ torch.linalg.pinv(css)
                   @ cs[..., m:, :].transpose(-1, -2))                     # [..., m, n]

        # ---- back to the softmax kernel: A = D_Q^{1/2} C D_K^{1/2} -----------------
        # the query factor is constant along a row and cancels in the normalization;
        # the key factor does not, and is what makes kernelized attention a different
        # operator from softmax attention.
        col = 0.5 * beta * (sq[..., m:] - sq[..., m:].amax(-1, keepdim=True))
        logmag = torch.log(c_tilde.abs()) + col.unsqueeze(-2)
        pi, z_rel = weights_from_log_magnitude(logmag, sign=torch.sign(c_tilde))
        pi = pi.to(q.dtype)
        y = pi @ v
        return AttentionOutput(y=y, weights=pi if return_weights else None,
                               extras={"n_landmarks": r, "Z_rel": z_rel.to(q.dtype),
                                       "min_weight": pi.amin(-1),
                                       "neg_mass": pi.clamp(max=0).sum(-1).abs()})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = min(self.r or (m + n), m + n)
        return {"flops": 2 * (m + n) * r * d + 2 * r ** 3 + 2 * (m + n) * r * dv,
                "state_bytes": r * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (d + dv)) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"r": None}
