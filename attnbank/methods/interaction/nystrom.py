"""Nystromformer: landmark Nystrom approximation of the row-stochastic matrix.

``sec:kernel-matrix`` and ``sec:factor-methods``.  With ``r`` landmark queries and keys
obtained by *segment means* of ``Q`` and ``K``,

    P_hat = softmax(beta Q Kt^T) * G^+ * softmax(beta Qt K^T),
    G     = softmax(beta Qt Kt^T),

where ``Qt, Kt`` are the landmarks and ``G^+`` the pseudo-inverse (the original uses an
iterative approximation of it; this reference implementation uses an exact pinv in
float64, so that the only error measured is the method's, not the solver's).

Three caveats the paper makes and this file honours.
(1) ``P`` is neither symmetric nor PSD when ``Q != K``, so Nystrom theory does not
    apply; there is no spectral guarantee here.
(2) The pseudo-inverse makes ``P_hat`` **signed**: ``pi_hat`` is a signed measure of
    unit mass, so the contract items ``nm`` and ``hl`` are lost -- ``row_stochastic`` is
    ``False``, and outputs can leave the convex hull of the values.  (Rows do sum to one
    when ``G`` is invertible, since ``G^{-1} 1 = 1`` for a stochastic ``G``; the library
    does not rely on that.)
(3) Segment-mean landmarks summarize the context, so the method also modifies ``mu``.

Exact limit.  With ``r = None`` the landmarks are the data themselves (``Qt = Q``,
``Kt = K``, which requires ``m = n``), giving ``P_hat = P P^+ P = P`` exactly by the
Moore--Penrose identity.  That is the correctness anchor.

Not causal: the segment means pool over the whole sequence.

Reference: Xiong et al. (2021); Williams and Seeger (2001).
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, AttentionOutput, Category, Objective, masked_softmax
from ...registry import register


def segment_means(x, r):
    """Mean-pool ``x`` [..., t, d] into ``r`` contiguous segments: [..., r, d]."""
    t = x.shape[-2]
    bounds = torch.linspace(0, t, r + 1).round().long().tolist()
    parts = [x[..., a:b, :].mean(dim=-2) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    return torch.stack(parts, dim=-2)


@register
class NystromAttention(AttentionMethod):
    """Landmark Nystrom factorization of the softmax matrix; signed weights."""

    name = "nystrom"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = False          # segment means pool over the whole sequence
    expresses_weights = True
    row_stochastic = False           # pseudo-inverse core => signed measure
    references = ("xiong2021nystromformer", "williams2001using")
    paper_section = "sec:kernel"

    def __init__(self, r: int | None = 16, **kw):
        """``r``: number of landmarks; ``None`` uses the data as landmarks (exact limit)."""
        super().__init__(r=r, **kw)
        self.r = r

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        beta = self._scale(d)
        dtype = q.dtype
        # the factorization is evaluated in float64 so that the measured error is the
        # method's and not the pseudo-inverse solver's
        q64, k64 = q.double(), k.double()
        if self.r is None:
            if m != n:
                raise ValueError("nystrom(r=None) uses the data as landmarks and needs m == n")
            qt, kt = q64, k64
        else:
            r = min(self.r, m, n)
            qt, kt = segment_means(q64, r), segment_means(k64, r)

        f = masked_softmax(torch.einsum("...md,...rd->...mr", q64, kt) * beta)    # [.., m, r]
        g = masked_softmax(torch.einsum("...rd,...sd->...rs", qt, kt) * beta)     # [.., r, r]
        b = masked_softmax(torch.einsum("...rd,...nd->...rn", qt, k64) * beta)    # [.., r, n]

        pi = (f @ torch.linalg.pinv(g) @ b).to(dtype)
        y = pi @ v
        return AttentionOutput(y=y, weights=pi if return_weights else None,
                               extras={"n_landmarks": g.shape[-1],
                                       "row_sum": pi.sum(-1),
                                       "min_weight": pi.amin(-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = self.r or min(m, n)
        return {"flops": 2 * (m + n) * r * d + 2 * r ** 3 + 2 * (m + n) * r * dv,
                "state_bytes": r * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (d + dv)) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"r": None}
