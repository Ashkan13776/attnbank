"""Hidden-dimension low rank: truncated SVD of the key and value channel matrices.

``sec:compress-lowrank``.  With ``Pi_r`` an orthogonal projector of rank ``r`` on the
channel space,

    k_hat_j = Pi_r^K k_j,   v_hat_j = Pi_r^V v_j,

where ``Pi_r^K`` and ``Pi_r^V`` are the leading right singular subspaces of ``K`` and
``V``.  The two errors are structurally different, which is the point of the file:

* Keys.  ``delta_l_j = -beta <(I - Pi_r) q, (I - Pi_r) k_j>``, so a subspace chosen from
  the keys alone fails exactly when the queries load on low-variance key directions --
  no amount of key energy captured says anything until the query is known.
* Values.  Since ``pi_hat`` is unaffected by a value projection, the value error is
  ``sum_j pi_j (I - Pi_r) v_j = (I - Pi_r) y(q)``: the projection residual of the
  **output**, not of the approximate output and not of the raw value matrix.  So the
  optimal value subspace is the leading eigenspace of the attention-weighted second
  moment of the *outputs*.

Not causal: the SVD pools over all tokens.  The factorization is computed in float64
so that the exact limit (``rank_k = rank_v = None``) is exact to machine precision.

Reference: Xu et al. (2025) ThinK; Chang et al. (2025) Palu; Wang et al. (2025) SVD-LLM.
"""
from __future__ import annotations

import torch

from ...core import (AttentionMethod, AttentionOutput, Category, Objective,
                     masked_softmax, scores)
from ...registry import register


def project_rank(x, r):
    """Project the rows of ``x`` [..., t, c] onto the top-``r`` right singular subspace."""
    if r is None or r >= x.shape[-1]:
        return x, None
    _, _, vh = torch.linalg.svd(x.double(), full_matrices=False)
    basis = vh[..., :r, :]                                    # [..., r, c]
    proj = basis.transpose(-1, -2) @ basis                    # [..., c, c]
    return (x.double() @ proj).to(x.dtype), basis.to(x.dtype)


@register
class LowRankKVAttention(AttentionMethod):
    """Truncated SVD of the key and/or value channel matrices."""

    name = "lowrank_kv"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False          # the SVD pools over the whole sequence
    expresses_weights = True
    row_stochastic = True
    references = ("xu2025think", "chang2025palu", "wang2025svdllm")
    paper_section = "sec:compress"

    def __init__(self, rank_k: int | None = 4, rank_v: int | None = 4, **kw):
        super().__init__(rank_k=rank_k, rank_v=rank_v, **kw)
        self.rank_k, self.rank_v = rank_k, rank_v

    def _forward(self, q, k, v, causal, return_weights):
        beta = self._scale(q.shape[-1])
        k_hat, basis_k = project_rank(k, self.rank_k)
        v_hat, basis_v = project_rank(v, self.rank_v)
        s, s_hat = scores(q, k, beta, causal), scores(q, k_hat, beta, causal)
        pi = masked_softmax(s_hat)
        y = pi @ v_hat
        extras = {"s": s, "s_hat": s_hat, "basis_k": basis_k, "basis_v": basis_v,
                  "q_residual": (q - (q @ (basis_k.transpose(-1, -2) @ basis_k))).norm(dim=-1)
                  if basis_k is not None else torch.zeros(q.shape[:-1], device=q.device)}
        return AttentionOutput(y=y, weights=pi if return_weights else None, v_hat=v_hat,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        rk, rv = min(self.rank_k or d, d), min(self.rank_v or dv, dv)
        state = n * (rk + rv) * bytes_per_el
        return {"flops": 2 * m * n * (d + dv) + 2 * n * (d * rk + dv * rv),
                "state_bytes": state,
                "bytes_moved": m * (d + dv) * bytes_per_el + state}

    @classmethod
    def limit_config(cls):
        return {"rank_k": None, "rank_v": None}
