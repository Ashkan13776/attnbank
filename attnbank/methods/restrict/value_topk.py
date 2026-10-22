"""ORACLE: output-aware greedy selection of the support.

``sec:sparse-operator`` ("Score-optimal is not output-optimal"), ``prop:error-topk``,
and the open problem of ``sec:open-retrieval``.  Top-k minimizes the omitted mass
``eps_q`` and therefore the *bound* ``eps_q osc(v)``; it does not minimize the error.
The right target, named in the text, is

    min_{|S| <= k}  eps_S ||m_{S^c} - m_S||.

That objective simplifies.  For a renormalized restriction ``y_hat = m_S``, so with
``W_S = pi(S)`` and ``U_S = sum_{j in S} pi_j v_j``,

    eps_S ||m_{S^c} - m_S|| = ||y - m_S|| = ||y - U_S / W_S||,

i.e. *the objective is just the realised output error*, and no conditional mean of the
complement needs to be formed.

Selection.  Forward greedy alone is badly myopic here: its first pick minimizes
``||y - v_j||`` and so can seed the support with a light outlier that later additions
cannot undo.  The selector therefore evaluates four cheap candidate supports and keeps,
*per query*, whichever realises the smallest error:

  1. score top-k (so the result is **never worse than** ``topk``, by construction);
  2. forward greedy from the empty set;
  3. forward greedy seeded with the single heaviest position;
  4. backward greedy from the full set, deleting the position whose removal costs least
     -- the elementary step of every eviction policy, and exactly ``prop:error-single``.

Complexity.  Each greedy step scores all ``n`` candidates, and a candidate is scored in
``O(d_v)`` from the running ``(U_S, W_S)`` rather than by recomputing the conditional
mean, so a forward pass is ``O(n k d_v)`` per query and the backward pass
``O(n (n - k) d_v)`` -- the naive form of either is ``O(n^2 k)``.  All queries are
processed in parallel, at a peak memory of ``B H m n d_v``; pass ``chunk`` to process
the query axis in slices if that is too large.

None of this is optimal: ``exhaustive=True`` enumerates every support of size ``k``
(guarded by ``max_subsets``) and gives the true minimum, which is what
``tests/test_restrict.py`` uses to bound the greedy gap.

This is an **oracle**: it reads ``v`` and the exact ``pi_q``, which no deployed
selector can.  Its purpose is to upper-bound what output-aware selection could buy, so
correctness matters and speed does not.

Reference: Ribar et al. (2024) SparQ; Feng et al. (2025) CriticalKV.
"""
from __future__ import annotations

import itertools
import math

import torch

from ...core import AttentionMethod, Category, Objective
from ...registry import register
from ._restriction import reference_pi, restrict, topk_support, visible


def _objective(pi, support, v, y):
    """``||y - U_S / W_S||`` per query: the realised restriction error of prop:sparse-identity."""
    tiny = torch.finfo(pi.dtype).tiny
    w = (pi * support).sum(-1, keepdim=True)
    u = (pi * support) @ v
    err = (y - u / w.clamp_min(tiny)).norm(dim=-1)
    return torch.where(w.squeeze(-1) > 0, err, torch.full_like(err, torch.finfo(pi.dtype).max))


def _step_costs(pi, v, y, w, u, sign):
    """Cost of adding (``sign = +1``) or removing (``sign = -1``) each position."""
    tiny = torch.finfo(pi.dtype).tiny
    cand_w = w + sign * pi                                          # [..., m, n]
    cand_u = u.unsqueeze(-2) + sign * pi.unsqueeze(-1) * v.unsqueeze(-3)
    resid = y.unsqueeze(-2) * cand_w.unsqueeze(-1) - cand_u
    return resid.norm(dim=-1) / cand_w.clamp_min(tiny), cand_w


def _forward_greedy(pi, vis, v, y, budget, seed_heaviest):
    """Add the position that most reduces ``||y - U_S/W_S||``, ``budget`` times."""
    big = torch.finfo(pi.dtype).max
    chosen = torch.zeros_like(vis)
    if seed_heaviest:
        top = pi.masked_fill(~vis, -1.0).argmax(-1, keepdim=True)
        chosen = torch.zeros_like(vis).scatter_(-1, top, True) & vis
    u = (pi * chosen) @ v
    w = (pi * chosen).sum(-1, keepdim=True)
    for _ in range(budget - int(seed_heaviest)):
        cost, cand_w = _step_costs(pi, v, y, w, u, +1)
        cost = torch.where(vis & ~chosen & (cand_w > 0), cost, torch.full_like(cost, big))
        j = cost.argmin(dim=-1, keepdim=True)
        fresh = torch.zeros_like(chosen).scatter_(-1, j, True) & vis & ~chosen
        if not bool(fresh.any()):
            break
        pick = pi * fresh
        w = w + pick.sum(-1, keepdim=True)
        u = u + pick @ v
        chosen = chosen | fresh
    return chosen


def _backward_greedy(pi, vis, v, y, budget):
    """Delete the position whose removal costs least, down to ``budget`` survivors."""
    big = torch.finfo(pi.dtype).max
    chosen = vis.clone()
    u = (pi * chosen) @ v
    w = (pi * chosen).sum(-1, keepdim=True)
    n_vis = int(vis.sum(-1).amax())
    for _ in range(max(0, n_vis - budget)):
        cost, cand_w = _step_costs(pi, v, y, w, u, -1)
        keep_needed = chosen.sum(-1, keepdim=True) <= budget
        cost = torch.where(chosen & (cand_w > 0) & ~keep_needed, cost,
                           torch.full_like(cost, big))
        j = cost.argmin(dim=-1, keepdim=True)
        gone = torch.zeros_like(chosen).scatter_(-1, j, True) & chosen & ~keep_needed
        if not bool(gone.any()):
            break
        drop = pi * gone
        w = w - drop.sum(-1, keepdim=True)
        u = u - drop @ v
        chosen = chosen & ~gone
    return chosen


def _greedy_support(pi, vis, v, budget):
    """Best of four candidate supports, chosen per query by the realised error."""
    y = pi @ v
    candidates = [
        topk_support(pi, vis, budget),                            # never worse than top-k
        _forward_greedy(pi, vis, v, y, budget, seed_heaviest=False),
        _forward_greedy(pi, vis, v, y, budget, seed_heaviest=True),
        _backward_greedy(pi, vis, v, y, budget),
    ]
    best = candidates[0]
    best_err = _objective(pi, best, v, y)
    for cand in candidates[1:]:
        err = _objective(pi, cand, v, y)
        take = (err < best_err).unsqueeze(-1)
        best = torch.where(take, cand, best)
        best_err = torch.minimum(err, best_err)
    return best


def _exhaustive_support(pi, vis, v, budget, max_subsets):
    """True minimizer over all supports of size ``budget`` (tiny ``n`` only)."""
    n, dv = pi.shape[-1], v.shape[-1]
    if math.comb(n, budget) > max_subsets:
        raise ValueError(f"exhaustive selection needs C({n},{budget}) <= {max_subsets} subsets")
    y = pi @ v
    flat_pi = pi.reshape(-1, n)
    flat_vis = vis.reshape(-1, n)
    v_rows = v.unsqueeze(-3).expand(*pi.shape[:-1], n, dv).reshape(-1, n, dv)
    y_rows = y.reshape(-1, dv)
    best = torch.full((flat_pi.shape[0],), float("inf"), dtype=pi.dtype, device=pi.device)
    out = torch.zeros_like(flat_vis)
    for comb in itertools.combinations(range(n), budget):
        sel = torch.zeros(n, dtype=torch.bool, device=pi.device)
        sel[list(comb)] = True
        ok = (flat_vis | ~sel).all(-1)                          # subset must be visible
        w = (flat_pi * sel).sum(-1)
        u = torch.einsum("bn,bnd->bd", flat_pi * sel, v_rows)
        cost = torch.where(w > 0, (y_rows - u / w.clamp_min(1e-30).unsqueeze(-1)).norm(dim=-1),
                           torch.full_like(w, float("inf")))
        cost = torch.where(ok, cost, torch.full_like(cost, float("inf")))
        better = cost < best
        best = torch.where(better, cost, best)
        out[better] = sel
    return out.reshape(vis.shape)


@register
class ValueTopKAttention(AttentionMethod):
    """Greedy (or exhaustive) output-aware support selection. Oracle."""

    name = "value_topk"
    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    references = ("ribar2024sparq", "feng2025criticalkv", "gupta2021memory")
    paper_section = "sec:sparse"

    def __init__(self, k: int | None = 8, exhaustive: bool = False,
                 max_subsets: int = 200_000, chunk: int | None = None, **kw):
        super().__init__(k=k, exhaustive=exhaustive, max_subsets=max_subsets, chunk=chunk, **kw)
        self.k, self.exhaustive, self.max_subsets, self.chunk = k, exhaustive, max_subsets, chunk

    def _forward(self, q, k, v, causal, return_weights):
        pi, s = reference_pi(self, q, k, causal)
        vis = visible(q, k, causal)
        n = pi.shape[-1]
        if self.k is None or self.k >= n:
            return restrict(pi, vis, v, return_weights, {"objective": "exact limit"})
        if self.exhaustive:
            sup = _exhaustive_support(pi, vis, v, self.k, self.max_subsets)
        elif self.chunk:
            parts = [_greedy_support(pi[..., i:i + self.chunk, :], vis[..., i:i + self.chunk, :],
                                     v, self.k)
                     for i in range(0, pi.shape[-2], self.chunk)]
            sup = torch.cat(parts, dim=-2)
        else:
            sup = _greedy_support(pi, vis, v, self.k)
        return restrict(pi, sup, v, return_weights,
                        {"selector": "exhaustive" if self.exhaustive else "greedy"})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        kk = self.k or n
        return {"flops": 2 * m * n * d + 2 * m * n * kk * dv + 2 * m * kk * dv,
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"k": None}
