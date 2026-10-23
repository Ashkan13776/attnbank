"""Weights of a *gated* linear recurrence, shared by the data-dependent family.

``sec:memory-state``, ``eq:memory-generalrec``, ``tab:memory-recurrences``.  Every
method in this family is

    H_t = Diag(a_t) H_{t-1} + k_t v_t^T,      y_hat_t = H_{p_t}^T q_t,

with ``H in R^{r x d_v}`` and a gate ``a_t in (0, 1]^r`` that is either fixed
(retention) or a function of the input (GLA, HGRN2, RWKV-6, mLSTM).  Unrolling,

    H_s = sum_{j <= s} (prod_{u = j+1}^{s} a_u) k_j v_j^T,

so the state is linear in ``V`` and the operator has exact weights on the original
positions,

    w_{tj} = < q_t , (prod_{u = j+1}^{p_t} a_u) * k_j >,     j <= p_t,   else 0.

``p_t`` is the state a query reads: its own position under causal masking, the last
position otherwise (``readout_positions``), and ``p_t - 1`` for RWKV, whose current
token bypasses the state and is added back with its own bonus weight.

The product is accumulated in log space -- ``prod_{u>j}^{s} a_u = exp(c_s - c_j)`` with
``c = cumsum(log a)`` -- which is what makes the gate usable at long context: ``a_u < 1``
gives ``c_s - c_j <= 0``, so the exponential cannot overflow, while the naive ratio
``A_s / A_j`` underflows and then divides by zero.  This is the same device the chunked
GLA kernel uses inside a chunk; here it is applied over the whole prefix, because a
reference implementation materialises ``pi_hat`` anyway (DESIGN.md Sec. 6).

Nothing here normalises.  These recurrences drop the denominator of
``eq:attention_expectation`` and lose normalization invariance, constant preservation
and convex-hull confinement with it; every method in the family therefore declares
``row_stochastic = False`` and returns the row sums it actually has.
"""
from __future__ import annotations

import torch

from ...components.recurrence import readout_positions

#: query rows per block in :func:`gated_weights`.  The gate tensor is the only
#: ``O(m n r)`` object in the family, so it is built a block of queries at a time.
CHUNK = 64


def _visible(read: torch.Tensor, n: int) -> torch.Tensor:
    """``[m, n]`` boolean: key ``j`` is in the state query ``t`` reads."""
    j = torch.arange(n, device=read.device)
    return j.unsqueeze(0) <= read.unsqueeze(1)


def state_indices(m: int, n: int, causal: bool, device, lag: int = 0) -> torch.Tensor:
    """Index of the state each query reads, ``[m]``, shifted back by ``lag``.

    ``lag = 0`` is the usual convention (``readout_positions``); ``lag = 1`` is RWKV,
    whose readout uses ``H_{t-1}``.  An index of ``-1`` means "the empty state", which
    :func:`gated_weights` renders as a row of zeros rather than a ``nan``.
    """
    return readout_positions(m, n, causal, device) - lag


def gated_weights(q: torch.Tensor, k: torch.Tensor, log_gate: torch.Tensor,
                  read: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """``w_{tj} = <q_t, exp(c_{read_t} - c_j) * k_j>`` for ``j <= read_t``, else 0.

    Parameters
    ----------
    q, k:
        ``[..., m, r]`` and ``[..., n, r]``.
    log_gate:
        ``[..., n, r]``, the per-step log gate ``log a_t``; must be ``<= 0`` for the
        exponential to be bounded, which every gate in the family guarantees.
    read:
        ``[m]`` state indices from :func:`state_indices`.

    Masked entries are set to ``-inf`` *before* the exponential, so a key outside the
    state contributes exactly ``0.0`` and never an overflowed ``exp`` of a positive
    difference.
    """
    n = k.shape[-2]
    c = log_gate.cumsum(-2)                                    # [..., n, r]
    vis = _visible(read, n)                                    # [m, n]
    safe = read.clamp_min(0)
    out = []
    for s in range(0, q.shape[-2], chunk):
        sl = slice(s, s + chunk)
        cr = c.index_select(-2, safe[sl])                      # [..., c, r]
        diff = cr.unsqueeze(-2) - c.unsqueeze(-3)              # [..., c, n, r]
        diff = diff.masked_fill(~vis[sl].unsqueeze(-1), float("-inf"))
        out.append(torch.einsum("...cr,...cnr->...cn",
                                q[..., sl, :], diff.exp() * k.unsqueeze(-3)))
    return torch.cat(out, dim=-2)


def scalar_gate_matrix(log_gate: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
    """``[..., m, n]`` with ``exp(c_{read_t} - c_j)`` for ``j <= read_t``, else 0.

    The cheap path for a gate that is a *scalar per step* (retention, mLSTM) rather than
    a vector per channel: the decay then factors out of the inner product and the
    ``O(m n r)`` tensor of :func:`gated_weights` is never built.
    """
    n = log_gate.shape[-1]
    c = log_gate.cumsum(-1)                                    # [..., n]
    cr = c.index_select(-1, read.clamp_min(0))                 # [..., m]
    diff = cr.unsqueeze(-1) - c.unsqueeze(-2)                  # [..., m, n]
    return diff.masked_fill(~_visible(read, n), float("-inf")).exp()


def normalized(w: torch.Tensor, on: bool) -> torch.Tensor:
    """Optionally put the denominator back, flooring it so a dead row stays finite."""
    if not on:
        return w
    z = w.sum(-1, keepdim=True)
    return w / torch.where(z.abs() < torch.finfo(w.dtype).tiny,
                           torch.full_like(z, torch.finfo(w.dtype).tiny), z)


def recurrent_cost(m: int, n: int, r: int, dv: int, bytes_per_el: int,
                   gate_flops: int = 1) -> dict:
    """The shared cost model: a state of ``r x d_v``, independent of ``n``."""
    return {"flops": 2 * n * r * dv + 2 * m * r * dv + gate_flops * n * r,
            "state_bytes": r * dv * bytes_per_el,
            "bytes_moved": (m * r + n * (r + dv) + m * dv) * bytes_per_el}


def truncate(state_rank, *tensors):
    """Restrict a tuple of ``[..., ., d]`` tensors to their first ``state_rank`` channels.

    How the state size is made configurable across this family: the recurrent state is
    ``r x d_v`` where ``r`` is the key/query dimension, so running with ``state_rank = r``
    *is* running the operator with ``d_k = r``.  ``None`` keeps the full ``d``.  The
    capacity studies sweep this against the number of items to be recalled
    (``prop:memory-rank``).
    """
    if state_rank is None:
        return tensors
    d = tensors[0].shape[-1]
    if not (isinstance(state_rank, int) and 1 <= state_rank <= d):
        raise ValueError(f"state_rank must be an int in [1, {d}] (the query/key "
                         f"dimension), got {state_rank!r}")
    return tuple(t[..., :state_rank] for t in tensors)


__all__ = ["CHUNK", "gated_weights", "normalized", "recurrent_cost",
           "scalar_gate_matrix", "state_indices", "truncate"]
