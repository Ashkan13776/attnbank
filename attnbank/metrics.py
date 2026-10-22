"""Error metrics matching Section 4 of the paper (``sec:output_error``).

All functions take reference pi [.., m, n], approximate pi_hat [.., m, n],
values v [.., n, dv] and (optionally) v_hat [.., n, dv], and reduce over the
last axes per query, returning tensors of shape [.., m].

Naming: a function called ``*_bound`` returns an upper bound on a quantity and is
tested in ``tests/test_metrics.py`` to actually bound it over randomized inputs.
Everything else is an identity or a definition.
"""
from __future__ import annotations

import torch
from torch import Tensor

TINY = 1e-30


# ---------------------------------------------------------------------------
# Output error and its exact decomposition
# ---------------------------------------------------------------------------

def output_error(y: Tensor, y_hat: Tensor, relative: bool = False) -> Tensor:
    e = (y - y_hat).norm(dim=-1)
    return e / y.norm(dim=-1).clamp_min(1e-12) if relative else e


def error_decomposition(pi: Tensor, pi_hat: Tensor, v: Tensor, v_hat: Tensor | None = None) -> dict:
    """y - y_hat = (pi - pi_hat) v  +  pi_hat (v - v_hat)   (eq:error_decomposition)."""
    v_hat = v if v_hat is None else v_hat
    dist = (pi - pi_hat) @ v
    val = pi_hat @ (v - v_hat)
    return {"distributional": dist.norm(dim=-1), "value": val.norm(dim=-1),
            "total": (dist + val).norm(dim=-1)}


def symmetric_decomposition(pi: Tensor, pi_hat: Tensor, v: Tensor,
                            v_hat: Tensor | None = None) -> dict:
    """y - y_hat = (pi - pi_hat) v_hat + pi (v - v_hat)   (eq:error-decomp-sym).

    The distributional error is measured through the *compressed* values and the
    value error is weighted by the *reference* distribution.
    """
    v_hat = v if v_hat is None else v_hat
    dist = (pi - pi_hat) @ v_hat
    val = pi @ (v - v_hat)
    return {"distributional": dist.norm(dim=-1), "value": val.norm(dim=-1),
            "total": (dist + val).norm(dim=-1)}


# ---------------------------------------------------------------------------
# Divergences
# ---------------------------------------------------------------------------

def total_variation(pi: Tensor, pi_hat: Tensor) -> Tensor:
    """TV(pi, pi_hat) = 1/2 ||pi - pi_hat||_1."""
    return 0.5 * (pi - pi_hat).abs().sum(-1)


def kl(pi: Tensor, pi_hat: Tensor) -> Tensor:
    """KL(pi || pi_hat) = sum_j pi_j log(pi_j / pi_hat_j).

    Returns ``+inf`` for rows where ``pi_hat`` drops support of ``pi``
    (``pi_j > 0`` and ``pi_hat_j == 0``), which is the mathematically correct
    value: this is how ``KL(pi || pi_hat) = inf`` for a renormalized restriction
    (``prop:sparse-identity``) is reproduced numerically rather than clamped away.
    Terms with ``pi_j == 0`` contribute ``0`` by the usual convention.
    """
    drops = (pi > 0) & (pi_hat <= 0)
    safe_pi = pi.clamp_min(TINY)
    safe_hat = pi_hat.clamp_min(TINY)
    r = pi * (torch.log(safe_pi) - torch.log(safe_hat))
    r = torch.where(pi > 0, r, torch.zeros_like(r))
    out = r.sum(-1)
    return torch.where(drops.any(-1), torch.full_like(out, float("inf")), out)


def chi2(pi_hat: Tensor, pi: Tensor) -> Tensor:
    """chi^2(pi_hat || pi) = sum_j (pi_hat_j - pi_j)^2 / pi_j.

    Returns ``+inf`` where ``pi_hat`` is not absolutely continuous with respect to
    ``pi`` (``pi_j == 0 < pi_hat_j``), matching the definition in Sec. 4, which
    declares chi^2 only for ``pi_hat << pi``.
    """
    bad = (pi <= 0) & (pi_hat > 0)
    num = (pi_hat - pi) ** 2
    term = torch.where(pi > 0, num / pi.clamp_min(TINY), torch.zeros_like(num))
    out = term.sum(-1)
    return torch.where(bad.any(-1), torch.full_like(out, float("inf")), out)


def spearman(x: Tensor, y: Tensor, dim: int = -1) -> Tensor:
    """Spearman rank correlation between ``x`` and ``y`` along ``dim``.

    Ties receive their average rank, so the coefficient equals Pearson's on the
    ranks in all cases.  Used by study S1 to compare matrix norms with output error.
    """
    xr, yr = _average_ranks(x, dim), _average_ranks(y, dim)
    return _pearson(xr, yr, dim)


def _average_ranks(x: Tensor, dim: int) -> Tensor:
    """Ranks along the last axis after ``movedim``, ties averaged within each run."""
    x = x.movedim(dim, -1).double()
    n = x.shape[-1]
    order = x.argsort(dim=-1)
    srt = x.gather(-1, order)
    pos = torch.arange(n, dtype=x.dtype, device=x.device).expand_as(x).contiguous()
    # group id of a sorted entry = number of strict increases to its left
    inc = torch.zeros_like(srt)
    inc[..., 1:] = (srt[..., 1:] > srt[..., :-1]).to(srt.dtype)
    gid = inc.cumsum(-1).long()
    counts = torch.zeros_like(srt).scatter_add_(-1, gid, torch.ones_like(srt))
    sums = torch.zeros_like(srt).scatter_add_(-1, gid, pos)
    mean_rank = (sums / counts.clamp_min(1.0)).gather(-1, gid)
    avg = torch.empty_like(x)
    avg.scatter_(-1, order, mean_rank)
    return avg.movedim(-1, dim)


def _pearson(x: Tensor, y: Tensor, dim: int) -> Tensor:
    x = x.double() - x.double().mean(dim, keepdim=True)
    y = y.double() - y.double().mean(dim, keepdim=True)
    num = (x * y).sum(dim)
    den = x.norm(dim=dim) * y.norm(dim=dim)
    return num / den.clamp_min(1e-300)


# ---------------------------------------------------------------------------
# Value geometry
# ---------------------------------------------------------------------------

def value_oscillation(v: Tensor, support: Tensor | None = None) -> Tensor:
    """osc(v) = diam of {v_j} in l2 (optionally over a boolean support [.., m, n]).

    Shape [..] without a support and [.., m] with one.
    """
    if support is None:
        return torch.cdist(v, v).amax(dim=(-1, -2))
    d = torch.cdist(v, v)  # [.., n, n]
    msk = support[..., :, None] & support[..., None, :]  # [.., m, n, n]
    return (d.unsqueeze(-3) * msk).amax(dim=(-1, -2))


def value_spread(pi: Tensor, v: Tensor) -> Tensor:
    """sigma_q = E_{pi}||v - y||, the attention-weighted spread of cor:kernel-rel-add."""
    y = pi @ v
    dev = (v.unsqueeze(-3) - y.unsqueeze(-2)).norm(dim=-1)  # [.., m, n]
    return (pi * dev).sum(-1)


def value_covariance_op(pi: Tensor, v: Tensor) -> Tensor:
    """||Cov_pi(v)||_op^{1/2} per query, the scale in prop:error-chi2."""
    y = pi @ v                                   # [.., m, dv]
    c = v.unsqueeze(-3) - y.unsqueeze(-2)        # [.., m, n, dv]
    cov = torch.einsum("...mn,...mna,...mnb->...mab", pi, c, c)
    return torch.linalg.matrix_norm(cov, ord=2).clamp_min(0).sqrt()


def n_eff(pi: Tensor) -> Tensor:
    """n_eff(pi) = (sum_j pi_j^2)^{-1}, the effective sample size of prop:error-stochastic."""
    return 1.0 / (pi ** 2).sum(-1).clamp_min(TINY)


def effective_support(pi: Tensor) -> Tensor:
    """exp(entropy) of pi_q: effective number of attended tokens (perplexity form)."""
    h = -(pi * torch.log(pi.clamp_min(TINY))).sum(-1)
    return torch.exp(h)


# ---------------------------------------------------------------------------
# Bounds (each one is tested to bound what it claims to bound)
# ---------------------------------------------------------------------------

def tv_osc_bound(pi: Tensor, pi_hat: Tensor, v: Tensor) -> Tensor:
    """prop:error-tv: ||(pi - pi_hat) v|| <= TV(pi, pi_hat) * osc(v)."""
    return total_variation(pi, pi_hat) * value_oscillation(v).unsqueeze(-1)


def chi2_variance_factor(pi: Tensor, pi_hat: Tensor, v: Tensor) -> dict:
    """prop:error-chi2: ||(pi - pi_hat) v|| <= sqrt(chi2(pi_hat||pi)) ||Cov_pi(v)||_op^{1/2}.

    Returns the two factors separately (``chi2`` and ``sd``) and their product
    (``bound``), because Sec. 4.2 reads the bound as "displaced mass times the
    *standard deviation* of the values", not as a single number: the chi^2 factor is
    the mechanism's, the sd factor the data's.  The bound is attained when ``v - y``
    is proportional to ``dpi_hat/dpi - 1``.
    """
    c2 = chi2(pi_hat, pi)
    sd = value_covariance_op(pi, v)
    return {"chi2": c2, "sd": sd, "bound": c2.clamp_min(0).sqrt() * sd}


def tanh_tv_bound(s: Tensor, s_hat: Tensor, v: Tensor | None = None) -> dict:
    """prop:error-score: a logit error ``delta`` gives TV <= tanh(delta/2).

    ``s`` and ``s_hat`` are the *scaled* logits (``beta <q, k_j>``), shape [.., m, n];
    masked entries may be ``-inf`` in both and are ignored.  With ``delta`` the
    per-query max absolute logit error (:func:`logit_error`),

        TV(pi, pi_hat) <= tanh(delta / 2),
        chi2(pi_hat||pi) <= sinh(delta)^2,
        KL(pi_hat||pi)   <= delta^2 / 2,
        ||y - y_hat||    <= tanh(delta / 2) * osc(v).

    The sharp constant uses ``delta = osc(s_hat - s)/2`` rather than the pointwise
    max, since a uniform shift of all logits is absorbed by the normalizer; both are
    returned (``delta`` and ``delta_osc``) and the bounds use ``min`` of the two.
    """
    delta = logit_error(s, s_hat)
    delta_osc = 0.5 * logit_oscillation(s, s_hat)
    d = torch.minimum(delta, delta_osc)
    out = {"delta": delta, "delta_osc": delta_osc,
           "tv_bound": torch.tanh(d / 2),
           "chi2_bound": torch.sinh(d) ** 2,
           "kl_bound": d ** 2 / 2}
    if v is not None:
        out["output_bound"] = out["tv_bound"] * value_oscillation(v).unsqueeze(-1)
    return out


def logit_error(s: Tensor, s_hat: Tensor) -> Tensor:
    """delta_q = max_j |s_hat_j - s_j| over the visible keys, shape [.., m]."""
    d = (s_hat - s).abs()
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))  # both -inf => masked out
    return d.amax(-1)


def logit_oscillation(s: Tensor, s_hat: Tensor) -> Tensor:
    """osc_j(s_hat_j - s_j) over the visible keys: max - min, shape [.., m]."""
    d = s_hat - s
    finite = torch.isfinite(d)
    lo = torch.where(finite, d, torch.full_like(d, float("inf"))).amin(-1)
    hi = torch.where(finite, d, torch.full_like(d, float("-inf"))).amax(-1)
    rng = hi - lo
    return torch.where(torch.isfinite(rng), rng, torch.zeros_like(rng))


def mult_error_bound(pi: Tensor, pi_hat: Tensor, v: Tensor, eps: float) -> Tensor:
    """prop:error-mult: for (1-eps) g <= g_hat <= (1+eps) g,
    ||y - y_hat|| <= eps / (1 + sqrt(1 - eps^2)) * osc(v)."""
    import math
    c = eps / (1.0 + math.sqrt(max(0.0, 1.0 - eps ** 2)))
    return torch.full_like(total_variation(pi, pi_hat), c) * value_oscillation(v).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Restriction (sec:error-restrict)
# ---------------------------------------------------------------------------

def omitted_mass(pi: Tensor, support: Tensor) -> Tensor:
    """eps_q = pi_q(X \\ S_q) for boolean support [.., m, n]."""
    return (pi * (~support)).sum(-1)


def restriction_identity(pi: Tensor, support: Tensor, v: Tensor) -> dict:
    """prop:sparse-identity: y - y_hat = eps_q (m_{S^c} - m_S) for renormalized restriction.

    For ``eps_q == 0`` (full support) ``m_Sc`` is undefined and reported as ``0``; the
    predicted error is then ``0``, which is correct.  Likewise for ``eps_q == 1``.
    """
    eps = omitted_mass(pi, support)
    inS, outS = pi * support, pi * (~support)
    m_S = (inS @ v) / inS.sum(-1, keepdim=True).clamp_min(TINY)
    m_Sc = (outS @ v) / outS.sum(-1, keepdim=True).clamp_min(TINY)
    return {"eps": eps, "m_S": m_S, "m_Sc": m_Sc,
            "predicted_error": eps.unsqueeze(-1) * (m_Sc - m_S)}


def single_token_omission(pi: Tensor, v: Tensor) -> Tensor:
    """prop:error-single: removing token j and renormalizing costs
    p_j / (1 - p_j) * ||v_j - y||.  Returns the cost of every j, shape [.., m, n]."""
    y = pi @ v
    dev = (v.unsqueeze(-3) - y.unsqueeze(-2)).norm(dim=-1)  # [.., m, n]
    return pi / (1 - pi).clamp_min(TINY) * dev


# ---------------------------------------------------------------------------
# Merging and transport (prop:error-merge, prop:error-wasserstein)
# ---------------------------------------------------------------------------

def wasserstein1(pi: Tensor, pi_hat: Tensor, positions: Tensor | None = None) -> Tensor:
    """W_1 between two distributions over the *ordered* context positions.

    On the line the optimal transport cost has the closed form
    ``W_1 = sum_j |F_j - F_hat_j| * (x_{j+1} - x_j)`` with ``F`` the CDFs, so no LP is
    needed.  ``positions`` defaults to ``0, 1, ..., n-1`` (token index as the ground
    metric), the metric under which merging moves mass by the width of a cluster.
    """
    n = pi.shape[-1]
    if positions is None:
        positions = torch.arange(n, device=pi.device, dtype=pi.dtype)
    gaps = positions[1:] - positions[:-1]
    cdf = (pi - pi_hat).cumsum(-1)[..., :-1].abs()
    return (cdf * gaps).sum(-1)


def value_lipschitz(v: Tensor, positions: Tensor | None = None) -> Tensor:
    """The smallest ``L_v`` with ``||v_i - v_j|| <= L_v |x_i - x_j||``, shape [..]."""
    n = v.shape[-2]
    if positions is None:
        positions = torch.arange(n, device=v.device, dtype=v.dtype)
    dist = (positions[:, None] - positions[None, :]).abs()
    num = torch.cdist(v, v)
    ratio = torch.where(dist > 0, num / dist.clamp_min(TINY), torch.zeros_like(num))
    return ratio.amax(dim=(-1, -2))


def wasserstein1_bound(pi: Tensor, pi_hat: Tensor, v: Tensor,
                       positions: Tensor | None = None) -> Tensor:
    """prop:error-wasserstein: ||(pi - pi_hat) v|| <= L_v * W_1(pi, pi_hat).

    The transport bound charges an approximation for moving mass *far*, not for moving
    mass; it is the right bound for merging, clustering and pivot methods, where total
    variation is needlessly pessimistic (Sec. 4.1).  The ground metric is the token
    index unless ``positions`` says otherwise.
    """
    return value_lipschitz(v, positions).unsqueeze(-1) * wasserstein1(pi, pi_hat, positions)


def merge_bound(pi: Tensor, assign: Tensor, v: Tensor, v_hat: Tensor,
                delta: Tensor) -> dict:
    """prop:error-merge / prop:compress-merge for a pushforward compression.

    ``assign`` [n] maps each original position to its cluster, ``v_hat`` [.., r, dv]
    holds the representative values, ``delta = max_i beta ||q|| R_i`` [.., m] the key
    radius term.  Returns

        ||y - y_hat|| <= sum_i pi(C_i) ||m_{C_i} - v_hat_i|| + tanh(delta/2) diam{v_hat_i}.
    """
    r = int(assign.max()) + 1
    onehot = torch.nn.functional.one_hot(assign, r).to(pi.dtype)      # [n, r]
    mass = pi @ onehot                                                 # [.., m, r]
    num = torch.einsum("...mn,nr,...nd->...mrd", pi, onehot, v)        # [.., m, r, dv]
    m_C = num / mass.unsqueeze(-1).clamp_min(TINY)
    repr_term = (mass * (m_C - v_hat.unsqueeze(-3)).norm(dim=-1)).sum(-1)
    dist_term = torch.tanh(delta / 2) * value_oscillation(v_hat).unsqueeze(-1)
    return {"representation": repr_term, "distributional": dist_term,
            "bound": repr_term + dist_term}


# ---------------------------------------------------------------------------
# Matrix-level diagnostics (Sec. 4.2)
# ---------------------------------------------------------------------------

def attention_matrix_errors(pi: Tensor, pi_hat: Tensor, v: Tensor) -> dict:
    """Matrix-level errors used in the literature, per (batch, head).

    Sec. 4.2 argues these are the wrong target; they are computed so that study S1 can
    show it.  ``AV_fro`` is the one that does control the output.
    """
    D = pi - pi_hat
    return {"fro": D.flatten(-2).norm(dim=-1),
            "spectral": torch.linalg.matrix_norm(D, ord=2),
            "max_entry": D.abs().flatten(-2).amax(-1),
            "inf_to_inf": D.abs().sum(-1).amax(-1),
            "AV_fro": (D @ v).flatten(-2).norm(dim=-1)}
