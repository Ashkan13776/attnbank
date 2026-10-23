"""OVAL: an output-aware local page basis for page selection.

``sec:sparse-query`` ("Blocks and pages"), alongside ``blocktopk``.  Keys are
partitioned into contiguous pages; a page is summarised once, offline, and ranked at
decode time by an *estimate of its attention mass*.  What OVAL changes is the summary.

Page score.  For a page with keys ``K in R^{m x d}``, write ``mu = K^T 1 / m`` and
``X = K - 1 mu^T``.  The log of the page's unnormalized mass is exactly

    l(q) = log sum_i exp(beta k_i^T q) = beta mu^T q + lse(z(q)),   z(q) = beta X q,

and since the softmax denominator is shared by all pages, ranking by ``l`` ranks them
by total attention probability.  A rank-``r`` orthonormal basis ``B in R^{d x r}``
gives the storable surrogate ``l_B(q) = beta mu^T q + lse(beta X P_B q)``, evaluated as
``R (B^T q)`` from the stored ``R = X B in R^{m x r}``.  Values never enter scoring.

Which basis.  The key-only answer -- page-local key PCA, i.e. the leading eigenvectors
of ``X^T X`` (``page_pca``, LOCKS / Loki) -- keeps the directions with the most key
energy.  OVAL's observation is that a logit error only matters when it moves mass
between tokens with *different values*.  Centering the values as ``Y = V - 1 nu^T``,
the conditional within-page output is ``f(z) = nu + Y^T softmax(z)``; linearized at
uniform attention its sensitivity to a logit perturbation ``h`` is ``Y^T h / m``.
Measuring that in a PSD seminorm ``G`` and the logit residual in ``l_2``, and taking
the expectation over isotropic queries (``E[q q^T] = I``), the two risks are

    R_key(B) = tr((I - P_B) X^T X (I - P_B)) / d,
    R_out(B) = tr((I - P_B) M_out (I - P_B)) / (m^2 d),   M_out = X^T Y G Y^T X >= 0.

Both are the energy of a PSD matrix outside the retained subspace, so both are
minimized by leading invariant subspaces.  Scaling each by its total energy and mixing
with ``eta in [0, 1]`` gives the matrix this file diagonalizes,

    M_eta = (1 - eta) c_key X^T X + eta c_out M_out,
    c_key = (tr(X^T X) + eps)^-1,   c_out = (tr(M_out) + eps)^-1,

whose leading ``r`` eigenvectors are ``B_eta``.  ``eta = 0`` *is* key PCA; ``eta = 1``
minimizes the linearized output surrogate alone.  Values are read once, when the page
summary is built; the stored summary ``(mu, B_eta, R_eta)`` and the decode-time cost
are those of a key-only spectral summary of the same rank.

What this buys and what it does not.  The basis is chosen for a *linearization at
uniform within-page attention* under an isotropic-query assumption, so nothing here is
a bound on the realised output error.  The bound that *does* hold is on retrieval:
lse is 1-Lipschitz in the sup norm, so ``|l(q) - l_B(q)| <= ||beta X (I - P_B) q||_inf``
for any basis, OVAL's included -- and its expected square is ``R_key``, which is what
``eta = 0`` minimizes.  Moving ``eta`` off zero deliberately gives retrieval accuracy
up; whether that is a good trade depends on the key-value coupling actually present,
and on isotropic data with none it is not (``tests/test_restrict_learned.py``).
Selection is still at page granularity, so the omitted mass ``eps_q`` of
``prop:sparse-identity`` is still at least the page-level minimum and the realised
error is still ``eps_q (m_{S^c} - m_S)``.

Differences from the paper, stated rather than hidden:

* ``G = W_O^T W_O`` in the paper, which measures the perturbation after the head's
  output projection.  The bank has no ``W_O``, so ``g=None`` (the identity: the
  perturbation measured in value space) is the default.  Pass ``g=`` a PSD
  ``[dv, dv]`` tensor to recover the paper's choice.
* Algorithm 1 reduces the ``d x d`` eigenproblem to ``m x m`` for short pages and
  completes a rank-deficient basis to exactly ``r`` columns.  This file diagonalizes
  ``M_eta`` directly (clarity over speed, DESIGN.md Sec. 6) and performs the same
  completion from the residual ``X - X P_B``; columns beyond that are reached only
  when ``span(B)`` already contains the row space of ``X``, so they contribute exactly
  zero to ``R_eta`` and are stored as zeros here instead of as arbitrary orthonormal
  directions.  The scored function of ``q`` is identical.
* The paper's harness builds on FreeKV and adds sink and recent regions to the
  selected pages.  Those are orthogonal to the basis and live in ``sink_window``;
  this file is the page selector alone, so that what it is compared against -- the
  same selector with ``eta = 0`` -- differs in nothing but the basis.

Causality follows ``blocktopk``: only *complete past* pages are rankable, the page
containing the query is always selected, and later pages are excluded.

Reference: Shahbazi, Thrash & Kolouri (2026) OVAL; Hwang (2026) LOCKS;
Singhania et al. (2024) Loki.
"""
from __future__ import annotations

import torch

from ...core import AttentionMethod, Category, Objective, query_positions
from ...registry import register
from ._restriction import reference_pi, restrict, visible

# ---------------------------------------------------------------------------
# Page summaries. Shared with ``page_pca``, which is this construction at eta = 0.
# ---------------------------------------------------------------------------


def _paged(t, page):
    """Reshape ``[..., n, c]`` to ``[..., nb, page, c]``, zero padding the last page."""
    n = t.shape[-2]
    nb = (n + page - 1) // page
    pad = nb * page - n
    if pad:
        t = torch.nn.functional.pad(t, (0, 0, 0, pad))
    return t.unflatten(-2, (nb, page)), nb, pad


def _members(n, nb, page, device):
    """Boolean ``[nb, page]``: which slots of each page are real positions."""
    keep = torch.ones(nb * page, dtype=torch.bool, device=device)
    keep[n:] = False
    return keep.view(nb, page)


def _top_subspace(mat, x, rank, tol_scale):
    """Leading ``rank`` eigenvectors of PSD ``mat``, with near-null columns zeroed.

    Returns ``(B, R, kept)`` with ``B`` of shape ``[..., d, rank]``, ``R = x @ B`` and
    ``kept`` the boolean ``[..., rank]`` marking the columns that carry energy.  A
    zeroed column contributes nothing to ``B B^T`` or to ``R``, so dropping a direction
    this way is the same as not having selected it.
    """
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    lam, vec = torch.linalg.eigh(mat)                       # ascending
    lam, vec = lam.flip(-1), vec.flip(-1)
    tol = lam[..., :1].clamp_min(0.0) * tol_scale
    kept = lam[..., :rank] > tol
    basis = vec[..., :, :rank] * kept.unsqueeze(-2).to(vec.dtype)
    return basis, x @ basis, kept


def page_summary(k, v, page, eta, rank, g=None, eps=1e-12):
    """Build ``(mu, B_eta, R_eta)`` for every page of ``k`` -- Algorithm 1 of OVAL.

    ``eta = 0`` gives page-local key PCA; ``eta = 1`` the output-sensitivity basis.
    The basis is returned split in two pieces, ``(B_a, R_a)`` from ``M_eta`` and
    ``(B_b, R_b)`` from the key residual used to complete a rank-deficient basis;
    scoring sums their contributions, so the split is bookkeeping, not two bases.
    """
    n, d, device = k.shape[-2], k.shape[-1], k.device
    kb, nb, _ = _paged(k, page)
    vb, _, _ = _paged(v, page)
    member = _members(n, nb, page, device)                  # [nb, page]
    mk = member.unsqueeze(-1).to(k.dtype)
    cnt = member.sum(-1).clamp_min(1).to(k.dtype)[:, None]  # [nb, 1]

    mu = (kb * mk).sum(-2) / cnt                            # [..., nb, d]
    nu = (vb * mk).sum(-2) / cnt                            # [..., nb, dv]
    x = (kb - mu.unsqueeze(-2)) * mk                        # [..., nb, page, d]
    y = (vb - nu.unsqueeze(-2)) * mk                        # [..., nb, page, dv]

    xtx = x.transpose(-1, -2) @ x                           # [..., nb, d, d]
    p = x.transpose(-1, -2) @ y                             # X^T Y, [..., nb, d, dv]
    m_out = (p if g is None else p @ g) @ p.transpose(-1, -2)
    m_out = 0.5 * (m_out + m_out.transpose(-1, -2))
    tr = lambda a: a.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
    c_key = 1.0 / (tr(xtx) + eps)
    c_out = 1.0 / (tr(m_out) + eps)
    m_eta = (1.0 - eta) * c_key * xtx + eta * c_out * m_out

    r = min(rank, d)
    tol_scale = d * torch.finfo(k.dtype).eps
    b_a, r_a, kept = _top_subspace(m_eta, x, r, tol_scale)
    # Algorithm 1, lines 9-11: complete the basis from the key directions the hybrid
    # subspace missed -- which is only possible when M_eta is rank deficient (eta > 0).
    x_perp = x - r_a @ b_a.transpose(-1, -2)
    b_b, r_b, kept_b = _top_subspace(x_perp.transpose(-1, -2) @ x_perp, x_perp, r,
                                     tol_scale)
    room = torch.arange(r, device=device) < (r - kept.sum(-1, keepdim=True))
    b_b = b_b * room.unsqueeze(-2).to(b_b.dtype)
    r_b = r_b * room.unsqueeze(-2).to(r_b.dtype)
    return {"mu": mu, "B": (b_a, b_b), "R": (r_a, r_b), "member": member,
            "X": x, "Y": y, "M": m_eta,
            "rank_used": (kept.sum(-1) + (kept_b & room).sum(-1))}


def page_log_mass(q, summary, beta):
    """``l_B(q) = beta mu^T q + lse(beta R (B^T q))`` for every (query, page): ``[..., m, nb]``."""
    mu, (b_a, b_b), (r_a, r_b), member = (summary["mu"], summary["B"], summary["R"],
                                          summary["member"])
    z = (torch.einsum("...bpr,...bmr->...bmp", r_a, torch.einsum("...md,...bdr->...bmr", q, b_a))
         + torch.einsum("...bpr,...bmr->...bmp", r_b,
                        torch.einsum("...md,...bdr->...bmr", q, b_b)))
    z = (beta * z).masked_fill(~member[:, None, :], float("-inf"))
    lse = torch.logsumexp(z, dim=-1)                        # [..., nb, m]
    return beta * torch.einsum("...md,...bd->...mb", q, mu) + lse.transpose(-1, -2)


def select_pages(score, m, n, page, n_pages, causal):
    """Top-``n_pages`` complete pages per query, plus the query's own page: ``[..., m, nb]``.

    Only pages that lie entirely in the query's past may be ranked, so the support
    depends on nothing after the query; the page holding the query is resident anyway
    and is always kept (``blocktopk``, Quest).
    """
    nb = score.shape[-1]
    bidx = torch.arange(nb, device=score.device)
    if causal:
        own = (query_positions(m, n, device=score.device) // page)[:, None]
        complete = bidx[None, :] < own
    else:
        own = torch.full((m, 1), -1, dtype=torch.long, device=score.device)
        complete = torch.ones(m, nb, dtype=torch.bool, device=score.device)
    sel = score.masked_fill(~complete, float("-inf"))
    take = min(n_pages, nb)
    chosen = torch.zeros_like(sel, dtype=torch.bool).scatter_(
        -1, sel.topk(take, dim=-1).indices, True) & complete
    return chosen | (bidx[None, :] == own)


class _SpectralPageSelector(AttentionMethod):
    """Shared machinery: summarise pages spectrally, rank by estimated log-mass, restrict."""

    category = Category.RESTRICT
    objective = Objective.APPROXIMATE
    supports_causal = True
    expresses_weights = True
    row_stochastic = True
    paper_section = "sec:sparse"
    eta: float = 0.0

    def __init__(self, page: int = 4, n_pages: int | None = 2, rank: int = 2,
                 eps: float = 1e-12, **kw):
        super().__init__(page=page, n_pages=n_pages, rank=rank, eps=eps, **kw)
        self.page, self.n_pages, self.rank, self.eps = page, n_pages, rank, eps

    def _g(self):
        return None

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        nb = (n + self.page - 1) // self.page
        pi, _ = reference_pi(self, q, k, causal)
        vis = visible(q, k, causal)
        base = {"n_pages": nb, "page_size": self.page, "eta": self.eta,
                "rank": min(self.rank, d)}
        if self.n_pages is None or self.n_pages >= nb:
            return restrict(pi, vis, v, return_weights, base)

        summary = page_summary(k, v, self.page, self.eta, self.rank, self._g(), self.eps)
        score = page_log_mass(q, summary, self._scale(d))
        chosen = select_pages(score, m, n, self.page, self.n_pages, causal)
        sup = chosen.repeat_interleave(self.page, dim=-1)[..., :n] & vis
        return restrict(pi, sup, v, return_weights,
                        dict(base, page_scores=score, pages_selected=chosen,
                             rank_used=summary["rank_used"]))

    def cost(self, m, n, d, dv, bytes_per_el=2):
        nb = (n + self.page - 1) // self.page
        r = min(self.rank, d)
        kept = min(self.n_pages or nb, nb) * self.page
        summary_el = nb * (d + d * r + self.page * r)      # mu, B, R per page
        return {"flops": 2 * m * nb * (d + d * r + self.page * r)
                         + 2 * m * kept * (d + dv),
                "state_bytes": (n * (d + dv) + summary_el) * bytes_per_el,
                "bytes_moved": (m * d + summary_el + m * kept * (d + dv) + m * dv)
                               * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"n_pages": None}


@register
class OvalAttention(_SpectralPageSelector):
    """Page selection by a rank-r basis built from the joint key-value structure."""

    name = "oval"
    references = ("shahbazi2026oval", "hwang2026locks", "tang2024quest")

    def __init__(self, page: int = 4, n_pages: int | None = 2, rank: int = 2,
                 eta: float = 0.5, g: torch.Tensor | None = None, eps: float = 1e-12,
                 **kw):
        """``eta`` mixes key fidelity (0) with output sensitivity (1); ``g`` is the PSD
        value-space metric, ``None`` meaning the identity (see the module docstring)."""
        super().__init__(page=page, n_pages=n_pages, rank=rank, eps=eps, eta=eta, g=g, **kw)
        self.eta, self.g = eta, g

    def _g(self):
        return self.g
