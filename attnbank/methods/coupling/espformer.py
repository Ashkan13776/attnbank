"""ESPFormer: doubly stochastic attention from *expected sliced transport plans*.

``sec:ot``, "Doubly stochastic attention in transformers".  Sinkformers reach the
column constraint by iterating ``eq:ot-sinkhorn``; the iteration is sequential, and
its Hilbert-metric rate degrades exponentially in ``||c||_inf / eps``.  ESPFormer
-- Shahbazi et al. (2025), building on the expected sliced transport plan of Liu
et al. (2025) -- reaches it in a *single* pass instead, by averaging one-dimensional
transport plans.

The construction, for one slice ``theta`` in ``R^d``:

1. project the queries and the keys on ``theta``:  ``q_i^theta = <q_i, theta>``,
   ``k_j^theta = <k_j, theta>``;
2. between two *uniform* measures on the line the optimal plan is the monotone
   rearrangement: sort both sides and match rank to rank.  With ``A`` and ``B`` the
   (descending) sorting permutations of the projected queries and keys,

       U_theta = (1/m) A^T I B,

   where ``I`` is the ``m x n`` rank-to-rank coupling (``eq:ot-nwcorner`` in spirit:
   the north-west-corner plan between the uniform marginals ``1/m`` and ``1/n``, with
   rows rescaled to sum to one).  ``I`` *is* the identity when ``m == n``, which is
   the form stated in the paper.  ``U_theta`` has row marginal ``1/m`` and column
   marginal ``1/n``: it is a transport plan, not a row normalization;
3. the plans of the individual slices are averaged with weights that favour the
   cheap slices,

       G = sum_l sigma_l U_{theta_l},
       sigma_l = softmax_l( -tau_slice * sum_ij C_ij [U_{theta_l}]_ij ),
       C_ij = ||beta q_i - beta k_j||^2 .

A convex combination of couplings in ``Pi(nu, mubar)`` is again in ``Pi(nu, mubar)``,
so ``G`` is doubly stochastic exactly, with no iteration at all.  The attention
operator is ``pi_hat = m G`` (row-stochastic) and ``y = pi_hat v``.

**Soft sorting.**  ``argsort`` has zero gradient almost everywhere, so ``A`` and ``B``
are replaced by ``SoftSort_t`` (Prillo and Eisenschlos, 2020),
``SoftSort_t(x)[r, i] = softmax_i( -(x_i - sort(x)_r)^2 / t )``.  This is the one
place where the construction stops being exact: at finite ``t`` the relaxed ``A``
is row-stochastic but not a permutation, so its *column* sums -- and hence the row
sums of ``G`` -- deviate from one.  The operator is renormalized here so that
``pi_hat`` is a probability measure, and the deviation that was removed is reported
as ``extras["row_sum_error"]``.  It goes to zero as ``t -> 0``; the test suite checks
that it does.

**Two conventions, stated so that numbers transfer.**  The slice cost here is that of
the unit-mass plan ``U_theta``, as in the paper; the released code omits the ``1/n`` and
so weights the slices with an inverse temperature larger by that factor (its default is
``10`` against the paper's ``0.1``).  And the paper's Algorithm 1 uses the *squared*
Euclidean cost while the released code uses the distance itself; both are available
through ``cost=``, with the paper's as the default.  Neither choice changes the support
of the operator, only how sharply the cheap slices are preferred.  Under those two
conventions ``extras["plan"]`` reproduces the authors' released module bit for bit;
``tests/test_coupling_transport.py`` writes out that computation and checks it.

**No key potential.**  ``prop:ot-conditional`` says the *entropic*-OT conditional is
softmax tilted by one number per key.  ESP's plan is a mixture of permutations, not
a Gibbs kernel ``exp(s + f + g)``, so no such ``psi`` exists and ``extras`` deliberately
contains no ``"potential"``: study S7's tilt analysis does not apply to it.  What it
does expose -- ``gamma``, ``row_marginal``, ``col_marginal``, ``marginal_error`` --
matches ``sinkhorn``, so the marginal diagnostics are shared.

**Cost.**  Not sub-quadratic, and not claimed to be.  The paper reports
``O(N_sl * n(n + d))`` for ``N_sl`` slices and credits the method with being *parallel*
over slices where Sinkhorn is sequential over iterations.  In this reference
implementation the dominant term is larger still: ``A^T I B`` is a product of dense
``m x m``, ``m x n`` and ``n x n`` matrices once the sort is relaxed, so forming the
``N_sl`` plans costs ``O(N_sl * mn(m + n))``.  A hard sort would make the product a
permutation composition and recover the quoted figure, at the price of the gradient
the soft sort exists to provide.  ``cost()`` reports what this file executes.

**Slices.**  The paper sets ``Theta = I_{d x d}`` -- axis-aligned slices, so that the
mechanism adds no parameters at all; that is the default here (``n_slices=None``) and
the reason ``trainable = False``.  Its Table 6 ablates *learnable* slicers, which help
when the number of slices is small; ``n_slices=L`` draws ``L`` directions uniformly on
the sphere through a seeded generator, which is that ablation's initialization but not
its training.

Not causal, for the reason every doubly stochastic operator is not: ``prop:ot-causal``,
a lower-triangular matrix with equal row and column sums is diagonal.  Sorting is in
any case a global operation over the sequence.

Reference: Shahbazi et al. (2025) ESPFormer; Liu et al. (2025) expected sliced
transport plans; Prillo and Eisenschlos (2020) SoftSort; Sander et al. (2022)
Sinkformers.
"""
from __future__ import annotations

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    make_generator,
    renormalize,
)
from ...errors import ConfigError
from ...registry import register

COSTS = ("sqeuclidean", "euclidean")


def soft_sort(x, tau: float):
    """``SoftSort_t`` of Prillo and Eisenschlos (2020), squared-distance variant.

    ``x`` is ``[..., n]``; the result is ``[..., n, n]`` and row ``r`` is a probability
    distribution over the positions ``i``, concentrated on the ``r``-th largest entry.
    As ``tau -> 0`` it converges to the descending-sort permutation matrix.  It is a
    *continuous* function of ``x`` even though the sort is not, which is why the fp32
    and fp64 paths agree.
    """
    srt = x.sort(dim=-1, descending=True).values
    gap = (x.unsqueeze(-2) - srt.unsqueeze(-1)) ** 2
    return torch.softmax(-gap / tau, dim=-1)


def rank_coupling(m: int, n: int, dtype, device):
    """The ``m x n`` rank-to-rank coupling ``I`` of the ESP plan, rows summing to one.

    Between uniform measures on ``m`` and ``n`` points of the line, the optimal plan is
    the north-west-corner (monotone) one: rank ``a`` of the first and rank ``b`` of the
    second are coupled by the overlap of the intervals ``[a/m, (a+1)/m]`` and
    ``[b/n, (b+1)/n]``.  Scaled by ``m`` so that ``U = (1/m) A^T I B`` has row marginal
    ``1/m`` and column marginal ``1/n``; for ``m == n`` this is the identity, the form
    the paper states.
    """
    a = torch.arange(m, dtype=dtype, device=device).unsqueeze(-1)
    b = torch.arange(n, dtype=dtype, device=device).unsqueeze(-2)
    lo = torch.maximum(a / m, b / n)
    hi = torch.minimum((a + 1) / m, (b + 1) / n)
    return (hi - lo).clamp_min(0.0) * m


@register
class ESPFormerAttention(AttentionMethod):
    """Doubly stochastic attention in one pass: the expectation of sliced OT plans."""

    name = "espformer"
    category = Category.COUPLING
    objective = Objective.MODIFY
    supports_causal = False          # prop:ot-causal, and the sort is global anyway
    expresses_weights = True
    row_stochastic = True
    trainable = False                # Theta = I by design: the paper adds no parameters
    references = ("shahbazi2025espformer", "liu2025expected", "prillo2020softsort",
                  "sander2022sinkformers")
    paper_section = "sec:ot"

    def __init__(self, n_slices: int | None = None, tau: float = 1e-3,
                 tau_slice: float = 0.1, cost: str = "sqeuclidean", seed: int = 0, **kw):
        """``n_slices``: ``None`` for the paper's axis-aligned ``Theta = I`` (``d`` slices),
        or an integer for that many random directions on the sphere.
        ``tau``: SoftSort temperature (the paper's ``t = 1e-3``); ``-> 0`` is a hard sort.
        ``tau_slice``: inverse temperature of the slice weights (the paper's ``tau = 0.1``),
        applied to the cost of the unit-mass plan ``U_theta``.  ``0`` averages the slices.
        ``cost``: ``'sqeuclidean'`` (Algorithm 1) or ``'euclidean'`` (the released code).
        """
        if cost not in COSTS:
            raise ConfigError(f"espformer: cost must be one of {COSTS}; got {cost!r}")
        if tau <= 0:
            raise ConfigError(f"espformer: tau is a SoftSort temperature and must be > 0; "
                              f"got {tau}. Use a small value (1e-3) for a near-hard sort.")
        if n_slices is not None and n_slices < 1:
            raise ConfigError(f"espformer: n_slices must be >= 1, or None for the "
                              f"paper's axis-aligned Theta = I; got {n_slices}")
        super().__init__(n_slices=n_slices, tau=tau, tau_slice=tau_slice, cost=cost,
                         seed=seed, **kw)
        self.n_slices, self.tau, self.tau_slice = n_slices, tau, tau_slice
        self.cost_kind, self.seed = cost, seed

    def _project(self, qs, ks, d):
        """Projections on the slices: the coordinates themselves when ``Theta = I``."""
        if self.n_slices is None:
            return qs, ks, None
        g = make_generator(self.seed)
        theta = torch.randn(self.n_slices, d, generator=g, dtype=torch.float64)
        theta = theta / theta.norm(dim=-1, keepdim=True)
        theta = theta.to(qs.device, qs.dtype)
        return (torch.einsum("...md,ld->...ml", qs, theta),
                torch.einsum("...nd,ld->...nl", ks, theta), theta)

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        beta = self._scale(d)
        qs, ks = q * beta, k * beta                      # the paper slices the scaled q, k
        pq, pk, theta = self._project(qs, ks, d)
        n_sl = pq.shape[-1]

        # one 1-D plan per slice: U = (1/m) A^T I B, held as the row-stochastic m A^T I B
        a_sort = soft_sort(pq.transpose(-1, -2), self.tau)       # [..., L, m, m]
        b_sort = soft_sort(pk.transpose(-1, -2), self.tau)       # [..., L, n, n]
        rank = rank_coupling(m, n, q.dtype, q.device)            # [m, n], rows sum to 1
        plans = a_sort.transpose(-1, -2) @ (rank @ b_sort)       # [..., L, m, n]

        # eq (9): weight the slices by the transport cost of their own plan
        cost = torch.cdist(qs, ks, p=2)                          # [..., m, n]
        if self.cost_kind == "sqeuclidean":
            cost = cost ** 2
        slice_cost = (cost.unsqueeze(-3) * plans).sum((-2, -1)) / m        # [..., L]
        sigma = torch.softmax(-self.tau_slice * slice_cost, dim=-1)        # [..., L]
        mixture = (sigma[..., None, None] * plans).sum(-3)                 # [..., m, n]

        # exact at t -> 0; renormalized here, and the removed deviation is reported
        row_sum_error = (mixture.sum(-1) - 1.0).abs().amax(-1)
        pi = renormalize(mixture)
        gamma = pi / m
        extras = {"gamma": gamma, "plan": mixture,
                  "row_marginal": gamma.sum(-1), "col_marginal": gamma.sum(-2),
                  "target_row": torch.full_like(gamma[..., 0], 1.0 / m),
                  "target_col": torch.full_like(gamma[..., 0, :], 1.0 / n),
                  "marginal_error": (gamma.sum(-1) - 1.0 / m).abs().amax(-1)
                                    + (gamma.sum(-2) - 1.0 / n).abs().amax(-1),
                  "row_sum_error": row_sum_error,
                  "slice_weights": sigma, "slice_cost": slice_cost,
                  "slices": theta, "n_slices": n_sl}
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        """What this file executes, not the quoted asymptotics.

        ``n_sl`` soft sorts (``m^2 + n^2`` each), ``n_sl`` triple products
        ``A^T (I B)`` (``mn(m + n)`` each), the cost matrix, and one ``m x n`` apply.
        The ``mn(m + n)`` term is the price of relaxing the sort: with hard
        permutations the same product is a relabelling in ``O(mn)`` and the total
        collapses to the paper's ``O(n_sl * n(n + d))``.
        """
        n_sl = self.n_slices if self.n_slices is not None else d
        return {"flops": 2 * n_sl * (m + n) * d + n_sl * (m * m + n * n)
                         + 2 * n_sl * m * n * (m + n) + 3 * m * n + 2 * m * n * dv,
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv
                                + n_sl * m * n) * bytes_per_el}
