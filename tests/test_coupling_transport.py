"""Mathematics specific to the transport-based couplings: ``sec:ot``.

``espformer`` (expected sliced transport plans), ``otke`` (transport onto a trainable
reference), ``unbalanced_sinkhorn`` (KL-relaxed marginals, ``eq:ot-unbalanced``) and
``dustbin_sinkhorn`` (SuperGlue's partial assignment).  ``tests/test_coupling.py``
covers ``sinkhorn`` and ``lowrank_ot``; this file never repeats it.

The through-line is the one ``sec:ot`` draws.  Every method here imposes something on
the *columns* of the attention matrix, and the question each test asks is what exactly
is imposed, on what pair of index sets, and what is given up for it.
"""
from __future__ import annotations

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M
from attnbank.methods.coupling.espformer import rank_coupling, soft_sort

TRANSPORT = ["espformer", "otke", "unbalanced_sinkhorn", "dustbin_sinkhorn"]
#: the ones whose plan is of Gibbs form ``exp(s + f + g)``, so prop:ot-conditional holds
GIBBS = ["sinkhorn", "unbalanced_sinkhorn", "dustbin_sinkhorn", "otke"]


def qkv(b=2, h=2, m=12, n=12, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def logits(q, k):
    return ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)


# ===========================================================================
# ESPFormer -- the expected sliced transport plan
# ===========================================================================
@pytest.mark.parametrize("m,n", [(6, 6), (5, 9), (9, 5), (1, 7), (4, 12)])
def test_the_rank_coupling_is_the_one_dimensional_plan_between_uniform_marginals(m, n):
    """``I`` in ``U = (1/m) A^T I B``: the north-west-corner plan, rows rescaled to one.

    It must have the marginals of a transport plan between ``1/m`` and ``1/n``, be
    monotone (each rank's support is a contiguous block, and the blocks advance), and
    collapse to the identity the paper writes for the equal-size case.
    """
    r = rank_coupling(m, n, torch.float64, "cpu")
    assert r.shape == (m, n)
    assert (r >= 0).all()
    assert torch.allclose(r.sum(-1), torch.ones(m, dtype=torch.float64), atol=1e-14)
    assert torch.allclose(r.sum(-2), torch.full((n,), m / n, dtype=torch.float64),
                          atol=1e-13)
    if m == n:
        assert torch.allclose(r, torch.eye(m, dtype=torch.float64), atol=1e-14)
    # monotone: the first and last supported column both increase with the rank
    occ = r > 1e-12
    first = occ.float().argmax(-1)
    last = n - 1 - occ.flip(-1).float().argmax(-1)
    assert (first[1:] >= first[:-1]).all() and (last[1:] >= last[:-1]).all()


def test_soft_sort_converges_to_the_descending_permutation():
    """SoftSort is a relaxation: it is row-stochastic at every ``t`` and a permutation
    only in the limit.  The gap it leaves is what ``extras['row_sum_error']`` reports."""
    x = torch.tensor([[0.3, -1.2, 2.0, 0.4, -0.1]], dtype=torch.float64)
    perm = torch.zeros(1, 5, 5, dtype=torch.float64)
    perm[0, torch.arange(5), x.argsort(dim=-1, descending=True)[0]] = 1.0
    errs = []
    for tau in (1.0, 1e-1, 1e-2, 1e-3, 1e-4):
        a = soft_sort(x, tau)
        assert torch.allclose(a.sum(-1), torch.ones(1, 5, dtype=torch.float64), atol=1e-14)
        errs.append(float((a - perm).abs().max()))
    assert all(p > q for p, q in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-6, errs


def test_the_expected_sliced_plan_is_doubly_stochastic_in_the_hard_sort_limit():
    """The whole point: a convex combination of 1-D OT plans is a transport plan, so
    ``G`` is doubly stochastic with no iteration.  Exactly, once the sort is exact."""
    q, k, v = qkv(m=10, n=10, seed=1)
    errs, rows = [], []
    for tau in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        e = ab.build("espformer", tau=tau)(q, k, v, return_weights=True).extras
        errs.append(float(e["marginal_error"].max()))
        rows.append(float(e["row_sum_error"].max()))
    assert all(p > q for p, q in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-14, errs
    # Both marginals reach machine precision; only the column one does so monotonically.
    # The row deviation wiggles, and the reason is worth recording: a pair of projected
    # values that happen to be close contributes *nothing* to it (the two ranks exchange
    # exactly the mass they lose), so which configuration dominates the maximum changes
    # with tau.  The limit is what the construction promises, not the path to it.
    assert rows[-1] < 1e-14 < rows[0], rows
    assert rows[0] > 1e10 * rows[-1], rows


def test_each_slice_contributes_a_permutation_and_the_mixture_is_their_average():
    """``U_theta = (1/n) A^T B``: with a hard sort every slice plan is a permutation
    matrix, and ``G`` is the ``sigma``-weighted average of them (eq. 8)."""
    q, k, v = qkv(b=1, h=1, m=8, n=8, seed=2)
    tau = 1e-10
    attn = ab.build("espformer", tau=tau)
    out = attn(q, k, v, return_weights=True)
    pq, pk, _ = attn._project(q * q.shape[-1] ** -0.5, k * k.shape[-1] ** -0.5, q.shape[-1])
    a_sort, b_sort = soft_sort(pq.transpose(-1, -2), tau), soft_sort(pk.transpose(-1, -2), tau)
    plans = a_sort.transpose(-1, -2) @ b_sort                 # m == n, so I is the identity
    for p in plans.reshape(-1, 8, 8):
        assert torch.allclose(p.sum(-1), torch.ones(8, dtype=torch.float64), atol=1e-9)
        assert torch.allclose(p.sum(-2), torch.ones(8, dtype=torch.float64), atol=1e-9)
        assert float((p - (p > 0.5).to(p.dtype)).abs().max()) < 1e-9   # a permutation
    sigma = out.extras["slice_weights"]
    assert torch.allclose(sigma.sum(-1), torch.ones_like(sigma[..., 0]), atol=1e-12)
    assert torch.allclose((sigma[..., None, None] * plans).sum(-3), out.weights, atol=1e-9)


def test_a_one_dimensional_head_reproduces_the_monotone_matching():
    """With ``d = 1`` there is one slice and no mixture: ESP *is* the 1-D OT plan, which
    matches the largest query to the largest key.  This is the construction, stripped."""
    q = torch.tensor([[[[0.0], [3.0], [1.0], [2.0]]]], dtype=torch.float64)
    k = torch.tensor([[[[5.0], [-1.0], [7.0], [2.0]]]], dtype=torch.float64)
    v = torch.eye(4, dtype=torch.float64).reshape(1, 1, 4, 4)
    w = ab.build("espformer", tau=1e-10)(q, k, v, return_weights=True).weights[0, 0]
    # rank of each query (descending) -> key of the same rank: 3>2>1>0 and 7>5>2>-1
    assert w.argmax(-1).tolist() == [1, 2, 3, 0]
    assert float((w - torch.eye(4, dtype=torch.float64)[[1, 2, 3, 0]]).abs().max()) < 1e-9


def test_the_raw_plan_matches_the_authors_released_implementation():
    """Faithfulness, checked rather than asserted.

    The right-hand side below is the computation in the released ESPFormer module
    (``dariansal/ESPFormer``, ``attentions/esp.py``), written out: soft-sort the scaled
    queries and keys per feature, form ``Pu^T Pv``, weight the slices by the Euclidean
    transport cost of that unweighted product, and sum.  ``extras["plan"]`` -- the
    mixture *before* this library renormalizes it -- must equal it to machine precision.

    The one difference is the renormalization, and this test also measures it: the
    released rows are several percent off unity at the recommended ``t``, which is why
    ``pi_hat`` here is the renormalized object and ``row_sum_error`` reports the gap.
    """
    b, h, n, d = 2, 3, 9, 6
    q, k, v = qkv(b=b, h=h, m=n, n=n, d=d, dv=4, seed=24)
    scale = d ** -0.5
    for tau, temperature in ((1e-3, 10.0), (1e-2, 0.5), (0.1, 3.0)):
        x, y = q * scale, k * scale
        pu = soft_sort(x.transpose(-1, -2), tau)            # [b, h, d, n, n]
        pv = soft_sort(y.transpose(-1, -2), tau)
        gamma = pu.transpose(-1, -2) @ pv
        cost = torch.cdist(x, y, p=2)
        swd = (cost.unsqueeze(-3) * gamma).sum((-2, -1))
        released = (torch.softmax(-temperature * swd, -1)[..., None, None] * gamma).sum(-3)

        out = ab.build("espformer", tau=tau, tau_slice=temperature * n,
                       cost="euclidean")(q, k, v, return_weights=True)
        assert torch.allclose(out.extras["plan"], released, atol=1e-13)
        assert torch.allclose(out.weights, released / released.sum(-1, keepdim=True),
                              atol=1e-13)
        assert float((released.sum(-1) - 1).abs().max()) > 1e-3     # and it does deviate


def test_the_slice_weights_prefer_the_cheap_slices():
    """eq. 9: ``sigma = softmax(-tau * D)``, so the ordering of the weights is the
    reverse of the ordering of the slice transport costs, and ``tau = 0`` averages."""
    q, k, v = qkv(b=1, h=1, m=10, n=10, seed=3, scale=1.3)
    e = ab.build("espformer", tau_slice=2.0)(q, k, v, return_weights=True).extras
    cost, sigma = e["slice_cost"][0, 0], e["slice_weights"][0, 0]
    assert float(M.spearman(cost, sigma)) == pytest.approx(-1.0)
    flat = ab.build("espformer", tau_slice=0.0)(q, k, v, return_weights=True).extras
    assert torch.allclose(flat["slice_weights"],
                          torch.full_like(flat["slice_weights"], 1 / 8), atol=1e-14)


def test_espformer_has_no_key_potential_and_says_so():
    """prop:ot-conditional characterises the *entropic*-OT conditional as softmax tilted
    by a query-independent ``psi``.  ESP's plan is a mixture of permutations, not a Gibbs
    kernel, so no such ``psi`` exists.  The contrast with ``sinkhorn`` is the test: the
    same residual that is constant across queries there is not here."""
    q, k, v = qkv(b=1, h=1, m=10, n=10, seed=4, scale=1.2)
    s = logits(q, k)

    def tilt_spread(weights):
        resid = torch.log(weights.clamp_min(1e-300)) - s
        centred = resid - resid.mean(-1, keepdim=True)
        return float((centred - centred[..., :1, :]).abs().max())

    esp = ab.build("espformer", tau=0.5)(q, k, v, return_weights=True)
    snk = ab.build("sinkhorn", n_iter=200)(q, k, v, return_weights=True)
    assert tilt_spread(snk.weights) < 1e-9                  # prop:ot-conditional holds
    assert tilt_spread(esp.weights) > 1.0                   # and fails here, by a lot
    assert "potential" not in esp.extras, \
        "espformer must not advertise a potential it does not have (study S7 reads it)"


def test_random_slices_are_an_option_and_the_default_is_parameter_free():
    """The paper sets ``Theta = I`` to avoid extra parameters -- hence
    ``trainable = False``.  Its Table 6 ablates learnable slicers; ``n_slices`` gives
    their initialization, so the two paths must both run and must differ."""
    q, k, v = qkv(m=9, n=9, seed=5)
    base = ab.build("espformer")(q, k, v, return_weights=True)
    assert base.extras["n_slices"] == q.shape[-1] and base.extras["slices"] is None
    rnd = ab.build("espformer", n_slices=3, seed=1)(q, k, v, return_weights=True)
    assert rnd.extras["slices"].shape == (3, 8)
    assert torch.allclose(rnd.extras["slices"].norm(dim=-1),
                          torch.ones(3, dtype=torch.float64), atol=1e-12)
    assert float((rnd.weights - base.weights).abs().max()) > 1e-3
    assert ab.get_class("espformer").trainable is False


def test_espformer_is_quadratic_and_does_not_pretend_otherwise():
    """sec:ot records that ESPFormer removes the *iteration*, not the ``n^2``."""
    c = ab.build("espformer").cost
    assert c(256, 256, 64, 64)["flops"] > 3.5 * c(128, 128, 64, 64)["flops"]
    assert c(1, 4096, 128, 128)["state_bytes"] == ab.build("softmax").cost(
        1, 4096, 128, 128)["state_bytes"]


def test_the_cost_convention_is_selectable_and_only_reweights_the_slices():
    q, k, v = qkv(b=1, h=1, m=8, n=8, seed=6)
    paper = ab.build("espformer", cost="sqeuclidean")(q, k, v, return_weights=True)
    code = ab.build("espformer", cost="euclidean")(q, k, v, return_weights=True)
    assert float((paper.extras["slice_cost"] - code.extras["slice_cost"]).abs().max()) > 1e-6
    assert float((paper.weights - code.weights).abs().max()) > 0        # different mixture
    with pytest.raises(ab.ConfigError, match="cost must be one of"):
        ab.build("espformer", cost="cosine")


# ===========================================================================
# unbalanced_sinkhorn -- eq:ot-unbalanced, the column constraint as a dial
# ===========================================================================
def test_a_zero_column_penalty_is_exactly_softmax():
    """eq:ot-unbalanced at ``lam_col = 0`` is the semi-relaxed problem of
    prop:ot-gibbs: ``psi = 0``, and softmax is recovered, not approximated."""
    q, k, v = qkv(seed=7, scale=1.4)
    ref = ab.build("softmax")(q, k, v, return_weights=True)
    out = ab.build("unbalanced_sinkhorn", lam_col=0.0, n_iter=25)(q, k, v,
                                                                  return_weights=True)
    assert torch.allclose(out.weights, ref.weights, atol=1e-14)
    assert torch.allclose(out.y, ref.y, atol=1e-14)
    assert float(out.extras["g"].abs().max()) == 0.0


def test_the_column_penalty_interpolates_between_softmax_and_the_balanced_plan():
    """sec:ot: ``lam_col`` is "a single knob between sinks allowed and sinks
    forbidden".  Moving it must monotonically buy column balance and monotonically
    spend fidelity to softmax, and the far end must reach ``sinkhorn``."""
    q, k, v = qkv(m=16, n=16, seed=8, scale=1.4)
    soft = ab.build("softmax")(q, k, v, return_weights=True)
    cols, tvs = [], []
    lams = [0.0, 0.25, 1.0, 4.0, 64.0]
    for lam in lams:
        out = ab.build("unbalanced_sinkhorn", lam_col=lam, n_iter=300)(q, k, v,
                                                                       return_weights=True)
        cols.append(float((out.extras["col_marginal"] - 1 / 16).abs().max()))
        tvs.append(float(M.total_variation(soft.weights, out.weights).mean()))
    assert all(p > q for p, q in zip(cols, cols[1:])), cols
    assert all(p < q for p, q in zip(tvs, tvs[1:])), tvs
    assert tvs[0] == pytest.approx(0.0, abs=1e-14)
    hard = ab.build("unbalanced_sinkhorn", lam_col=1e6, n_iter=300)(q, k, v,
                                                                    return_weights=True)
    bal = ab.build("sinkhorn", n_iter=300)(q, k, v, return_weights=True)
    assert float((hard.weights - bal.weights).abs().max()) < 1e-6


def test_the_damping_factor_is_chizats_and_the_row_constraint_stays_hard():
    """Chizat et al.: each potential update is multiplied by ``lam / (lam + eps)``."""
    q, k, v = qkv(m=10, n=10, seed=9)
    for lam, eps in ((1.0, 1.0), (3.0, 0.5), (0.0, 2.0)):
        out = ab.build("unbalanced_sinkhorn", lam_col=lam, eps=eps, n_iter=300)(
            q, k, v, return_weights=True)
        e = out.extras
        assert e["rho_col"] == pytest.approx(lam / (lam + eps))
        assert e["rho_row"] == 1.0                      # lam_row = None means infinity
        # the conditional is a probability measure at every iteration ...
        assert torch.allclose(out.weights.sum(-1), torch.ones_like(out.weights[..., 0]),
                              atol=1e-14)
        # ... and at convergence the plan's own row marginal is the prescribed one
        assert torch.allclose(e["row_marginal"],
                              torch.full_like(e["row_marginal"], 1 / 10), atol=1e-12)
    soft = ab.build("unbalanced_sinkhorn", lam_row=1.0, eps=1.0, n_iter=300)(
        q, k, v, return_weights=True).extras
    assert soft["rho_row"] == pytest.approx(0.5)
    assert float((soft["row_marginal"] - 1 / 10).abs().max()) > 1e-6   # no longer exact


def test_the_relaxed_conditional_is_still_softmax_tilted_by_a_key_potential():
    """prop:ot-conditional survives the relaxation: the plan is still ``exp(s + f + g)``,
    so the tilt is still one number per key, independent of the query.  Only its size
    changes with ``lam_col`` -- which is what makes the family a continuum."""
    q, k, v = qkv(b=1, h=1, m=14, n=14, seed=10, scale=1.2)
    s = logits(q, k)
    spreads = []
    for lam in (0.5, 2.0, 50.0):
        out = ab.build("unbalanced_sinkhorn", lam_col=lam, n_iter=300)(q, k, v,
                                                                       return_weights=True)
        resid = torch.log(out.extras["gamma"]) - s
        centred = resid - resid.mean(-1, keepdim=True)
        assert torch.allclose(centred, centred[..., :1, :].expand_as(centred), atol=1e-10)
        psi = centred[..., 0, :]
        assert torch.allclose(torch.softmax(s + psi.unsqueeze(-2), -1), out.weights,
                              atol=1e-10)
        spreads.append(float((psi - psi.mean()).abs().max()))
    assert all(p < q for p, q in zip(spreads, spreads[1:])), spreads


def test_the_penalised_divergence_is_what_shrinks():
    """``lam_col KL(gamma_X || mubar)`` is the term being paid for; raising the price
    must lower the quantity bought."""
    q, k, v = qkv(m=12, n=12, seed=11, scale=1.5)
    kls = [float(ab.build("unbalanced_sinkhorn", lam_col=lam, n_iter=200)(
        q, k, v, return_weights=True).extras["col_kl"].max())
        for lam in (0.0, 0.5, 2.0, 16.0, 1e3)]
    assert all(p > q for p, q in zip(kls, kls[1:])), kls
    assert kls[-1] < 1e-6 < kls[0]


def test_unbalanced_declares_the_one_exact_limit_in_the_family():
    cls = ab.get_class("unbalanced_sinkhorn")
    assert cls.objective is ab.Objective.MODIFY          # a different operator in general
    assert cls.limit_config() == {"lam_col": 0.0, "eps": 1.0}
    assert cls.convergence_configs() is None


# ===========================================================================
# dustbin_sinkhorn -- SuperGlue's partial assignment, read as attention
# ===========================================================================
def test_the_augmented_plan_meets_the_superglue_marginals():
    """eq. 9 of Sarlin et al.: ``a = [1_m, n]/(m+n)``, ``b = [1_n, m]/(m+n)``, so both
    sides carry unit mass and the bin holds exactly the surplus."""
    q, k, v = qkv(m=7, n=11, seed=12, scale=1.2)
    e = ab.build("dustbin_sinkhorn", n_iter=400)(q, k, v, return_weights=True).extras
    gb, total = e["gamma_bar"], 7 + 11
    assert gb.shape[-2:] == (8, 12) and (gb >= 0).all()
    want_a = torch.cat([torch.ones(7), torch.tensor([11.0])]).double() / total
    want_b = torch.cat([torch.ones(11), torch.tensor([7.0])]).double() / total
    assert torch.allclose(gb.sum(-1), want_a.expand_as(gb[..., 0]), atol=1e-9)
    assert torch.allclose(gb.sum(-2), want_b.expand_as(gb[..., 0, :]), atol=1e-12)
    assert float(gb.sum((-1, -2)).max() - 1.0) < 1e-9


def test_a_deep_dustbin_recovers_the_balanced_coupling():
    """``z -> -inf``: nothing is worth abstaining for, the surplus goes to the bin-to-bin
    cell, and the real block is the doubly stochastic plan of ``sinkhorn``.

    The abstention vanishes immediately, but the *plan* approaches the balanced one only
    slowly: a deep bin drives the augmented matrix out of total support (Sinkhorn, 1967),
    and the scaling loses its geometric rate.  The measured decay is ``1/n_iter``.  That
    is the real behaviour of the limit, so it is what is asserted.
    """
    q, k, v = qkv(m=10, n=10, seed=13, scale=1.2)
    bal = ab.build("sinkhorn", n_iter=2000)(q, k, v, return_weights=True)
    mass = []
    for z in (0.0, -5.0, -15.0, -40.0):
        out = ab.build("dustbin_sinkhorn", dustbin=z, n_iter=400)(q, k, v,
                                                                  return_weights=True)
        mass.append(float(out.extras["dustbin_mass"].max()))
    assert all(p > q for p, q in zip(mass, mass[1:])), mass
    assert mass[-1] < 1e-12, mass

    gaps = [float((ab.build("dustbin_sinkhorn", dustbin=-40.0, n_iter=it)(
        q, k, v, return_weights=True).weights - bal.weights).abs().max())
        for it in (200, 600, 1800, 5400)]
    assert all(p > 2.0 * q for p, q in zip(gaps, gaps[1:])), gaps   # ~ 1 / n_iter
    assert gaps[-1] < 1e-4, gaps


def test_the_dustbin_is_abstention_and_is_reported_not_hidden():
    """The renormalization that turns a partial assignment into an attention row throws
    mass away.  ``extras['dustbin_mass']`` is exactly what it throws away."""
    q, k, v = qkv(m=9, n=9, seed=14, scale=1.2)
    prev = -1.0
    for z in (-4.0, -1.0, 0.0, 2.0, 5.0):
        out = ab.build("dustbin_sinkhorn", dustbin=z, n_iter=300)(q, k, v,
                                                                  return_weights=True)
        e = out.extras
        cur = float(e["dustbin_mass"].mean())
        assert cur > prev, (z, cur, prev)
        prev = cur
        assert (e["dustbin_mass"] >= 0).all() and (e["dustbin_mass"] < 1).all()
        # `weights` is the augmented conditional with the bin dropped and renormalized
        s = logits(q, k)
        bin_col = (e["g_bin"] + z).reshape(e["g_bin"].shape + (1, 1)).expand_as(s[..., :1])
        full = torch.softmax(torch.cat([s + e["g"].unsqueeze(-2), bin_col], -1), dim=-1)
        assert torch.allclose(full[..., -1], e["dustbin_mass"], atol=1e-12)
        assert torch.allclose(full[..., :-1] / (1 - full[..., -1:]), out.weights,
                              atol=1e-12)
    assert prev > 0.5, prev


def test_the_dustbin_conditional_is_softmax_tilted_by_the_restricted_potential():
    """prop:ot-conditional, read on the real block: the bin changes ``psi``, not the
    form of the conditional."""
    q, k, v = qkv(b=1, h=1, m=12, n=12, seed=15, scale=1.2)
    s = logits(q, k)
    out = ab.build("dustbin_sinkhorn", dustbin=0.5, n_iter=400)(q, k, v,
                                                                return_weights=True)
    psi = out.extras["potential"]
    assert torch.allclose(torch.softmax(s + psi.unsqueeze(-2), -1), out.weights, atol=1e-12)
    resid = torch.log(out.extras["gamma"]) - s
    centred = resid - resid.mean(-1, keepdim=True)
    assert torch.allclose(centred, centred[..., :1, :].expand_as(centred), atol=1e-10)


def test_abstention_buys_back_the_sinks_the_column_constraint_forbids():
    """sec:ot calls the dustbin "a natural way to retain abstention".  Concretely: the
    real block's column mass is free to be uneven again, between softmax's spread and
    the balanced plan's zero."""
    q, k, v = qkv(b=1, h=1, m=16, n=16, seed=16, scale=1.6)
    def spread(w):
        return float((w.mean(-2) - 1 / 16).abs().max())
    soft = spread(ab.build("softmax")(q, k, v, return_weights=True).weights)
    bal = spread(ab.build("sinkhorn", n_iter=400)(q, k, v, return_weights=True).weights)
    bin_ = spread(ab.build("dustbin_sinkhorn", dustbin=3.0, n_iter=400)(
        q, k, v, return_weights=True).weights)
    assert bal < bin_ < soft, (bal, bin_, soft)


# ===========================================================================
# OTKE -- transport onto a trainable reference
# ===========================================================================
def test_the_key_to_reference_plan_is_balanced_and_the_keys_get_their_share():
    """OTKE's constraint is between the set and the reference, not between queries and
    keys: every key receives exactly ``1/n`` of the reference's total attention."""
    q, k, v = qkv(m=14, n=14, seed=17, scale=1.2)
    out = ab.build("otke", p=5, n_iter=400)(q, k, v, return_weights=True)
    e = out.extras
    assert torch.allclose(e["row_marginal"], torch.full_like(e["row_marginal"], 1 / 5),
                          atol=1e-10)
    assert torch.allclose(e["col_marginal"], torch.full_like(e["col_marginal"], 1 / 14),
                          atol=1e-12)
    assert torch.allclose(e["slot_col_marginal"],
                          torch.full_like(e["slot_col_marginal"], 1 / 14), atol=1e-12)
    assert float(e["marginal_error"].max()) < 1e-10


def test_otke_is_a_pivot_factorization_of_rank_at_most_p():
    """def:factor-stochastic / prop:factor-closure: both halves are row-stochastic, so
    the operator is a probability measure by construction and has rank at most ``p``."""
    for p in (2, 4, 7):
        q, k, v = qkv(m=12, n=12, seed=18)
        out = ab.build("otke", p=p, n_iter=50)(q, k, v, return_weights=True)
        a, w = out.extras["a"], out.extras["w"]
        for f in (a, w):
            assert (f >= 0).all()
            assert torch.allclose(f.sum(-1), torch.ones_like(f[..., 0]), atol=1e-12)
        assert torch.allclose(a @ w, out.weights, atol=1e-12)
        assert int(torch.linalg.matrix_rank(out.weights).max()) <= p


def test_otke_constrains_the_reference_where_lowrank_ot_constrains_the_queries():
    """The contrast that puts both in ``coupling`` and keeps them distinct.  LOTFormer
    Sinkhorns both halves, so the *query* column marginal is uniform; OTKE Sinkhorns
    only the key half, so it is not -- the balance it enforces is one level in."""
    q, k, v = qkv(b=1, h=1, m=16, n=16, seed=19, scale=1.3)
    otke = ab.build("otke", p=4, n_iter=400)(q, k, v, return_weights=True)
    lot = ab.build("lowrank_ot", r=4, n_iter=400)(q, k, v, return_weights=True)
    assert float((lot.weights.mean(-2) - 1 / 16).abs().max()) < 1e-6
    assert float((otke.weights.mean(-2) - 1 / 16).abs().max()) > 1e-3
    assert float((otke.extras["slot_col_marginal"] - 1 / 16).abs().max()) < 1e-12


def test_the_slot_conditional_is_softmax_tilted_by_a_key_potential():
    """prop:ot-conditional on the coupling OTKE actually computes."""
    q, k, v = qkv(b=1, h=1, m=10, n=12, seed=20, scale=1.2)
    out = ab.build("otke", p=4, n_iter=400)(q, k, v, return_weights=True)
    e = out.extras
    s_ref = torch.einsum("pd,...nd->...pn", e["reference"], k) * k.shape[-1] ** -0.5
    assert torch.allclose(torch.softmax(s_ref + e["potential"].unsqueeze(-2), -1),
                          e["w"], atol=1e-12)
    resid = torch.log(e["gamma"]) - s_ref
    centred = resid - resid.mean(-1, keepdim=True)
    assert torch.allclose(centred, centred[..., :1, :].expand_as(centred), atol=1e-10)


def test_the_reference_is_the_method_and_is_declared_trainable():
    """Mialon et al. initialize ``Z`` by k-means and learn it end to end.  Untrained it
    is arbitrary, and the operator moves with it -- which is what ``trainable`` means."""
    q, k, v = qkv(m=10, n=10, seed=21)
    a = ab.build("otke", p=4, seed=0)(q, k, v, return_weights=True)
    b = ab.build("otke", p=4, seed=1)(q, k, v, return_weights=True)
    assert float((a.extras["reference"] - b.extras["reference"]).abs().max()) > 0.1
    assert float(M.total_variation(a.weights, b.weights).mean()) > 0.05
    assert ab.get_class("otke").trainable is True
    assert a.extras["reference"].shape == (4, 8)


# ===========================================================================
# family-level declarations
# ===========================================================================
@pytest.mark.parametrize("name", TRANSPORT)
def test_every_transport_coupling_is_an_operator_modification(name):
    """tab:comparative "mod.": fidelity to softmax is not the goal, so the conformance
    suite exempts them from the anchor -- and here the exemption is asserted, not
    assumed.  ``unbalanced_sinkhorn`` is the one that still declares an exact limit,
    because softmax is an interior point of its own parameter range."""
    cls = ab.get_class(name)
    assert cls.objective is ab.Objective.MODIFY
    assert cls.category is ab.Category.COUPLING
    if name == "unbalanced_sinkhorn":
        assert cls.limit_config() is not None
    else:
        assert cls.limit_config() is None and cls.convergence_configs() is None


@pytest.mark.parametrize("name", TRANSPORT)
def test_none_of_them_claims_to_be_causal(name):
    """prop:ot-causal: a lower-triangular doubly stochastic matrix is the identity.  The
    refusal is a statement about the operator, and the error message says so."""
    cls = ab.get_class(name)
    assert cls.supports_causal is False
    q, k, v = qkv()
    with pytest.raises(ValueError, match="statement about the operator"):
        ab.build(name)(q, k, v, causal=True)


@pytest.mark.parametrize("name", TRANSPORT)
def test_backends_are_declared_honestly(name):
    """Every one of these needs the full score matrix for its ``extras``; none is a
    dense softmax over a mask.  ``reference`` only is the correct answer."""
    assert ab.get_class(name).backends == ("reference",)


@pytest.mark.parametrize("name", GIBBS)
def test_the_gibbs_couplings_share_the_sinkhorn_extras_contract(name):
    """Study S7 reads ``potential``, ``f``, ``g``, ``gamma`` and the marginals off
    ``sinkhorn``.  Every method here whose plan is of Gibbs form exposes the same keys
    with the same meaning, so the same analysis runs on all of them."""
    q, k, v = qkv(m=10, n=10, seed=22)
    e = ab.build(name)(q, k, v, return_weights=True).extras
    for key in ("potential", "f", "g", "gamma", "row_marginal", "col_marginal",
                "target_row", "target_col", "marginal_error"):
        assert key in e, f"{name}: extras is missing {key!r}"
    assert torch.isfinite(e["potential"]).all()
    assert torch.allclose(e["gamma"].sum(-1), e["row_marginal"], atol=1e-14)
    assert torch.allclose(e["gamma"].sum(-2), e["col_marginal"], atol=1e-14)


@pytest.mark.parametrize("name", TRANSPORT)
def test_every_transport_coupling_differs_from_softmax_at_its_defaults(name):
    """They are in the bank to be different operators; a default that happened to be
    softmax would make the category vacuous."""
    q, k, v = qkv(m=12, n=12, seed=23, scale=1.4)
    ref = ab.build("softmax")(q, k, v, return_weights=True)
    out = ab.build(name)(q, k, v, return_weights=True)
    assert float(M.total_variation(ref.weights, out.weights).mean()) > 0.02


@pytest.mark.parametrize("name", TRANSPORT)
def test_the_module_docstring_names_the_section_and_the_source(name):
    cls = ab.get_class(name)
    import importlib
    doc = importlib.import_module(cls.__module__).__doc__
    assert cls.paper_section == "sec:ot"
    assert "sec:ot" in doc and "Reference:" in doc
    assert len(cls.references) >= 2
