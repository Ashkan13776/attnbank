"""Mathematics specific to the *deterministic* surrogates for g (sec:kernel).

``performer`` and ``trigrff`` approximate ``g`` by sampling features; the three
methods tested here approximate it by kernel algebra instead -- a PSD lift and a
Nystrom factorization (``skyformer``), a truncation of the logit spectrum
(``lowrank_logit``), and a truncation of the exponential series with an optional
sketch of each tensor power (``polysketch``).  What each one keeps of the contract,
and the regime in which it stops working, is what these tests pin down.
"""
from __future__ import annotations

import math

import pytest
import torch

import attnbank as ab
from attnbank.methods.interaction.polysketch import PolySketchAttention, tensor_sketch
from attnbank.methods.interaction.polysketch import countsketch_matrices


def qkv(b=1, h=1, m=8, n=12, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def mean_error(name, cfg, q, k, v, ref_y, seeds=range(6)):
    """Mean ``||y - y_softmax||`` over seeds; a deterministic method ignores them."""
    import inspect

    seeded = "seed" in inspect.signature(ab.get_class(name).__init__).parameters
    errs = []
    for s in seeds:
        out = ab.build(name, **({"seed": s} if seeded else {}), **cfg)(q, k, v)
        errs.append((out.y - ref_y).norm(dim=-1).mean().item())
    return sum(errs) / len(errs)


# ---------------------------------------------------------------------------
# skyformer: the point is the lift, not the formula (Chen et al., 2021)
# ---------------------------------------------------------------------------
def gaussian_lift(q, k, beta):
    """``Cbar = kappa([Q; K] / d^{1/4}, [Q; K] / d^{1/4})``, the matrix Skyformer lifts to."""
    z = torch.cat([q, k], dim=-2)
    d2 = torch.cdist(z, z) ** 2
    return torch.exp(-0.5 * beta * d2)


def test_the_lift_is_psd_while_the_matrix_nystromformer_factors_is_not():
    """The whole difference from ``nystrom``: the premise of the Nystrom bound holds."""
    q, k, _ = qkv(m=6, n=9)
    beta = 1 / math.sqrt(q.shape[-1])
    lift = gaussian_lift(q, k, beta)[0, 0]
    assert torch.allclose(lift, lift.T, atol=1e-12)
    assert float(torch.linalg.eigvalsh(lift).min()) > 0          # strictly PD, in fact

    # Nystromformer instead factors P itself, through a landmark core that is not even
    # symmetric -- so it is not PSD and no Nystrom guarantee applies to it.
    from attnbank.methods.interaction.nystrom import segment_means
    qt, kt = segment_means(q, 4), segment_means(k, 4)
    core = ab.build("softmax")(qt, kt, torch.zeros(*kt.shape[:-1], 1, dtype=kt.dtype),
                               return_weights=True).weights[0, 0]
    assert not torch.allclose(core, core.T, atol=1e-6)


def test_skyformer_is_exact_when_every_position_is_a_landmark():
    """``Cbar Cbar^+ Cbar = Cbar``: the correctness anchor, and it does not need m == n."""
    for m, n in ((16, 16), (5, 17), (1, 9)):
        q, k, v = qkv(m=m, n=n)
        ref = ab.build("softmax")(q, k, v, return_weights=True)
        for cfg in ({"r": None}, {"r": m + n}, {"r": 10 * (m + n)}):
            out = ab.build("skyformer", **cfg)(q, k, v, return_weights=True)
            assert torch.allclose(out.weights, ref.weights, atol=1e-10), (m, n, cfg)
            assert torch.allclose(out.y, ref.y, atol=1e-10), (m, n, cfg)


def test_skyformer_error_decreases_as_the_sub_sample_grows():
    q, k, v = qkv(m=12, n=12, d=16, scale=0.7)
    ref = ab.build("softmax")(q, k, v).y
    errs = [mean_error("skyformer", {"r": r}, q, k, v, ref) for r in (4, 8, 12, 20)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    assert mean_error("skyformer", {"r": None}, q, k, v, ref) < 1e-12


def test_skyformer_weights_are_signed_but_still_sum_to_one():
    """``nm`` survives the library's explicit normalization; ``hl`` does not."""
    worst = 0.0
    for seed in range(8):
        q, k, v = qkv(m=10, n=10, seed=seed, scale=2.0)
        out = ab.build("skyformer", r=6, seed=seed)(q, k, v, return_weights=True)
        sums = out.weights.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-10)
        worst = min(worst, float(out.weights.min()))
        assert float(out.extras["neg_mass"].max()) >= 0
    assert worst < 0, "expected a signed measure somewhere in this regime"


def test_kernelized_attention_equals_softmax_only_for_equinormal_keys():
    """Why this file normalizes: ``C = D_Q^{-1/2} A D_K^{-1/2}`` and only ``D_Q`` cancels.

    The paper's own operator is ``C V`` -- a different operator, by exactly the key-norm
    factor.  Make the keys equinormal and the difference disappears.
    """
    q, k, v = qkv(m=6, n=7)
    beta = 1 / math.sqrt(q.shape[-1])
    ref = ab.build("softmax")(q, k, v, return_weights=True).weights

    def kernelized(key):
        c = torch.exp(-0.5 * beta * torch.cdist(q, key) ** 2)
        return c / c.sum(-1, keepdim=True)

    assert not torch.allclose(kernelized(k), ref, atol=1e-3)
    k_eq = k / k.norm(dim=-1, keepdim=True)
    ref_eq = ab.build("softmax")(q, k_eq, v, return_weights=True).weights
    assert torch.allclose(kernelized(k_eq), ref_eq, atol=1e-10)


# ---------------------------------------------------------------------------
# lowrank_logit: low rank in the right space (Bhojanapalli et al., 2020)
# ---------------------------------------------------------------------------
def test_lowrank_logit_is_exact_at_the_head_dimension():
    """``rank(beta Q K^T) <= d``, so the truncation only bites below ``d``."""
    q, k, v = qkv(m=16, n=20, d=6)
    ref = ab.build("softmax")(q, k, v, return_weights=True)
    for r in (6, 7, 12, None):
        out = ab.build("lowrank_logit", r=r)(q, k, v, return_weights=True)
        assert torch.allclose(out.weights, ref.weights, atol=1e-12), r
    out = ab.build("lowrank_logit", r=3)(q, k, v, return_weights=True)
    assert not torch.allclose(out.weights, ref.weights, atol=1e-3)


def test_lowrank_logit_keeps_the_whole_contract_where_svd_oracle_loses_it():
    """Same rank, different space: ``exp`` after the truncation is still a measure."""
    q, k, v = qkv(m=12, n=12, d=10)
    mine = ab.build("lowrank_logit", r=3)(q, k, v, return_weights=True).weights
    theirs = ab.build("svd_oracle", r=3)(q, k, v, return_weights=True).weights
    assert (mine >= 0).all()
    assert torch.allclose(mine.sum(-1), torch.ones_like(mine[..., 0]), atol=1e-12)
    assert float(theirs.min()) < 0
    # and the outputs stay inside the convex hull of the values, which signed ones need not
    y = mine @ v
    assert bool((y <= v.amax(-2, keepdim=True) + 1e-12).all())
    assert bool((y >= v.amin(-2, keepdim=True) - 1e-12).all())


def test_lowrank_logit_error_decreases_with_rank():
    q, k, v = qkv(m=16, n=16, d=12, scale=1.2)
    ref = ab.build("softmax")(q, k, v).y
    errs = [mean_error("lowrank_logit", {"r": r}, q, k, v, ref, seeds=[0]) for r in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs


def test_a_low_rank_interaction_is_not_a_low_rank_attention_matrix():
    """``rank(S_r) = r`` says nothing about ``rank(exp(S_r))``; this is not sec:factor."""
    q, k, v = qkv(m=16, n=16, d=12, scale=1.5)
    out = ab.build("lowrank_logit", r=2)(q, k, v, return_weights=True)
    sv = torch.linalg.svdvals(out.weights[0, 0])
    assert int((sv > 1e-10 * sv[0]).sum()) > 2


# ---------------------------------------------------------------------------
# polysketch: a deterministic truncation, plus an optional sketch of each power
# ---------------------------------------------------------------------------
def test_the_exact_path_is_the_taylor_truncation_of_the_softmax_kernel():
    q, k, _ = qkv(m=5, n=6, scale=0.6)
    beta = 1 / math.sqrt(q.shape[-1])
    s = torch.einsum("...md,...nd->...mn", q, k) * beta
    for degree in (1, 2, 5):
        want = sum(s ** p / math.factorial(p) for p in range(degree + 1))
        got = PolySketchAttention(degree=degree, sketch=None)._surrogate(q, k, beta)
        assert torch.allclose(got, want, atol=1e-12), degree
    deep = PolySketchAttention(degree=24, sketch=None)._surrogate(q, k, beta)
    assert torch.allclose(deep, torch.exp(s), atol=1e-12)


def test_tensorsketch_is_unbiased_for_the_polynomial_kernel():
    """``E <TS_p(x), TS_p(y)> = <x, y>^p`` (Pham and Pagh, 2013)."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 6, generator=g, dtype=torch.float64) * 0.7
    y = torch.randn(4, 6, generator=g, dtype=torch.float64) * 0.7
    p, s = 3, 8
    est = []
    for seed in range(3000):
        mats = countsketch_matrices(p, 6, s, seed, x.device)
        est.append(tensor_sketch(x, mats) @ tensor_sketch(y, mats).T)
    est = torch.stack(est)
    want = (x @ y.T) ** p
    sem = est.std(0) / math.sqrt(est.shape[0])
    assert float(((est.mean(0) - want).abs() / sem.clamp_min(1e-12)).max()) < 4.5


def test_the_sketch_variance_falls_like_one_over_the_number_of_buckets():
    q, k, _ = qkv(m=4, n=4, d=6, scale=0.5)
    beta = 1 / math.sqrt(6)
    sds = []
    for s in (8, 32, 128):
        est = torch.stack([PolySketchAttention(degree=3, sketch=s, seed=i)._surrogate(q, k, beta)
                           for i in range(200)])
        sds.append(float(est.std(0).mean()))
    assert sds[0] > sds[1] > sds[2]
    for a, b in zip(sds, sds[1:]):
        assert 1.3 < a / b < 3.0, sds          # a factor of four in s, a factor of two in sd


def test_an_even_truncation_is_a_measure_and_an_odd_one_need_not_be():
    """``sum_{p<=P} s^p/p!`` has no real root for even ``P`` -- positivity, for free."""
    q, k, v = qkv(m=8, n=10, scale=2.0)
    for degree in (2, 4, 6):
        w = ab.build("polysketch", degree=degree, sketch=None)(
            q, k, v, return_weights=True).weights
        assert (w >= 0).all(), degree
        assert torch.allclose(w.sum(-1), torch.ones_like(w[..., 0]), atol=1e-12)
    odd = ab.build("polysketch", degree=3, sketch=None)(q, k, v, return_weights=True).weights
    assert float(odd.min()) < 0


def test_the_sketch_destroys_the_positivity_the_truncation_had():
    q, k, v = qkv(m=8, n=10, scale=1.5)
    worst = 0.0
    for seed in range(6):
        w = ab.build("polysketch", degree=4, sketch=8, seed=seed)(
            q, k, v, return_weights=True).weights
        worst = min(worst, float(w.min()))
    assert worst < 0, "row_stochastic = False is declared for this reason"


def test_the_unsketched_truncation_converges_to_softmax_in_the_degree():
    """The deterministic half of the error: no randomness, and it reaches the dtype."""
    q, k, v = qkv(m=10, n=10, scale=0.8)
    ref = ab.build("softmax")(q, k, v).y
    errs = [mean_error("polysketch", {"degree": p, "sketch": None}, q, k, v, ref, seeds=[0])
            for p in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < errs[0] / 1000, errs
    assert mean_error("polysketch", {"degree": 20, "sketch": None}, q, k, v, ref,
                      seeds=[0]) < 1e-12


def test_polysketch_converges_as_the_feature_budget_grows():
    """The declared anchor: ``(P, s)`` is one budget and neither half converges alone."""
    q, k, v = qkv(m=10, n=10, scale=0.5)
    ref = ab.build("softmax")(q, k, v).y
    errs = [mean_error("polysketch", cfg, q, k, v, ref, seeds=range(8))
            for cfg in PolySketchAttention.convergence_configs()]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs


def test_polysketch_is_not_invariant_to_a_shift_of_the_logits():
    """The caveat that matters: ``exp`` can be shifted before evaluation, a polynomial cannot.

    Appending ``(c, 1)`` to ``(q, k)`` adds ``beta c`` to every logit in a row, which
    softmax cannot see at all.
    """
    q, k, v = qkv(m=6, n=9, d=7, scale=1.0)
    c = torch.full((*q.shape[:-1], 1), 3.0, dtype=q.dtype)
    q_s = torch.cat([q, c], dim=-1)
    k_s = torch.cat([k, torch.ones(*k.shape[:-1], 1, dtype=k.dtype)], dim=-1)
    # the extra channel changes d, hence beta: pin the scale so only the shift moves
    beta = 1 / math.sqrt(q.shape[-1])
    ref = ab.build("softmax", scale=beta)(q, k, v).y
    ref_s = ab.build("softmax", scale=beta)(q_s, k_s, v).y
    assert torch.allclose(ref, ref_s, atol=1e-12)

    poly = ab.build("polysketch", degree=4, sketch=None, scale=beta)
    assert not torch.allclose(poly(q, k, v).y, poly(q_s, k_s, v).y, atol=1e-3)


@pytest.mark.parametrize("name", ["skyformer", "lowrank_logit", "polysketch"])
def test_the_surrogate_is_reported_honestly_in_extras(name):
    q, k, v = qkv()
    out = ab.build(name)(q, k, v, return_weights=True)
    assert out.extras, name
    assert torch.allclose(out.y, out.weights @ v, atol=1e-12)
