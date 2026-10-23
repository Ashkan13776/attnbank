"""Mathematics specific to the random-feature / fitted-feature-map surrogates.

``favorpp``, ``favorsharp``, ``based``, ``rebased``, ``polaformer``, ``hedgehog``.
``tests/test_conformance.py`` already checks that these obey the contract; what is
checked here is that they are the estimators their papers describe -- unbiased for
the exponential kernel, ordered the way the variance theory says, positive where
positivity is claimed, and converging where an anchor claims convergence.
"""
from __future__ import annotations

import math

import pytest
import torch

import attnbank as ab
from attnbank.methods.interaction.based import taylor_exp
from attnbank.methods.interaction.favorpp import optimal_a, pair_scale
from attnbank.methods.interaction.polaformer import polarity_features
from attnbank.methods.interaction.rebased import rebased_feature

RF = ["performer", "favorpp", "favorsharp"]


def qkv(b=1, h=1, m=8, n=12, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def true_g(q, k):
    """``g(q, x) = exp(<qt, kt_x>)`` with ``qt = q / d^{1/4}``."""
    return torch.exp(torch.einsum("...md,...nd->...mn", q, k) / math.sqrt(q.shape[-1]))


def single_feature_moments(name, q, k, draws=4000):
    """Mean and variance of the one-feature estimator over ``draws`` seeds."""
    zero = torch.zeros(*k.shape[:-1], 1, dtype=q.dtype)
    est = torch.stack([
        torch.exp(ab.build(name, r=1, orthogonal=False, seed=s)(q, k, zero)
                  .extras["log_g_hat"])
        for s in range(draws)])
    return est.mean(0), est.var(0)


# ---------------------------------------------------------------------------
# The generalized-exponential family: unbiased for every A, with the stated order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", RF)
def test_generalized_exponential_features_are_unbiased(name):
    """Thm. 3.1 / 4.1: B, C, D are chosen so that E[f(w,qt) f(w,kt)] = exp(<qt,kt>)."""
    q, k, _ = qkv(m=3, n=3, scale=0.4)
    mean, _ = single_feature_moments(name, q, k)
    rel = (mean - true_g(q, k)).abs() / true_g(q, k)
    assert float(rel.max()) < 0.12, (name, float(rel.max()))


def test_variance_is_ordered_favor_plus_then_pp_then_sharp():
    """The point of the family: each optimisation strictly reduces relative variance.

    FAVOR+ fixes ``A = 0``; FAVOR++ optimises one scalar ``A``; FAVOR# optimises a
    diagonal in the eigenbasis of the pair covariance, of which the scalar is the
    isotropic special case -- so the inequality must be the nesting order.
    """
    q, k, _ = qkv(m=3, n=3, scale=1.0)
    g2 = true_g(q, k) ** 2
    relvar = {}
    for name in RF:
        _, var = single_feature_moments(name, q, k, draws=6000)
        relvar[name] = float((var / g2).mean())
    assert relvar["favorsharp"] < relvar["favorpp"] < relvar["performer"], relvar


def test_favorpp_at_a_zero_is_exactly_performer():
    """``A = 0`` gives B = D = 1: the family contains FAVOR+ verbatim, bit for bit."""
    q, k, v = qkv()
    a = ab.build("performer", r=32, seed=3)(q, k, v, return_weights=True)
    b = ab.build("favorpp", r=32, seed=3, a=0.0)(q, k, v, return_weights=True)
    assert torch.equal(a.y, b.y)
    assert torch.equal(a.weights, b.weights)


def test_favorsharp_reduces_to_favorpp_when_the_pair_covariance_is_isotropic():
    """``tr(Sigma_i) = S_i``, so FAVOR++ is FAVOR# with every eigenvalue at the mean."""
    q, k, v = qkv(m=4, n=6)
    d = q.shape[-1]
    qt, kt = q / d ** 0.25, k / d ** 0.25
    lam = ab.build("favorsharp", r=4)(q, k, v).extras["eigenvalues"]
    assert torch.allclose(lam.sum(-1), pair_scale(qt, kt, q.shape[-2], causal=False),
                          atol=1e-10)


def test_optimal_a_is_zero_at_the_origin_and_decreasing():
    """``A*(0) = 0`` recovers FAVOR+; variance reduction only ever shrinks ``A``."""
    lam = torch.linspace(0.0, 50.0, 64, dtype=torch.float64)
    a = optimal_a(lam)
    assert abs(float(a[0])) < 1e-15
    assert bool((a[1:] < a[:-1]).all())
    assert bool((a <= 0).all()) and bool((1 - 4 * a > 0).all())


@pytest.mark.parametrize("name", ["favorpp", "favorsharp"])
def test_positive_features_keep_a_probability_measure_where_trig_loses_it(name):
    """``prop:error-additive``: positivity survives the logit scale that breaks trigrff."""
    q, k, v = qkv(scale=1.5)
    out = ab.build(name, r=8, seed=0)(q, k, v, return_weights=True)
    assert bool((out.weights >= 0).all())
    assert torch.allclose(out.weights.sum(-1), torch.ones_like(out.weights[..., 0]),
                          atol=1e-10)
    trig = ab.build("trigrff", r=8, seed=0)(q, k, v, return_weights=True)
    assert bool((trig.weights < 0).any())


def test_variance_reduction_grows_with_the_logit_scale():
    """FAVOR+ is optimal only as ``||qt + kt|| -> 0``; the gap is what grows."""
    gaps = []
    for scale in (0.25, 1.0):
        q, k, _ = qkv(m=2, n=2, scale=scale)
        g2 = true_g(q, k) ** 2
        v0 = float((single_feature_moments("performer", q, k, 4000)[1] / g2).mean())
        v1 = float((single_feature_moments("favorpp", q, k, 4000)[1] / g2).mean())
        gaps.append(v0 / v1)
    assert gaps[1] > gaps[0] > 1.0, gaps


# ---------------------------------------------------------------------------
# Based: the truncated exponential, and the floor it cannot get under
# ---------------------------------------------------------------------------
def test_based_is_the_second_order_taylor_polynomial():
    q, k, v = qkv()
    z = torch.einsum("...md,...nd->...mn", q, k) / math.sqrt(q.shape[-1])
    g_hat = ab.build("based")(q, k, v).extras["g_hat"]
    assert torch.allclose(g_hat, 1 + z + z ** 2 / 2, atol=1e-12)


def test_even_truncations_of_exp_are_strictly_positive_and_odd_ones_are_not():
    """Why ``degree`` must be even: it is exactly the condition for ``g_hat > 0``."""
    z = torch.linspace(-30.0, 10.0, 400, dtype=torch.float64)
    for even in (2, 4, 6):
        assert bool((taylor_exp(z, even) > 0).all())
    for odd in (1, 3, 5):
        assert bool((taylor_exp(z, odd) < 0).any())
    with pytest.raises(ab.ConfigError, match="even"):
        ab.build("based", degree=3)


def test_based_converges_to_softmax_with_the_truncation_order():
    q, k, v = qkv(scale=0.5)
    ref = ab.build("softmax")(q, k, v).y
    errs = [float((ab.build("based", degree=p)(q, k, v).y - ref).norm())
            for p in (2, 4, 6, 8, 10)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs


def test_based_cannot_assign_a_near_zero_weight_but_rebased_can():
    """The ``1/2`` floor of ``1 + z + z^2/2`` is exactly what ReBased removes."""
    z = torch.linspace(-50.0, 0.0, 501, dtype=torch.float64)
    assert float(taylor_exp(z, 2).min()) >= 0.5
    assert float(taylor_exp(torch.tensor([-1.0], dtype=torch.float64), 2)) == 0.5
    # a movable root: beta = -gamma * norm(x)_c makes the ReBased feature vanish
    x = torch.tensor([[-1.0, 1.0, -1.0, 1.0]], dtype=torch.float64)
    nx = torch.nn.functional.layer_norm(x, x.shape[-1:], eps=1e-5)
    assert float(rebased_feature(x, 1.0, -float(nx[0, 0])).min()) < 1e-8


def test_rebased_cannot_sharpen_with_the_query_norm():
    """``rem:kernel-homogeneous`` holds exactly: LayerNorm discards ``||q||``."""
    q, k, v = qkv()
    base = ab.build("rebased")(q, k, v, return_weights=True).weights
    for factor in (0.1, 10.0):
        scaled = ab.build("rebased")(factor * q, k, v, return_weights=True).weights
        assert torch.allclose(base, scaled, atol=1e-8)
    soft = ab.build("softmax")(q, k, v, return_weights=True).weights
    sharp = ab.build("softmax")(10.0 * q, k, v, return_weights=True).weights
    assert float(sharp.max()) > float(soft.max()) + 0.1       # softmax does sharpen


# ---------------------------------------------------------------------------
# PolaFormer: two measures, and why no single one exists
# ---------------------------------------------------------------------------
def test_polaformer_recovers_the_opposite_sign_interaction_relu_deletes():
    """``<q,k> = (same) - (opposite)``: the split is exact before the power is applied."""
    q, k, _ = qkv(m=4, n=4)
    qp, qn = polarity_features(q, 1.0)
    kp, kn = polarity_features(k, 1.0)
    same = qp @ kp.transpose(-1, -2) + qn @ kn.transpose(-1, -2)
    opp = qn @ kp.transpose(-1, -2) + qp @ kn.transpose(-1, -2)
    assert torch.allclose(same - opp, q @ k.transpose(-1, -2), atol=1e-12)
    assert float(opp.min()) >= 0.0 and float(opp.max()) > 0.0   # relu features drop this


def test_polaformer_output_is_not_an_average_against_a_single_measure():
    """Invariant W is declined for a reason: the two halves of ``v`` see different pi."""
    q, k, _ = qkv(m=5, n=7, dv=4)
    u = torch.arange(7.0, dtype=torch.float64).reshape(1, 1, 7, 1)
    v = torch.cat([u, u], dim=-1)                     # both halves identical
    out = ab.build("polaformer")(q, k, v, return_weights=True)
    assert out.weights is None
    half = v.shape[-1] // 2
    assert not torch.allclose(out.y[..., :half], out.y[..., half:], atol=1e-6)
    pi_same = out.extras["weights_same_sign"]
    pi_opp = out.extras["weights_opposite_sign"]
    for pi in (pi_same, pi_opp):                      # each half IS a probability measure
        assert bool((pi >= 0).all())
        assert torch.allclose(pi.sum(-1), torch.ones_like(pi[..., 0]), atol=1e-12)
    assert not torch.allclose(pi_same, pi_opp, atol=1e-6)


def test_polaformer_power_sharpens_the_same_sign_measure():
    """``p > 1`` is their entropy-reduction condition; higher ``p``, lower entropy."""
    q, k, v = qkv()
    ent = []
    for power in (1.0, 3.0, 6.0):
        pi = ab.build("polaformer", power=power)(q, k, v).extras["weights_same_sign"]
        ent.append(float(-(pi * pi.clamp_min(1e-300).log()).sum(-1).mean()))
    assert all(a > b for a, b in zip(ent, ent[1:])), ent


# ---------------------------------------------------------------------------
# Hedgehog: the fit is what makes it work
# ---------------------------------------------------------------------------
def test_hedgehog_distillation_lowers_the_cross_entropy_it_minimises():
    q, k, v = qkv(m=6, n=6)
    ce = [float(ab.build("hedgehog", r=4, steps=s, seed=0)(q, k, v)
                .extras["cross_entropy"].mean())
          for s in (0, 20, 100)]
    assert all(a > b for a, b in zip(ce, ce[1:])), ce


def test_a_fitted_feature_map_beats_the_fixed_ones_it_replaces():
    """The claim the method exists to make: ``phi`` chosen by fitting, not by hand."""
    q, k, v = qkv(m=6, n=6)
    ref = ab.build("softmax")(q, k, v).y
    err = lambda out: float((out - ref).norm())
    fitted = err(ab.build("hedgehog", r=8, steps=200, seed=0)(q, k, v).y)
    assert fitted < err(ab.build("linear_elu")(q, k, v).y)
    assert fitted < err(ab.build("hedgehog", r=8, steps=0, seed=0)(q, k, v).y)
