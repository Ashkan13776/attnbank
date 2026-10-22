"""Mathematics specific to ``factorize/``: prop:factor-closure and prop:factor-rank."""
from __future__ import annotations

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M

STOCHASTIC = ["latent_stochastic", "clustered", "nmf_oracle"]


def qkv(b=2, h=2, m=16, n=16, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def softmax_out(q, k, v):
    return ab.build("softmax")(q, k, v, return_weights=True)


# ---------------------------------------------------------------------------
# prop:factor-closure: a composition of Markov kernels preserves the contract
# ---------------------------------------------------------------------------
def test_latent_factors_are_both_row_stochastic():
    """def:factor-stochastic: A 1 = 1, B 1 = 1, A, B >= 0, and P_hat = A B."""
    q, k, v = qkv()
    out = ab.build("latent_stochastic", r=5)(q, k, v, return_weights=True)
    a, b = out.extras["a"], out.extras["b"]
    for f in (a, b):
        assert (f >= 0).all()
        assert torch.allclose(f.sum(-1), torch.ones_like(f[..., 0]), atol=1e-12)
    assert torch.allclose(a @ b, out.weights, atol=1e-12)


@pytest.mark.parametrize("name", STOCHASTIC + ["lowrank_ot"])
def test_closure_constant_preservation_and_convex_hull(name):
    """prop:factor-closure (i)-(iii)."""
    q, k, v = qkv()
    out = ab.build(name)(q, k, v, return_weights=True)
    assert (out.weights >= -1e-12).all()
    assert torch.allclose(out.weights.sum(-1), torch.ones_like(out.weights[..., 0]),
                          atol=1e-10)
    # (ii) a constant value field is reproduced exactly
    c = torch.full_like(v, 1.7)
    y_const = ab.build(name)(q, k, c).y
    assert torch.allclose(y_const, torch.full_like(y_const, 1.7), atol=1e-9)
    # (iii) the output stays in the convex hull of the values
    assert (out.y >= v.amin(-2, keepdim=True) - 1e-9).all()
    assert (out.y <= v.amax(-2, keepdim=True) + 1e-9).all()


def test_total_variation_contracts_through_the_write_kernel():
    """prop:factor-closure (iv): TV(a b, a' b) <= TV(a, a') -- data processing."""
    q, k, v = qkv()
    out = ab.build("latent_stochastic", r=5)(q, k, v, return_weights=True)
    b = out.extras["b"]
    g = torch.Generator().manual_seed(1)
    a1 = torch.softmax(torch.randn(*out.extras["a"].shape, generator=g, dtype=v.dtype), -1)
    a2 = torch.softmax(torch.randn(*out.extras["a"].shape, generator=g, dtype=v.dtype), -1)
    assert (M.total_variation(a1 @ b, a2 @ b) <= M.total_variation(a1, a2) + 1e-12).all()


# ---------------------------------------------------------------------------
# prop:factor-rank: rank <= rank_+ = rank_s, so the signed oracle is a lower bound
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r", [1, 2, 4, 8])
def test_svd_oracle_is_eckart_young_optimal(r):
    """The Frobenius residual equals the tail of the singular values."""
    q, k, v = qkv(seed=2)
    pi = softmax_out(q, k, v).weights
    out = ab.build("svd_oracle", r=r)(q, k, v, return_weights=True)
    s = out.extras["singular_values"]
    residual = (pi - out.weights).flatten(-2).norm(dim=-1)
    assert torch.allclose(residual, (s[..., r:] ** 2).sum(-1).sqrt(), atol=1e-9)


@pytest.mark.parametrize("r", [1, 2, 3, 4, 6])
def test_nonnegative_rank_costs_something(r):
    """The signed oracle is optimal, so the nonnegative one can never beat it; the gap
    is the empirical nonnegative-rank penalty of study S5."""
    q, k, v = qkv(seed=3, scale=1.2)
    pi = softmax_out(q, k, v).weights
    svd = ab.build("svd_oracle", r=r)(q, k, v, return_weights=True)
    nmf = ab.build("nmf_oracle", r=r, iters=400, restarts=4)(q, k, v, return_weights=True)
    e_svd = (pi - svd.weights).flatten(-2).norm(dim=-1)
    e_nmf = (pi - nmf.extras["raw"]).flatten(-2).norm(dim=-1)
    assert (e_svd <= e_nmf + 1e-9).all(), (r, e_svd.max(), e_nmf.max())


def test_the_nonnegative_gap_is_strict_for_a_rank_three_example():
    q, k, v = qkv(b=1, h=1, m=12, n=12, seed=4, scale=1.5)
    pi = softmax_out(q, k, v).weights
    r = 3
    svd = ab.build("svd_oracle", r=r)(q, k, v, return_weights=True)
    nmf = ab.build("nmf_oracle", r=r, iters=800, restarts=6)(q, k, v, return_weights=True)
    e_svd = float((pi - svd.weights).flatten(-2).norm(dim=-1).mean())
    e_nmf = float((pi - nmf.extras["raw"]).flatten(-2).norm(dim=-1).mean())
    assert e_nmf > e_svd * 1.001, (e_svd, e_nmf)


def test_svd_oracle_breaks_the_contract_that_nmf_keeps():
    """The price of the lower bound: signed weights and outputs outside the hull."""
    q, k, v = qkv(seed=5, scale=1.5)
    svd = ab.build("svd_oracle", r=2)(q, k, v, return_weights=True)
    nmf = ab.build("nmf_oracle", r=2)(q, k, v, return_weights=True)
    assert bool((svd.weights < -1e-6).any())
    assert float((svd.extras["row_sum"] - 1).abs().max()) > 1e-6
    assert (nmf.weights >= -1e-12).all()
    assert torch.allclose(nmf.weights.sum(-1), torch.ones_like(nmf.weights[..., 0]),
                          atol=1e-10)
    # constant preservation, the sharpest form of the convex-hull property: a constant
    # value field must come back unchanged, and for a signed factorization it does not
    const = torch.full_like(v, 1.0)
    y_svd = ab.build("svd_oracle", r=2)(q, k, const).y
    y_nmf = ab.build("nmf_oracle", r=2)(q, k, const).y
    assert float((y_svd - 1.0).abs().max()) > 1e-6, "expected constant preservation to fail"
    assert torch.allclose(y_nmf, torch.ones_like(y_nmf), atol=1e-9)
    # and so the output leaves the convex hull of the (single) value
    assert bool(((y_svd < 1.0 - 1e-6) | (y_svd > 1.0 + 1e-6)).any())
    lo, hi = v.amin(-2, keepdim=True), v.amax(-2, keepdim=True)
    assert ((nmf.y >= lo - 1e-9) & (nmf.y <= hi + 1e-9)).all()


@pytest.mark.parametrize("name", ["svd_oracle", "nmf_oracle", "clustered"])
def test_error_decreases_with_rank(name):
    q, k, v = qkv(seed=6)
    pi = softmax_out(q, k, v).weights
    errs = [float(M.total_variation(pi, ab.build(name, r=r)(
        q, k, v, return_weights=True).weights).mean()) for r in (1, 2, 4, 8, 16)]
    assert errs[0] > errs[-1], errs
    assert errs[-1] < 1e-6


def test_nmf_restarts_do_not_make_the_fit_worse():
    q, k, v = qkv(b=1, h=1, seed=7)
    pi = softmax_out(q, k, v).weights
    e = [float((pi - ab.build("nmf_oracle", r=3, restarts=n, iters=300)(
        q, k, v, return_weights=True).extras["raw"]).flatten(-2).norm(dim=-1).mean())
        for n in (1, 2, 6)]
    assert e[-1] <= e[0] + 1e-12, e


# ---------------------------------------------------------------------------
# clustered: the factorization is the cluster assignment
# ---------------------------------------------------------------------------
def test_clustered_is_a_rank_r_stochastic_factorization():
    q, k, v = qkv(seed=8)
    out = ab.build("clustered", r=4)(q, k, v, return_weights=True)
    assign, b = out.extras["assign"], out.extras["b"]
    a = torch.nn.functional.one_hot(assign, b.shape[-2]).to(v.dtype)   # [..., m, r]
    assert torch.allclose(a.sum(-1), torch.ones_like(a[..., 0]), atol=1e-12)
    assert torch.allclose(a @ b, out.weights, atol=1e-12)
    assert torch.linalg.matrix_rank(out.weights.double()).max() <= 4


def test_clustered_shares_one_row_within_a_cluster():
    q, k, v = qkv(seed=9)
    out = ab.build("clustered", r=3)(q, k, v, return_weights=True)
    assign = out.extras["assign"]
    for bb in range(assign.shape[0]):
        for hh in range(assign.shape[1]):
            for c in range(3):
                rows = out.weights[bb, hh][assign[bb, hh] == c]
                if rows.shape[0] > 1:
                    assert torch.allclose(rows, rows[:1].expand_as(rows), atol=1e-12)


def test_clustered_error_is_controlled_by_the_distance_to_the_centroid():
    """prop:error-score with delta = beta ||q - c|| max_j ||k_j||."""
    q, k, v = qkv(b=1, h=1, m=16, n=16, seed=10)
    out = ab.build("clustered", r=4)(q, k, v, return_weights=True)
    ref = softmax_out(q, k, v)
    beta = q.shape[-1] ** -0.5
    delta = beta * out.extras["centroid_dist"] * k.norm(dim=-1).amax(-1, keepdim=True)
    bound = torch.tanh(delta / 2) * M.value_oscillation(v).unsqueeze(-1)
    assert ((ref.y - out.y).norm(dim=-1) <= bound + 1e-9).all()


def test_latent_stochastic_is_declared_an_operator_modification():
    """tab:comparative: 'Stochastic latent factorization (Luna, Latte) ... mod.'"""
    cls = ab.get_class("latent_stochastic")
    assert cls.objective is ab.Objective.MODIFY
    assert cls.limit_config() is None and cls.convergence_configs() is None
    assert cls.trainable is True
