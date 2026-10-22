"""Numerical proof of every identity and every bound in ``attnbank.metrics``.

The rule for this file: an *identity* is checked to machine precision in float64, and a
*bound* is checked to actually bound the thing it claims to bound over randomized
inputs -- not merely to be computable.  Each test names the proposition it realizes.
"""
from __future__ import annotations

import math

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M

TOL = 1e-10


def rand(*shape, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype)


def qkv(b=2, h=3, m=10, n=16, d=8, dv=5, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: scale * torch.randn(*s, generator=g, dtype=torch.float64)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def softmax_pi(q, k, causal=False):
    return ab.build("softmax")(q, k, torch.zeros(*k.shape[:-1], 1, dtype=k.dtype),
                               causal=causal, return_weights=True).weights


def rand_pi(shape, seed=0):
    return torch.softmax(rand(*shape, seed=seed), dim=-1)


# ---------------------------------------------------------------------------
# Decompositions (eq:error_decomposition, eq:error-decomp-sym)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sym", [False, True])
def test_decompositions_are_exact(sym):
    q, k, v = qkv()
    pi = softmax_pi(q, k)
    pi_hat = rand_pi(pi.shape, seed=1)
    v_hat = v + 0.3 * rand(*v.shape, seed=2)
    f = M.symmetric_decomposition if sym else M.error_decomposition
    d = f(pi, pi_hat, v, v_hat)
    direct = (pi @ v - pi_hat @ v_hat).norm(dim=-1)
    assert torch.allclose(d["total"], direct, atol=TOL)


def test_both_decompositions_agree_on_the_total():
    q, k, v = qkv()
    pi, pi_hat = softmax_pi(q, k), rand_pi((2, 3, 10, 16), seed=4)
    v_hat = v + 0.3 * rand(*v.shape, seed=5)
    a = M.error_decomposition(pi, pi_hat, v, v_hat)
    b = M.symmetric_decomposition(pi, pi_hat, v, v_hat)
    assert torch.allclose(a["total"], b["total"], atol=TOL)
    # they differ only by the second-order cross term
    cross = ((pi - pi_hat) @ (v - v_hat)).norm(dim=-1)
    assert (a["distributional"] - b["distributional"]).abs().max() <= cross.max() + TOL


# ---------------------------------------------------------------------------
# Divergences
# ---------------------------------------------------------------------------
def test_total_variation_is_half_l1_and_bounded():
    p, qq = rand_pi((4, 7), seed=6), rand_pi((4, 7), seed=7)
    assert torch.allclose(M.total_variation(p, qq), 0.5 * (p - qq).abs().sum(-1), atol=TOL)
    assert (M.total_variation(p, qq) <= 1 + TOL).all()
    assert torch.allclose(M.total_variation(p, p), torch.zeros(4, dtype=p.dtype), atol=TOL)


def test_kl_is_infinite_where_support_is_dropped():
    """prop:sparse-identity: KL(pi || pi_hat) = inf whenever eps_q > 0."""
    p = rand_pi((5, 9), seed=8)
    idx = p.topk(4, dim=-1).indices
    S = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, idx, True)
    p_hat = (p * S) / (p * S).sum(-1, keepdim=True)
    assert torch.isinf(M.kl(p, p_hat)).all(), "kl must return +inf, not a clamped value"
    # the other direction is finite and equals -log(1 - eps)
    eps = M.omitted_mass(p, S)
    assert torch.allclose(M.kl(p_hat, p), -torch.log1p(-eps), atol=TOL)


def test_kl_matches_the_definition_and_vanishes_at_equality():
    p, qq = rand_pi((6, 11), seed=9), rand_pi((6, 11), seed=10)
    manual = (p * (p.log() - qq.log())).sum(-1)
    assert torch.allclose(M.kl(p, qq), manual, atol=TOL)
    assert torch.allclose(M.kl(p, p), torch.zeros(6, dtype=p.dtype), atol=TOL)
    assert (M.kl(p, qq) >= -TOL).all()


def test_pinskers_inequality():
    p, qq = rand_pi((20, 11), seed=11), rand_pi((20, 11), seed=12)
    assert (M.total_variation(p, qq) <= (M.kl(p, qq) / 2).sqrt() + 1e-9).all()


def test_chi2_of_a_restriction_is_eps_over_one_minus_eps():
    """prop:sparse-identity / prop:error-restrict (i)."""
    p = rand_pi((5, 12), seed=13)
    idx = p.topk(5, dim=-1).indices
    S = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, idx, True)
    p_hat = (p * S) / (p * S).sum(-1, keepdim=True)
    eps = M.omitted_mass(p, S)
    assert torch.allclose(M.chi2(p_hat, p), eps / (1 - eps), atol=TOL)


def test_chi2_is_infinite_without_absolute_continuity():
    p = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64)
    p_hat = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.float64)
    assert torch.isinf(M.chi2(p_hat, p)).all()


def test_kl_le_log1p_chi2():
    p, qq = rand_pi((20, 9), seed=14), rand_pi((20, 9), seed=15)
    assert (M.kl(qq, p) <= torch.log1p(M.chi2(qq, p)) + 1e-9).all()


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------
def test_spearman_endpoints_and_ties():
    x = torch.arange(10, dtype=torch.float64)
    assert torch.allclose(M.spearman(x, x), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(M.spearman(x, -x), torch.tensor(-1.0, dtype=torch.float64), atol=1e-12)
    # a monotone nonlinear map is rank-preserving
    assert torch.allclose(M.spearman(x, torch.exp(x)), torch.tensor(1.0, dtype=torch.float64),
                          atol=1e-12)


def test_spearman_matches_scipy_including_ties():
    scipy_stats = pytest.importorskip("scipy.stats")
    x, y = rand(5, 30, seed=16), rand(5, 30, seed=17)
    x[0, :8] = 1.0                      # a run of ties
    y[1, 4:12] = -2.0
    mine = M.spearman(x, y)
    for i in range(5):
        ref = scipy_stats.spearmanr(x[i].numpy(), y[i].numpy()).statistic
        assert abs(float(mine[i]) - float(ref)) < 1e-10


# ---------------------------------------------------------------------------
# Restriction identities (prop:sparse-identity, prop:error-single)
# ---------------------------------------------------------------------------
def test_restriction_identity_is_exact_and_tv_equals_eps():
    q, k, v = qkv()
    pi = softmax_pi(q, k)
    for budget in (1, 3, 8, 16):
        out = ab.build("topk", k=budget)(q, k, v, return_weights=True)
        S = out.extras["support"]
        r = M.restriction_identity(pi, S, v)
        assert torch.allclose(pi @ v - out.y, r["predicted_error"], atol=TOL)
        assert torch.allclose(M.total_variation(pi, out.weights), r["eps"], atol=TOL)
        assert (r["predicted_error"].norm(dim=-1)
                <= r["eps"] * M.value_oscillation(v).unsqueeze(-1) + TOL).all()


def test_single_token_omission_matches_recomputation():
    """prop:error-single: ||y - y_hat|| = p_j/(1-p_j) ||v_j - y||."""
    q, k, v = qkv(b=1, h=1, m=4, n=9)
    pi = softmax_pi(q, k)
    y = pi @ v
    pred = M.single_token_omission(pi, v)
    for j in range(9):
        S = torch.ones_like(pi, dtype=torch.bool)
        S[..., j] = False
        pi_hat = (pi * S) / (pi * S).sum(-1, keepdim=True)
        actual = (y - pi_hat @ v).norm(dim=-1)
        assert torch.allclose(actual, pred[..., j], atol=TOL)


# ---------------------------------------------------------------------------
# BOUNDS -- each must actually bound its target over randomized inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_tv_osc_bound_holds(seed):
    """prop:error-tv: ||(pi - pi_hat) v|| <= TV osc(v)."""
    q, k, v = qkv(seed=seed)
    pi, pi_hat = softmax_pi(q, k), rand_pi((2, 3, 10, 16), seed=seed + 50)
    err = ((pi - pi_hat) @ v).norm(dim=-1)
    assert (err <= M.tv_osc_bound(pi, pi_hat, v) + 1e-9).all()


def test_tv_osc_bound_is_attained():
    """The constant is sharp: two-valued v and a mass-t transfer attain it."""
    for t in (0.1, 0.4, 0.9):
        pi = torch.tensor([[t, 1 - t, 0.0]], dtype=torch.float64)
        pi_hat = torch.tensor([[0.0, 1 - t, t]], dtype=torch.float64)
        v = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
        err = ((pi - pi_hat) @ v).norm(dim=-1)
        assert torch.allclose(err, M.tv_osc_bound(pi, pi_hat, v), atol=TOL)


@pytest.mark.parametrize("seed", range(6))
def test_chi2_variance_factor_bounds_the_distributional_error(seed):
    """prop:error-chi2: ||(pi - pi_hat) v|| <= sqrt(chi2) ||Cov_pi(v)||_op^{1/2}."""
    q, k, v = qkv(seed=seed)
    pi = softmax_pi(q, k)
    pi_hat = torch.softmax(torch.log(pi) + 0.5 * rand(*pi.shape, seed=seed + 60), dim=-1)
    err = ((pi - pi_hat) @ v).norm(dim=-1)
    b = M.chi2_variance_factor(pi, pi_hat, v)
    assert (err <= b["bound"] + 1e-9).all()
    # the weaker form of the proposition, with the second moment instead of the covariance
    y = pi @ v
    second = (pi * (v.unsqueeze(-3) - y.unsqueeze(-2)).pow(2).sum(-1)).sum(-1).sqrt()
    assert (b["sd"] <= second + 1e-9).all()


def test_chi2_variance_bound_is_attained_when_v_aligns_with_the_ratio():
    """Equality holds when v - y is proportional to dpi_hat/dpi - 1."""
    pi = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    w = torch.tensor([[2.0, 1.0, 0.6]], dtype=torch.float64)       # E_pi[w] = 1
    pi_hat = pi * w
    assert torch.allclose(pi_hat.sum(-1), torch.ones(1, dtype=torch.float64))
    v = (w - 1).squeeze(0).unsqueeze(-1)                           # v - y proportional to w - 1
    err = ((pi - pi_hat) @ v).norm(dim=-1)
    b = M.chi2_variance_factor(pi, pi_hat, v)
    assert torch.allclose(err, b["bound"], atol=1e-9)


@pytest.mark.parametrize("delta", [0.05, 0.25, 1.0, 3.0])
def test_tanh_tv_bound_holds(delta):
    """prop:error-score: |l_hat - l| <= delta  =>  TV <= tanh(delta/2), chi2 <= sinh^2 delta,
    KL <= delta^2/2 and ||y - y_hat|| <= tanh(delta/2) osc(v)."""
    q, k, v = qkv(seed=3)
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)
    pert = delta * torch.tanh(rand(*s.shape, seed=71))             # |pert| <= delta
    s_hat = s + pert
    pi, pi_hat = torch.softmax(s, -1), torch.softmax(s_hat, -1)
    b = M.tanh_tv_bound(s, s_hat, v)
    assert (b["delta"] <= delta + 1e-12).all()
    assert (M.total_variation(pi, pi_hat) <= b["tv_bound"] + 1e-9).all()
    assert (M.chi2(pi_hat, pi) <= b["chi2_bound"] + 1e-9).all()
    assert (M.kl(pi_hat, pi) <= b["kl_bound"] + 1e-9).all()
    assert (((pi - pi_hat) @ v).norm(dim=-1) <= b["output_bound"] + 1e-9).all()


def test_tanh_tv_bound_is_insensitive_to_a_uniform_logit_shift():
    """Adding the same constant to every logit changes nothing; the osc form sees that."""
    q, k, v = qkv(seed=4)
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)
    shifted = s + 5.0
    b = M.tanh_tv_bound(s, shifted, v)
    assert (b["delta"] >= 4.9).all()                   # the pointwise bound is blind to it
    assert (b["delta_osc"] <= 1e-12).all()             # the sharp one is not
    assert (b["tv_bound"] <= 1e-12).all()


@pytest.mark.parametrize("eps", [0.05, 0.3, 0.7])
def test_mult_error_bound_holds(eps):
    """prop:error-mult: (1 +- eps) g  =>  ||y - y_hat|| <= eps/(1+sqrt(1-eps^2)) osc(v)."""
    q, k, v = qkv(seed=5)
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)
    g = torch.exp(s - s.amax(-1, keepdim=True))
    ratio = 1 + eps * torch.tanh(rand(*s.shape, seed=81))
    pi = g / g.sum(-1, keepdim=True)
    g_hat = g * ratio
    pi_hat = g_hat / g_hat.sum(-1, keepdim=True)
    assert (((pi - pi_hat) @ v).norm(dim=-1)
            <= M.mult_error_bound(pi, pi_hat, v, eps) + 1e-9).all()


@pytest.mark.parametrize("seed", range(5))
def test_wasserstein1_matches_the_lp(seed):
    scipy_stats = pytest.importorskip("scipy.stats")
    p, qq = rand_pi((4, 12), seed=seed), rand_pi((4, 12), seed=seed + 90)
    pos = torch.arange(12, dtype=torch.float64)
    mine = M.wasserstein1(p, qq)
    for i in range(4):
        ref = scipy_stats.wasserstein_distance(pos.numpy(), pos.numpy(),
                                               p[i].numpy(), qq[i].numpy())
        assert abs(float(mine[i]) - float(ref)) < 1e-9


@pytest.mark.parametrize("seed", range(6))
def test_wasserstein1_bound_holds(seed):
    """prop:error-wasserstein: ||(pi - pi_hat) v|| <= L_v W_1(pi, pi_hat)."""
    q, k, v = qkv(seed=seed)
    # values that are genuinely Lipschitz in position, the regime the bound is for
    pos = torch.arange(v.shape[-2], dtype=v.dtype)
    v = v * 0.05 + pos[:, None] * torch.linspace(0.1, 0.3, v.shape[-1], dtype=v.dtype)
    pi, pi_hat = softmax_pi(q, k), rand_pi((2, 3, 10, 16), seed=seed + 100)
    err = ((pi - pi_hat) @ v).norm(dim=-1)
    assert (err <= M.wasserstein1_bound(pi, pi_hat, v) + 1e-9).all()


def test_wasserstein1_is_tighter_than_tv_for_local_moves():
    """The point of the transport bound: moving mass one step is cheap, TV says it is not."""
    n = 64
    p = torch.zeros(1, n, dtype=torch.float64)
    p[0, 10] = 1.0
    qq = torch.zeros(1, n, dtype=torch.float64)
    qq[0, 11] = 1.0
    pos = torch.arange(n, dtype=torch.float64)
    v = pos[:, None] * 0.5
    err = ((p - qq) @ v).norm(dim=-1)
    assert float(M.wasserstein1_bound(p, qq, v)) < float(M.tv_osc_bound(p, qq, v))
    assert (err <= M.wasserstein1_bound(p, qq, v) + 1e-12).all()


@pytest.mark.parametrize("seed", range(4))
def test_merge_bound_holds(seed):
    """prop:compress-merge / prop:error-merge, checked against the real merge operator."""
    q, k, v = qkv(seed=seed, n=16, scale=0.7)
    pi = softmax_pi(q, k)
    y = pi @ v
    out = ab.build("merge_kv", n_clusters=4)(q, k, v, return_weights=True)
    assign = out.extras["assign"]
    v_bar, radius = out.extras["v_bar"], out.extras["cluster_radius"]
    beta = q.shape[-1] ** -0.5
    delta = beta * q.norm(dim=-1) * radius.amax(-1, keepdim=True)        # [.., m]
    b = M.merge_bound(pi, assign, v, v_bar, delta)
    assert ((y - out.y).norm(dim=-1) <= b["bound"] + 1e-9).all()


# ---------------------------------------------------------------------------
# Value geometry and matrix norms
# ---------------------------------------------------------------------------
def test_n_eff_endpoints():
    """prop:error-stochastic: n_eff ranges from 1 (one-hot) to n (uniform)."""
    n = 8
    one_hot = torch.zeros(1, n, dtype=torch.float64)
    one_hot[0, 2] = 1.0
    uniform = torch.full((1, n), 1 / n, dtype=torch.float64)
    assert torch.allclose(M.n_eff(one_hot), torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(M.n_eff(uniform), torch.tensor([float(n)], dtype=torch.float64))
    p = rand_pi((20, n), seed=33)
    assert ((M.n_eff(p) >= 1 - TOL) & (M.n_eff(p) <= n + TOL)).all()


def test_effective_support_endpoints():
    n = 8
    uniform = torch.full((1, n), 1 / n, dtype=torch.float64)
    assert torch.allclose(M.effective_support(uniform),
                          torch.tensor([float(n)], dtype=torch.float64), atol=1e-9)


def test_stochastic_rounding_variance_formula():
    """prop:error-stochastic: E||e||^2 = sum_j pi_j^2 sigma_j^2 <= max sigma^2 / n_eff."""
    torch.manual_seed(0)
    n, dv, trials = 6, 3, 40000
    pi = rand_pi((1, n), seed=44)
    sigma2 = 0.25
    g = torch.Generator().manual_seed(5)
    noise = math.sqrt(sigma2 / dv) * torch.randn(trials, n, dv, generator=g,
                                                 dtype=torch.float64)
    e = torch.einsum("n,tnd->td", pi[0], noise)
    empirical = (e ** 2).sum(-1).mean()
    predicted = (pi[0] ** 2).sum() * sigma2
    assert abs(float(empirical) - float(predicted)) < 0.05 * float(predicted)
    assert float(predicted) <= sigma2 / float(M.n_eff(pi)) + 1e-12


def test_value_spread_and_covariance_are_consistent():
    q, k, v = qkv()
    pi = softmax_pi(q, k)
    sd = M.value_covariance_op(pi, v)
    spread = M.value_spread(pi, v)
    # ||Cov||_op^{1/2} <= (E||v - y||^2)^{1/2}, and E||v-y|| <= (E||v-y||^2)^{1/2}
    y = pi @ v
    rms = (pi * (v.unsqueeze(-3) - y.unsqueeze(-2)).pow(2).sum(-1)).sum(-1).sqrt()
    assert (sd <= rms + 1e-9).all()
    assert (spread <= rms + 1e-9).all()


def test_matrix_inf_to_inf_is_twice_the_worst_total_variation():
    """Sec. 4.2: row i of P - P_hat has l1 norm 2 TV(pi_i, pi_hat_i)."""
    q, k, v = qkv()
    pi, pi_hat = softmax_pi(q, k), rand_pi((2, 3, 10, 16), seed=55)
    mat = M.attention_matrix_errors(pi, pi_hat, v)
    assert torch.allclose(mat["inf_to_inf"], 2 * M.total_variation(pi, pi_hat).amax(-1),
                          atol=TOL)
    assert torch.allclose(mat["AV_fro"], ((pi - pi_hat) @ v).flatten(-2).norm(dim=-1), atol=TOL)


def test_matrix_error_and_output_error_are_incomparable():
    """ex:error-matrix: large Frobenius error with zero output error, and the converse."""
    n = 2 * 32 + 1
    pi = torch.zeros(1, n, dtype=torch.float64)
    pi_hat = torch.zeros(1, n, dtype=torch.float64)
    pi[0, 0] = pi_hat[0, 0] = 0.5
    pi[0, 1:33] = 0.5 / 32
    pi_hat[0, 33:] = 0.5 / 32
    v = torch.zeros(n, 1, dtype=torch.float64)
    v[1:33] = 1.0
    v[33:] = -1.0
    mat = M.attention_matrix_errors(pi, pi_hat, v)
    assert float(mat["fro"]) < 0.2                      # tiny matrix error
    assert float(mat["AV_fro"]) == pytest.approx(1.0)   # maximal output error
    assert float(M.total_variation(pi, pi_hat)) == pytest.approx(0.5)
    # converse: moving mass between equal values is free
    v_flat = torch.ones(n, 1, dtype=torch.float64)
    mat2 = M.attention_matrix_errors(pi, pi_hat, v_flat)
    assert float(mat2["fro"]) > 0.1 and float(mat2["AV_fro"]) < 1e-12


def test_logit_error_ignores_jointly_masked_entries():
    s = torch.tensor([[[0.0, 1.0, float("-inf")]]], dtype=torch.float64)
    s_hat = torch.tensor([[[0.5, 1.0, float("-inf")]]], dtype=torch.float64)
    assert torch.allclose(M.logit_error(s, s_hat), torch.tensor([[0.5]], dtype=torch.float64))
    assert torch.allclose(M.logit_oscillation(s, s_hat),
                          torch.tensor([[0.5]], dtype=torch.float64))
