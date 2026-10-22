"""Mathematics specific to ``interaction/``: surrogates for g (sec:kernel)."""
from __future__ import annotations

import math

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M
from attnbank.methods.interaction.performer import gaussian_features


def qkv(b=1, h=1, m=8, n=12, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def true_g(q, k):
    """g(q, x) = exp(<qt, kt>) with qt = q / d^{1/4}."""
    d = q.shape[-1]
    return torch.exp(torch.einsum("...md,...nd->...mn", q, k) / math.sqrt(d))


# ---------------------------------------------------------------------------
# sec:kernel-rf: the estimators are unbiased for g, with the stated variances
# ---------------------------------------------------------------------------
def test_performer_is_unbiased_for_the_exponential_kernel():
    q, k, _ = qkv(m=4, n=4, scale=0.4)
    g = true_g(q, k)
    est = torch.stack([torch.exp(ab.build("performer", r=64, orthogonal=False, seed=s)(
        q, k, torch.zeros(1, 1, 4, 1, dtype=q.dtype), return_weights=True
    ).extras["log_g_hat"]) for s in range(600)])
    # the log_g_hat is the estimator itself (no normalization applied to it)
    rel = (est.mean(0) - g).abs() / g
    assert float(rel.max()) < 0.12, float(rel.max())


def test_performer_relative_variance_matches_eq_kernel_pos_var():
    """Var xi / g^2 = exp(||qt + kt||^2) - 1  (eq:kernel-pos-var)."""
    q, k, _ = qkv(m=3, n=3, scale=0.3)
    d = q.shape[-1]
    qt, kt = q / d ** 0.25, k / d ** 0.25
    predicted = torch.exp((qt.unsqueeze(-2) + kt.unsqueeze(-3)).pow(2).sum(-1)) - 1
    g = true_g(q, k)
    r = 1                                   # one feature: the raw single-draw variance
    est = torch.stack([torch.exp(ab.build("performer", r=r, orthogonal=False, seed=s)(
        q, k, torch.zeros(1, 1, 3, 1, dtype=q.dtype), return_weights=True
    ).extras["log_g_hat"]) for s in range(4000)])
    empirical = est.var(0) / g ** 2
    rel = (empirical - predicted).abs() / predicted.clamp_min(1e-6)
    assert float(rel.max()) < 0.2, (float(rel.max()), predicted.max())


def test_trig_relative_variance_matches_eq_kernel_trig_var():
    """Var xi^trig / g^2 = 2 sinh^2(||qt - kt||^2 / 2)."""
    q, k, _ = qkv(m=3, n=3, scale=0.3)
    d = q.shape[-1]
    qt, kt = q / d ** 0.25, k / d ** 0.25
    sq = (qt.unsqueeze(-2) - kt.unsqueeze(-3)).pow(2).sum(-1)
    predicted = 2 * torch.sinh(sq / 2) ** 2
    om = torch.stack([gaussian_features(1, d, s, False, q.device, q.dtype)[0]
                      for s in range(20000)])                        # [S, d]
    pref = torch.exp((qt.pow(2).sum(-1).unsqueeze(-1) + kt.pow(2).sum(-1).unsqueeze(-2)) / 2)
    cos = torch.cos(torch.einsum("sd,...md->s...m", om, qt).unsqueeze(-1)
                    - torch.einsum("sd,...nd->s...n", om, kt).unsqueeze(-2))
    xi = pref * cos
    g = true_g(q, k)
    empirical = xi.var(0) / g ** 2
    rel = (empirical - predicted).abs() / predicted.clamp_min(1e-4)
    assert float(rel.max()) < 0.15, float(rel.max())


def test_performer_weights_are_a_probability_measure_but_trig_need_not_be():
    """prop:error-additive: positivity of g_hat is what makes pi_hat a measure."""
    q, k, v = qkv(scale=1.5)
    p = ab.build("performer", r=8, seed=0)(q, k, v, return_weights=True)
    assert (p.weights >= 0).all()
    assert torch.allclose(p.weights.sum(-1), torch.ones_like(p.weights[..., 0]), atol=1e-10)
    assert (p.extras["Z_rel"] > 0).all()

    seen_negative = False
    for s in range(40):
        t = ab.build("trigrff", r=4, orthogonal=False, seed=s)(
            *qkv(scale=3.0, seed=7), return_weights=True)
        seen_negative |= bool((t.weights < 0).any())
    assert seen_negative, "trigonometric features must be able to produce signed weights"


def test_trig_normalizer_can_go_non_positive_at_large_logits():
    """sec:kernel-error (iii): a signed g_hat lets Z_hat approach or cross zero."""
    bad = 0
    for s in range(60):
        q, k, v = qkv(scale=4.0, seed=s)
        out = ab.build("trigrff", r=2, orthogonal=False, seed=s)(q, k, v, return_weights=True)
        bad += int(out.extras["negative_Z"].sum())
        assert torch.isfinite(out.y).all()        # finite even then, by the signed floor
    assert bad > 0, "no non-positive normalizer observed; raise the logit scale"


def test_random_feature_error_degrades_with_logit_scale():
    """sec:kernel-limits: the cost of a fixed accuracy grows like exp(Theta(R^2))."""
    errs = []
    for scale in (0.3, 0.8, 1.5, 2.5):
        q, k, v = qkv(scale=scale, seed=1)
        ref = ab.build("softmax")(q, k, v).y
        e = [float((ab.build("performer", r=32, seed=s)(q, k, v).y - ref).norm(dim=-1).mean()
                   / ref.norm(dim=-1).mean()) for s in range(12)]
        errs.append(sum(e) / len(e))
    assert errs[-1] > errs[0], errs


def test_orthogonal_features_reduce_variance():
    q, k, v = qkv(scale=0.8, seed=2)
    ref = ab.build("softmax")(q, k, v).y
    def mean_err(orth):
        return sum(float((ab.build("performer", r=16, orthogonal=orth, seed=s)(q, k, v).y
                          - ref).norm(dim=-1).mean()) for s in range(40)) / 40
    assert mean_err(True) < mean_err(False)


# ---------------------------------------------------------------------------
# lem:kernel-mixture and rem:kernel-homogeneous
# ---------------------------------------------------------------------------
def test_positive_separable_attention_is_a_mixture_of_fixed_context_measures():
    """lem:kernel-mixture: pi_hat_q = sum_l a_l(q) kappa_l with kappa_l query-independent."""
    q, k, v = qkv(m=6, n=10)
    phi = lambda x: torch.nn.functional.elu(x) + 1.0
    pk = phi(k)                                     # psi(k_j), [.., n, r]
    z = pk.sum(-2)                                  # z_l
    kappa = (pk / z.unsqueeze(-2)).transpose(-1, -2)         # kappa_l(x_j): [.., r, n]
    assert torch.allclose(kappa.sum(-1), torch.ones_like(kappa[..., 0]), atol=1e-12)
    a = phi(q) * z.unsqueeze(-2)
    a = a / a.sum(-1, keepdim=True)                 # a_l(q)
    out = ab.build("linear_elu")(q, k, v, return_weights=True)
    assert torch.allclose(a @ kappa, out.weights, atol=1e-12)
    # entropy is bounded below by the least-entropic prototype
    ent = lambda p: -(p * p.clamp_min(1e-300).log()).sum(-1)
    assert (ent(out.weights) >= ent(kappa).amin(-1, keepdim=True) - 1e-10).all()


def test_relu_features_cannot_sharpen_with_query_norm():
    """rem:kernel-homogeneous: phi positively homogeneous => pi_hat_{alpha q} = pi_hat_q."""
    q, k, v = qkv()
    base = ab.build("cosformer")(q, k, v, return_weights=True).weights
    for alpha in (0.1, 3.0, 50.0):
        scaled = ab.build("cosformer")(alpha * q, k, v, return_weights=True).weights
        assert torch.allclose(scaled, base, atol=1e-11), alpha
    # softmax, by contrast, sharpens
    s_base = ab.build("softmax")(q, k, v, return_weights=True).weights
    s_hot = ab.build("softmax")(50 * q, k, v, return_weights=True).weights
    assert float(M.effective_support(s_hot).mean()) < float(M.effective_support(s_base).mean())


def test_cosformer_decay_is_nonnegative_and_monotone_in_distance():
    q, k, v = qkv(m=12, n=12)
    out = ab.build("cosformer")(q, k, v, return_weights=True)
    decay = out.extras["decay"]
    assert (decay >= -1e-12).all()
    row = decay[6]                                  # query at position 6
    assert float(row[6]) == pytest.approx(1.0)
    assert float(row[0]) < float(row[3]) < float(row[6])


def test_linear_elu_is_an_operator_not_an_approximation():
    """def:kernel-regimes: no resource parameter, no limit, no convergence sequence."""
    for name in ("linear_elu", "cosformer"):
        cls = ab.get_class(name)
        assert cls.objective is ab.Objective.MODIFY
        assert cls.limit_config() is None and cls.convergence_configs() is None


# ---------------------------------------------------------------------------
# Nystrom (sec:kernel-matrix)
# ---------------------------------------------------------------------------
def test_nystrom_weights_are_signed_and_leave_the_convex_hull():
    """tab:comparative: Nystrom loses nm and hl."""
    q, k, v = qkv(m=16, n=16, scale=1.5, seed=5)
    out = ab.build("nystrom", r=3)(q, k, v, return_weights=True)
    assert bool((out.weights < -1e-9).any()), "expected signed weights from the pinv core"
    lo, hi = v.amin(-2, keepdim=True), v.amax(-2, keepdim=True)
    assert bool(((out.y < lo - 1e-9) | (out.y > hi + 1e-9)).any()), \
        "expected an output outside the convex hull of the values"


def test_nystrom_improves_with_more_landmarks():
    q, k, v = qkv(m=24, n=24, scale=0.8, seed=6)
    ref = ab.build("softmax")(q, k, v, return_weights=True)
    errs = [float(M.total_variation(ref.weights,
                                    ab.build("nystrom", r=r)(q, k, v, return_weights=True).weights
                                    ).mean()) for r in (2, 4, 8, 16, 24)]
    assert errs[0] > errs[-1]
    assert errs[-1] < 1e-8                   # r = n reproduces P exactly


def test_nystrom_rows_sum_to_one_when_the_core_is_invertible():
    """G^{-1} 1 = 1 for a stochastic G, so unit mass survives even though positivity does not."""
    q, k, v = qkv(m=16, n=16, scale=0.5, seed=8)
    out = ab.build("nystrom", r=4)(q, k, v, return_weights=True)
    assert torch.allclose(out.weights.sum(-1), torch.ones_like(out.weights[..., 0]), atol=1e-8)
