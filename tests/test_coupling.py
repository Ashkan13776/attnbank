"""Mathematics specific to ``coupling/``: sec:ot."""
from __future__ import annotations

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M


def qkv(b=2, h=2, m=12, n=12, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


# ---------------------------------------------------------------------------
# Sinkhorn converges to the prescribed marginals
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("eps", [2.0, 1.0, 0.5])
def test_sinkhorn_converges_to_the_prescribed_marginals(eps):
    q, k, v = qkv(seed=1)
    errs = []
    for it in (1, 2, 5, 20, 80):
        e = ab.build("sinkhorn", n_iter=it, eps=eps)(q, k, v, return_weights=True).extras
        errs.append(float(e["marginal_error"].max()))
        assert torch.allclose(e["col_marginal"], e["target_col"], atol=1e-12), it
    assert all(a >= b for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-8 * max(1.0, 1 / eps), errs


def test_the_converged_coupling_is_doubly_stochastic_up_to_scale():
    q, k, v = qkv(m=10, n=10, seed=2)
    e = ab.build("sinkhorn", n_iter=200)(q, k, v, return_weights=True).extras
    gamma = e["gamma"]
    assert torch.allclose(gamma.sum(-1), torch.full_like(gamma[..., 0], 1 / 10), atol=1e-9)
    assert torch.allclose(gamma.sum(-2), torch.full_like(gamma[..., 0], 1 / 10), atol=1e-12)
    assert (gamma >= 0).all()


def test_zero_iterations_is_exactly_softmax():
    """One step from psi = 0, stopped after the row update (eq:ot-sinkhorn)."""
    q, k, v = qkv(seed=3)
    ref = ab.build("softmax")(q, k, v, return_weights=True)
    out = ab.build("sinkhorn", n_iter=0, eps=1.0)(q, k, v, return_weights=True)
    assert torch.allclose(out.weights, ref.weights, atol=1e-14)
    assert torch.allclose(out.y, ref.y, atol=1e-14)


# ---------------------------------------------------------------------------
# prop:ot-conditional: softmax tilted by a query-independent key potential
# ---------------------------------------------------------------------------
def test_the_conditional_is_softmax_tilted_by_a_key_potential():
    """prop:ot-conditional (c): pi^OT_q(j) ∝ exp(beta <q, k_j> + psi_j), with psi
    *independent of the query*.  The potential is recovered from the coupling itself,
    not from the construction, so this is a check and not a tautology."""
    q, k, v = qkv(m=12, n=12, seed=4, scale=1.2)
    out = ab.build("sinkhorn", n_iter=200, eps=1.0)(q, k, v, return_weights=True)
    gamma = out.extras["gamma"]
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)

    # recover the tilt from the coupling: log Gamma_ij - s_ij must split as f_i + psi_j,
    # i.e. after removing each row's mean, every row of the residual must be identical
    resid = torch.log(gamma) - s                            # [..., m, n]
    centred = resid - resid.mean(-1, keepdim=True)
    assert torch.allclose(centred, centred[..., :1, :].expand_as(centred), atol=1e-9), \
        "the tilt is not query-independent: prop:ot-conditional would be misstated"

    potential = centred[..., 0, :]                          # psi_j up to an additive gauge
    direct = torch.softmax(s + potential.unsqueeze(-2), dim=-1)
    assert torch.allclose(direct, out.weights, atol=1e-10)


def test_the_potential_suppresses_over_subscribed_keys():
    """sec:ot: psi is a price, negative on keys over-subscribed under plain softmax."""
    q, k, v = qkv(b=1, h=1, m=24, n=24, seed=5, scale=1.5)
    ref = ab.build("softmax")(q, k, v, return_weights=True)
    out = ab.build("sinkhorn", n_iter=300)(q, k, v, return_weights=True)
    col_mass = ref.weights.mean(-2)[0, 0]                   # softmax column mass
    psi = out.extras["potential"][0, 0]
    psi = psi - psi.mean()
    assert float(M.spearman(col_mass, psi)) < -0.5, float(M.spearman(col_mass, psi))


def test_doubly_stochastic_attention_differs_from_softmax():
    q, k, v = qkv(seed=6, scale=1.5)
    ref = ab.build("softmax")(q, k, v, return_weights=True)
    tvs = [float(M.total_variation(ref.weights,
                                   ab.build("sinkhorn", n_iter=it)(q, k, v,
                                                                   return_weights=True).weights
                                   ).mean()) for it in (0, 1, 5, 50)]
    assert tvs[0] == pytest.approx(0.0, abs=1e-14)
    assert all(a <= b + 1e-9 for a, b in zip(tvs, tvs[1:])), tvs
    assert tvs[-1] > 0.05


def test_sharper_attention_needs_more_iterations():
    """Hilbert-metric contraction degrades exponentially in ||c||_inf / eps."""
    q, k, v = qkv(m=16, n=16, seed=7, scale=1.5)
    def err(eps, it):
        return float(ab.build("sinkhorn", n_iter=it, eps=eps)(
            q, k, v, return_weights=True).extras["marginal_error"].max())
    assert err(0.2, 10) > err(2.0, 10)


# ---------------------------------------------------------------------------
# prop:ot-causal: causal + uniform marginals = the identity
# ---------------------------------------------------------------------------
def test_causal_doubly_stochastic_attention_is_the_identity():
    """prop:ot-causal, verified directly: a lower-triangular nonnegative matrix whose row
    and column sums agree is diagonal, so causal attention with uniform marginals has
    every token attending only to itself.  This is why ``sinkhorn`` is not causal.
    The off-diagonal mass is driven to zero as the scaling is iterated."""
    n = 10
    g = torch.Generator().manual_seed(8)
    logits = torch.randn(1, 1, n, n, generator=g, dtype=torch.float64)
    logits = logits.masked_fill(~ab.causal_mask(n, n), float("-inf"))
    k_mat = torch.exp(logits - logits.amax())
    a = b = torch.full((n,), 1.0 / n, dtype=torch.float64)

    offs = []
    u = torch.ones(1, 1, n, dtype=torch.float64)
    w = torch.ones(1, 1, n, dtype=torch.float64)
    for total in (100, 1000, 20000):
        while True:
            u = a / (k_mat @ w.unsqueeze(-1)).squeeze(-1).clamp_min(1e-300)
            w = b / (k_mat.transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1).clamp_min(1e-300)
            total -= 1
            if total <= 0:
                break
        gamma = u.unsqueeze(-1) * k_mat * w.unsqueeze(-2)
        offs.append(float((gamma - torch.diag_embed(
            torch.diagonal(gamma, dim1=-2, dim2=-1))).abs().max()))
    assert all(a > 5 * b for a, b in zip(offs, offs[1:])), offs   # geometric decay to 0
    assert offs[-1] < 1e-4, offs
    assert torch.allclose(torch.diagonal(gamma, dim1=-2, dim2=-1),
                          torch.full((1, 1, n), 1 / n, dtype=torch.float64), atol=1e-4)


def test_sinkhorn_declares_itself_non_causal():
    assert ab.get_class("sinkhorn").supports_causal is False
    q, k, v = qkv()
    with pytest.raises(ValueError, match="does not support causal"):
        ab.build("sinkhorn")(q, k, v, causal=True)


# ---------------------------------------------------------------------------
# eq:ot-glued: the pivot factorization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r", [2, 4, 6])
def test_glued_coupling_has_rank_at_most_r_and_is_row_stochastic(r):
    q, k, v = qkv(m=12, n=12, seed=9)
    out = ab.build("lowrank_ot", r=r, n_iter=50)(q, k, v, return_weights=True)
    a, b = out.extras["a"], out.extras["b"]
    for f in (a, b):
        assert (f >= 0).all()
        assert torch.allclose(f.sum(-1), torch.ones_like(f[..., 0]), atol=1e-12)
    assert torch.allclose(a @ b, out.weights, atol=1e-12)
    assert int(torch.linalg.matrix_rank(out.weights.double()).max()) <= r


def test_glued_coupling_balances_the_column_marginal():
    """eq:ot-glued lies in Pi(nu, mubar): the pivot plan transmits the column constraint."""
    q, k, v = qkv(m=16, n=16, seed=10, scale=1.0)
    out = ab.build("lowrank_ot", r=4, n_iter=400)(q, k, v, return_weights=True)
    col = out.weights.mean(-2)                      # (1/m) sum_i pi_hat_i(j)
    assert torch.allclose(col, torch.full_like(col, 1 / 16), atol=1e-6)
    ref = ab.build("softmax")(q, k, v, return_weights=True)
    soft_col = ref.weights.mean(-2)
    assert float((soft_col - 1 / 16).abs().max()) > float((col - 1 / 16).abs().max())


def test_lowrank_ot_is_declared_an_operator_modification():
    cls = ab.get_class("lowrank_ot")
    assert cls.objective is ab.Objective.MODIFY
    assert cls.limit_config() is None and cls.convergence_configs() is None
    assert cls.trainable is True
