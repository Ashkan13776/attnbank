"""Mathematics specific to ``recurrent/``: sec:memory-state."""
from __future__ import annotations

import pytest
import torch

import attnbank as ab
from attnbank.methods.recurrent._state import feature_map

PHI = feature_map("elu")


def qkv(b=2, h=2, m=12, n=16, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


# ---------------------------------------------------------------------------
# The recurrences really are recurrences: compare with an explicit state loop
# ---------------------------------------------------------------------------
def test_linear_rnn_matches_an_explicit_state_loop():
    """H_t = H_{t-1} + psi(k_t) v_t^T,  y_t = H_t^T phi(q_t)."""
    q, k, v = qkv(m=16, n=16)
    out = ab.build("linear_rnn")(q, k, v, causal=True)
    h = torch.zeros(*k.shape[:-2], k.shape[-1], v.shape[-1], dtype=v.dtype)
    ys = []
    for t in range(16):
        h = h + PHI(k[..., t, :]).unsqueeze(-1) * v[..., t, :].unsqueeze(-2)
        ys.append(torch.einsum("...rd,...r->...d", h, PHI(q[..., t, :])))
    assert torch.allclose(out.y, torch.stack(ys, dim=-2), atol=1e-11)


@pytest.mark.parametrize("gamma", [1.0, 0.9, 0.5])
def test_decay_rnn_matches_an_explicit_state_loop(gamma):
    """H_t = gamma H_{t-1} + psi(k_t) v_t^T."""
    q, k, v = qkv(m=16, n=16)
    out = ab.build("decay_rnn", decay=gamma)(q, k, v, causal=True)
    h = torch.zeros(*k.shape[:-2], k.shape[-1], v.shape[-1], dtype=v.dtype)
    ys = []
    for t in range(16):
        h = gamma * h + PHI(k[..., t, :]).unsqueeze(-1) * v[..., t, :].unsqueeze(-2)
        ys.append(torch.einsum("...rd,...r->...d", h, PHI(q[..., t, :])))
    assert torch.allclose(out.y, torch.stack(ys, dim=-2), atol=1e-11)


@pytest.mark.parametrize("beta", [1.0, 0.5])
def test_delta_rnn_matches_eq_memory_delta(beta):
    """H_t = (I - beta psi_t psi_t^T) H_{t-1} + beta psi_t v_t^T (eq:memory-delta)."""
    q, k, v = qkv(m=16, n=16)
    out = ab.build("delta_rnn", beta=beta)(q, k, v, causal=True)
    psi = PHI(k)
    psi = psi / psi.norm(dim=-1, keepdim=True)
    h = torch.zeros(*k.shape[:-2], k.shape[-1], v.shape[-1], dtype=v.dtype)
    ys = []
    for t in range(16):
        p = psi[..., t, :].unsqueeze(-1)                       # [..., r, 1]
        h = h - beta * p * (p.transpose(-1, -2) @ h)
        h = h + beta * p * v[..., t, :].unsqueeze(-2)
        ys.append(torch.einsum("...rd,...r->...d", h, PHI(q[..., t, :])))
    assert torch.allclose(out.y, torch.stack(ys, dim=-2), atol=1e-11)


def test_delta_rule_is_one_gradient_step_on_the_regression_loss():
    """sec:memory-state: H_t = H_{t-1} - beta psi_t (psi_t^T H_{t-1} - v_t^T)."""
    g = torch.Generator().manual_seed(1)
    r, dv, beta = 6, 4, 0.3
    h = torch.randn(r, dv, generator=g, dtype=torch.float64, requires_grad=True)
    psi = torch.nn.functional.normalize(torch.randn(r, generator=g, dtype=torch.float64),
                                        dim=0)
    v = torch.randn(dv, generator=g, dtype=torch.float64)
    loss = 0.5 * ((h.T @ psi) - v).pow(2).sum()
    loss.backward()
    sgd = (h - beta * h.grad).detach()
    closed = ((torch.eye(r, dtype=torch.float64) - beta * torch.outer(psi, psi)) @ h.detach()
              + beta * torch.outer(psi, v))
    assert torch.allclose(sgd, closed, atol=1e-12)


# ---------------------------------------------------------------------------
# Invariant W holds even though the weights are not a probability measure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["linear_rnn", "decay_rnn", "delta_rnn"])
@pytest.mark.parametrize("causal", [False, True])
def test_weights_reproduce_the_output(name, causal):
    q, k, v = qkv()
    out = ab.build(name)(q, k, v, causal=causal, return_weights=True)
    assert torch.allclose(out.weights @ v, out.y, atol=1e-11)


def test_linear_rnn_weights_are_the_separable_interaction():
    q, k, v = qkv()
    out = ab.build("linear_rnn")(q, k, v, causal=True, return_weights=True)
    expected = (PHI(q) @ PHI(k).transpose(-1, -2)) * ab.causal_mask(12, 16)
    assert torch.allclose(out.weights, expected, atol=1e-12)


def test_decay_rnn_weights_decay_geometrically_in_the_lag():
    q, k, v = qkv(m=16, n=16)
    gamma = 0.7
    out = ab.build("decay_rnn", decay=gamma)(q, k, v, causal=True, return_weights=True)
    content = PHI(q) @ PHI(k).transpose(-1, -2)
    for t in (5, 10, 15):
        for lag in (0, 1, 3):
            ratio = out.weights[..., t, t - lag] / content[..., t, t - lag]
            assert torch.allclose(ratio, torch.full_like(ratio, gamma ** lag), atol=1e-11)


def test_decay_one_is_the_additive_recurrence():
    q, k, v = qkv()
    a = ab.build("decay_rnn", decay=1.0)(q, k, v, causal=True).y
    b = ab.build("linear_rnn")(q, k, v, causal=True).y
    assert torch.allclose(a, b, atol=1e-11)


# ---------------------------------------------------------------------------
# What is lost when the denominator is dropped (sec:memory-state)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["linear_rnn", "decay_rnn", "delta_rnn"])
def test_constant_preservation_fails_without_the_normalizer(name):
    """y_hat = c <phi(q), z> scales with the context, so v == c does not give y == c."""
    q, k, v = qkv()
    c = torch.full_like(v, 1.0)
    y = ab.build(name)(q, k, c, causal=True).y
    assert float((y - 1.0).abs().max()) > 0.1, name
    assert ab.get_class(name).row_stochastic is False


def test_output_grows_with_context_length_for_the_additive_recurrence():
    """The convex-hull bound is gone: ||y_hat|| is not bounded uniformly in n."""
    norms = []
    for n in (8, 32, 128):
        q, k, v = qkv(m=1, n=n, seed=2)
        norms.append(float(ab.build("linear_rnn")(q, k, v, causal=True).y.norm(dim=-1).mean()))
    assert norms[0] < norms[-1] / 2, norms


def test_decay_restores_boundedness_but_not_normalization():
    """The effective mass sum_i gamma^{t-i} is bounded by 1/(1 - gamma)."""
    masses = []
    for n in (8, 32, 128, 512):
        q, k, v = qkv(m=1, n=n, seed=3)
        out = ab.build("decay_rnn", decay=0.8)(q, k, v, causal=True, return_weights=True)
        masses.append(float(out.extras["mass"].mean()))
    assert masses[-1] < 2 * masses[1], masses          # saturates
    assert abs(masses[-1] - 1.0) > 0.1                 # but is not 1


def test_normalized_linear_rnn_is_the_linear_transformer():
    q, k, v = qkv()
    a = ab.build("linear_rnn", normalize=True)(q, k, v, causal=True, return_weights=True)
    b = ab.build("linear_elu")(q, k, v, causal=True, return_weights=True)
    assert torch.allclose(a.weights, b.weights, atol=1e-11)
    assert torch.allclose(a.y, b.y, atol=1e-11)
    assert torch.allclose(a.weights.sum(-1), torch.ones_like(a.weights[..., 0]), atol=1e-11)


# ---------------------------------------------------------------------------
# prop:memory-rank: recall is bounded by the state
# ---------------------------------------------------------------------------
def test_delta_rule_recalls_exactly_while_the_keys_fit_in_the_state():
    """With orthonormal keys and beta = 1 the delta rule stores each association
    exactly -- until the number of items exceeds the state rank r."""
    dv, r = 4, 8
    for n, exact in ((4, True), (8, True), (12, False)):
        if n <= r:
            k = torch.eye(r, dtype=torch.float64)[:n]                  # orthonormal keys
        else:                                                          # more items than state
            g0 = torch.Generator().manual_seed(n)
            k = torch.nn.functional.normalize(
                torch.randn(n, r, generator=g0, dtype=torch.float64), dim=-1)
        k = k.reshape(1, 1, n, r)
        g = torch.Generator().manual_seed(4)
        v = torch.randn(1, 1, n, dv, generator=g, dtype=torch.float64)
        y = ab.build("delta_rnn", beta=1.0, feature="identity")(
            k, k, v, causal=False).y                       # query j = key j
        err = float((y - v).abs().max())
        assert (err < 1e-9) == exact, (n, err)


def test_additive_recurrence_interferes_where_the_delta_rule_does_not():
    """Plain linear attention never corrects interference between non-orthogonal keys."""
    n, r, dv = 6, 8, 4
    g = torch.Generator().manual_seed(5)
    k = torch.nn.functional.normalize(
        torch.randn(1, 1, n, r, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(1, 1, n, dv, generator=g, dtype=torch.float64)
    e_delta = float((ab.build("delta_rnn", beta=1.0, feature="identity")(k, k, v).y
                     - v).norm(dim=-1).mean())
    e_lin = float((ab.build("linear_rnn", feature="identity")(k, k, v).y
                   - v).norm(dim=-1).mean())
    assert e_delta < e_lin, (e_delta, e_lin)


def test_recurrent_state_is_independent_of_the_context_length():
    """prop:memory-rank: the state is r x d_v whatever n is -- that is the whole point."""
    for name in ("linear_rnn", "decay_rnn", "delta_rnn"):
        meth = ab.build(name)
        a = meth.cost(1, 64, 32, 32)["state_bytes"]
        b = meth.cost(1, 65536, 32, 32)["state_bytes"]
        assert a == b, name
        assert b < ab.build("softmax").cost(1, 65536, 32, 32)["state_bytes"]
