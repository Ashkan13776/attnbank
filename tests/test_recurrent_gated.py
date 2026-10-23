"""Mathematics specific to the *gated* linear recurrences: sec:memory-state.

``retnet``, ``gla``, ``hgrn2``, ``rwkv`` and ``mlstm`` are all

    H_t = A_t(H_{t-1}) + k_t v_t^T,      y_hat_t = H_{p_t}^T q_t,

and differ only in ``A_t`` and in what, if anything, is left of the denominator.  Each
one is checked against an explicit state loop written straight from its paper -- that
is the only test in this file that really matters, because everything else is a
property of a recurrence the loop has already pinned down.
"""
from __future__ import annotations

import pytest
import torch

import attnbank as ab

GATED = ["retnet", "gla", "hgrn2", "rwkv", "mlstm"]


def qkv(b=2, h=4, m=16, n=16, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def outer(a, b):
    """``[..., r] x [..., dv] -> [..., r, dv]``."""
    return a.unsqueeze(-1) * b.unsqueeze(-2)


def stack(ys):
    return torch.stack(ys, dim=-2)


# ---------------------------------------------------------------------------
# Each operator against an explicit state loop
# ---------------------------------------------------------------------------
def test_retnet_matches_the_multiscale_state_loop():
    """S_n = gamma_h S_{n-1} + k_n v_n^T, one gamma per head (Sun et al. 2023)."""
    q, k, v = qkv()
    meth = ab.build("retnet")
    out = meth(q, k, v, causal=True)
    gamma = meth.gammas(q.shape[1], dtype=q.dtype).view(1, -1, 1, 1)
    scale = q.shape[-1] ** -0.5
    s = torch.zeros(*k.shape[:-2], k.shape[-1], v.shape[-1], dtype=v.dtype)
    ys = []
    for t in range(k.shape[-2]):
        s = gamma * s + outer(k[..., t, :], v[..., t, :])
        ys.append(torch.einsum("...rd,...r->...d", s, q[..., t, :]) * scale)
    assert torch.allclose(out.y, stack(ys), atol=1e-11)


def test_gla_matches_the_diagonal_gated_state_loop():
    """S_t = Diag(alpha_t) S_{t-1} + k_t v_t^T, alpha_t = sigmoid(k_t)^{1/tau}."""
    q, k, v = qkv()
    meth = ab.build("gla")
    out = meth(q, k, v, causal=True)
    alpha = meth.log_gate(k).exp()
    scale = q.shape[-1] ** -0.5
    s = torch.zeros(*k.shape[:-2], k.shape[-1], v.shape[-1], dtype=v.dtype)
    ys = []
    for t in range(k.shape[-2]):
        s = alpha[..., t, :].unsqueeze(-1) * s + outer(k[..., t, :], v[..., t, :])
        ys.append(torch.einsum("...rd,...r->...d", s, q[..., t, :]) * scale)
    assert torch.allclose(out.y, stack(ys), atol=1e-11)


def test_hgrn2_matches_the_tied_gate_state_loop():
    """H_t = Diag(f_t) H_{t-1} + (1 - f_t) v_t^T: the key IS the input gate."""
    q, k, v = qkv()
    meth = ab.build("hgrn2")
    out = meth(q, k, v, causal=True)
    f = meth.forget_gate(k)
    s = torch.zeros(*k.shape[:-2], k.shape[-1], v.shape[-1], dtype=v.dtype)
    ys = []
    for t in range(k.shape[-2]):
        s = f[..., t, :].unsqueeze(-1) * s + outer(1.0 - f[..., t, :], v[..., t, :])
        ys.append(torch.einsum("...rd,...r->...d", s, q[..., t, :]))
    assert torch.allclose(out.y, stack(ys), atol=1e-11)


@pytest.mark.parametrize("data_dependent", [True, False])
def test_rwkv_matches_the_eagle_finch_state_loop(data_dependent):
    """wkv_t = S_{t-1} + Diag(u) k_t v_t^T, then S_t = Diag(w_t) S_{t-1} + k_t v_t^T."""
    q, k, v = qkv()
    meth = ab.build("rwkv", data_dependent=data_dependent, bonus=0.7)
    out = meth(q, k, v, causal=True)
    decay = meth.log_decay(k).exp()
    s = torch.zeros(*k.shape[:-2], k.shape[-1], v.shape[-1], dtype=v.dtype)
    ys = []
    for t in range(k.shape[-2]):
        write = outer(k[..., t, :], v[..., t, :])
        wkv = s + 0.7 * write                       # the bonus bypasses the state
        ys.append(torch.einsum("...rd,...r->...d", wkv, q[..., t, :]))
        s = decay[..., t, :].unsqueeze(-1) * s + write
    assert torch.allclose(out.y, stack(ys), atol=1e-11)


def test_mlstm_matches_the_unstabilised_state_loop():
    """C_t, n_t and the max(., 1) readout, run without the stabiliser as a check on it.

    The stabiliser is an exactness device, not an approximation: at these magnitudes
    the naive exponential recursion does not overflow, and the two must agree to
    machine precision.
    """
    q, k, v = qkv(scale=0.5)
    meth = ab.build("mlstm")
    out = meth(q, k, v, causal=True)
    log_f, log_i = meth.gates(k)
    scale = q.shape[-1] ** -0.5
    ks = k * scale
    c = torch.zeros(*k.shape[:-2], k.shape[-1], v.shape[-1], dtype=v.dtype)
    nvec = torch.zeros(*k.shape[:-2], k.shape[-1], dtype=v.dtype)
    ys = []
    for t in range(k.shape[-2]):
        f, i = log_f[..., t].exp(), log_i[..., t].exp()
        c = f[..., None, None] * c + i[..., None, None] * outer(ks[..., t, :], v[..., t, :])
        nvec = f[..., None] * nvec + i[..., None] * ks[..., t, :]
        num = torch.einsum("...rd,...r->...d", c, q[..., t, :])
        den = torch.einsum("...r,...r->...", nvec, q[..., t, :]).abs().clamp_min(1.0)
        ys.append(num / den.unsqueeze(-1))
    assert torch.allclose(out.y, stack(ys), atol=1e-11)


def test_mlstm_stabiliser_is_the_papers_recurrence():
    """m_t = max(log f_t + m_{t-1}, log i_t), written here as a prefix scan."""
    _, k, _ = qkv()
    meth = ab.build("mlstm")
    log_f, log_i = meth.gates(k)
    m = log_i[..., 0]
    want = [m]
    for t in range(1, k.shape[-2]):
        m = torch.maximum(log_f[..., t] + m, log_i[..., t])
        want.append(m)
    got = meth(*qkv(), causal=True, return_weights=True).extras["stabilizer"]
    assert torch.allclose(got, torch.stack(want, dim=-1), atol=1e-12)


# ---------------------------------------------------------------------------
# Invariant W: the state is linear in V, so exact weights exist on the n positions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", GATED)
@pytest.mark.parametrize("causal", [False, True])
def test_weights_reproduce_the_output(name, causal):
    q, k, v = qkv(m=12, n=16)
    out = ab.build(name)(q, k, v, causal=causal, return_weights=True)
    assert torch.allclose(out.weights @ v, out.y, atol=1e-11)
    assert ab.get_class(name).expresses_weights is True


@pytest.mark.parametrize("name", GATED)
def test_objective_is_modify_and_the_anchors_are_waived(name):
    """sec:memory-state: these define a different operator; softmax fidelity is not it."""
    cls = ab.get_class(name)
    assert cls.objective is ab.Objective.MODIFY
    assert cls.limit_config() is None and cls.convergence_configs() is None
    assert cls.supports_causal is True
    assert cls.backends == ("reference",)


# ---------------------------------------------------------------------------
# Causality: the strong form -- truncating the context cannot change the past
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", GATED)
def test_a_truncated_context_gives_the_same_prefix(name):
    """Running on the first p tokens must reproduce the first p outputs.

    Stronger than the suite's perturbation test, which keeps the shapes fixed: this one
    rules out a cumulative scan whose *earlier* entries depend on later ones, which is
    how a gated recurrence actually leaks.  Equality is to 1e-13 rather than bit for
    bit only because ``weights @ v`` reduces over ``n`` columns, and the columns beyond
    the prefix -- exactly zero, and asserted so below -- still change the summation
    order.  The bit-exact version, at fixed shape, is the conformance suite's.
    """
    q, k, v = qkv(m=16, n=16)
    full = ab.build(name)(q, k, v, causal=True, return_weights=True)
    for p in (1, 3, 8, 15):
        part = ab.build(name)(q[..., :p, :], k[..., :p, :], v[..., :p, :],
                              causal=True, return_weights=True)
        assert torch.equal(full.weights[..., :p, p:],
                           torch.zeros_like(full.weights[..., :p, p:]))
        assert torch.allclose(full.weights[..., :p, :p], part.weights, atol=1e-13,
                              rtol=0), f"{name}: prefix weights p={p}"
        assert torch.allclose(full.y[..., :p, :], part.y, atol=1e-13, rtol=0), \
            f"{name}: prefix p={p}"


@pytest.mark.parametrize("name", GATED)
def test_the_future_never_enters_the_weights(name):
    q, k, v = qkv(m=12, n=16)
    out = ab.build(name)(q, k, v, causal=True, return_weights=True)
    future = ~ab.causal_mask(12, 16)
    assert torch.equal(out.weights[..., future], torch.zeros_like(out.weights[..., future]))


# ---------------------------------------------------------------------------
# A_t: input-dependent for four of the five, and not for retention
# ---------------------------------------------------------------------------
def test_retention_is_the_input_independent_member():
    """A_t = gamma_h Id depends on the head and the lag, never on the token."""
    meth = ab.build("retnet")
    _, k1, _ = qkv(seed=0)
    _, k2, _ = qkv(seed=1)
    q, _, v = qkv()
    d1 = meth(q, k1, v, causal=True, return_weights=True).extras["decay"]
    d2 = meth(q, k2, v, causal=True, return_weights=True).extras["decay"]
    assert torch.equal(d1, d2)


@pytest.mark.parametrize("name,key", [("gla", "gate"), ("hgrn2", "forget_gate"),
                                      ("rwkv", "decay"), ("mlstm", "forget_gate")])
def test_the_gate_depends_on_the_input(name, key):
    meth = ab.build(name)
    q, k1, v = qkv(seed=0)
    _, k2, _ = qkv(seed=1)
    g1 = meth(q, k1, v, causal=True).extras[key]
    g2 = meth(q, k2, v, causal=True).extras[key]
    assert not torch.allclose(g1, g2)


def test_retention_runs_several_time_constants_at_once():
    """gamma = 1 - 2^{-5-h}: the multi-scale schedule is what decay_rnn cannot express."""
    meth = ab.build("retnet")
    gamma = meth.gammas(8, dtype=torch.float64)
    assert torch.allclose(gamma, 1.0 - 2.0 ** (-5.0 - torch.arange(8, dtype=torch.float64)))
    assert bool((gamma[1:] > gamma[:-1]).all())          # later heads forget slower
    q, k, v = qkv(h=8)
    w = meth(q, k, v, causal=True, return_weights=True).weights
    content = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    for h in (0, 3, 7):
        for t, lag in ((10, 1), (10, 4), (15, 7)):
            ratio = w[:, h, t, t - lag] / content[:, h, t, t - lag]
            assert torch.allclose(ratio, torch.full_like(ratio, float(gamma[h]) ** lag),
                                  atol=1e-11), (h, lag)


def test_retention_with_one_rate_is_decay_rnn_on_identity_features():
    """Collapsing the schedule to a scalar recovers the row above it in the table."""
    q, k, v = qkv()
    a = ab.build("retnet", decay=0.9, scale=1.0)(q, k, v, causal=True).y
    b = ab.build("decay_rnn", decay=0.9, feature="identity")(q, k, v, causal=True).y
    assert torch.allclose(a, b, atol=1e-11)


def test_gla_without_a_gate_is_the_additive_recurrence():
    """alpha -> 1 as tau -> infinity: the gate is the only thing GLA adds."""
    q, k, v = qkv()
    a = ab.build("gla", gate_temp=1e12, scale=1.0)(q, k, v, causal=True).y
    b = ab.build("linear_rnn", feature="identity")(q, k, v, causal=True).y
    assert torch.allclose(a, b, atol=1e-8)


def test_eagle_and_finch_differ_only_in_whether_the_decay_moves():
    q, k, v = qkv()
    eagle = ab.build("rwkv", data_dependent=False)(q, k, v, causal=True,
                                                   return_weights=True)
    finch = ab.build("rwkv", data_dependent=True)(q, k, v, causal=True,
                                                  return_weights=True)
    flat = eagle.extras["decay"].reshape(-1)
    assert torch.equal(flat, flat[:1].expand_as(flat))        # one constant rate
    assert float(finch.extras["decay"].std()) > 0.0
    # the bonus column is identical: it never passes through the state
    t = torch.arange(16)
    assert torch.allclose(eagle.weights[..., t, t], finch.weights[..., t, t], atol=1e-12)


def test_rwkv_bonus_is_the_whole_diagonal():
    """w_tt = u <q_t, k_t> exactly: the current token bypasses the recurrence."""
    q, k, v = qkv()
    for u in (0.0, 0.7, 2.0):
        w = ab.build("rwkv", bonus=u)(q, k, v, causal=True, return_weights=True).weights
        t = torch.arange(16)
        want = u * (q * k).sum(-1)
        assert torch.allclose(w[..., t, t], want, atol=1e-11), u


# ---------------------------------------------------------------------------
# What is left of eq:attention_expectation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", GATED)
def test_none_of_them_is_row_stochastic(name):
    """The denominator is gone (or clipped): rows do not sum to one and may be signed."""
    assert ab.get_class(name).row_stochastic is False
    q, k, v = qkv()
    w = ab.build(name)(q, k, v, causal=True, return_weights=True).weights
    s = w.sum(-1)
    assert float((s - 1.0).abs().max()) > 0.1, name
    if name != "hgrn2":                       # hgrn2's write vector 1 - f is >= 0
        assert float(w.min()) < 0.0, f"{name}: expected signed weights"


@pytest.mark.parametrize("name", GATED)
def test_constant_preservation_fails(name):
    """v == c does not give y == c once the normalizer is dropped."""
    q, k, _ = qkv()
    c = torch.ones(2, 4, 16, 5, dtype=torch.float64)
    y = ab.build(name)(q, k, c, causal=True).y
    assert float((y - 1.0).abs().max()) > 0.1, name


def test_hgrn2_mass_per_channel_is_one_minus_the_product_of_the_gates():
    """sum_j (1 - f_j) prod_{u>j} f_u = 1 - prod_{u<=t} f_u <= 1, channel by channel."""
    q, k, v = qkv()
    meth = ab.build("hgrn2")
    f = meth.forget_gate(k)
    mass = []
    for t in range(16):
        g = torch.ones_like(f[..., 0, :])
        for u in range(t + 1):
            g = g * f[..., u, :]
        mass.append(1.0 - g)
    mass = torch.stack(mass, dim=-2)                 # [B, H, n, r], all in [0, 1]
    assert bool((mass >= -1e-12).all()) and bool((mass <= 1.0 + 1e-12).all())
    # row sum of the weights is <q_t, mass_t>
    got = ab.build("hgrn2")(q, k, v, causal=True, return_weights=True).weights.sum(-1)
    assert torch.allclose(got, (q * mass).sum(-1), atol=1e-11)


def test_hgrn2_at_lower_bound_one_writes_nothing():
    """f == 1 means 1 - f == 0: the degenerate top of the hierarchical schedule."""
    q, k, v = qkv()
    out = ab.build("hgrn2", lower_bound=1.0)(q, k, v, causal=True, return_weights=True)
    assert torch.equal(out.y, torch.zeros_like(out.y))
    assert torch.isfinite(out.weights).all()


def test_mlstm_normalises_exactly_when_the_mass_exceeds_one():
    """max(|n^T q|, 1): a probability-like row only where the denominator binds."""
    q, k, v = qkv()
    out = ab.build("mlstm")(q, k, v, causal=True, return_weights=True)
    s, clipped = out.weights.sum(-1), out.extras["clipped"]
    assert bool(clipped.any()) and bool((~clipped).any()), "need both regimes"
    assert torch.allclose(s[~clipped].abs(), torch.ones_like(s[~clipped]), atol=1e-11)
    assert bool((s[clipped].abs() < 1.0 + 1e-11).all())
    assert float(s.abs().max()) <= 1.0 + 1e-11       # bounded, which linear_rnn is not


def test_mlstm_is_bounded_where_the_additive_recurrence_is_not():
    """The normaliser state buys back the uniform bound the convex hull used to give."""
    gated, plain = [], []
    for n in (8, 64, 512):
        q, k, v = qkv(m=1, n=n, seed=2)
        gated.append(float(ab.build("mlstm")(q, k, v, causal=True).y.norm(dim=-1).mean()))
        plain.append(float(ab.build("linear_rnn")(q, k, v, causal=True).y.norm(dim=-1).mean()))
    assert plain[-1] > 4 * plain[0], plain          # grows with the context
    assert gated[-1] < 4 * gated[0], gated          # does not


@pytest.mark.parametrize("name", ["retnet", "gla", "hgrn2", "rwkv"])
def test_normalize_restores_a_measure(name):
    """Putting the denominator back is one keyword, and it is not the method."""
    q, k, v = qkv()
    out = ab.build(name, normalize=True)(q, k, v, causal=True, return_weights=True)
    s = out.weights.sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-10)
    plain = ab.build(name)(q, k, v, causal=True).y
    assert not torch.allclose(out.y, plain, atol=1e-6)


# ---------------------------------------------------------------------------
# prop:memory-rank: the state is finite, and it is configurable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", GATED)
def test_the_state_does_not_grow_with_the_context(name):
    meth = ab.build(name)
    short = meth.cost(1, 64, 32, 32)["state_bytes"]
    long = meth.cost(1, 65536, 32, 32)["state_bytes"]
    assert short == long == meth.cost(1, 1 << 20, 32, 32)["state_bytes"]
    assert long < ab.build("softmax").cost(1, 65536, 32, 32)["state_bytes"]


@pytest.mark.parametrize("name", GATED)
def test_state_rank_is_the_state_size_and_the_cost_model_says_so(name):
    """``state_rank`` is the ``r`` of ``H in R^{r x d_v}`` -- what S6 and S9 sweep."""
    q, k, v = qkv(d=16)
    sizes = [ab.build(name, state_rank=r).cost(16, 16, 16, 5)["state_bytes"]
             for r in (2, 4, 8, 16)]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]
    assert sizes[-1] == ab.build(name).cost(16, 16, 16, 5)["state_bytes"]
    full = ab.build(name)(q, k, v, causal=True, return_weights=True)
    cut = ab.build(name, state_rank=16)(q, k, v, causal=True, return_weights=True)
    assert torch.allclose(full.y, cut.y, atol=1e-12)
    assert ab.build(name, state_rank=4)(q, k, v, causal=True).extras["state_rank"] == 4


@pytest.mark.parametrize("name", GATED)
def test_a_bad_state_rank_is_refused(name):
    q, k, v = qkv(d=8)
    for bad in (0, -1, 9):
        with pytest.raises(ValueError, match="state_rank"):
            ab.build(name, state_rank=bad)(q, k, v, causal=True)


# hgrn2 is excluded: its "key" is the tied input gate 1 - f_t, not the key argument,
# so it is not a content-addressed associative memory and the construction below says
# nothing about it.  That is a property of the method, not a gap in the test.
@pytest.mark.parametrize("name", [n for n in GATED if n != "hgrn2"])
def test_recall_degrades_as_the_state_shrinks(name):
    """prop:memory-rank: fewer state variables, less of the context recoverable.

    Keys are orthonormal and each query repeats a key, so a large enough state can
    return the matching value; the error is monotone in how much state is left.
    """
    n, dv = 24, 4
    g = torch.Generator().manual_seed(11)
    k = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))[0]
    k = k.reshape(1, 1, n, n)
    v = torch.randn(1, 1, n, dv, generator=g, dtype=torch.float64)
    err = []
    for r in (2, 6, 24):
        y = ab.build(name, state_rank=r)(k, k, v, causal=False).y
        err.append(float((y - v).norm(dim=-1).mean()))
    assert err[0] >= err[-1], (name, err)
