"""Mathematics specific to the *time-invariant* state-space lineage (``sec:memory-state``).

``s4d``, ``s5``, ``lru`` and ``hyena`` share one defining property: the state dynamics
are **input-independent**.  ``A`` is a parameter, not a function of ``x_t``, so the
operator is a convolution in the lag and the queries and the keys are never read.  The
first block of tests pins that down, because it is the claim the family is in the
library to make and the thing the gated linear RNNs (``retnet``, ``gla``, ``mamba``-like
``decay_rnn``) gave up.

``mega`` is the deliberate partial exception, and the last block says exactly how far
the property holds for it.

The conformance suite already covers shapes, Invariant W, causal leakage, determinism
and the rest; nothing here repeats it.
"""
from __future__ import annotations

import pytest
import torch

import attnbank as ab
from attnbank.methods.recurrent._ssm_kernels import (
    channel_conv,
    diagonal_scan,
    lag_and_visibility,
    s4d_poles,
)

#: the four operators whose mixing is a fixed function of the lag
LTI = ["s4d", "s5", "lru", "hyena"]
#: the three with a genuine finite-dimensional linear state
LINEAR_STATE = ["s4d", "s5", "lru"]
FAMILY = LTI + ["mega"]


def qkv(b=2, h=2, m=12, n=16, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


# ---------------------------------------------------------------------------
# The family's defining property: the dynamics do not depend on the input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", LTI)
@pytest.mark.parametrize("causal", [False, True])
def test_the_operator_never_reads_the_queries_or_the_keys(name, causal):
    """An LTI layer mixes positions by lag; there is nowhere for q and k to enter."""
    q, k, v = qkv()
    meth = ab.build(name)
    base = meth(q, k, v, causal=causal)
    other = meth(torch.randn_like(q), torch.randn_like(k), v, causal=causal)
    assert torch.equal(base.y, other.y), name
    assert base.extras["reads_qk"] is False


@pytest.mark.parametrize("name", LINEAR_STATE)
def test_the_mixing_is_time_invariant(name):
    """y_t = sum_l K[l] v_{t-l}: delaying the input delays the output unchanged.

    This is what "the dynamics are input-independent" buys and what a data-dependent
    gate destroys -- ``decay_rnn`` with a learned gate would fail this.
    """
    n, shift = 24, 5
    g = torch.Generator().manual_seed(3)
    v = torch.randn(1, 1, n, 4, generator=g, dtype=torch.float64)
    v_delayed = torch.cat([torch.zeros_like(v[..., :shift, :]), v[..., :-shift, :]], -2)
    q = k = torch.zeros(1, 1, n, 4, dtype=torch.float64)
    meth = ab.build(name)
    y = meth(q, k, v, causal=True).y
    y_delayed = meth(q, k, v_delayed, causal=True).y
    assert torch.allclose(y_delayed[..., shift:, :], y[..., :-shift, :], atol=1e-12), name


@pytest.mark.parametrize("name", LINEAR_STATE)
def test_the_linear_state_models_are_linear_in_the_values(name):
    """No gate: scaling the values scales the output, and superposition holds."""
    q, k, v = qkv()
    meth = ab.build(name)
    assert torch.allclose(meth(q, k, 2.0 * v, causal=True).y,
                          2.0 * meth(q, k, v, causal=True).y, atol=1e-12), name


def test_hyena_is_not_linear_in_the_values_because_the_gate_is_data_controlled():
    """Order-N Hyena is a degree-(N+1) form in v: the gating is the data dependence."""
    q, k, v = qkv()
    for order in (1, 2):
        meth = ab.build("hyena", order=order)
        y1 = meth(q, k, v, causal=True).y
        y2 = meth(q, k, 2.0 * v, causal=True).y
        assert torch.allclose(y2, (2.0 ** (order + 1)) * y1, atol=1e-11), order
        assert not torch.allclose(y2, 2.0 * y1, atol=1e-6)


@pytest.mark.parametrize("name", LTI)
def test_input_independent_dynamics_cannot_do_associative_recall(name):
    """``prop:memory-rank``'s companion failure: no content addressing at all.

    Softmax retrieves the value whose key matches the query; an operator that never
    reads the query returns the same thing whichever item is asked for.  This is the
    gap ``sec:memory-state`` says the data-dependent recurrences were introduced to
    close, and it is not a matter of state *size*.
    """
    n, dv = 8, 4
    keys = torch.eye(n, dtype=torch.float64).reshape(1, 1, n, n) * 8.0
    g = torch.Generator().manual_seed(11)
    v = torch.randn(1, 1, n, dv, generator=g, dtype=torch.float64)
    ask = lambda i: keys[:, :, i:i + 1, :]

    soft = ab.build("softmax")
    assert float((soft(ask(2), keys, v).y - v[:, :, 2:3, :]).abs().max()) < 1e-3
    assert float((soft(ask(5), keys, v).y - v[:, :, 5:5 + 1, :]).abs().max()) < 1e-3

    meth = ab.build(name)
    assert torch.equal(meth(ask(2), keys, v).y, meth(ask(5), keys, v).y), name


# ---------------------------------------------------------------------------
# s4d: the kernel is the one S4D writes down, and the FFT path agrees
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("init", ["lin", "inv"])
def test_s4d_matches_an_explicit_per_channel_state_loop(init):
    """h_t = lam h_{t-1} + Bbar v_t per channel, y_t = 2 Re(C h_t)."""
    q, k, v = qkv(m=16, n=16, dv=4)
    meth = ab.build("s4d", init=init)
    out = meth(q, k, v, causal=True)
    lam, b_bar, c = meth._params(v.shape[-1], v.device)       # [dv, N] complex128
    h = torch.zeros(*v.shape[:-2], v.shape[-1], lam.shape[-1], dtype=lam.dtype)
    ys = []
    for t in range(v.shape[-2]):
        h = lam * h + b_bar * v[..., t, :].unsqueeze(-1).to(lam.dtype)
        ys.append(2.0 * (c * h).sum(-1).real)
    assert torch.allclose(out.y, torch.stack(ys, dim=-2), atol=1e-11)


def test_s4d_frequency_domain_evaluation_agrees_with_the_direct_one():
    """The FFT path is the same operator; it is opt-in only because it is not
    bit-exactly causal, and the conformance suite checks causality bit for bit."""
    q, k, v = qkv(m=16, n=16)
    for causal in (False, True):
        direct = ab.build("s4d")(q, k, v, causal=causal).y
        fourier = ab.build("s4d", fft=True)(q, k, v, causal=causal).y
        assert torch.allclose(direct, fourier, atol=1e-10), causal


def test_s4d_poles_are_stable_and_placed_where_the_paper_puts_them():
    lin = s4d_poles(4, "lin")
    assert torch.allclose(lin.real, torch.full((4,), -0.5, dtype=torch.float64))
    assert torch.allclose(lin.imag, torch.tensor([0.0, 1.0, 2.0, 3.0],
                                                 dtype=torch.float64) * torch.pi)
    assert (s4d_poles(8, "inv").real < 0).all()
    with pytest.raises(ValueError, match="init must be one of"):
        s4d_poles(4, "legs")


def test_s4d_kernels_differ_across_channels_which_is_why_it_has_no_weights():
    """The honest reason for ``expresses_weights = False``: one kernel per channel."""
    q, k, v = qkv(dv=6)
    kern = ab.build("s4d")(q, k, v, causal=True).extras["kernel"]
    assert kern.shape == (6, v.shape[-2])
    assert float((kern[0] - kern[1]).abs().max()) > 1e-6
    assert ab.get_class("s4d").expresses_weights is False


# ---------------------------------------------------------------------------
# s5 and lru: one recurrence, two derivations
# ---------------------------------------------------------------------------
def _run_sequentially(extras, v):
    """h_t = lam h_{t-1} + Bbar v_t,  y_t = Re(C_eff h_t) + D v_t -- the whole operator."""
    lam, b_bar = extras["lam"], extras["b_bar"]
    c_eff, d_skip = extras["c_eff"], extras["d_skip"]
    h = torch.zeros(*v.shape[:-2], lam.shape[-1], dtype=lam.dtype)
    ys = []
    for t in range(v.shape[-2]):
        h = lam * h + torch.einsum("...c,kc->...k", v[..., t, :].to(lam.dtype), b_bar)
        ys.append(torch.einsum("...k,ck->...c", h, c_eff).real + d_skip * v[..., t, :])
    return torch.stack(ys, dim=-2)


@pytest.mark.parametrize("name", ["s5", "lru"])
def test_the_parallel_scan_computes_the_sequential_recurrence(name):
    """S5's contribution: a diagonal recurrence is an associative scan, exactly."""
    q, k, v = qkv(m=16, n=16)
    out = ab.build(name)(q, k, v, causal=True)
    assert torch.allclose(out.y, _run_sequentially(out.extras, v), atol=1e-11), name


def test_s5_and_lru_are_the_same_recurrence_with_different_parametrisations():
    """``sec:memory-state``: it is the structure, not the state-space derivation.

    One loop -- the loop above -- reproduces both methods from their own parameters.
    All that differs is where ``lambda`` comes from: a discretised HiPPO-style pole for
    S5, a polar parametrisation initialised on a ring for the LRU.
    """
    q, k, v = qkv(m=16, n=16)
    e5 = ab.build("s5")(q, k, v, causal=True).extras
    el = ab.build("lru")(q, k, v, causal=True).extras
    assert set(e5) >= {"lam", "b_bar", "c_eff", "d_skip"}
    assert {key: e5[key].shape for key in ("lam", "b_bar", "c_eff", "d_skip")} == \
           {key: el[key].shape for key in ("lam", "b_bar", "c_eff", "d_skip")}
    # and the two are genuinely different operators, not the same numbers twice
    assert not torch.allclose(e5["lam"], el["lam"])


@pytest.mark.parametrize("r_max", [0.5, 0.99, 1.0])
def test_lru_is_stable_by_construction_for_every_ring(r_max):
    """|lambda| = exp(-exp(nu)) < 1 for any real nu: no projection, no clipping."""
    q, k, v = qkv()
    out = ab.build("lru", r_max=r_max, r_min=0.0)(q, k, v, causal=True)
    radius = out.extras["radius"]
    assert (radius < 1.0).all() and (radius > 0.0).all(), float(radius.max())
    assert torch.isfinite(out.y).all()


def test_lru_normalisation_keeps_the_state_bounded_as_the_ring_approaches_one():
    """gamma = sqrt(1 - |lambda|^2) is what stops long memory from blowing the forward
    pass up -- the ``proper normalization`` of Orvieto et al."""
    norms = {}
    for r_min in (0.0, 0.9, 0.99):
        q, k, v = qkv(m=1, n=256, seed=4)
        y = ab.build("lru", r_min=r_min, r_max=1.0, feedthrough=False)(
            q, k, v, causal=True).y
        norms[r_min] = float(y.norm(dim=-1).mean())
    assert max(norms.values()) < 8 * min(norms.values()), norms


def test_the_scan_component_matches_a_sequential_loop():
    g = torch.Generator().manual_seed(7)
    for n in (1, 2, 5, 16, 17):
        lam = torch.complex(torch.rand(3, generator=g, dtype=torch.float64) * 0.8,
                            torch.rand(3, generator=g, dtype=torch.float64) * 0.3)
        u = torch.complex(torch.randn(n, 3, generator=g, dtype=torch.float64),
                          torch.randn(n, 3, generator=g, dtype=torch.float64))
        h, ref = torch.zeros(3, dtype=lam.dtype), []
        for t in range(n):
            h = lam * h + u[t]
            ref.append(h)
        assert torch.allclose(diagonal_scan(lam, u), torch.stack(ref), atol=1e-12), n


def test_channel_conv_is_exactly_zero_off_the_causal_support():
    """The reason the direct evaluation is the default: causality holds bit for bit."""
    lag, visible = lag_and_visibility(4, 4, causal=True)
    kernel = torch.arange(1, 5, dtype=torch.float64).reshape(1, 4)
    x = torch.zeros(4, 1, dtype=torch.float64)
    x[3, 0] = 1.0                                   # only the last input is nonzero
    y = channel_conv(kernel, x, lag, visible).squeeze(-1)
    assert torch.equal(y[:3], torch.zeros(3, dtype=torch.float64))
    assert float(y[3]) == 1.0


# ---------------------------------------------------------------------------
# hyena: an implicit filter, and no finite state
# ---------------------------------------------------------------------------
def test_hyena_filters_are_normalised_and_independent_of_the_input():
    q, k, v = qkv(dv=4)
    meth = ab.build("hyena", order=2)
    a = meth(q, k, v, causal=True).extras["filters"]
    b = meth(q, k, 3.0 * v + 1.0, causal=True).extras["filters"]
    assert a.shape == (2, 4, v.shape[-2])
    assert torch.equal(a, b)                                   # the filter is a parameter
    assert torch.allclose(a.abs().sum(-1), torch.ones_like(a[..., 0]), atol=1e-12)


def test_hyena_filter_length_follows_the_sequence_because_it_is_implicit():
    """A filter parametrised as a function of the lag has no fixed length."""
    for n in (8, 32):
        q, k, v = qkv(m=4, n=n)
        got = ab.build("hyena")(q, k, v, causal=True).extras
        assert got["filter_length"] == n


def test_hyena_has_no_finite_state_and_says_so_in_its_cost():
    """Every other method here keeps ``state_bytes`` fixed as the context grows; an
    arbitrary long convolution cannot, and ``prop:memory-rank`` has nothing to bound."""
    hy = ab.build("hyena")
    assert hy.cost(1, 4096, 64, 64)["state_bytes"] > 10 * hy.cost(1, 64, 64, 64)["state_bytes"]
    for name in LINEAR_STATE:
        meth = ab.build(name)
        assert meth.cost(1, 64, 64, 64)["state_bytes"] == \
               meth.cost(1, 65536, 64, 64)["state_bytes"], name


def test_s5_carries_one_state_for_all_channels_where_s4d_carries_one_each():
    """The MIMO simplification is a capacity statement, not a bookkeeping one."""
    big = dict(m=1, n=4096, d=64, dv=64)
    per_channel = ab.build("s4d", n_state=16).cost(**big)["state_bytes"]
    shared = ab.build("s5", n_state=16).cost(**big)["state_bytes"]
    assert shared * 64 == per_channel
    assert shared == ab.build("lru", n_state=16).cost(**big)["state_bytes"]
    assert per_channel < ab.build("softmax").cost(**big)["state_bytes"]


@pytest.mark.parametrize("name", LINEAR_STATE)
def test_state_size_is_configurable_because_the_capacity_sweeps_vary_it(name):
    q, k, v = qkv()
    for n_state in (4, 8, 32):
        out = ab.build(name, n_state=n_state)(q, k, v, causal=True)
        assert out.extras["n_state"] == n_state
        assert torch.isfinite(out.y).all()
    assert ab.build(name, n_state=4).cost(1, 64, 8, 8)["state_bytes"] < \
           ab.build(name, n_state=32).cost(1, 64, 8, 8)["state_bytes"]


# ---------------------------------------------------------------------------
# mega: the partial exception
# ---------------------------------------------------------------------------
def test_mega_without_its_ema_or_its_chunking_is_dense_softmax():
    """Not declared as ``limit_config`` -- Mega does not target softmax -- but it is
    the cleanest check that the attention half is implemented correctly."""
    q, k, v = qkv(m=16, n=16)
    for causal in (False, True):
        got = ab.build("mega", chunk=None, ema=False)(q, k, v, causal=causal,
                                                      return_weights=True)
        ref = ab.build("softmax")(q, k, v, causal=causal, return_weights=True)
        assert torch.equal(got.y, ref.y), causal
        assert torch.equal(got.weights, ref.weights), causal


def test_mega_does_read_the_queries_and_keys_unlike_the_rest_of_the_family():
    """The caveat in the family's story: Mega's attention branch is fully
    input-dependent; only its moving average has input-independent dynamics."""
    q, k, v = qkv()
    meth = ab.build("mega")
    assert not torch.allclose(meth(q, k, v, causal=True).y,
                              meth(torch.randn_like(q), k, v, causal=True).y)


def test_the_mega_ema_has_input_independent_dynamics():
    """EMA(a x + b z) == a EMA(x) + b EMA(z): the coefficients are parameters."""
    meth = ab.build("mega")
    g = torch.Generator().manual_seed(9)
    x = torch.randn(1, 1, 12, 5, generator=g, dtype=torch.float64)
    z = torch.randn(1, 1, 12, 5, generator=g, dtype=torch.float64)
    assert torch.allclose(meth._smooth(2.0 * x - 3.0 * z),
                          2.0 * meth._smooth(x) - 3.0 * meth._smooth(z), atol=1e-12)


def test_the_mega_ema_is_a_decaying_causal_filter():
    meth = ab.build("mega", ndim=4)
    kernel = meth._ema_kernel(3, 64, torch.device("cpu"), torch.float64)
    assert kernel.shape == (3, 64)
    assert float(kernel[:, -1].abs().max()) < float(kernel[:, 0].abs().max())
    assert torch.isfinite(kernel).all()


@pytest.mark.parametrize("chunk", [2, 4, 8])
def test_mega_attention_is_confined_to_its_chunk(chunk):
    q, k, v = qkv(m=16, n=16)
    out = ab.build("mega", chunk=chunk)(q, k, v, causal=True, return_weights=True)
    w = out.weights
    pos = torch.arange(16)
    same = (pos[:, None] // chunk) == (pos[None, :] // chunk)
    assert torch.equal(w[..., ~same], torch.zeros_like(w[..., ~same]))
    assert torch.allclose(w.sum(-1), torch.ones_like(w[..., 0]), atol=1e-12)
    assert float(out.extras["support_size"].max()) <= chunk


def test_mega_recalls_inside_its_chunk_and_cannot_outside_it():
    """The chunked variant is linear-cost and therefore locally blind: that is the
    trade ``sec:memory-state`` records about the headline model."""
    n, dv, chunk = 16, 4, 4
    keys = torch.eye(n, dtype=torch.float64).reshape(1, 1, n, n) * 8.0
    g = torch.Generator().manual_seed(13)
    v = torch.randn(1, 1, n, dv, generator=g, dtype=torch.float64)
    meth = ab.build("mega", chunk=chunk, ema=False)
    # a single query is aligned with the LAST key, so it sits at position 15, chunk 3
    inside = meth(keys[:, :, 15:16, :], keys, v).y
    assert float((inside - v[:, :, 15:16, :]).abs().max()) < 1e-3
    outside = meth(keys[:, :, 2:3, :], keys, v, return_weights=True)   # key 2: chunk 0
    assert float(outside.weights[..., 0, 2]) == 0.0
    assert float((outside.y - v[:, :, 2:3, :]).abs().max()) > 0.5


# ---------------------------------------------------------------------------
# The declarations the family makes about itself
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", FAMILY)
def test_the_family_modifies_the_operator_and_claims_no_softmax_anchor(name):
    """``MODIFY``: these were never approximations of softmax, so neither anchor
    applies, and DESIGN.md Sec. 2 exempts them rather than letting them skip quietly."""
    cls = ab.get_class(name)
    assert cls.objective is ab.Objective.MODIFY, name
    assert cls.limit_config() is None and cls.convergence_configs() is None, name
    assert cls.category is ab.Category.RECURRENT
    assert cls.supports_causal is True
    assert cls.backends == ("reference",), name
    assert cls.paper_section == "sec:memory-state"


@pytest.mark.parametrize("name", LTI)
def test_the_convolutional_members_decline_to_invent_a_measure(name):
    """A per-channel (or matrix-valued) kernel is not a measure on the ``n`` positions,
    and the library says so rather than averaging one into existence."""
    cls = ab.get_class(name)
    assert cls.expresses_weights is False, name
    q, k, v = qkv()
    assert ab.build(name)(q, k, v, causal=True, return_weights=True).weights is None


def test_mega_is_the_member_that_can_be_scored_by_the_decomposition_studies():
    cls = ab.get_class("mega")
    assert cls.expresses_weights is True and cls.row_stochastic is True
    q, k, v = qkv()
    out = ab.build("mega")(q, k, v, causal=True, return_weights=True)
    assert torch.allclose(out.weights @ v, out.y, atol=1e-11)
