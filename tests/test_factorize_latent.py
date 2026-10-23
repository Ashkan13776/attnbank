"""Latent bottlenecks: where the kernel composition survives, and where it does not.

``sec:factor`` draws a line that the four methods tested here sit on either side of.

* A **stochastic factorization** composes two Markov kernels, ``pi_hat_q = a(q, .)
  b(., x)``.  Nothing happens to the latents between the hops, so the composition is
  again a probability measure on the original ``n`` positions: ``abc`` and ``latte``
  return it as ``weights`` and satisfy ``prop:factor-closure``.
* A **nonlinear latent bottleneck** processes the latents between the hops.  There is
  then no kernel composition, and the induced measure may not exist at all:
  ``perceiver`` and ``isab`` are not even linear in the value field, so no matrix
  ``W(q, k)`` with ``y = W v`` exists, let alone a stochastic one.  They declare
  ``expresses_weights = False``, and the tests below are the evidence for that
  declaration rather than a convention.
"""
from __future__ import annotations

import pytest
import torch

import attnbank as ab

#: two Markov kernels, composed -- pi_hat exists on the n original positions
COMPOSITION = ["abc", "latte"]
#: processing between the hops -- no induced measure
BOTTLENECK = ["perceiver", "isab"]


def qkv(b=2, h=2, m=16, n=16, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


# ---------------------------------------------------------------------------
# the composition side: def:factor-stochastic and prop:factor-closure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", COMPOSITION)
def test_both_factors_are_markov_kernels_and_their_product_is_the_measure(name):
    """def:factor-stochastic: A 1 = 1, B 1 = 1, A, B >= 0 and P_hat = A B."""
    q, k, v = qkv()
    out = ab.build(name, r=4)(q, k, v, return_weights=True)
    a, b = out.extras["a"], out.extras["b"]
    for f in (a, b):
        assert (f >= 0).all()
        assert torch.allclose(f.sum(-1), torch.ones_like(f[..., 0]), atol=1e-12)
    assert torch.allclose(a @ b, out.weights, atol=1e-12)


@pytest.mark.parametrize("name", COMPOSITION)
def test_closure_constant_preservation_and_convex_hull(name):
    """prop:factor-closure (i)-(iii): the composition is still a measure."""
    q, k, v = qkv()
    out = ab.build(name, r=4)(q, k, v, return_weights=True)
    assert (out.weights >= -1e-12).all()
    assert torch.allclose(out.weights.sum(-1), torch.ones_like(out.weights[..., 0]),
                          atol=1e-10)
    c = torch.full_like(v, 1.7)
    y_const = ab.build(name, r=4)(q, k, c).y
    assert torch.allclose(y_const, torch.full_like(y_const, 1.7), atol=1e-9)
    assert (out.y >= v.amin(-2, keepdim=True) - 1e-9).all()
    assert (out.y <= v.amax(-2, keepdim=True) + 1e-9).all()


@pytest.mark.parametrize("name", COMPOSITION)
@pytest.mark.parametrize("r", [1, 2, 5])
def test_the_measure_has_rank_at_most_r(name, r):
    """Routing through r latents bounds the rank of P_hat by r -- the whole point."""
    q, k, v = qkv()
    out = ab.build(name, r=r)(q, k, v, return_weights=True)
    assert int(torch.linalg.matrix_rank(out.weights).max()) <= r


@pytest.mark.parametrize("name", COMPOSITION)
def test_the_composition_is_linear_in_the_value_field(name):
    """y = P_hat v with P_hat independent of v: the property the bottlenecks lose."""
    q, k, v = qkv()
    meth = ab.build(name, r=4)
    y1, y2 = meth(q, k, v).y, meth(q, k, 2 * v).y
    assert torch.allclose(2 * y1, y2, atol=1e-12)


# ---------------------------------------------------------------------------
# causality: the write kernel is normalized over the prefix
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", COMPOSITION)
def test_causal_measure_is_supported_on_the_prefix(name):
    q, k, v = qkv(m=12, n=16)
    out = ab.build(name, r=4)(q, k, v, causal=True, return_weights=True)
    mask = ab.causal_mask(12, 16)
    assert float(out.weights.masked_select(~mask).abs().max()) == 0.0
    assert torch.allclose(out.weights.sum(-1), torch.ones_like(out.weights[..., 0]),
                          atol=1e-10)


@pytest.mark.parametrize("name", COMPOSITION)
def test_the_causal_row_equals_the_method_run_on_that_prefix(name):
    """Prefix normalization, stated as a property rather than as an implementation:
    row t of the causal run is the non-causal operator applied to the first t+1 keys."""
    m = n = 12
    r = 4
    q, k, v = qkv(b=1, h=2, m=m, n=n, seed=3)
    meth = ab.build(name, r=r)
    full = meth(q, k, v, causal=True, return_weights=True)
    for t in range(r + 1, m):          # r+1 keys onward: the memory is really bounded
        part = meth(q[..., t:t + 1, :], k[..., :t + 1, :], v[..., :t + 1, :],
                    return_weights=True)
        assert torch.allclose(full.weights[..., t:t + 1, :t + 1], part.weights,
                              atol=1e-12), t
        assert torch.allclose(full.y[..., t:t + 1, :], part.y, atol=1e-12), t


# ---------------------------------------------------------------------------
# abc: bounded memory, and the unbounded-memory limit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
def test_abc_unbounded_memory_is_exactly_softmax(causal):
    """Peng et al., Sec. 3: with n = N slots and phi_i = e_i, ABC *is* softmax."""
    q, k, v = qkv()
    ref = ab.build("softmax")(q, k, v, causal=causal, return_weights=True)
    for cfg in ({"r": None}, {"r": 16}, {"r": 64}):      # r >= n is also unbounded
        out = ab.build("abc", **cfg)(q, k, v, causal=causal, return_weights=True)
        assert torch.allclose(out.weights, ref.weights, atol=1e-12), cfg
        assert torch.allclose(out.y, ref.y, atol=1e-12), cfg
        assert out.extras["unbounded"] is True


def test_abc_sees_the_context_only_through_its_r_memory_slots():
    """The bottleneck, tested directly: a value perturbation the write kernel cannot
    see -- delta in the null space of B -- leaves the output exactly where it was."""
    q, k, v = qkv(m=16, n=16, seed=4)
    meth = ab.build("abc", r=4)
    out = meth(q, k, v, return_weights=True)
    b = out.extras["b"]                                   # [..., r, n]
    g = torch.Generator().manual_seed(11)
    u = torch.randn(v.shape, generator=g, dtype=v.dtype)
    gram = b @ b.transpose(-1, -2)
    delta = u - b.transpose(-1, -2) @ torch.linalg.solve(gram, b @ u)
    assert float((b @ delta).abs().max()) < 1e-10         # delta is invisible to B
    assert float(delta.abs().max()) > 1e-3                # and is not itself trivial
    assert torch.allclose(meth(q, k, v + delta).y, out.y, atol=1e-9)


def test_abc_reads_through_context_summarized_memory_keys_but_latte_does_not():
    """The two parameterizations of the read kernel, and the difference between them:
    ABC's latent keys are k_bar_z = int k db(z, .); Latte's are free parameters."""
    q, k, v = qkv(seed=5)
    g = torch.Generator().manual_seed(12)
    k2 = k + torch.randn(k.shape, generator=g, dtype=k.dtype)
    a_abc = [ab.build("abc", r=4)(q, kk, v, return_weights=True).extras["a"]
             for kk in (k, k2)]
    a_latte = [ab.build("latte", r=4)(q, kk, v, return_weights=True).extras["a"]
               for kk in (k, k2)]
    assert float((a_abc[0] - a_abc[1]).abs().max()) > 1e-3
    assert torch.equal(a_latte[0], a_latte[1])            # bit for bit: p(l | q) only


def test_latte_is_declared_an_operator_modification():
    """Free latents never see the keys, so nothing converges to softmax."""
    cls = ab.get_class("latte")
    assert cls.objective is ab.Objective.MODIFY
    assert cls.limit_config() is None and cls.convergence_configs() is None
    assert cls.trainable is True


def test_abc_is_an_approximation_with_an_anchor():
    cls = ab.get_class("abc")
    assert cls.objective is ab.Objective.APPROXIMATE
    assert cls.limit_config() == {"r": None}


@pytest.mark.parametrize("name", COMPOSITION)
def test_the_two_causal_factorizations_declare_it(name):
    """``factorize`` had no causal method before these two; the claim is tested by the
    conformance suite's leakage check, and declared here."""
    assert ab.get_class(name).supports_causal is True
    assert ab.get_class(name).expresses_weights is True
    assert ab.get_class(name).row_stochastic is True


# ---------------------------------------------------------------------------
# the bottleneck side: no measure, and the evidence for saying so
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", BOTTLENECK)
def test_a_bottleneck_returns_no_weights_even_when_asked(name):
    q, k, v = qkv()
    out = ab.build(name, r=4)(q, k, v, return_weights=True)
    assert ab.get_class(name).expresses_weights is False
    assert out.weights is None and out.v_hat is None


@pytest.mark.parametrize("name", BOTTLENECK)
def test_a_bottleneck_is_not_linear_in_the_value_field(name):
    """The decisive fact.  If ``y = W(q, k) v`` for *any* matrix W -- stochastic or
    not -- then y is linear in v.  It is not, so there is nothing to return."""
    q, k, v = qkv(seed=6)
    meth = ab.build(name, r=4)
    y1 = meth(q, k, v).y
    y2 = meth(q, k, 2 * v).y
    y3 = meth(q, k, -v).y
    assert float((2 * y1 - y2).abs().max()) > 1e-3        # not homogeneous
    assert float((y1 + y3).abs().max()) > 1e-3            # not odd, so not additive


@pytest.mark.parametrize("name", BOTTLENECK)
def test_a_bottleneck_does_not_preserve_a_constant_value_field(name):
    """prop:factor-closure (ii) stated in reverse: every measure reproduces a constant
    value field exactly, so an operator that does not cannot be one."""
    q, k, _ = qkv(seed=7)
    g = torch.Generator().manual_seed(13)
    c = torch.randn(5, generator=g, dtype=torch.float64)
    v = c.expand(2, 2, 16, 5).contiguous()
    y = ab.build(name, r=4)(q, k, v).y
    assert float((y - c).abs().max()) > 1e-2


@pytest.mark.parametrize("name", BOTTLENECK)
def test_a_bottleneck_still_exposes_its_two_kernels_but_they_do_not_compose(name):
    """``extras`` carries the read and write attentions, as the category requires; the
    test is that multiplying them would *not* reproduce the output."""
    q, k, v = qkv(seed=8)
    out = ab.build(name, r=4)(q, k, v)
    a, b = out.extras["a"], out.extras["b"]
    assert a.shape[-2:] == (16, 4) and b.shape[-2:] == (4, 16)
    for f in (a, b):                                      # each hop is a Markov kernel
        assert (f >= 0).all()
        assert torch.allclose(f.sum(-1), torch.ones_like(f[..., 0]), atol=1e-12)
    fake = (a @ b) @ v                                    # the fabrication we refuse
    err = (out.y - fake).norm(dim=-1) / out.y.norm(dim=-1).clamp_min(1e-12)
    assert float(err.min()) > 1e-2, "a @ b happened to reproduce y; check the block"
    assert "NOT the induced measure" in out.extras["note"]


@pytest.mark.parametrize("name", BOTTLENECK)
def test_a_bottleneck_is_declared_modify_and_is_exempt_from_the_anchor(name):
    cls = ab.get_class(name)
    assert cls.objective is ab.Objective.MODIFY
    assert cls.limit_config() is None and cls.convergence_configs() is None
    assert cls.trainable is True and cls.row_stochastic is False


def test_perceiver_refuses_the_configuration_that_would_make_it_a_factorization():
    with pytest.raises(ab.ConfigError, match="latent_stochastic"):
        ab.build("perceiver", depth=0)


def test_perceiver_depth_is_a_real_stack():
    q, k, v = qkv(seed=9)
    ys = [ab.build("perceiver", r=4, depth=dep)(q, k, v).y for dep in (1, 2, 3)]
    assert float((ys[0] - ys[1]).abs().max()) > 1e-3
    assert float((ys[1] - ys[2]).abs().max()) > 1e-3
