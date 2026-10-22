"""Mathematics specific to ``exact/``: the online-softmax monoid (prop:exact-monoid)."""
from __future__ import annotations

import random

import pytest
import torch

import attnbank as ab
from attnbank.methods.exact.flash import merge_states, online_state
from attnbank.methods.exact.lowprec import hadamard

ID = (float("-inf"), 0.0, 0.0)


def qkv(b=2, h=2, m=12, n=24, d=8, dv=6, seed=0, dtype=torch.float64, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def state_of(idx, s, v):
    return online_state(s[..., idx], v[..., idx, :])


def close(a, b, atol=1e-12):
    return all(torch.allclose(x, y, atol=atol) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# prop:exact-monoid (i): (Sigma, +, 0) is a commutative monoid
# ---------------------------------------------------------------------------
def test_monoid_identity_commutativity_associativity():
    q, k, v = qkv()
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)
    a = state_of([0, 1, 2], s, v)
    b = state_of([3, 4], s, v)
    c = state_of([5, 6, 7, 8], s, v)
    zero = tuple(torch.full_like(t, f) for t, f in zip(a, ID))

    assert close(merge_states(a, zero), a)
    assert close(merge_states(zero, a), a)
    assert close(merge_states(a, b), merge_states(b, a))
    assert close(merge_states(merge_states(a, b), c), merge_states(a, merge_states(b, c)))


def test_monoid_homomorphism_on_disjoint_blocks():
    """prop:exact-monoid (ii): sigma_{J1 u J2} = sigma_{J1} + sigma_{J2}."""
    q, k, v = qkv()
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)
    j1, j2 = [1, 4, 7, 11], [0, 2, 3, 20]
    assert close(state_of(sorted(j1 + j2), s, v), merge_states(state_of(j1, s, v),
                                                               state_of(j2, s, v)))


def test_state_bounds():
    """prop:exact-monoid (iv): 1 <= l_J <= |J| and ||u_J|| <= |J| max ||v_j||."""
    q, k, v = qkv(scale=8.0)                      # large logits: the bound must still hold
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)
    for idx in ([0], list(range(5)), list(range(24))):
        _, ell, u = state_of(idx, s, v)
        assert (ell >= 1 - 1e-12).all() and (ell <= len(idx) + 1e-12).all()
        assert (u.norm(dim=-1) <= len(idx) * v.norm(dim=-1).amax(-1, keepdim=True) + 1e-9).all()


# ---------------------------------------------------------------------------
# prop:exact-monoid (iii): every partition and every bracketing gives y(q)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
def test_splitk_is_partition_independent(causal):
    q, k, v = qkv()
    n = k.shape[-2]
    ref = ab.build("softmax")(q, k, v, causal=causal, return_weights=True)

    rng = random.Random(0)
    perm = list(range(n))
    rng.shuffle(perm)
    partitions = [
        [list(range(n))],                                     # one block
        [[j] for j in range(n)],                              # singletons
        [list(range(0, 7)), list(range(7, 9)), list(range(9, n))],   # uneven contiguous
        [perm[:5], perm[5:6], perm[6:19], perm[19:]],         # non-contiguous, shuffled
        [perm[i::5] for i in range(5)],                       # strided
    ]
    for part in partitions:
        for tree in (False, True):
            out = ab.build("splitk", partition=part, tree=tree)(
                q, k, v, causal=causal, return_weights=True)
            assert torch.allclose(out.y, ref.y, atol=1e-13), f"partition {part[:2]}..."
            assert torch.allclose(out.weights, ref.weights, atol=1e-14)


@pytest.mark.parametrize("block_n", [1, 2, 3, 5, 16, 64])
@pytest.mark.parametrize("causal", [False, True])
def test_flash_equals_softmax_for_every_block_size(block_n, causal):
    q, k, v = qkv()
    ref = ab.build("softmax")(q, k, v, causal=causal, return_weights=True)
    out = ab.build("flash", block_n=block_n)(q, k, v, causal=causal, return_weights=True)
    assert torch.allclose(out.y, ref.y, atol=1e-13)
    assert torch.allclose(out.weights, ref.weights, atol=1e-14)


@pytest.mark.parametrize("scale", [10.0, 40.0, 200.0])
def test_flash_survives_large_logits_where_naive_exp_overflows(scale):
    """The running maximum is the entire numerical content of part (iv): every
    intermediate is bounded independently of the logit scale, so overflow is impossible.
    Un-shifted exp(s) overflows fp32 well before these scales."""
    q, k, v = qkv(dtype=torch.float32, scale=scale)
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=True)
    assert not torch.isfinite(torch.exp(s.masked_fill(torch.isneginf(s), 0.0))).all(), \
        "pick a larger scale: un-shifted exp must actually overflow for this test to bite"
    out = ab.build("flash")(q, k, v, causal=True, return_weights=True)
    assert torch.isfinite(out.y).all() and torch.isfinite(out.weights).all()
    # fp32 rounding on an LSE of magnitude |s| gives a row-sum error of order |s| * 2^-24
    tol = max(1e-5, 4e-7 * float(s[torch.isfinite(s)].abs().max()))
    assert torch.allclose(out.weights.sum(-1), torch.ones_like(out.weights[..., 0]), atol=tol)


def test_lse_is_a_sufficient_statistic():
    """pi_q(j) = exp(s_j - LSE(q)), so the backward pass needs only LSE."""
    q, k, v = qkv()
    out = ab.build("flash")(q, k, v, causal=True, return_weights=True)
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=True)
    assert torch.allclose(torch.exp(s - out.extras["lse"].unsqueeze(-1)), out.weights,
                          atol=1e-14)


# ---------------------------------------------------------------------------
# sec:exact-boundary: the exact invariances, and where exactness ends
# ---------------------------------------------------------------------------
def test_hadamard_is_orthogonal():
    for d in (1, 2, 4, 8, 6, 12):
        r = hadamard(d, "cpu", torch.float64)
        assert torch.allclose(r @ r.T, torch.eye(d, dtype=torch.float64), atol=1e-12)


def test_exact_reparametrizations_change_nothing_without_quantization():
    """Rotation and key smoothing are exact invariances of the operator."""
    q, k, v = qkv()
    ref = ab.build("softmax")(q, k, v, return_weights=True)
    for cfg in ({"bits": None, "hadamard_qk": True}, {"bits": None, "smooth_k": True},
                {"bits": None, "hadamard_qk": True, "smooth_k": True}):
        out = ab.build("lowprec", fmt=None, bits_p=None, **cfg)(q, k, v, return_weights=True)
        assert torch.allclose(out.weights, ref.weights, atol=1e-12), cfg
        assert torch.allclose(out.y, ref.y, atol=1e-12), cfg


def test_low_precision_error_decreases_with_bit_width():
    q, k, v = qkv(dtype=torch.float32)
    ref = ab.build("softmax")(q, k, v).y
    errs = [float((ab.build("lowprec", bits=b)(q, k, v).y - ref).norm(dim=-1).mean())
            for b in (2, 3, 4, 6, 8, 12)]
    assert all(a > b for a, b in zip(errs, errs[1:])), errs


def test_low_precision_is_not_exact_and_says_so():
    """sec:exact-boundary: quantized kernels are approximations, not exact acceleration."""
    assert ab.get_class("lowprec").objective is ab.Objective.APPROXIMATE
    assert ab.get_class("flash").objective is ab.Objective.ACCELERATE
    assert ab.get_class("splitk").objective is ab.Objective.ACCELERATE


def test_splitk_cost_counts_the_merge():
    c1 = ab.build("splitk", splits=1).cost(8, 64, 16, 16)
    c8 = ab.build("splitk", splits=8).cost(8, 64, 16, 16)
    assert c8["flops"] > c1["flops"]            # the second-stage merge is extra arithmetic
    assert c8["state_bytes"] == c1["state_bytes"]
