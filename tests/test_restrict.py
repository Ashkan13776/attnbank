"""Mathematics specific to ``restrict/``: prop:sparse-identity and what it implies."""
from __future__ import annotations

import itertools

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M

SELECTORS = ["topk", "window", "sink_window", "blocktopk", "value_topk", "lsh_sample"]


def qkv(b=2, h=2, m=10, n=16, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def softmax_out(q, k, v, causal=False):
    return ab.build("softmax")(q, k, v, causal=causal, return_weights=True)


# ---------------------------------------------------------------------------
# prop:sparse-identity holds for every selector that restricts and renormalizes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["topk", "window", "sink_window", "blocktopk", "value_topk"])
@pytest.mark.parametrize("causal", [False, True])
def test_restriction_identity_for_every_selector(name, causal):
    q, k, v = qkv()
    ref = softmax_out(q, k, v, causal)
    out = ab.build(name)(q, k, v, causal=causal, return_weights=True)
    r = M.restriction_identity(ref.weights, out.extras["support"], v)
    assert torch.allclose(ref.y - out.y, r["predicted_error"], atol=1e-11), name
    assert torch.allclose(r["eps"], out.extras["eps"], atol=1e-12)
    assert torch.allclose(M.total_variation(ref.weights, out.weights), r["eps"], atol=1e-11)
    assert (r["predicted_error"].norm(dim=-1)
            <= r["eps"] * M.value_oscillation(v).unsqueeze(-1) + 1e-10).all()


@pytest.mark.parametrize("name", SELECTORS)
def test_the_output_is_the_retained_conditional_mean(name):
    """y_hat = m_{S_q}: the whole of prop:sparse-identity in one line."""
    q, k, v = qkv()
    ref = softmax_out(q, k, v)
    out = ab.build(name)(q, k, v, return_weights=True)
    if name == "lsh_sample":
        pytest.skip("sampling reweights by 1/u; it is not a restriction (sec:sparse-content)")
    S = out.extras["support"]
    m_S = (ref.weights * S) @ v / (ref.weights * S).sum(-1, keepdim=True)
    assert torch.allclose(out.y, m_S, atol=1e-11)


# ---------------------------------------------------------------------------
# prop:error-topk: top-k is optimal for mass, not for output
# ---------------------------------------------------------------------------
def test_topk_minimizes_omitted_mass_over_all_supports():
    q, k, v = qkv(b=1, h=1, m=5, n=10)
    pi = softmax_out(q, k, v).weights
    for budget in (2, 3, 5):
        eps_topk = ab.build("topk", k=budget)(q, k, v).extras["eps"]
        best = torch.full_like(eps_topk, float("inf"))
        for comb in itertools.combinations(range(10), budget):
            S = torch.zeros(10, dtype=torch.bool)
            S[list(comb)] = True
            best = torch.minimum(best, M.omitted_mass(pi, S.expand_as(pi)))
        assert torch.allclose(eps_topk, best, atol=1e-12), budget


def test_topk_is_not_output_optimal_the_papers_counterexample():
    """prop:error-topk / sec:sparse-operator: pi = (.45,.35,.20), v = (0,0,1), k = 2."""
    pi = torch.tensor([[0.45, 0.35, 0.20]], dtype=torch.float64)
    v = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float64)
    y = pi @ v
    def err(sel):
        S = torch.zeros(1, 3, dtype=torch.bool)
        S[0, list(sel)] = True
        p = (pi * S) / (pi * S).sum(-1, keepdim=True)
        return float((y - p @ v).norm())
    assert err((0, 1)) == pytest.approx(0.2)         # top-2 by mass
    assert err((0, 2)) == pytest.approx(0.2 / 0.65 - 0.2, abs=1e-12)
    assert err((0, 2)) < err((0, 1))


@pytest.mark.parametrize("budget", [2, 3, 4])
def test_value_topk_greedy_is_bounded_by_the_exhaustive_optimum(budget):
    """The greedy selector is an upper bound on the true min of eps_S ||m_{S^c} - m_S||."""
    q, k, v = qkv(b=1, h=1, m=6, n=9, seed=3)
    ref = softmax_out(q, k, v)
    greedy = ab.build("value_topk", k=budget)(q, k, v)
    exact = ab.build("value_topk", k=budget, exhaustive=True)(q, k, v)
    e_greedy = (ref.y - greedy.y).norm(dim=-1)
    e_exact = (ref.y - exact.y).norm(dim=-1)
    assert (e_exact <= e_greedy + 1e-12).all()
    assert float(e_greedy.mean()) < 1.6 * float(e_exact.mean()) + 1e-12


@pytest.mark.parametrize("budget", [2, 3, 5, 8])
def test_value_topk_beats_topk_on_output_error(budget):
    """The claim of sec:3.3 and sec:7 that the library exists to test."""
    q, k, v = qkv(b=4, h=4, m=10, n=16, seed=4)
    ref = softmax_out(q, k, v)
    e_top = (ref.y - ab.build("topk", k=budget)(q, k, v).y).norm(dim=-1)
    e_val = (ref.y - ab.build("value_topk", k=budget)(q, k, v).y).norm(dim=-1)
    assert float(e_val.mean()) < float(e_top.mean()), (budget, e_val.mean(), e_top.mean())
    # top-k is one of the candidate supports, so the guarantee is per query, not on average
    assert (e_val <= e_top + 1e-12).all()


def test_value_topk_pays_for_it_in_omitted_mass():
    """And it omits *more* mass than top-k, which is the point: mass is the wrong target."""
    q, k, v = qkv(b=4, h=4, seed=4)
    a = ab.build("topk", k=4)(q, k, v).extras["eps"]
    b = ab.build("value_topk", k=4)(q, k, v).extras["eps"]
    assert float(b.mean()) > float(a.mean())


def test_value_topk_objective_equals_the_realised_output_error():
    """eps_S ||m_{S^c} - m_S|| = ||y - m_S||, the simplification the greedy uses."""
    q, k, v = qkv(seed=5)
    pi = softmax_out(q, k, v).weights
    out = ab.build("value_topk", k=4)(q, k, v)
    r = M.restriction_identity(pi, out.extras["support"], v)
    objective = r["eps"].unsqueeze(-1) * (r["m_Sc"] - r["m_S"])
    assert torch.allclose(objective.norm(dim=-1), (pi @ v - out.y).norm(dim=-1), atol=1e-11)


# ---------------------------------------------------------------------------
# prop:sparse-certificate
# ---------------------------------------------------------------------------
def test_a_posteriori_certificate_bounds_the_omitted_mass():
    """eps_q <= N_c e^tau / (Z_S + N_c e^tau) with tau an upper bound on the omitted scores."""
    q, k, v = qkv(seed=6)
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)
    for budget in (2, 4, 8):
        out = ab.build("topk", k=budget)(q, k, v)
        S = out.extras["support"]
        tau = s.masked_fill(S, float("-inf")).amax(-1)           # the true max outside S
        z_s = (torch.exp(s - s.amax(-1, keepdim=True)) * S).sum(-1)
        n_c = (~S).sum(-1).to(s.dtype)
        shift = s.amax(-1)
        bound = n_c * torch.exp(tau - shift) / (z_s + n_c * torch.exp(tau - shift))
        assert (out.extras["eps"] <= bound + 1e-12).all(), budget


# ---------------------------------------------------------------------------
# Structure of the fixed patterns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
def test_window_always_contains_the_diagonal(causal):
    q, k, v = qkv(m=16, n=16)
    for w in (1, 2, 5, 16):
        S = ab.build("window", w=w)(q, k, v, causal=causal).extras["support"]
        diag = torch.arange(16)
        assert bool(S[..., diag, diag].all()), w
        assert bool((S.sum(-1) >= 1).all())


def test_sink_window_contains_the_window():
    q, k, v = qkv(m=16, n=16)
    sw = ab.build("sink_window", sinks=3, w=4)(q, k, v, causal=True).extras["support"]
    w = ab.build("window", w=4)(q, k, v, causal=True).extras["support"]
    assert bool((sw | w == sw).all())


def test_keeping_the_sink_beats_a_longer_window_when_the_sink_carries_mass():
    """ex:sparse-sink: the sink carries normalization, not information."""
    b, h, m, n, d, dv = 1, 1, 8, 32, 4, 3
    g = torch.Generator().manual_seed(11)
    q = torch.randn(b, h, m, d, generator=g, dtype=torch.float64)
    k = torch.randn(b, h, n, d, generator=g, dtype=torch.float64)
    v = torch.randn(b, h, n, dv, generator=g, dtype=torch.float64)
    k[..., 0, :] = 6.0 * q.mean(-2)        # token 0 is a sink: every query scores it highly
    v[..., 0, :] = 0.0                     # ... with a null value
    ref = softmax_out(q, k, v, causal=True)
    assert float(ref.weights[..., 0].mean()) > 0.3, "construct a stronger sink"
    e_win = float((ref.y - ab.build("window", w=5)(q, k, v, causal=True).y).norm(dim=-1).mean())
    e_sink = float((ref.y - ab.build("sink_window", sinks=1, w=4)(
        q, k, v, causal=True).y).norm(dim=-1).mean())
    assert e_sink < e_win, (e_sink, e_win)


# ---------------------------------------------------------------------------
# Granularity: blocks cannot beat tokens at the same budget
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("block", [2, 4, 8])
def test_block_selection_omits_at_least_as_much_mass_as_token_topk(block):
    q, k, v = qkv(b=4, h=4, m=8, n=32, seed=7)
    n_blocks = 2
    bt = ab.build("blocktopk", block=block, n_blocks=n_blocks)(q, k, v)
    tk = ab.build("topk", k=int(bt.extras["k_realized"].max()))(q, k, v)
    assert float(tk.extras["eps"].mean()) <= float(bt.extras["eps"].mean()) + 1e-12


def test_quest_bound_dominates_the_true_block_max():
    """Quest's elementwise bound never underestimates a page's largest score."""
    q, k, v = qkv(b=2, h=2, m=8, n=32, seed=8)
    meth = ab.build("blocktopk", block=4, n_blocks=3, score="bound")
    beta = q.shape[-1] ** -0.5
    bound = meth._block_scores(q, k, beta)
    exact = ab.build("blocktopk", block=4, n_blocks=3, score="max")._block_scores(q, k, beta)
    assert (bound >= exact - 1e-10).all()


# ---------------------------------------------------------------------------
# Sampling is not restriction (sec:sparse-content, prop:error-ratio-est)
# ---------------------------------------------------------------------------
def test_snis_numerator_and_normalizer_are_unbiased():
    """eq:sparse-snis: E[sum_{j in S} g_j/u_j] = Z, and likewise for the numerator."""
    q, k, v = qkv(b=1, h=1, m=4, n=10, seed=9)
    s = ab.scores(q, k, q.shape[-1] ** -0.5, causal=False)
    g = torch.exp(s - s.amax(-1, keepdim=True))
    z = g.sum(-1)
    num = g @ v
    zs, nums = [], []
    for seed in range(1200):
        out = ab.build("lsh_sample", n_samples=3, alpha=0.6, seed=seed)(q, k, v,
                                                                        return_weights=True)
        take, u = out.extras["support"], out.extras["u"]
        w = torch.where(take & (u > 0), g / u.clamp_min(1e-30), torch.zeros_like(g))
        zs.append(w.sum(-1))
        nums.append(w @ v)
    z_hat = torch.stack(zs).mean(0)
    n_hat = torch.stack(nums).mean(0)
    assert float(((z_hat - z).abs() / z).max()) < 0.10
    assert float(((n_hat - num).norm(dim=-1) / num.norm(dim=-1)).max()) < 0.25


def test_snis_bias_shrinks_with_the_sample_size():
    """prop:error-ratio-est: the ratio has O(1/N_s) bias, of lower order than its sd."""
    q, k, v = qkv(b=1, h=1, m=6, n=12, seed=10)
    y = softmax_out(q, k, v).y
    biases = []
    for ns in (2, 4, 8):
        ys = torch.stack([ab.build("lsh_sample", n_samples=ns, alpha=0.7, seed=s)(q, k, v).y
                          for s in range(600)])
        biases.append(float((ys.mean(0) - y).norm(dim=-1).mean()))
    assert biases[0] > biases[-1], biases


def test_sampling_does_not_systematically_reallocate_tail_mass():
    """Unlike restriction, SNIS keeps the tail in expectation rather than folding it in."""
    q, k, v = qkv(b=1, h=1, m=6, n=12, seed=10)
    y = softmax_out(q, k, v).y
    ns = 4
    bias_sample = (torch.stack([ab.build("lsh_sample", n_samples=ns, alpha=0.7, seed=s)(
        q, k, v).y for s in range(600)]).mean(0) - y).norm(dim=-1).mean()
    bias_topk = (ab.build("topk", k=ns)(q, k, v).y - y).norm(dim=-1).mean()
    assert float(bias_sample) < float(bias_topk)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("budget", [2, 3, 4])
def test_exhaustive_oracle_equals_brute_force(causal, budget):
    """The oracle must be an oracle: it reproduces the true minimum over all supports."""
    q, k, v = qkv(b=2, h=2, m=5, n=9, d=6, dv=3, seed=21)
    ref = softmax_out(q, k, v, causal)
    pi, y = ref.weights, ref.y
    mine = (y - ab.build("value_topk", k=budget, exhaustive=True)(
        q, k, v, causal=causal).y).norm(dim=-1)
    vis = ab.causal_mask(5, 9) if causal else torch.ones(5, 9, dtype=torch.bool)
    best = torch.full_like(mine, float("inf"))
    for comb in itertools.combinations(range(9), budget):
        S = torch.zeros(9, dtype=torch.bool)
        S[list(comb)] = True
        ok = (vis | ~S).all(-1)
        w = (pi * S).sum(-1)
        u = (pi * S) @ v
        e = (y - u / w.clamp_min(1e-300).unsqueeze(-1)).norm(dim=-1)
        best = torch.minimum(best, torch.where(ok & (w > 0), e,
                                               torch.full_like(e, float("inf"))))
    assert torch.allclose(mine, best, atol=1e-12)
