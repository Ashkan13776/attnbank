"""Mathematics specific to the learned and query-dependent selectors.

These are all supports chosen *without* reading ``pi_q``: by a spectral page summary
(``oval``, ``page_pca``), by a random hash (``reformer``), by a trained partition
(``routing``), or by a hash plus an unbiased sample of the rest (``hyperattention``).
The conformance suite already checks that each is a well-formed method; what is
checked here is that each implements the mechanism its paper describes.
"""
from __future__ import annotations

import math

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M
from attnbank.methods.restrict.oval import page_log_mass, page_summary

#: the four that are pure renormalized restrictions. ``hyperattention`` is not: it
#: reweights its sampled keys by ``1 / u``, exactly as ``lsh_sample`` does.
SELECTORS = ["oval", "page_pca", "reformer", "routing"]
FAMILY = SELECTORS + ["hyperattention"]


def qkv(b=2, h=2, m=10, n=16, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def softmax_out(q, k, v, causal=False):
    return ab.build("softmax")(q, k, v, causal=causal, return_weights=True)


# ---------------------------------------------------------------------------
# prop:sparse-identity -- they are restrictions, whatever chose the support
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", SELECTORS)
@pytest.mark.parametrize("causal", [False, True])
def test_restriction_identity(name, causal):
    q, k, v = qkv()
    ref = softmax_out(q, k, v, causal)
    out = ab.build(name)(q, k, v, causal=causal, return_weights=True)
    r = M.restriction_identity(ref.weights, out.extras["support"], v)
    assert torch.allclose(ref.y - out.y, r["predicted_error"], atol=1e-11), name
    assert torch.allclose(r["eps"], out.extras["eps"], atol=1e-12)
    assert torch.allclose(M.total_variation(ref.weights, out.weights), r["eps"], atol=1e-11)


@pytest.mark.parametrize("name", SELECTORS)
def test_the_output_is_the_retained_conditional_mean(name):
    q, k, v = qkv()
    ref = softmax_out(q, k, v)
    out = ab.build(name)(q, k, v, return_weights=True)
    S = out.extras["support"]
    m_S = (ref.weights * S) @ v / (ref.weights * S).sum(-1, keepdim=True)
    assert torch.allclose(out.y, m_S, atol=1e-11)


@pytest.mark.parametrize("name", FAMILY)
def test_no_selector_beats_top_k_on_omitted_mass(name):
    """prop:error-topk: at the same realised budget, top-k minimizes eps_q."""
    q, k, v = qkv(b=4, h=4, seed=2)
    out = ab.build(name)(q, k, v)
    budget = int(out.extras["k_realized"].max())
    tk = ab.build("topk", k=budget)(q, k, v)
    assert float(tk.extras["eps"].mean()) <= float(out.extras["eps"].mean()) + 1e-12


# ---------------------------------------------------------------------------
# OVAL: the page summary is the one Algorithm 1 specifies
# ---------------------------------------------------------------------------
def _projector(summary):
    b_a, b_b = summary["B"]
    return b_a @ b_a.transpose(-1, -2) + b_b @ b_b.transpose(-1, -2)


def _trace(a):
    return a.diagonal(dim1=-2, dim2=-1).sum(-1)


def test_the_stored_basis_is_an_orthonormal_projector_of_the_declared_rank():
    q, k, v = qkv(n=24, d=6)
    for eta in (0.0, 0.5, 1.0):
        s = page_summary(k, v, page=4, eta=eta, rank=3)
        p = _projector(s)
        assert torch.allclose(p, p.transpose(-1, -2), atol=1e-12)
        assert torch.allclose(p @ p, p, atol=1e-10), eta        # idempotent
        assert torch.allclose(_trace(p), torch.full_like(_trace(p), 3.0), atol=1e-10)


def test_a_complete_basis_reproduces_the_page_log_mass_exactly():
    """l_B(q) = beta mu^T q + lse(beta X P_B q) is exact when P_B = I."""
    q, k, v = qkv(m=5, n=24, d=6)
    beta = 6 ** -0.5
    for page in (4, 8):
        nb = 24 // page
        exact = torch.logsumexp((beta * (q @ k.transpose(-1, -2))).unflatten(-1, (nb, page)),
                                dim=-1)
        got = page_log_mass(q, page_summary(k, v, page, 0.5, rank=6), beta)
        assert torch.allclose(got, exact, atol=1e-12), page
        partial = page_log_mass(q, page_summary(k, v, page, 0.5, rank=2), beta)
        assert float((partial - exact).abs().max()) > 1e-3     # truncation is real


def test_the_retrieval_error_obeys_the_lipschitz_bound():
    """lse is 1-Lipschitz in the sup norm, so |l - l_B| <= ||(I - P_B) beta X q||_inf."""
    q, k, v = qkv(m=6, n=24, d=6)
    beta = 6 ** -0.5
    for eta in (0.0, 1.0):
        s = page_summary(k, v, page=4, eta=eta, rank=2)
        resid = torch.einsum("...bpd,...md->...bmp", s["X"] - s["X"] @ _projector(s), beta * q)
        resid = resid.masked_fill(~s["member"][:, None, :], 0.0).abs().amax(-1)
        exact = torch.logsumexp((beta * (q @ k.transpose(-1, -2))).unflatten(-1, (6, 4)), -1)
        got = page_log_mass(q, s, beta)
        assert ((got - exact).abs() <= resid.transpose(-1, -2) + 1e-10).all(), eta


@pytest.mark.parametrize("eta", [0.0, 0.3, 1.0])
def test_the_basis_minimizes_the_hybrid_risk(eta):
    """Corollary 3.3: B_eta spans a leading invariant subspace of M_eta."""
    q, k, v = qkv(n=24, d=6)
    s = page_summary(k, v, page=4, eta=eta, rank=2)
    eye = torch.eye(6, dtype=k.dtype)
    risk = lambda p: _trace((eye - p) @ s["M"] @ (eye - p))
    mine = risk(_projector(s))
    g = torch.Generator().manual_seed(0)
    for trial in range(24):                       # no random subspace does better
        a = torch.randn(6, 2, generator=g, dtype=k.dtype)
        b = torch.linalg.qr(a).Q
        assert (mine <= risk(b @ b.T) + 1e-10).all(), (eta, trial)


def test_eta_trades_key_fidelity_for_output_sensitivity():
    """The whole content of the hybrid: R_key rises and R_out falls, monotonically."""
    q, k, v = qkv(n=32, d=8, dv=5, seed=1)
    eye = torch.eye(8, dtype=k.dtype)
    keys, outs = [], []
    for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = page_summary(k, v, page=8, eta=eta, rank=2)
        x, y, p = s["X"], s["Y"], _projector(s)
        xtx = x.transpose(-1, -2) @ x
        xty = x.transpose(-1, -2) @ y
        m_out = xty @ xty.transpose(-1, -2)
        keys.append(float(_trace((eye - p) @ xtx @ (eye - p)).mean()))
        outs.append(float(_trace((eye - p) @ m_out @ (eye - p)).mean()))
    assert all(a <= b + 1e-9 for a, b in zip(keys, keys[1:])), keys
    assert all(a >= b - 1e-9 for a, b in zip(outs, outs[1:])), outs
    assert keys[0] < keys[-1] and outs[0] > outs[-1]            # the trade is not trivial


def test_eta_zero_is_exactly_the_key_only_baseline():
    """``page_pca`` is the endpoint ``oval`` argues against, with nothing else changed."""
    q, k, v = qkv(n=32, seed=4)
    for causal in (False, True):
        a = ab.build("oval", eta=0.0, page=8, n_pages=2, rank=2)(q, k, v, causal=causal,
                                                                 return_weights=True)
        b = ab.build("page_pca", page=8, n_pages=2, rank=2)(q, k, v, causal=causal,
                                                            return_weights=True)
        assert torch.equal(a.y, b.y)
        assert torch.equal(a.weights, b.weights)
    assert (ab.build("oval", page=8, rank=2).cost(1, 4096, 128, 128)
            == ab.build("page_pca", page=8, rank=2).cost(1, 4096, 128, 128))


# ---------------------------------------------------------------------------
# OVAL: what the output-aware term buys, and what it does not
# ---------------------------------------------------------------------------
def _coupled_pages(seed, n_pages=3, page=8, d=4, dv=3, hot=1, spread=0.8, qscale=8.0):
    """Pages whose logits move along a *low key energy* direction that separates values.

    ``e0`` carries most of the key variance in every page, so a rank-1 key PCA keeps
    ``e0`` and drops ``e1``.  Page ``hot`` is the only one whose keys vary along ``e1``,
    and its values are a multiple of that same coefficient.  The query is ``e1``, so
    page ``hot`` carries nearly all the mass and a summary blind to ``e1`` cannot see it.
    """
    g = torch.Generator().manual_seed(seed)
    e0, e1 = torch.eye(d, dtype=torch.float64)[0], torch.eye(d, dtype=torch.float64)[1]
    vdir = torch.eye(dv, dtype=torch.float64)[0] * 2.0
    ks, vs = [], []
    for p in range(n_pages):
        b = spread if p == hot else 0.0
        c0 = 2.0 * torch.randn(page, 1, generator=g, dtype=torch.float64)
        c1 = b * torch.randn(page, 1, generator=g, dtype=torch.float64)
        ks.append(c0 * e0 + c1 * e1
                  + 0.05 * torch.randn(page, d, generator=g, dtype=torch.float64))
        vs.append(c1 * vdir + 0.02 * torch.randn(page, dv, generator=g,
                                                 dtype=torch.float64))
    return ((qscale * e1).view(1, 1, 1, d), torch.cat(ks).view(1, 1, -1, d),
            torch.cat(vs).view(1, 1, -1, dv))


def test_the_output_aware_basis_recovers_pages_key_pca_misses():
    """sec:sparse-query: the claim OVAL makes, on data with the structure it assumes."""
    cfg = dict(page=8, n_pages=1, rank=1)
    e_pca, e_oval, hit_pca, hit_oval = [], [], 0, 0
    for seed in range(32):
        q, k, v = _coupled_pages(seed)
        y = softmax_out(q, k, v).y
        a = ab.build("page_pca", **cfg)(q, k, v)
        b = ab.build("oval", eta=1.0, **cfg)(q, k, v)
        e_pca.append(float((y - a.y).norm()))
        e_oval.append(float((y - b.y).norm()))
        hit_pca += int(a.extras["pages_selected"].view(-1)[1])
        hit_oval += int(b.extras["pages_selected"].view(-1)[1])
    assert hit_oval > hit_pca + 5, (hit_oval, hit_pca)
    assert sum(e_oval) < 0.75 * sum(e_pca), (sum(e_oval) / 32, sum(e_pca) / 32)


def test_the_output_aware_term_buys_nothing_without_key_value_coupling():
    """And it is not free: on isotropic data eta > 0 only moves the basis off the optimum."""
    ratios = []
    for seed in range(8):
        q, k, v = qkv(b=2, h=2, m=8, n=32, d=8, dv=5, seed=seed)
        y = softmax_out(q, k, v).y
        err = lambda name, **kw: float(
            (y - ab.build(name, page=8, n_pages=1, rank=2, **kw)(q, k, v).y)
            .norm(dim=-1).mean())
        ratios.append(err("oval", eta=1.0) / err("page_pca"))
    assert sum(ratios) / len(ratios) > 0.95, sum(ratios) / len(ratios)


def test_the_page_summary_is_the_size_of_a_key_only_one():
    """The paper's efficiency claim, as a statement about the stored summary."""
    n, d, dv, page, r = 4096, 128, 128, 32, 8
    dense = ab.build("softmax").cost(1, n, d, dv)["state_bytes"]
    for name in ("oval", "page_pca"):
        c = ab.build(name, page=page, rank=r, n_pages=4).cost(1, n, d, dv)
        summary = c["state_bytes"] - dense
        assert summary == (n // page) * (d + d * r + page * r) * 2
        assert summary < 0.5 * dense                 # an index, not a second cache


# ---------------------------------------------------------------------------
# Reformer: the support is an angular hash bucket
# ---------------------------------------------------------------------------
def test_the_collision_probability_is_one_minus_the_angle_over_pi():
    """The defining property of hyperplane LSH, which is why the bucket tracks the score."""
    q, k, v = qkv(b=1, h=1, m=8, n=20, d=16, seed=5)
    qn, kn = q / q.norm(dim=-1, keepdim=True), k / k.norm(dim=-1, keepdim=True)
    predicted = 1.0 - torch.arccos((qn @ kn.transpose(-1, -2)).clamp(-1, 1)) / math.pi
    trials = 200
    empirical = torch.zeros_like(predicted)
    for seed in range(trials):
        e = ab.build("reformer", n_buckets=2, n_rounds=1, seed=seed)(q, k, v).extras
        empirical += (e["bucket_q"][0].unsqueeze(-1)
                      == e["bucket_k"][0].unsqueeze(-2)).to(predicted.dtype)
    empirical /= trials
    assert float((empirical - predicted).abs().max()) < 0.15
    assert float(torch.corrcoef(torch.stack([empirical.flatten(),
                                             predicted.flatten()]))[0, 1]) > 0.9


def test_more_hash_rounds_recover_more_mass():
    """Amplification: unioning independent buckets is how Reformer buys recall."""
    q, k, v = qkv(b=2, h=2, m=8, n=32, d=16, seed=6)
    eps = [float(torch.stack([ab.build("reformer", n_buckets=8, n_rounds=r, seed=s)(
        q, k, v).extras["eps"].mean() for s in range(4)]).mean()) for r in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(eps, eps[1:])), eps


@pytest.mark.parametrize("name", ["reformer", "routing"])
@pytest.mark.parametrize("causal", [False, True])
def test_the_query_always_sees_its_own_position(name, causal):
    """The nonempty-row guarantee: shared QK makes it automatic, here it is explicit."""
    q, k, v = qkv(m=16, n=16, seed=7)
    S = ab.build(name)(q, k, v, causal=causal).extras["support"]
    diag = torch.arange(16)
    assert bool(S[..., diag, diag].all())
    assert bool((S.sum(-1) >= 1).all())


# ---------------------------------------------------------------------------
# Routing Transformer: the partition is a parameter, and training is what makes it work
# ---------------------------------------------------------------------------
def test_routing_partitions_the_keys():
    """Every key is routed to exactly one centroid, so two queries in different
    clusters share only the positions the diagonal forces."""
    q, k, v = qkv(b=1, h=1, m=8, n=24, d=8, seed=8)
    out = ab.build("routing", n_clusters=4)(q, k, v)
    cq, ck = out.extras["cluster_q"], out.extras["cluster_k"]
    S = out.extras["support"]
    for i in range(8):
        routed = (ck[0, 0] == cq[0, 0, i])
        assert bool((S[0, 0, i] | routed == (routed | (torch.arange(24) == i + 16))).all())


def test_trained_centroids_are_what_make_routing_work():
    """``trainable = True`` is a claim: the untrained partition is the floor, not the method."""
    g = torch.Generator().manual_seed(9)
    d, c, per = 8, 4, 8
    dirs = torch.randn(c, d, generator=g, dtype=torch.float64)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    k = (3 * dirs.repeat_interleave(per, 0)
         + 0.3 * torch.randn(c * per, d, generator=g, dtype=torch.float64)).view(1, 1, -1, d)
    q = (3 * dirs.repeat_interleave(2, 0)
         + 0.3 * torch.randn(2 * c, d, generator=g, dtype=torch.float64)).view(1, 1, -1, d)
    v = torch.randn(1, 1, c * per, 5, generator=g, dtype=torch.float64)
    y = softmax_out(q, k, v).y
    err = lambda meth: float((y - meth(q, k, v).y).norm(dim=-1).mean())
    untrained = sum(err(ab.build("routing", n_clusters=c, seed=s)) for s in range(8)) / 8
    trained = err(ab.build("routing", n_clusters=c, centroids=dirs))
    assert trained < untrained, (trained, untrained)
    assert ab.get_class("routing").trainable
    assert not any(ab.get_class(n).trainable for n in FAMILY if n != "routing")


# ---------------------------------------------------------------------------
# HyperAttention: exact on the hashed block, unbiased on the rest
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("causal", [False, True])
def test_the_hashed_block_has_the_declared_size(causal):
    """sortLSH keeps exactly ``block`` keys per query, or every visible key if fewer."""
    q, k, v = qkv(m=10, n=16, seed=10)
    for block in (1, 3, 6):
        e = ab.build("hyperattention", block=block)(q, k, v, causal=causal).extras
        vis = ab.causal_mask(10, 16) if causal else torch.ones(10, 16, dtype=torch.bool)
        want = vis.sum(-1).clamp(max=block)
        assert torch.equal(e["heavy"].sum(-1), want.expand_as(e["heavy"].sum(-1)))
        assert bool((e["heavy"] & ~vis).sum() == 0)


def test_with_no_sampling_it_is_the_restriction_to_the_hashed_block():
    q, k, v = qkv(seed=11)
    ref = softmax_out(q, k, v)
    out = ab.build("hyperattention", n_sampled=0)(q, k, v, return_weights=True)
    H = out.extras["heavy"]
    assert torch.equal(out.extras["support"], H)
    m_H = (ref.weights * H) @ v / (ref.weights * H).sum(-1, keepdim=True)
    assert torch.allclose(out.y, m_H, atol=1e-11)


def test_the_residual_estimator_is_unbiased_at_a_fixed_hash():
    """eq:sparse-snis on the complement of the block: E[sum_{sampled} pi_j / u_j] = pi(H^c)."""
    q, k, v = qkv(b=1, h=1, m=4, n=12, seed=12)
    pi = softmax_out(q, k, v).weights
    heavy = ab.build("hyperattention", n_sampled=2)(q, k, v).extras["heavy"]
    trials, acc = 800, torch.zeros_like(pi)
    for seed in range(trials):
        e = ab.build("hyperattention", n_sampled=2, seed=seed)(q, k, v).extras
        assert torch.equal(e["heavy"], heavy)        # hash_seed is a separate knob
        take = e["support"] & ~heavy
        acc += torch.where(take, pi / e["u"].clamp_min(1e-30), torch.zeros_like(pi))
    target = (pi * ~heavy).sum(-1)
    assert float(((acc / trials).sum(-1) - target).abs().max()) < 0.08


def test_its_bias_shrinks_with_the_sample_size():
    """prop:error-ratio-est: only the self-normalization is biased, at O(1/n_sampled)."""
    q, k, v = qkv(b=1, h=1, m=6, n=16, seed=13)
    y = softmax_out(q, k, v).y
    bias = []
    for ns in (1, 4, 16):
        ys = torch.stack([ab.build("hyperattention", n_sampled=ns, seed=s)(q, k, v).y
                          for s in range(200)])
        bias.append(float((ys.mean(0) - y).norm(dim=-1).mean()))
    assert bias[0] > bias[-1], bias


def test_sampling_the_tail_beats_discarding_it():
    """sec:sparse-content: the contrast with ``blocktopk`` at a comparable budget."""
    q, k, v = qkv(b=2, h=2, m=8, n=32, seed=14)
    y = softmax_out(q, k, v).y
    hyper = torch.stack([(y - ab.build("hyperattention", block=4, n_sampled=4, seed=s)(
        q, k, v).y).norm(dim=-1) for s in range(10)]).mean(0)
    kept = int(ab.build("hyperattention", block=4, n_sampled=4)(
        q, k, v).extras["k_realized"].to(torch.float64).mean())
    window = (y - ab.build("window", w=kept // 2)(q, k, v).y).norm(dim=-1)
    assert float(hyper.mean()) < float(window.mean()), (hyper.mean(), window.mean())


def test_the_hashed_block_beats_a_random_block_of_the_same_size():
    """The mask is the half of HyperAttention that is not sampling: it must carry signal.

    And it still loses to ``topk`` on omitted mass, which is the whole reason the
    sampled residual is there.
    """
    q, k, v = qkv(b=2, h=4, m=8, n=64, d=16, seed=15)
    pi = softmax_out(q, k, v).weights
    for block in (4, 8, 16):
        hashed = sum(float(ab.build("hyperattention", block=block, n_sampled=0,
                                    hash_seed=s)(q, k, v).extras["eps"].mean())
                     for s in range(8)) / 8
        chance = 0.0
        for s in range(8):
            g = torch.Generator().manual_seed(100 + s)
            idx = torch.rand(pi.shape, generator=g, dtype=pi.dtype).topk(block, -1).indices
            S = torch.zeros_like(pi, dtype=torch.bool).scatter_(-1, idx, True)
            chance += float((1.0 - (pi * S).sum(-1)).mean()) / 8
        oracle = float(ab.build("topk", k=block)(q, k, v).extras["eps"].mean())
        assert oracle < hashed < chance, (block, oracle, hashed, chance)
