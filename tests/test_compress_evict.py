"""Eviction and merging: what the second wave of ``compress/`` adds, and why.

Two claims are on trial here.  ``prop:compress-noeviction``: eviction has no uniform
guarantee, and -- the part that is easy to lose in a reference implementation --
eviction is *not* restriction.  For a single query the two are indistinguishable, so
every test below is built so that the query that chose the support is not the query
that pays for it.  ``prop:compress-merge``: merging is a pushforward, and mass that is
carried is mass that is not lost.
"""
from __future__ import annotations

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M
from attnbank.methods.compress.evict_pyramidkv import observation_scores, pyramid_budgets
from attnbank.methods.compress.merge_tome import bipartite_soft_match


def qkv(b=2, h=4, m=10, n=16, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def softmax_out(q, k, v, causal=False):
    return ab.build("softmax")(q, k, v, causal=causal, return_weights=True)


# ---------------------------------------------------------------------------
# PyramidKV: the budget is allocated across layers, and the total is conserved
# ---------------------------------------------------------------------------
def test_the_pyramid_moves_cache_between_layers_without_creating_any():
    """The schedule is a decreasing arithmetic sequence whose mean is the uniform budget."""
    budget, layers, recent = 64, 8, 4
    sched = pyramid_budgets(budget, layers, recent=recent, beta=20)
    assert all(a > b for a, b in zip(sched, sched[1:])), sched
    assert sum(sched) == pytest.approx(layers * budget, abs=layers / 2)
    assert sched[0] > budget > sched[-1]


def test_beta_one_is_the_uniform_allocation_every_other_method_uses():
    sched = pyramid_budgets(64, 8, recent=4, beta=1)
    assert sched == [64] * 8


def test_the_pyramid_is_capped_by_the_context_and_compensates_at_the_top():
    """The bottom layer cannot keep more tokens than exist; the total still holds."""
    sched = pyramid_budgets(budget=10, n_layers=4, recent=2, beta=20, n=12)
    assert max(sched) <= 12
    assert sum(sched) == pytest.approx(4 * 10, abs=4)


@pytest.mark.parametrize("layer", [0, 1, 2, 3])
def test_lower_layers_really_keep_more_tokens(layer):
    q, k, v = qkv(m=12, n=24, seed=1)
    out = ab.build("evict_pyramidkv", budget=8, n_layers=4, layer=layer)(q, k, v)
    kept = int(out.extras["support"].sum(-1).amax())
    assert kept == min(out.extras["layer_budgets"][layer], 24)
    assert kept == max(1, int(out.extras["k_realized"].amax()))


def test_the_schedule_is_monotone_in_the_layer_index():
    q, k, v = qkv(m=12, n=24, seed=1)
    kept = [int(ab.build("evict_pyramidkv", budget=8, n_layers=4, layer=la)(q, k, v)
                .extras["support"].sum(-1).amax()) for la in range(4)]
    assert all(a > b for a, b in zip(kept, kept[1:])), kept


# ---------------------------------------------------------------------------
# Pooling: the SnapKV variant this file adds
# ---------------------------------------------------------------------------
def test_max_pooling_keeps_the_neighbourhood_of_a_spike_and_averaging_dilutes_it():
    """SnapKV's released code max-pools; ``evict_snapkv`` averages. This is the gap."""
    pi = torch.zeros(1, 1, 1, 8)
    pi[..., 4] = 1.0
    mx = observation_scores(pi, obs=1, pool=3, pooling="max")[0, 0]
    av = observation_scores(pi, obs=1, pool=3, pooling="avg")[0, 0]
    assert torch.allclose(mx[3:6], torch.ones(3))               # the spike spreads
    assert float(av[3]) == pytest.approx(1 / 3)                 # ... or is divided
    assert float(mx[0]) == 0.0
    # and so the selected set is a neighbourhood rather than an isolated token
    assert set(mx.topk(3).indices.tolist()) == {3, 4, 5}


def test_a_bad_pooling_name_is_rejected_with_the_accepted_ones():
    with pytest.raises(TypeError, match="max"):
        ab.build("evict_pyramidkv", pooling="median")(*qkv())


def test_the_layer_index_must_lie_in_the_stack():
    with pytest.raises(TypeError, match="layer"):
        ab.build("evict_pyramidkv", layer=9, n_layers=4)


# ---------------------------------------------------------------------------
# Eviction is not restriction: nesting
# ---------------------------------------------------------------------------
def test_tova_supports_are_nested_and_topk_supports_are_not():
    """``S_{t+1} subset S_t u {t+1}``.  A restriction recomputes its set per query and
    is free to come back to a position it ignored a step ago; an eviction cannot."""
    q, k, v = qkv(m=16, n=16, seed=2)
    ev = ab.build("evict_tova", budget=6)(q, k, v, causal=True).extras["support"]
    re = ab.build("topk", k=6)(q, k, v, causal=True).extras["support"]
    idx = torch.arange(16)
    nested_ev, nested_re = True, True
    for t in range(15):
        new = (idx == t + 1)
        nested_ev &= bool((ev[..., t + 1, :] <= (ev[..., t, :] | new)).all())
        nested_re &= bool((re[..., t + 1, :] <= (re[..., t, :] | new)).all())
    assert nested_ev, "eviction must be nested"
    assert not nested_re, "top-k must NOT be nested, or the comparison is vacuous"


def test_a_token_tova_evicts_is_gone_for_every_later_query():
    q, k, v = qkv(m=16, n=16, seed=2)
    out = ab.build("evict_tova", budget=6)(q, k, v, causal=True)
    S, when = out.extras["support"], out.extras["evicted_at"]
    assert int((when >= 0).sum()) > 0, "the policy must actually evict something"
    for t in range(16):
        dead = when.clamp_min(-1)
        assert not bool((S[..., t, :] & (dead >= 0) & (dead <= t)).any()), t


def test_tova_keeps_the_current_token_so_no_row_is_ever_empty():
    q, k, v = qkv(m=16, n=16, seed=2)
    S = ab.build("evict_tova", budget=4)(q, k, v, causal=True).extras["support"]
    for t in range(16):
        assert bool(S[..., t, t].all()), t
    assert int(S.sum(-1).amin()) >= 1


def test_tova_respects_its_budget():
    q, k, v = qkv(m=16, n=16, seed=2)
    S = ab.build("evict_tova", budget=5)(q, k, v, causal=True).extras["support"]
    assert int(S.sum(-1).amax()) <= 5


# ---------------------------------------------------------------------------
# Eviction is not restriction: the damage lands on the queries that did not choose
# ---------------------------------------------------------------------------
def test_eviction_costs_more_than_the_same_budget_restriction_and_the_gap_grows():
    """The sets coincide for the query that made the choice and diverge after it.
    ``topk`` hides the context and can look again; ``evict_tova`` deleted it."""
    q, k, v = qkv(b=4, h=4, m=24, n=24, seed=3)
    ref = softmax_out(q, k, v, causal=True).y
    ev = ab.build("evict_tova", budget=6)(q, k, v, causal=True).y
    re = ab.build("topk", k=6)(q, k, v, causal=True).y
    e_ev = (ref - ev).norm(dim=-1).mean((0, 1))               # [m]
    e_re = (ref - re).norm(dim=-1).mean((0, 1))
    assert float(e_ev.mean()) > float(e_re.mean())
    early, late = slice(6, 12), slice(18, 24)
    assert float((e_ev - e_re)[late].mean()) > float((e_ev - e_re)[early].mean())


def test_a_frozen_support_serves_queries_that_had_no_say_in_it():
    """SnapKV/PyramidKV/Ada-KV choose once, from the last ``obs`` queries.  Every other
    query is then answered out of a cache pruned without knowing its question."""
    q, k, v = qkv(b=4, h=4, m=12, n=20, seed=4)
    ref = softmax_out(q, k, v).y
    for name in ("evict_pyramidkv", "evict_adakv", "evict_snapkv"):
        out = ab.build(name, budget=8)(q, k, v, return_weights=True)
        S = out.extras["support"]
        assert bool((S[..., 0, :] == S[..., -1, :]).all()), f"{name}: support varies by query"
        err = (ref - out.y).norm(dim=-1).mean((0, 1))         # [m]
        assert float(err[:4].mean()) > float(err[-4:].mean()), \
            f"{name}: the observation window must be the cheapest place to be"


def test_no_uniform_guarantee_for_a_frozen_pyramid_cache():
    """prop:compress-noeviction.  The probe query is placed *before* the observation
    window, so the cache that answers it was pruned by other queries -- which is the
    only arrangement in which eviction is distinguishable from restriction.  Point the
    probe at a token the policy threw away, ``q = tau k_j``: ``eps_q -> 1`` while
    ``y_hat`` stays inside ``conv{v_i : i retained}``."""
    b, h, n, d, dv = 1, 1, 14, 6, 4
    g = torch.Generator().manual_seed(5)
    k = torch.nn.functional.normalize(
        torch.randn(b, h, n, d, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(b, h, n, dv, generator=g, dtype=torch.float64)
    warm = torch.randn(b, h, 8, d, generator=g, dtype=torch.float64)

    meth = ab.build("evict_pyramidkv", budget=5, n_layers=2, layer=1, recent=1,
                    pool=1, obs=4)
    S = meth(warm, k, v).extras["support"][0, 0, -1]
    evicted = (~S).nonzero().flatten()
    assert len(evicted) > 0, "the policy must actually evict something"
    j = int(evicted[0])

    errs = []
    for tau in (4.0, 20.0, 120.0):
        probe = (tau * k[:, :, j]).unsqueeze(-2)
        out = meth(torch.cat([probe, warm], dim=-2), k, v)
        ref = softmax_out(probe, k, v)
        kept = out.extras["support"][0, 0, 0]
        assert not bool(kept[j]), "the probe must be served from a cache without j"
        errs.append(float((ref.y[0, 0, 0] - out.y[0, 0, 0]).norm()))
    assert float(ref.weights[0, 0, 0, j]) > 0.9, "eps_q must approach 1"
    assert torch.allclose(ref.y[0, 0, 0], v[0, 0, j], atol=1e-2)
    assert errs[0] < errs[-1], errs
    retained = v[0, 0][kept]
    assert (out.y[0, 0, 0] >= retained.amin(0) - 1e-9).all()
    assert (out.y[0, 0, 0] <= retained.amax(0) + 1e-9).all()


# ---------------------------------------------------------------------------
# Ada-KV: one pool, many heads
# ---------------------------------------------------------------------------
def test_adakv_moves_budget_between_heads_without_creating_any():
    q, k, v = qkv(b=2, h=8, m=10, n=24, seed=6)
    out = ab.build("evict_adakv", budget=8, floor_alpha=0.0)(q, k, v)
    per_head = out.extras["head_budget"]
    assert per_head.shape == (2, 8)
    assert int(per_head.sum(-1).amax()) == pytest.approx(8 * 8, abs=8)
    assert int(per_head.amax()) > int(per_head.amin()), "allocation must be adaptive"
    assert int(out.extras["support"].sum(-1).amax()) <= int(per_head.amax())


def test_floor_alpha_one_recovers_the_uniform_allocation():
    q, k, v = qkv(b=2, h=8, m=10, n=24, seed=6)
    per_head = ab.build("evict_adakv", budget=8, floor_alpha=1.0)(q, k, v) \
        .extras["head_budget"]
    assert bool((per_head == 8).all()), per_head


def test_adakv_couples_heads_and_says_so():
    assert ab.get_class("evict_adakv").head_independent is False
    q, k, v = qkv(b=1, h=4, m=10, n=24, seed=7)
    meth = ab.build("evict_adakv", budget=8, floor_alpha=0.0)
    base = meth(q, k, v).extras["head_budget"]
    k2 = k.clone()
    k2[:, 0] = k2[:, 0] * 8.0                       # sharpen head 0 only
    pert = meth(q, k2, v).extras["head_budget"]
    assert not torch.equal(base[:, 1:], pert[:, 1:]), \
        "a head's budget must react to the other heads: that is the mechanism"
    assert int(base.sum()) == pytest.approx(int(pert.sum()), abs=4)


def test_the_diffuse_head_is_given_the_larger_cache():
    """Ada-KV's claim: a concentrated head wastes a uniform budget, a diffuse one
    starves on it."""
    d, n = 8, 24
    g = torch.Generator().manual_seed(8)
    k = torch.randn(1, 2, n, d, generator=g, dtype=torch.float64)
    k[:, 0] = torch.nn.functional.normalize(k[:, 0], dim=-1)
    v = torch.randn(1, 2, n, 4, generator=g, dtype=torch.float64)
    q = torch.empty(1, 2, 4, d, dtype=torch.float64)
    q[:, 0] = 30.0 * k[:, 0, 3]                     # head 0: all mass on one token
    q[:, 1] = 0.0                                   # head 1: uniform attention
    per_head = ab.build("evict_adakv", budget=8, floor_alpha=0.0, pool=1,
                        recent=1)(q, k, v).extras["head_budget"]
    assert int(per_head[0, 1]) > int(per_head[0, 0]), per_head


# ---------------------------------------------------------------------------
# CaM: the measure is restricted, the value field is repaired
# ---------------------------------------------------------------------------
def test_cam_deletes_exactly_the_mass_pure_eviction_deletes():
    """Keys are never merged, so pi_hat is bit-for-bit the eviction's; ``merge_window=0``
    is the same method with the repair switched off."""
    q, k, v = qkv(m=10, n=16, seed=9)
    cam = ab.build("merge_cam", budget=6, sinks=2, merge_window=6)(q, k, v,
                                                                   return_weights=True)
    raw = ab.build("merge_cam", budget=6, sinks=2, merge_window=0)(q, k, v,
                                                                   return_weights=True)
    assert torch.equal(cam.extras["support"], raw.extras["support"])
    assert torch.equal(cam.weights, raw.weights)
    assert torch.equal(cam.extras["eps"], raw.extras["eps"])
    assert torch.equal(raw.v_hat, v), "merge_window = 0 must not touch the values"


def test_the_cam_cache_is_frozen_where_a_restriction_slides():
    """At the same budget, ``topk`` is free to look wherever each query needs; CaM has
    one cache, chosen before those queries existed.  The gap is concentrated exactly on
    the queries that fell outside the retained window."""
    q, k, v = qkv(b=4, h=4, m=16, n=16, seed=20)
    ref = softmax_out(q, k, v).y
    cam = ab.build("merge_cam", budget=6, sinks=2)(q, k, v)
    sw = ab.build("sink_window", sinks=2, w=4)(q, k, v, return_weights=True)
    top = ab.build("topk", k=6)(q, k, v)
    assert bool((sw.extras["support"][..., 0, :] != sw.extras["support"][..., -1, :]).any()), \
        "a restriction recomputes its set per query"
    assert bool((cam.extras["support"][..., 0, :] == cam.extras["support"][..., -1, :]).all()), \
        "a cache does not"
    e_cam = (ref - cam.y).norm(dim=-1).mean((0, 1))
    e_top = (ref - top.y).norm(dim=-1).mean((0, 1))
    assert float(e_cam.mean()) > float(e_top.mean())
    assert float((e_cam - e_top)[:4].mean()) > float((e_cam - e_top)[-4:].mean())


def test_cam_repairs_the_value_field_and_only_the_value_field():
    """eq:error_decomposition: identical distributional term, a value term that is not
    zero.  This is what makes CaM a different object from the pushforward of merge_kv,
    where the mass itself moves and pi_hat picks up the log |C_i| bias."""
    q, k, v = qkv(m=10, n=16, seed=9)
    ref = softmax_out(q, k, v)
    cam = ab.build("merge_cam", budget=6, sinks=2, merge_window=6)(q, k, v,
                                                                   return_weights=True)
    raw = ab.build("merge_cam", budget=6, sinks=2, merge_window=0)(q, k, v,
                                                                   return_weights=True)
    d_cam = M.error_decomposition(ref.weights, cam.weights, v, cam.v_hat)
    d_raw = M.error_decomposition(ref.weights, raw.weights, v, raw.v_hat)
    assert torch.allclose(d_cam["distributional"], d_raw["distributional"], atol=1e-14)
    assert float(d_cam["value"].max()) > 1e-6
    assert float(d_raw["value"].max()) == 0.0


def test_cam_overwrites_the_cache_it_cannot_un_merge():
    """A restriction hides the context and leaves ``v`` alone -- ``sink_window`` returns
    ``v_hat = None``.  CaM has written the evicted values into the survivors."""
    q, k, v = qkv(m=10, n=16, seed=9)
    assert ab.build("sink_window", sinks=2, w=4)(q, k, v).v_hat is None
    assert ab.build("evict_tova", budget=6)(q, k, v).v_hat is None
    cam = ab.build("merge_cam", budget=6, sinks=2)(q, k, v)
    kept = cam.extras["retained_idx"]
    assert cam.v_hat is not None
    assert not torch.allclose(cam.v_hat.index_select(-2, kept), v.index_select(-2, kept))


def test_cam_is_better_than_throwing_the_same_tokens_away():
    q, k, v = qkv(b=4, h=4, m=12, n=24, seed=10)
    ref = softmax_out(q, k, v).y
    cam = ab.build("merge_cam", budget=8, sinks=2, merge_window=6)(q, k, v).y
    drop = ab.build("merge_cam", budget=8, sinks=2, merge_window=0)(q, k, v).y
    assert float((ref - cam).norm(dim=-1).mean()) < float((ref - drop).norm(dim=-1).mean())


def test_cam_carries_a_heavy_hitter_and_lets_an_ignored_token_go():
    """The gate is the prominence of the departing token in the window it leaves."""
    q, k, v = qkv(b=1, h=1, m=6, n=16, seed=11)
    out = ab.build("merge_cam", budget=6, sinks=2)(q, k, v)
    rho, gate = out.extras["prominence"], out.extras["merge_gate"]
    assert float(rho.min()) >= 0.0 and float(rho.max()) <= 1.0
    assert torch.equal(rho, gate)                 # mode='expected' is the gate itself
    sampled = ab.build("merge_cam", budget=6, sinks=2, mode="sample",
                       seed=1)(q, k, v).extras["merge_gate"]
    assert bool(((sampled == 0) | (sampled == 1)).all())
    assert torch.equal(sampled, ab.build("merge_cam", budget=6, sinks=2, mode="sample",
                                         seed=1)(q, k, v).extras["merge_gate"])


def test_cam_clusters_are_a_partition_of_the_context():
    q, k, v = qkv(m=8, n=16, seed=12)
    out = ab.build("merge_cam", budget=6, sinks=2)(q, k, v)
    assign, sizes, kept = (out.extras[key] for key in ("assign", "sizes", "retained_idx"))
    assert int(sizes.sum()) == 16
    assert int(assign.max()) + 1 == int(sizes.numel()) == int(kept.numel())
    assert bool((assign.index_select(0, kept) == torch.arange(kept.numel())).all())
    assert (out.extras["cluster_radius"] >= 0).all()


# ---------------------------------------------------------------------------
# ToMe: bipartite soft matching, and the pushforward it induces
# ---------------------------------------------------------------------------
def test_bipartite_matching_pairs_the_duplicates_first():
    """Two identical keys in opposite halves are the heaviest edge there is."""
    metric = torch.eye(8, dtype=torch.float64)
    metric[5] = metric[4]                       # positions 4 and 5 are identical
    assign = bipartite_soft_match(metric, target=7)
    assert int(assign[4]) == int(assign[5])
    assert int(assign.max()) + 1 == 7


def test_the_matching_reaches_exactly_the_requested_number_of_clusters():
    g = torch.Generator().manual_seed(13)
    metric = torch.randn(32, 6, generator=g, dtype=torch.float64)
    for target in (1, 2, 3, 7, 16, 31):
        assign = bipartite_soft_match(metric, target)
        assert int(assign.max()) + 1 == target
        assert bool((assign.sort().values.diff() <= 1).all()), "ids must stay compact"


def test_the_matching_is_deterministic_and_needs_no_seed():
    g = torch.Generator().manual_seed(14)
    metric = torch.randn(16, 6, generator=g, dtype=torch.float64)
    assert torch.equal(bipartite_soft_match(metric, 5), bipartite_soft_match(metric, 5))
    assert torch.equal(bipartite_soft_match(metric, 16), torch.arange(16))
    assert "seed" not in ab.build("merge_tome").config


def test_tome_is_a_pushforward_mass_is_carried_cluster_by_cluster():
    """Inside a cluster every original position carries the same weight pi_hat(i)/|C_i|,
    so the compressed-domain mass is the sum over the preimage: T_# mu loses nothing."""
    q, k, v = qkv(b=1, h=2, m=8, n=16, seed=15)
    out = ab.build("merge_tome", n_clusters=5)(q, k, v, return_weights=True)
    assign, sizes, pi = out.extras["assign"], out.extras["sizes"], out.weights
    assert torch.allclose(pi.sum(-1), torch.ones_like(pi[..., 0]), atol=1e-12)
    for h in range(2):
        for i in range(5):
            members = (assign[0, h] == i).nonzero().flatten()
            block = pi[0, h][:, members]
            assert torch.allclose(block, block[:, :1].expand_as(block), atol=1e-12)
            assert torch.allclose(block.sum(-1), block[:, 0] * sizes[0, h, i], atol=1e-12)


def test_tome_reproduces_the_compressed_domain_output_with_the_log_s_bias():
    """ToMe's proportional attention: softmax(QK^T/sqrt(d) + log s) on r tokens."""
    q, k, v = qkv(b=1, h=1, m=8, n=16, seed=16)
    out = ab.build("merge_tome", n_clusters=4)(q, k, v, return_weights=True)
    sizes, k_bar, v_bar = (out.extras[key] for key in ("sizes", "k_bar", "v_bar"))
    logits = torch.einsum("...md,...rd->...mr", q, k_bar) * q.shape[-1] ** -0.5
    assert torch.allclose(torch.softmax(logits + sizes.log(), -1) @ v_bar, out.y, atol=1e-11)
    plain = torch.softmax(logits, -1)
    biggest = int(sizes[0, 0].argmax())
    assert float(torch.softmax(logits + sizes.log(), -1)[..., biggest].mean()) > \
        float(plain[..., biggest].mean()), "dropping log s under-weights big clusters"


def test_the_realised_tome_error_respects_prop_compress_merge():
    q, k, v = qkv(b=1, h=1, m=8, n=16, seed=17)
    ref = softmax_out(q, k, v)
    out = ab.build("merge_tome", n_clusters=5)(q, k, v, return_weights=True)
    radius = out.extras["cluster_radius"][0, 0].amax()
    delta = q.shape[-1] ** -0.5 * q[0, 0].norm(dim=-1) * radius            # [m]
    bound = M.merge_bound(ref.weights[0, 0], out.extras["assign"][0, 0], v[0, 0],
                          out.extras["v_bar"][0, 0], delta)
    err = (ref.y[0, 0] - out.y[0, 0]).norm(dim=-1)
    assert (err <= bound["bound"] + 1e-9).all(), (err, bound["bound"])


def test_tome_error_decreases_as_the_clusters_shrink():
    q, k, v = qkv(b=4, h=4, m=8, n=16, seed=18)
    ref = softmax_out(q, k, v).y
    errs = [float((ab.build("merge_tome", n_clusters=r)(q, k, v).y - ref).norm(dim=-1).mean())
            for r in (2, 4, 8, 12, 16)]
    assert all(a >= b for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-12


def test_matching_beats_contiguous_blocks_when_similar_tokens_are_scattered():
    """The reason ToMe exists: the partition should follow similarity, not position."""
    b, h, n, d, dv, r = 1, 1, 16, 6, 4, 8
    g = torch.Generator().manual_seed(19)
    proto = torch.randn(r, d, generator=g, dtype=torch.float64)
    order = torch.randperm(n, generator=g)
    k = proto.repeat(2, 1)[order].reshape(b, h, n, d)
    v = torch.randn(b, h, n, dv, generator=g, dtype=torch.float64)
    q = torch.randn(b, h, 8, d, generator=g, dtype=torch.float64)
    ref = softmax_out(q, k, v).y
    tome = (ref - ab.build("merge_tome", n_clusters=r)(q, k, v).y).norm(dim=-1).mean()
    block = (ref - ab.build("merge_kv", n_clusters=r)(q, k, v).y).norm(dim=-1).mean()
    assert float(tome) < float(block), (float(tome), float(block))
    # matched clusters are tighter than positional ones -- greedy, so not perfect: an
    # A-slot token cannot be matched to its twin when the twin also landed in A
    r_tome = ab.build("merge_tome", n_clusters=r)(q, k, v).extras["cluster_radius"]
    r_block = ab.build("merge_kv", n_clusters=r)(q, k, v).extras["cluster_radius"]
    assert float(r_tome.mean()) < float(r_block.mean())


# ---------------------------------------------------------------------------
# What the family declares
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["evict_pyramidkv", "evict_adakv", "evict_tova"])
def test_evictions_expose_the_restriction_diagnostics(name):
    q, k, v = qkv()
    extras = ab.build(name)(q, k, v).extras
    assert {"support", "eps", "chi2", "k_realized"} <= set(extras)
    assert (extras["eps"] >= -1e-12).all() and (extras["eps"] <= 1 + 1e-12).all()


@pytest.mark.parametrize("name", ["merge_tome", "merge_cam", "merge_kv"])
def test_merges_expose_the_pushforward_triple(name):
    q, k, v = qkv()
    extras = ab.build(name)(q, k, v).extras
    assert {"assign", "sizes", "cluster_radius"} <= set(extras)


@pytest.mark.parametrize("name", ["evict_pyramidkv", "evict_adakv", "evict_tova",
                                  "merge_cam", "merge_tome"])
def test_every_new_method_declares_reference_only_and_an_exact_limit(name):
    cls = ab.get_class(name)
    assert cls.backends == ("reference",)
    assert cls.limit_config() is not None
