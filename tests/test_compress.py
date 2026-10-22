"""Mathematics specific to ``compress/``: sec:compress and its propositions."""
from __future__ import annotations

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M
from attnbank.methods.compress.quantkv import affine_quantize


def qkv(b=2, h=4, m=10, n=16, d=8, dv=5, seed=0, scale=1.0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def softmax_out(q, k, v, causal=False):
    return ab.build("softmax")(q, k, v, causal=causal, return_weights=True)


# ---------------------------------------------------------------------------
# prop:compress-sensitivity: keys and values are not symmetric
# ---------------------------------------------------------------------------
def test_quantizer_is_idempotent_and_bounded_by_its_step():
    x = qkv()[1]
    for bits in (2, 4, 8):
        qx = affine_quantize(x, bits, "token")
        assert torch.allclose(affine_quantize(qx, bits, "token"), qx, atol=1e-9)
        step = (x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)) / (2 ** bits - 1)
        assert ((x - qx).abs() <= step / 2 + 1e-9).all()


def test_key_quantization_is_purely_distributional_and_value_purely_representational():
    """eq:error_decomposition: quantizing only keys leaves v_hat = v, and conversely."""
    q, k, v = qkv()
    ref = softmax_out(q, k, v)
    k_only = ab.build("quantkv", bits_k=3, bits_v=None)(q, k, v, return_weights=True)
    v_only = ab.build("quantkv", bits_k=None, bits_v=3)(q, k, v, return_weights=True)

    dk = M.error_decomposition(ref.weights, k_only.weights, v, k_only.v_hat)
    assert float(dk["value"].max()) < 1e-12
    dv_ = M.error_decomposition(ref.weights, v_only.weights, v, v_only.v_hat)
    assert float(dv_["distributional"].max()) < 1e-12
    assert torch.allclose(v_only.weights, ref.weights, atol=1e-12)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_keys_hurt_more_than_values_at_equal_bit_width(bits):
    """prop:compress-sensitivity (i): key error is amplified, value error is averaged."""
    q, k, v = qkv(b=4, h=4, scale=1.0, seed=1)
    ref = softmax_out(q, k, v)
    # put keys and values on the same scale so the comparison is about the mechanism
    v = v * float(k.std() / v.std())
    e_k = (ref.y - ab.build("quantkv", bits_k=bits, bits_v=None)(q, k, v).y).norm(dim=-1)
    e_v = (ref.y - ab.build("quantkv", bits_k=None, bits_v=bits)(q, k, v).y).norm(dim=-1)
    assert float(e_k.mean()) > float(e_v.mean()), (bits, e_k.mean(), e_v.mean())


def test_realised_tv_respects_the_tanh_bound_for_key_quantization():
    """eq:error-joint: TV(pi, pi_hat) <= tanh(delta/2) with delta the logit error."""
    q, k, v = qkv(b=4, h=4, seed=2)
    ref = softmax_out(q, k, v)
    for bits in (2, 3, 4, 8):
        out = ab.build("quantkv", bits_k=bits, bits_v=None)(q, k, v, return_weights=True)
        b = M.tanh_tv_bound(out.extras["s"], out.extras["s_hat"], v)
        assert (M.total_variation(ref.weights, out.weights) <= b["tv_bound"] + 1e-12).all()
        assert ((ref.y - out.y).norm(dim=-1) <= b["output_bound"] + 1e-12).all()


def test_a_token_constant_key_error_has_no_effect():
    """prop:compress-sensitivity (ii): only the token-varying part of delta k matters."""
    q, k, v = qkv()
    ref = softmax_out(q, k, v)
    shift = torch.randn(k.shape[-1], generator=torch.Generator().manual_seed(3),
                        dtype=k.dtype)
    out = softmax_out(q, k + shift, v)
    assert torch.allclose(out.weights, ref.weights, atol=1e-11)


def test_per_channel_grouping_helps_when_one_key_channel_is_an_outlier():
    """sec:compress-quant: a large token-constant channel coarsens every channel under
    per-token grouping, and only its own channel under per-channel grouping."""
    q, k, v = qkv(b=1, h=1, m=8, n=32, d=8, seed=4)
    k = k.clone()
    k[..., 3] += 60.0                                 # a KIVI-style key outlier channel
    ref = softmax_out(q, k, v)
    e_token = (ref.y - ab.build("quantkv", bits_k=3, bits_v=None, axis="token")(
        q, k, v).y).norm(dim=-1).mean()
    e_chan = (ref.y - ab.build("quantkv", bits_k=3, bits_v=None, axis="channel")(
        q, k, v).y).norm(dim=-1).mean()
    assert float(e_chan) < float(e_token), (float(e_chan), float(e_token))


def test_per_channel_grouping_refuses_to_pretend_it_is_causal():
    q, k, v = qkv()
    with pytest.raises(ValueError, match="not causal"):
        ab.build("quantkv", axis="channel")(q, k, v, causal=True)


# ---------------------------------------------------------------------------
# sec:compress-lowrank: the value error is (I - Pi_r) y
# ---------------------------------------------------------------------------
def test_low_rank_value_error_is_the_projection_residual_of_the_output():
    """Not of the approximate output, and not of the raw value matrix."""
    q, k, v = qkv()
    ref = softmax_out(q, k, v)
    out = ab.build("lowrank_kv", rank_k=None, rank_v=2)(q, k, v, return_weights=True)
    assert torch.allclose(out.weights, ref.weights, atol=1e-12)     # keys untouched
    basis = out.extras["basis_v"]
    proj = basis.transpose(-1, -2) @ basis
    residual = ref.y - ref.y @ proj
    assert torch.allclose(ref.y - out.y, residual, atol=1e-10)


def test_low_rank_key_error_vanishes_when_the_query_lives_in_the_kept_subspace():
    """delta_l = -beta <(I - Pi_r) q, (I - Pi_r) k>: no query energy outside, no error."""
    q, k, v = qkv(b=1, h=1, m=6, n=20, d=8, seed=5)
    out = ab.build("lowrank_kv", rank_k=3, rank_v=None)(q, k, v, return_weights=True)
    basis = out.extras["basis_k"]
    q_in = q @ (basis.transpose(-1, -2) @ basis)                    # query inside span
    ref = softmax_out(q_in, k, v)
    out2 = ab.build("lowrank_kv", rank_k=3, rank_v=None)(q_in, k, v, return_weights=True)
    assert torch.allclose(out2.weights, ref.weights, atol=1e-10)


def test_low_rank_error_decreases_with_rank():
    q, k, v = qkv(seed=6)
    ref = softmax_out(q, k, v).y
    errs = [float((ab.build("lowrank_kv", rank_k=r, rank_v=r)(q, k, v).y - ref)
                  .norm(dim=-1).mean()) for r in (1, 2, 4, 8)]
    assert all(a >= b for a, b in zip(errs, errs[1:])), errs
    assert errs[-1] < 1e-12


# ---------------------------------------------------------------------------
# prop:compress-noeviction and the nested structure of eviction
# ---------------------------------------------------------------------------
def test_eviction_supports_are_nested():
    """S_{t+1} subset S_t union {t+1}: a deleted token is gone for every future query."""
    q, k, v = qkv(m=16, n=16, seed=7)
    S = ab.build("evict_h2o", budget=6, recent=2)(q, k, v, causal=True).extras["support"]
    pos = torch.arange(16)
    for t in range(15):
        newly = torch.zeros(16, dtype=torch.bool)
        newly[pos == t + 1] = True
        assert bool((S[..., t + 1, :] <= (S[..., t, :] | newly)).all()), t


def test_eviction_respects_its_budget_and_the_recent_window():
    q, k, v = qkv(m=16, n=16, seed=7)
    out = ab.build("evict_h2o", budget=6, recent=2)(q, k, v, causal=True)
    S = out.extras["support"]
    assert int(S.sum(-1).max()) <= 6
    for t in range(16):
        assert bool(S[..., t, t].all()), f"the current token must stay resident at t={t}"


def test_no_uniform_guarantee_for_eviction():
    """prop:compress-noeviction.  For any retained S and any evicted j, the query
    q = tau k_j drives eps_q -> 1 and y(q) -> v_j while y_hat stays in conv{v_i: i in S}.
    So the error is bounded below by dist(v_j, conv{v_i : i in S})."""
    b, h, n, d, dv, budget = 1, 1, 12, 6, 4, 4
    g = torch.Generator().manual_seed(8)
    k = torch.nn.functional.normalize(
        torch.randn(b, h, n, d, generator=g, dtype=torch.float64), dim=-1)
    v = torch.randn(b, h, n, dv, generator=g, dtype=torch.float64)
    warmup = torch.randn(b, h, 8, d, generator=g, dtype=torch.float64)

    S = ab.build("evict_h2o", budget=budget, recent=1)(
        warmup, k, v, causal=False).extras["support"][0, 0, -1]
    evicted = (~S).nonzero().flatten()
    assert len(evicted) > 0, "the policy must actually evict something"
    j = int(evicted[0])

    errs = []
    for tau in (4.0, 12.0, 40.0):
        probe = (tau * k[:, :, j]).unsqueeze(-2)                  # q = tau k_j
        ref = softmax_out(probe, k, v)
        out = ab.build("evict_h2o", budget=budget, recent=1)(probe, k, v)
        errs.append(float((ref.y - out.y).norm()))
        last_ref, last_out = ref, out
    assert float(last_ref.weights[0, 0, 0, j]) > 0.9, "eps_q must approach 1"
    assert torch.allclose(last_ref.y[0, 0, 0], v[0, 0, j], atol=1e-3)
    assert errs[0] < errs[-1], errs                               # the error grows with tau
    # y_hat never leaves the convex hull of the retained values
    retained = v[0, 0][S]
    assert (last_out.y[0, 0, 0] >= retained.amin(0) - 1e-9).all()
    assert (last_out.y[0, 0, 0] <= retained.amax(0) + 1e-9).all()


def test_snapkv_is_declared_non_causal_because_its_support_is_chosen_at_the_end():
    assert ab.get_class("evict_snapkv").supports_causal is False
    q, k, v = qkv()
    with pytest.raises(ValueError, match="does not support causal"):
        ab.build("evict_snapkv")(q, k, v, causal=True)


def test_snapkv_keeps_the_recent_window_and_its_budget():
    q, k, v = qkv(m=10, n=16, seed=9)
    S = ab.build("evict_snapkv", budget=6, recent=2)(q, k, v).extras["support"]
    assert int(S.sum(-1).max()) <= 6
    assert bool(S[..., -2:].all())
    assert bool((S[..., 0, :] == S[..., -1, :]).all()), "one support serves every query"


# ---------------------------------------------------------------------------
# prop:compress-merge: merging is a genuine pushforward
# ---------------------------------------------------------------------------
def test_merging_preserves_total_mass_cluster_by_cluster():
    """The lifted weights put exactly pi_hat(i) on the preimage of representative i,
    so T_# mu loses no mass -- the defining property of a pushforward."""
    q, k, v = qkv(m=10, n=16, seed=10)
    out = ab.build("merge_kv", n_clusters=5)(q, k, v, return_weights=True)
    assign, sizes = out.extras["assign"], out.extras["sizes"]
    pi = out.weights
    assert torch.allclose(pi.sum(-1), torch.ones_like(pi[..., 0]), atol=1e-12)
    # inside a cluster every position carries the same weight: pi_hat(i) / |C_i|
    for i in range(5):
        members = (assign == i).nonzero().flatten()
        block = pi[..., members]
        assert torch.allclose(block, block[..., :1].expand_as(block), atol=1e-12)
        # and the compressed-domain mass is the sum over the preimage
        assert torch.allclose(block.sum(-1), block[..., 0] * sizes[i], atol=1e-12)


def test_merging_reproduces_the_compressed_domain_output():
    """y_hat = sum_i mu(C_i) e^{<q, k_i>} v_i / sum_i mu(C_i) e^{<q, k_i>} (ToMe's
    proportional attention), recomputed independently on the r representatives."""
    q, k, v = qkv(m=10, n=16, seed=11)
    out = ab.build("merge_kv", n_clusters=4)(q, k, v, return_weights=True)
    sizes = out.extras["sizes"]
    k_bar, v_bar = out.extras["k_bar"], out.extras["v_bar"]
    beta = q.shape[-1] ** -0.5
    logits = torch.einsum("...md,...rd->...mr", q, k_bar) * beta + sizes.log()
    pi_hat = torch.softmax(logits, dim=-1)
    assert torch.allclose(pi_hat @ v_bar, out.y, atol=1e-11)


def test_dropping_the_multiplicity_underweights_large_clusters():
    """Without the log |C_i| gauge bias, T_# mu is replaced by counting measure."""
    q, k, v = qkv(m=8, n=16, seed=12)
    out = ab.build("merge_kv", n_clusters=3)(q, k, v, return_weights=True)
    sizes = out.extras["sizes"]
    k_bar = out.extras["k_bar"]
    beta = q.shape[-1] ** -0.5
    base = torch.einsum("...md,...rd->...mr", q, k_bar) * beta
    with_mass = torch.softmax(base + sizes.log(), dim=-1)
    without = torch.softmax(base, dim=-1)
    biggest = int(sizes.argmax())
    assert float(with_mass[..., biggest].mean()) > float(without[..., biggest].mean())


def test_merging_error_decreases_as_clusters_shrink():
    q, k, v = qkv(m=8, n=16, seed=13)
    ref = softmax_out(q, k, v).y
    errs = [float((ab.build("merge_kv", n_clusters=r)(q, k, v).y - ref).norm(dim=-1).mean())
            for r in (2, 4, 8, 16)]
    assert errs[0] > errs[-1] and errs[-1] < 1e-12


# ---------------------------------------------------------------------------
# sec:compress-arch: GQA / MLA couple heads, and nothing else
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("group_size", [2, 4])
def test_gqa_couples_exactly_one_group_of_heads(group_size):
    q, k, v = qkv(h=8, seed=14)
    meth = ab.build("gqa", group_size=group_size)
    base = meth(q, k, v).y
    k2 = k.clone()
    k2[:, 1] += 1.0
    pert = meth(q, k2, v).y
    group = [i for i in range(8) if i // group_size == 1 // group_size]
    outside = [i for i in range(8) if i not in group]
    assert torch.equal(base[:, outside], pert[:, outside])
    assert not torch.equal(base[:, group], pert[:, group])


def test_gqa_mean_pools_the_group():
    q, k, v = qkv(h=4, seed=15)
    out = ab.build("gqa", group_size=4)(q, k, v, return_weights=True)
    assert torch.allclose(out.v_hat, v.mean(-3, keepdim=True).expand_as(v), atol=1e-12)


def test_gqa_and_mla_save_capacity_in_proportion_to_their_sharing():
    dense = ab.build("softmax").cost(1, 4096, 128, 128)["state_bytes"]
    for g in (2, 4, 8):
        assert ab.build("gqa", group_size=g).cost(1, 4096, 128, 128)["state_bytes"] \
               == pytest.approx(dense / g)
    small = ab.build("mla", r=64).cost(1, 4096, 128, 128, heads=8)["state_bytes"]
    assert small < dense


def test_mla_latent_is_shared_across_heads():
    q, k, v = qkv(h=4, seed=16)
    meth = ab.build("mla", r=8)
    base = meth(q, k, v).y
    k2 = k.clone()
    k2[:, 0] += 1.0
    assert not torch.equal(base[:, 2], meth(q, k2, v).y[:, 2])
    assert ab.get_class("mla").head_independent is False


# ---------------------------------------------------------------------------
# Linformer: signed weights on the original positions
# ---------------------------------------------------------------------------
def test_linformer_weights_are_signed_and_do_not_sum_to_one():
    """tab:comparative: Linformer loses nm and hl."""
    q, k, v = qkv(m=10, n=16, seed=17)
    out = ab.build("linformer", r=4)(q, k, v, return_weights=True)
    assert bool((out.weights < -1e-9).any())
    s = out.weights.sum(-1)
    assert float((s - 1).abs().max()) > 1e-3
    # Invariant W still holds, which is why it can be studied at all
    assert torch.allclose(out.weights @ v, out.y, atol=1e-11)


def test_linformer_latent_weights_are_a_probability_measure_over_pseudo_tokens():
    q, k, v = qkv(seed=18)
    out = ab.build("linformer", r=4)(q, k, v, return_weights=True)
    p = out.extras["latent_weights"]
    assert (p >= 0).all()
    assert torch.allclose(p.sum(-1), torch.ones_like(p[..., 0]), atol=1e-12)
