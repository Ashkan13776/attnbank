"""The published KV-cache quantization schemes: what each one groups, and what it keeps.

``sec:compress-quant``.  ``quantkv`` is the generic uniform affine quantizer and
``tests/test_compress.py`` tests the generic facts about it.  The five schemes here are
*specific published answers* to the same question, and what distinguishes them is not
the bit width but **the axis they group along** and **what they refuse to round**:

===========  ===================================  =============================
scheme       grouping axis                        kept in high precision
===========  ===================================  =============================
``kivi``     keys per channel, values per token    the recent residual window
``kvquant``  keys per channel *pre-RoPE*,          ~1% per-vector outliers, the
             values per token, non-uniform levels  attention sink
``atom``     per head, after an energy reorder     the top-energy channels, INT8
``qjl``      none -- a 1-bit JL sketch             the key norm ``||k_j||``
``zipcache`` per token, channel-separable          the salient tokens, by score
===========  ===================================  =============================

Each block below tests the *mechanism* that entry names, not the fact that quantization
loses something.  The family-wide block tests what the monograph needs from all of them:
Invariant W with a real ``v_hat``, the two halves of ``eq:error_decomposition`` being
separable, and ``extras`` that feed ``metrics.tanh_tv_bound``.
"""
from __future__ import annotations

import math

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M
from attnbank.components.quantization import affine_quantize
from attnbank.methods.compress.atom import reorder_mixed
from attnbank.methods.compress.kvquant import (nuq_levels, quantize_vectors, rope)
from attnbank.methods.compress.qjl import qjl_keys
from attnbank.methods.compress.zipcache import channel_separable_quantize

#: keys quantized, values left alone -- the distributional half of eq:error_decomposition
KEY_ONLY = {
    "kivi": dict(bits_k=3, bits_v=None, group=4, residual=0),
    "kvquant": dict(bits_k=3, bits_v=None),
    "atom": dict(bits_k=3, bits_v=None),
    "qjl": dict(sketch=32, bits_v=None),
    "zipcache": dict(bits_salient=4, bits_regular=2, target="k"),
}
#: values quantized, keys left alone -- the value half
VALUE_ONLY = {
    "kivi": dict(bits_k=None, bits_v=3),
    "kvquant": dict(bits_k=None, bits_v=3),
    "atom": dict(bits_k=None, bits_v=3),
    "qjl": dict(sketch=None, bits_v=3),
    "zipcache": dict(bits_salient=4, bits_regular=2, target="v"),
}
FAMILY = sorted(KEY_ONLY)


def qkv(b=2, h=2, m=16, n=24, d=8, dv=6, seed=0, dtype=torch.float64,
        outlier=None, gain=12.0):
    """Random ``q, k, v``; ``outlier`` amplifies one *key* channel, as real keys do."""
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    q, k, v = mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)
    if outlier is not None:
        k = k.clone()
        k[..., outlier] *= gain
    return q, k, v


def softmax_out(q, k, v, causal=False):
    return ab.build("softmax")(q, k, v, causal=causal, return_weights=True)


# ===========================================================================
# the family contract: what sec:output_error needs from every one of them
# ===========================================================================
@pytest.mark.parametrize("name", FAMILY)
def test_each_scheme_is_an_approximating_compressor(name):
    cls = ab.get_class(name)
    assert cls.category is ab.Category.COMPRESS
    assert cls.objective is ab.Objective.APPROXIMATE
    assert cls.expresses_weights and cls.row_stochastic


@pytest.mark.parametrize("name", FAMILY)
def test_a_value_quantizer_must_return_v_hat(name):
    """The error decomposition is silently wrong if ``v_hat`` comes back ``None``."""
    q, k, v = qkv()
    out = ab.build(name, **VALUE_ONLY[name])(q, k, v, return_weights=True)
    assert out.v_hat is not None, f"{name} quantized the values but returned v_hat=None"
    assert out.v_hat.shape == v.shape
    assert not torch.allclose(out.v_hat, v), f"{name}: v_hat == v, nothing was quantized"
    assert ab.check_invariant(out, v) < 1e-10


@pytest.mark.parametrize("name", FAMILY)
def test_key_quantization_is_purely_distributional(name):
    """eq:error_decomposition: with ``v`` untouched the value term is exactly zero."""
    q, k, v = qkv(outlier=3)
    ref = softmax_out(q, k, v)
    out = ab.build(name, **KEY_ONLY[name])(q, k, v, return_weights=True)
    dec = M.error_decomposition(ref.weights, out.weights, v, out.v_hat)
    assert float(dec["value"].max()) < 1e-12
    assert float(dec["distributional"].max()) > 0


@pytest.mark.parametrize("name", FAMILY)
def test_value_quantization_is_purely_representational(name):
    """And conversely: with ``k`` untouched ``pi_hat`` is the reference measure."""
    q, k, v = qkv(outlier=3)
    ref = softmax_out(q, k, v)
    out = ab.build(name, **VALUE_ONLY[name])(q, k, v, return_weights=True)
    assert torch.allclose(out.weights, ref.weights, atol=1e-12)
    dec = M.error_decomposition(ref.weights, out.weights, v, out.v_hat)
    assert float(dec["distributional"].max()) < 1e-12


@pytest.mark.parametrize("name", FAMILY)
def test_extras_feed_the_tanh_bound(name):
    """``s`` and ``s_hat`` are what study S4 hands to :func:`metrics.tanh_tv_bound`."""
    q, k, v = qkv(outlier=3)
    ref = softmax_out(q, k, v)
    out = ab.build(name, **KEY_ONLY[name])(q, k, v, return_weights=True)
    for key in ("s", "s_hat", "delta_k", "delta_v"):
        assert key in out.extras, f"{name} does not expose extras[{key!r}]"
    assert out.extras["s"].shape == ref.weights.shape
    assert out.extras["delta_k"].shape == k.shape[:-1]
    bound = M.tanh_tv_bound(out.extras["s"], out.extras["s_hat"], v)
    assert (M.total_variation(ref.weights, out.weights)
            <= bound["tv_bound"] + 1e-12).all()
    assert ((ref.y - out.y).norm(dim=-1) <= bound["output_bound"] + 1e-12).all()


@pytest.mark.parametrize("name", FAMILY)
def test_disabling_quantization_is_the_identity_not_merely_accurate(name):
    """The anchor: at the limit config nothing is rounded at all."""
    cfg = ab.get_class(name).limit_config()
    assert cfg is not None, f"{name} declares no exact limit"
    q, k, v = qkv()
    out = ab.build(name, **cfg)(q, k, v, return_weights=True)
    ref = softmax_out(q, k, v)
    assert float(out.extras["delta_k"].abs().max()) == 0.0
    assert float(out.extras["delta_v"].abs().max()) == 0.0
    assert torch.allclose(out.y, ref.y, atol=1e-12)
    assert torch.allclose(out.weights, ref.weights, atol=1e-12)


@pytest.mark.parametrize("name", FAMILY)
def test_only_the_reference_backend_is_claimed(name):
    """No fused kernel can return the perturbed score matrix these methods expose."""
    assert ab.get_class(name).backends == ("reference",)


# ===========================================================================
# KIVI: keys per channel, values per token, recent window exact
# ===========================================================================
def test_kivi_per_channel_keys_beat_per_token_on_an_outlier_channel():
    """The grouping axis is the whole claim: confine the outlier's step to its channel."""
    for seed in range(4):
        q, k, v = qkv(seed=seed, outlier=3)
        ref = softmax_out(q, k, v)
        chan = ab.build("kivi", bits_k=3, bits_v=None, group=8, residual=0)
        tok = ab.build("quantkv", bits_k=3, bits_v=None, axis="token")
        e_chan = (ref.y - chan(q, k, v).y).norm(dim=-1).mean()
        e_tok = (ref.y - tok(q, k, v).y).norm(dim=-1).mean()
        assert float(e_chan) < float(e_tok), (seed, float(e_chan), float(e_tok))


def test_kivi_values_are_quantized_per_token():
    """A per-token scale confines a token's value error to its own summand."""
    q, k, v = qkv()
    v2 = v.clone()
    v2[..., 5, :] *= 30.0
    a = ab.build("kivi", bits_k=None, bits_v=3)(q, k, v, return_weights=True).v_hat
    b = ab.build("kivi", bits_k=None, bits_v=3)(q, k, v2, return_weights=True).v_hat
    moved = (a - b).abs().amax(-1) > 0
    assert bool(moved[..., 5].all()) and not bool(moved[..., :5].any())
    assert not bool(moved[..., 6:].any())


def test_kivi_sealed_prefix_follows_the_residual_rule():
    """``L(p) = group * max(0, (p + 1 - residual) // group)`` -- exact, column by column."""
    group, residual, n = 4, 4, 16
    q, k, v = qkv(b=1, h=1, m=n, n=n)
    out = ab.build("kivi", bits_k=2, bits_v=None, group=group,
                   residual=residual)(q, k, v, causal=True, return_weights=True)
    touched = (out.extras["s"] - out.extras["s_hat"]).abs()[0, 0] > 1e-12
    for p in range(n):
        want = group * max(0, (p + 1 - residual) // group)
        assert int(touched[p].sum()) == want, (p, int(touched[p].sum()), want)
        assert not bool(touched[p, want:].any()), f"a key past L({p}) was quantized"


def test_kivi_a_longer_residual_quantizes_less():
    q, k, v = qkv(m=24, n=24, outlier=3)
    ref = softmax_out(q, k, v, causal=True)
    errs = []
    for residual in (0, 4, 8, 16):
        out = ab.build("kivi", bits_k=2, bits_v=2, group=4, residual=residual)
        errs.append(float((ref.y - out(q, k, v, causal=True).y).norm(dim=-1).mean()))
    assert all(a >= b for a, b in zip(errs, errs[1:])), errs
    assert errs[0] > errs[-1]


# ===========================================================================
# KVQuant: pre-RoPE, non-uniform levels, sensitivity, dense-and-sparse
# ===========================================================================
def test_rope_round_trip_is_an_involution():
    q, k, v = qkv()
    assert torch.allclose(rope(rope(k, inverse=True)), k, atol=1e-12)


def test_pre_rope_quantization_reduces_the_key_error():
    """RoPE smears a pre-RoPE outlier channel across its pair; quantize before it."""
    for seed in range(4):
        q, k, v = qkv(seed=seed, n=32, outlier=3)
        k_rope = rope(k)                       # the cache is handed post-RoPE keys
        cfg = dict(bits_k=3, bits_v=None, nuq=False, sparsity=0.0, keep_first=0)
        d_pre = ab.build("kvquant", pre_rope=True, **cfg)(
            q, k_rope, v, return_weights=True).extras["delta_k"].mean()
        d_post = ab.build("kvquant", pre_rope=False, **cfg)(
            q, k_rope, v, return_weights=True).extras["delta_k"].mean()
        assert float(d_pre) < float(d_post), (seed, float(d_pre), float(d_post))


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_non_uniform_levels_beat_a_uniform_grid_on_their_own_objective(bits):
    """k-means signposts minimise the reconstruction error a uniform grid only bounds.

    Note what is *not* asserted: a smaller key reconstruction error does not imply a
    smaller output error.  Only the token-varying part of ``delta l`` survives the
    softmax (``prop:compress-sensitivity``), and a non-uniform grid concentrates its
    error on exactly the extreme tokens that carry the most mass -- which is why
    KVQuant weights the fit by a sensitivity rather than fitting the raw distribution.
    """
    g = torch.Generator().manual_seed(4)
    z = torch.randn(6, 64, generator=g, dtype=torch.float64)
    z[0] *= torch.exp(torch.randn(64, generator=g, dtype=torch.float64))
    uniform = affine_quantize(z, bits, "token", None)
    nuq = nuq_levels(z, torch.ones_like(z), 2 ** bits, iters=8)
    assert float(((z - nuq) ** 2).mean()) < float(((z - uniform) ** 2).mean())


def test_sensitivity_weighting_trades_plain_error_for_weighted_error():
    """It minimises ``sum_i F_i (x_i - Q(x_i))^2``, and that is a different optimum."""
    g = torch.Generator().manual_seed(9)
    z = torch.randn(4, 64, generator=g, dtype=torch.float64)
    w = torch.rand(4, 64, generator=g, dtype=torch.float64) ** 4
    plain = quantize_vectors(z, 3, None, 0.0, nuq=True)
    weighted = quantize_vectors(z, 3, w, 0.0, nuq=True)
    assert float((w * (z - weighted) ** 2).sum()) < float((w * (z - plain) ** 2).sum())
    assert float(((z - weighted) ** 2).mean()) > float(((z - plain) ** 2).mean())


def test_value_sensitivity_is_channel_constant_so_weighting_a_token_is_a_no_op():
    """``sum_i pi_ij^2`` does not depend on the channel, so NUQ on values cannot use it."""
    g = torch.Generator().manual_seed(11)
    v = torch.randn(3, 16, generator=g, dtype=torch.float64)
    per_token = torch.rand(3, 1, generator=g, dtype=torch.float64).expand_as(v)
    assert torch.allclose(quantize_vectors(v, 3, per_token.contiguous(), 0.0, nuq=True),
                          quantize_vectors(v, 3, None, 0.0, nuq=True), atol=1e-12)


def test_dense_and_sparse_keeps_the_isolated_outliers_exact():
    """1% of entries in fp16, excluded from the range the other levels have to cover."""
    g = torch.Generator().manual_seed(12)
    z = torch.randn(4, 100, generator=g, dtype=torch.float64)
    z[:, 7] = 60.0
    z[:, 11] = -60.0
    dense = quantize_vectors(z, 3, None, 0.0, nuq=False)
    sparse = quantize_vectors(z, 3, None, 0.04, nuq=False)
    assert torch.equal(sparse[:, 7], z[:, 7]) and torch.equal(sparse[:, 11], z[:, 11])
    keep = [c for c in range(100) if c not in (7, 11)]
    assert (float(((z[:, keep] - sparse[:, keep]) ** 2).mean())
            < float(((z[:, keep] - dense[:, keep]) ** 2).mean()))


def test_keep_first_pins_the_attention_sink():
    q, k, v = qkv()
    out = ab.build("kvquant", bits_k=3, bits_v=3, keep_first=2)(q, k, v,
                                                                return_weights=True)
    assert float(out.extras["delta_k"][..., :2].abs().max()) == 0.0
    assert float(out.extras["delta_v"][..., :2].abs().max()) == 0.0
    assert float(out.extras["delta_k"][..., 2:].min()) > 0.0


def test_the_local_fisher_is_the_first_order_term_of_the_error():
    """``extras['sensitivity']`` has one weight per (token, channel) and is nonnegative."""
    q, k, v = qkv()
    out = ab.build("kvquant", bits_k=3, nuq=True, sensitivity=True)(q, k, v,
                                                                    return_weights=True)
    f = out.extras["sensitivity"]
    assert f.shape == k.shape and bool((f >= 0).all())
    assert ab.build("kvquant", sensitivity=False)(q, k, v).y is not None


# ===========================================================================
# Atom: energy reorder, mixed precision, per head
# ===========================================================================
def test_atom_without_outliers_is_the_papers_section_4_4_cache_scheme():
    """Asymmetric quantization within each head -- which is ``quantkv(axis='token')``."""
    q, k, v = qkv(outlier=3)
    a = ab.build("atom", bits_k=4, bits_v=4, outliers=0, group=None)
    b = ab.build("quantkv", bits_k=4, bits_v=4, axis="token", group=None)
    assert torch.equal(a(q, k, v).y, b(q, k, v).y)


def test_the_permutation_is_exact_and_only_bites_through_the_grouping():
    """Reordering is a layout device: with one scale per token it changes nothing."""
    q, k, v = qkv(outlier=3)
    assert torch.equal(reorder_mixed(k, 4, 8, 0, None)[0],
                       affine_quantize(k, 4, "token", None))
    assert not torch.allclose(reorder_mixed(k, 4, 8, 0, 4)[0],
                              affine_quantize(k, 4, "token", 4))


def test_atom_mixed_precision_beats_a_larger_uniform_budget():
    """3 bits + one INT8 channel of 8 is 3.625 bits/entry, and still beats 4 everywhere."""
    for seed in range(4):
        q, k, v = qkv(seed=seed, outlier=3)
        mixed = reorder_mixed(k, 3, 8, 1, None)[0]
        uniform = affine_quantize(k, 4, "token", None)
        assert float((k - mixed).norm()) < float((k - uniform).norm())


def test_atom_spends_its_high_precision_on_the_high_energy_channels():
    q, k, v = qkv(outlier=3)
    out = ab.build("atom", bits_k=3, bits_v=None, bits_outlier=8,
                   outliers=1)(q, k, v, return_weights=True)
    assert out.extras["outlier_channels_k"] == 1
    err = (k - reorder_mixed(k, 3, 8, 1, None)[0]).abs().mean(dim=(0, 1, 2))
    assert int(err.argmin()) == 3, err
    assert float(err[3]) < 0.1 * float(err.mean())


# ===========================================================================
# QJL: a 1-bit sketch, unbiased for the logit and biased for the output
# ===========================================================================
def ensemble_keys(k, sketch, seeds, proj=None):
    """``seeds`` independent QJL sketches of ``k``, vectorised along a leading axis.

    Exactly the arithmetic of :func:`attnbank.methods.compress.qjl.qjl_keys`, with the
    seed axis folded into the einsum so that a Monte-Carlo study is two matmuls instead
    of ``seeds`` of them.  The test below pins the two together on a shared projection,
    so the statistics measured with this helper are statistics about ``qjl_keys``.
    """
    if proj is None:
        proj = torch.randn(seeds, sketch, k.shape[-1], dtype=k.dtype,
                           generator=torch.Generator().manual_seed(0))
    sign = torch.sign(torch.einsum("...nd,srd->s...nr", k, proj))
    nu = k.norm(dim=-1, keepdim=True)
    return (math.sqrt(math.pi / 2) / sketch) * nu * torch.einsum("s...nr,srd->s...nd",
                                                                 sign, proj)


def test_the_decoded_sketch_reproduces_the_estimator_exactly():
    """``<q, k_hat_j> == sqrt(pi/2) (nu_j / R) <S q, sign(S k_j)>`` -- the paper's formula."""
    q, k, _ = qkv(b=1, h=1, m=4, n=6)
    r = 32
    gen = torch.Generator().manual_seed(0)          # the generator ``qjl_keys`` uses
    s = torch.randn(r, k.shape[-1], generator=gen, dtype=torch.float64)
    k_hat, bits = qjl_keys(k, r, seed=0)
    assert set(bits.unique().tolist()) <= {-1.0, 0.0, 1.0}
    direct = (math.sqrt(math.pi / 2) / r) * k.norm(dim=-1, keepdim=True).transpose(-1, -2) \
        * ((q @ s.T) @ bits.transpose(-1, -2))
    assert torch.allclose(q @ k_hat.transpose(-1, -2), direct, atol=1e-12)
    assert torch.allclose(ensemble_keys(k, r, 1, s[None])[0], k_hat, atol=1e-12)


def _qjl_ensemble(q, k, v, sketch, seeds):
    """Scores and outputs of ``seeds`` independent sketches, as ``[seeds, B, H, ...]``."""
    k_hat = ensemble_keys(k, sketch, seeds)
    s = torch.einsum("...md,s...nd->s...mn", q, k_hat) * q.shape[-1] ** -0.5
    return s, torch.softmax(s, dim=-1) @ v


def test_qjl_logits_are_unbiased_but_its_outputs_are_not():
    """The estimator is unbiased for ``<q, k>``; softmax is convex, so ``y`` is not.

    Both biases are measured in units of the Monte-Carlo standard error of the mean, so
    the test is a statement about bias and not about variance.  The logit sits within
    noise of zero; the output does not, and that is the failure mode that makes an
    "unbiased" quantizer interesting (notes: QJL/TurboQuant tilt toward large-norm keys).
    """
    seeds, sketch = 192, 8
    q, k, v = qkv(b=1, h=2, m=6, n=10, dv=5, seed=3)
    k = k * torch.linspace(0.3, 2.0, k.shape[-2], dtype=k.dtype)[:, None]
    ref = softmax_out(q, k, v)
    s_true = torch.einsum("...md,...nd->...mn", q, k) * q.shape[-1] ** -0.5
    s, y = _qjl_ensemble(q, k, v, sketch, seeds)

    z_logit = ((s.mean(0) - s_true).abs() / (s.std(0) / seeds ** 0.5)).mean()
    z_out = ((y.mean(0) - ref.y).abs() / (y.std(0) / seeds ** 0.5)).mean()
    assert float(z_logit) < 2.0, f"logit estimator looks biased: z = {float(z_logit)}"
    assert float(z_out) > 2.0, f"output looks unbiased: z = {float(z_out)}"


def test_qjl_variance_falls_like_one_over_the_sketch_dimension():
    q, k, v = qkv(b=1, h=2, m=6, n=10, dv=5, seed=3)
    s_true = torch.einsum("...md,...nd->...mn", q, k) * q.shape[-1] ** -0.5
    sds = []
    for sketch in (8, 32, 128, 512):
        s, _ = _qjl_ensemble(q, k, v, sketch, 96)
        sds.append(float((s - s_true).std()))
    assert all(a > b for a, b in zip(sds, sds[1:])), sds
    ratios = [a / b for a, b in zip(sds, sds[1:])]           # 4x the sketch -> half
    assert all(1.7 < r < 2.3 for r in ratios), ratios


def test_qjl_stores_one_norm_per_token_and_no_scales():
    """The cost model is the claim: ``sketch`` bits plus a norm, no group constants."""
    c = ab.build("qjl", sketch=64, bits_v=4).cost(1, 4096, 128, 128)
    dense = ab.build("softmax").cost(1, 4096, 128, 128)
    assert c["state_bytes"] < dense["state_bytes"]


# ===========================================================================
# ZipCache: precision by normalised attention score
# ===========================================================================
@pytest.mark.parametrize("bits", [2, 3, 4])
def test_channel_separable_quantization_equalises_precision_across_channels(bits):
    """What the channel normalizer buys is *relative* precision, channel by channel.

    A plain per-token scale is set by whichever channel is largest in that token, so
    every channel gets the same **absolute** step: the outlier is fine and the small
    channels are obliterated -- here by a factor of order 100 between best and worst.
    Dividing by ``a_c = max_j |x_jc|`` first makes the step proportional to the
    channel's own scale, so the relative error is near-uniform and the *worst* channel
    -- the one a query might load on -- is several times more accurate.  It is not a
    smaller Frobenius error, and this test does not claim one.
    """
    for seed in range(3):
        q, k, v = qkv(seed=seed, outlier=3)
        def relative(hat):
            return ((k - hat).pow(2).sum((0, 1, 2)).sqrt()
                    / k.pow(2).sum((0, 1, 2)).sqrt())
        cs, pt = relative(channel_separable_quantize(k, bits)), \
            relative(affine_quantize(k, bits, "token", None))
        assert float(cs.max()) < float(pt.max())
        assert float(cs.max() / cs.min()) < 3.0
        assert float(pt.max() / pt.min()) > 20.0


def test_the_normalized_score_is_less_positionally_biased():
    """The accumulated column sum rewards being early twice over; dividing by nnz does not."""
    n = 32
    for seed in range(3):
        q, k, v = qkv(b=2, h=4, m=n, n=n, seed=seed)
        out = ab.build("zipcache")(q, k, v, return_weights=True)
        acc, nrm = out.extras["accumulated"], out.extras["saliency"]
        pos = torch.arange(n, dtype=torch.float64)
        top = lambda t: pos[t.topk(8, dim=-1).indices].mean()
        assert float(top(nrm)) > float(top(acc)), seed
        spread = lambda t: float((t.amax(-1) / t.amin(-1)).mean())
        assert spread(nrm) < 0.5 * spread(acc), (spread(nrm), spread(acc))


def test_salient_tokens_are_the_ones_with_the_small_value_error():
    q, k, v = qkv(m=24, n=24)
    out = ab.build("zipcache", bits_salient=4, bits_regular=2,
                   salient=0.5)(q, k, v, return_weights=True)
    keep, dv = out.extras["salient"], out.extras["delta_v"]
    assert 0 < int(keep.sum()) < keep.numel()
    assert float(dv[keep].mean()) < 0.5 * float(dv[~keep].mean())


def test_the_probe_sample_recovers_most_of_the_exact_salient_set():
    """The point of the probe trick: the ranking survives looking at a few query rows."""
    for seed in range(3):
        q, k, v = qkv(m=24, n=24, seed=seed)
        exact = ab.build("zipcache", salient=0.5)(q, k, v,
                                                  return_weights=True).extras["salient"]
        probed = ab.build("zipcache", salient=0.5, probe=0.25)(
            q, k, v, return_weights=True).extras["salient"]
        overlap = int((exact & probed).sum()) / int(exact.sum())
        assert overlap > 0.5, (seed, overlap)


def test_zipcache_spends_more_bits_than_eviction_and_fewer_than_dense():
    """Mixed precision sits between ``quantkv`` at the low width and at the high one."""
    args = dict(m=1, n=4096, d=128, dv=128)
    lo = ab.build("quantkv", bits_k=2, bits_v=2).cost(**args)["state_bytes"]
    hi = ab.build("quantkv", bits_k=4, bits_v=4).cost(**args)["state_bytes"]
    mid = ab.build("zipcache").cost(**args)["state_bytes"]
    assert lo < mid < hi
