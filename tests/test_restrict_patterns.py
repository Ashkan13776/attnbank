"""Geometry of the *fixed* sparsity patterns: longformer, bigbird, strided, shifted_window.

The conformance suite already checks that each of these is a well-formed method -- the
invariant, row stochasticity, the exact limit, no causal leak, no ``nan``.  What it
cannot check is whether the pattern is the one the paper describes, and that is what
is here: that a dilation actually dilates, that BigBird's random blocks are seeded and
land outside the window, that the Sparse Transformer's periodic head really is
periodic, that Swin's windows partition the sequence and the shift moves the
boundaries.

Two properties are asserted for all four, because they are what makes this family
*fixed*: the support is a function of positions only (perturbing ``q, k, v`` cannot
move it, which is what separates these from the learned selectors), and every row
keeps at least its own key, so renormalisation never divides by a clamped zero.
"""
from __future__ import annotations

import pytest
import torch

import attnbank as ab
from attnbank import metrics as M
from attnbank.core import causal_mask, query_positions

PATTERNS = ["longformer", "bigbird", "strided", "shifted_window"]

#: the tightest configuration of each: a single retained key per row where possible
MINIMAL = {
    "longformer": {"w": 1, "dilation": 1, "n_global": 0},
    "bigbird": {"block": 1, "window_blocks": 1, "random_blocks": 0, "global_blocks": 0},
    "strided": {"stride": 1, "pattern": "fixed", "c": 0},
    "shifted_window": {"w": 1, "shift": 0},
}


def qkv(b=2, h=4, m=12, n=24, d=8, dv=6, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g).to(dtype)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def support_of(name, cfg=None, causal=False, **kw):
    q, k, v = qkv(**kw)
    return ab.build(name, **(cfg or {}))(q, k, v, causal=causal).extras["support"]


def offsets(sup, m, n):
    """``p - j`` for every retained entry of a ``[m, n]`` support."""
    p = query_positions(m, n)[:, None].expand(m, n)
    j = torch.arange(n)[None, :].expand(m, n)
    return (p - j)[sup]


# ---------------------------------------------------------------------------
# what makes the family "fixed": the support is a function of positions alone
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("causal", [False, True])
def test_the_support_does_not_depend_on_the_content(name, causal):
    """Query-independent by construction -- the dividing line against learned selectors."""
    meth = ab.build(name)
    a = meth(*qkv(seed=0), causal=causal).extras["support"]
    b = meth(*qkv(seed=1), causal=causal).extras["support"]
    assert torch.equal(a, b), name
    c = meth(*qkv(seed=0, dtype=torch.float32), causal=causal).extras["support"]
    assert torch.equal(a, c), f"{name}: support moved with the dtype"


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("causal", [False, True])
def test_the_extras_contract(name, causal):
    """``support`` is a boolean [B, H, m, n] and ``eps`` is the mass it omits."""
    q, k, v = qkv()
    out = ab.build(name)(q, k, v, causal=causal, return_weights=True)
    sup = out.extras["support"]
    assert sup.dtype is torch.bool and sup.shape == (2, 4, 12, 24), name
    pi = ab.build("softmax")(q, k, v, causal=causal, return_weights=True).weights
    assert torch.allclose(out.extras["eps"], 1.0 - (pi * sup).sum(-1), atol=1e-12)
    assert torch.equal(out.extras["k_realized"], sup.sum(-1))


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("causal", [False, True])
def test_prop_sparse_identity(name, causal):
    """y - y_hat = eps_q (m_{S^c} - m_S): these are restrictions, nothing else."""
    q, k, v = qkv()
    ref = ab.build("softmax")(q, k, v, causal=causal, return_weights=True)
    out = ab.build(name)(q, k, v, causal=causal, return_weights=True)
    r = M.restriction_identity(ref.weights, out.extras["support"], v)
    assert torch.allclose(ref.y - out.y, r["predicted_error"], atol=1e-11), name
    assert torch.allclose(r["eps"], out.extras["eps"], atol=1e-12)
    assert torch.allclose(M.total_variation(ref.weights, out.weights), r["eps"], atol=1e-11)


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("causal", [False, True])
def test_the_output_is_the_retained_conditional_mean(name, causal):
    q, k, v = qkv()
    ref = ab.build("softmax")(q, k, v, causal=causal, return_weights=True)
    out = ab.build(name)(q, k, v, causal=causal)
    S = out.extras["support"]
    m_S = (ref.weights * S) @ v / (ref.weights * S).sum(-1, keepdim=True)
    assert torch.allclose(out.y, m_S, atol=1e-11), name


# ---------------------------------------------------------------------------
# non-empty support and causal safety -- the two ways a fixed pattern goes wrong
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("cfg", [None, "minimal"])
@pytest.mark.parametrize("causal", [False, True])
def test_every_query_keeps_at_least_its_own_key(name, cfg, causal):
    """No row is ever empty, and the retained mass is far from the TINY clamp."""
    conf = MINIMAL[name] if cfg == "minimal" else {}
    q, k, v = qkv()
    out = ab.build(name, **conf)(q, k, v, causal=causal, return_weights=True)
    sup = out.extras["support"]
    assert (sup.sum(-1) >= 1).all(), f"{name} {conf}: an empty row"
    diag = (query_positions(12, 24)[:, None] == torch.arange(24)[None, :])
    assert (sup & diag).any(-1).all(), f"{name} {conf}: the diagonal was dropped"
    assert torch.isfinite(out.y).all() and torch.isfinite(out.weights).all()
    assert float(out.extras["eps"].max()) < 1.0 - 1e-12, "retained mass hit the clamp"


@pytest.mark.parametrize("name", PATTERNS)
def test_the_minimal_configuration_returns_the_diagonal_value(name):
    """With one key per row the restriction is the identity read-out, exactly."""
    q, k, v = qkv()
    out = ab.build(name, **MINIMAL[name])(q, k, v, causal=True)
    assert torch.equal(out.extras["support"].sum(-1),
                       torch.ones_like(out.extras["support"].sum(-1)))
    expected = v[..., query_positions(12, 24), :]
    assert torch.allclose(out.y, expected, atol=1e-12), name


@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("cfg", [None, "minimal"])
def test_no_pattern_reaches_into_the_future(name, cfg):
    """Under the causal mask the support is a subset of what query p may see."""
    conf = MINIMAL[name] if cfg == "minimal" else {}
    sup = support_of(name, conf, causal=True)
    allowed = causal_mask(12, 24)
    assert not (sup & ~allowed).any(), f"{name} {conf}: support beyond the diagonal"


@pytest.mark.parametrize("name", ["longformer", "strided", "shifted_window"])
def test_the_causal_support_is_the_masked_bidirectional_one(name):
    """The pattern is built once and intersected with the mask; nothing else changes.

    BigBird is excluded on purpose: its random blocks are *drawn from the causally
    admissible candidates*, so that an early query is not left with a support the mask
    has emptied of everything but the window.  That is checked separately, below.
    """
    bi = support_of(name, causal=False)
    ca = support_of(name, causal=True)
    assert torch.equal(ca, bi & causal_mask(12, 24)), name


# ---------------------------------------------------------------------------
# longformer: the dilated sliding window and the global tokens
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dilation", [1, 2, 3, 5])
def test_dilation_actually_dilates(dilation):
    """Every retained offset is a multiple of d, and the reach is (w//2) * d."""
    w, n = 8, 64
    sup = support_of("longformer", {"w": w, "dilation": dilation}, causal=False, n=n, m=n)
    off = offsets(sup[0, 0], n, n)
    assert (off % dilation == 0).all(), dilation
    assert int(off.abs().max()) == (w // 2) * dilation
    mid = sup[0, 0, n // 2]                               # room for both sides of the window
    assert int(mid.sum()) == w + 1, (dilation, int(mid.sum()))


def test_dilation_trades_resolution_for_reach_at_a_fixed_budget():
    """The defining trade of a dilated window: same count, d times the span."""
    counts, reach = [], []
    for d in (1, 2, 4):
        sup = support_of("longformer", {"w": 8, "dilation": d}, causal=True, n=40, m=40)
        counts.append(int(sup[0, 0, 30].sum()))
        reach.append(int(offsets(sup[0, 0], 40, 40).max()))
    assert len(set(counts)) == 1, counts           # the budget does not move
    assert reach == [4, 8, 16], reach              # the span does


def test_undilated_longformer_is_the_sliding_window():
    """w//2 + 1 retained positions per causal row: exactly ``window`` at that width."""
    q, k, v = qkv()
    for width in (2, 3, 5):
        a = ab.build("longformer", w=2 * (width - 1), dilation=1)(q, k, v, causal=True)
        b = ab.build("window", w=width)(q, k, v, causal=True)
        assert torch.equal(a.extras["support"], b.extras["support"]), width
        assert torch.allclose(a.y, b.y, atol=1e-14)


@pytest.mark.parametrize("causal", [False, True])
def test_global_tokens_are_attended_by_everyone_and_attend_to_everyone(causal):
    """Beltagy et al.: A(i,:) = 1 and A(:,i) = 1. A sink only gets the second half."""
    g = 3
    sup = support_of("longformer", {"w": 4, "n_global": g}, causal=causal)
    assert sup[..., :g].all(), "a global column was not attended"
    p = query_positions(12, 24)
    rows = p < g                                   # the global queries, if any are in range
    allowed = causal_mask(12, 24) if causal else torch.ones(12, 24, dtype=torch.bool)
    if rows.any():
        assert torch.equal(sup[0, 0][rows], allowed[rows]), "a global row was not dense"
    # and a global *column* is strictly more than the sinks of sink_window only in rows
    assert int(sup[0, 0].sum()) > int(support_of("longformer", {"w": 4})[0, 0].sum())


def test_per_head_dilation_gives_each_head_its_own_reach():
    """Longformer runs undilated and dilated heads side by side."""
    meth = ab.build("longformer", w=8, dilation=3, head_dilation=True)
    assert meth.head_dilations(4) == [1, 2, 3, 1]
    sup = meth(*qkv(n=40, m=40), causal=True).extras["support"][0]
    for h, d in enumerate([1, 2, 3, 1]):
        single = support_of("longformer", {"w": 8, "dilation": d}, causal=True, n=40, m=40)
        assert torch.equal(sup[h], single[0, 0]), (h, d)
    assert not torch.equal(sup[0], sup[2]), "head_dilation did not change anything"


def test_dilation_alone_is_the_periodic_column_pattern():
    """w=None leaves only the residue condition -- LongNet's dilated attention."""
    sup = support_of("longformer", {"w": None, "dilation": 4}, causal=True, n=32, m=32)
    off = offsets(sup[0, 0], 32, 32)
    assert (off % 4 == 0).all() and int(off.max()) == 28


# ---------------------------------------------------------------------------
# bigbird: window + global + *random* blocks
# ---------------------------------------------------------------------------
BB = {"block": 4, "window_blocks": 3, "random_blocks": 2, "global_blocks": 1}


@pytest.mark.parametrize("causal", [False, True])
def test_bigbird_random_blocks_are_seeded_and_reproducible(causal):
    a = support_of("bigbird", {**BB, "seed": 0}, causal=causal, m=64, n=64)
    b = support_of("bigbird", {**BB, "seed": 0}, causal=causal, m=64, n=64)
    c = support_of("bigbird", {**BB, "seed": 1}, causal=causal, m=64, n=64)
    assert torch.equal(a, b), "the same seed gave a different pattern"
    assert not torch.equal(a, c), "the seed does not reach the random blocks"


def test_bigbird_random_draw_does_not_touch_the_global_rng():
    torch.manual_seed(0)
    before = torch.randn(3)
    torch.manual_seed(0)
    support_of("bigbird", BB, m=64, n=64)
    assert torch.equal(before, torch.randn(3)), "the draw advanced the global RNG"


@pytest.mark.parametrize("causal", [False, True])
def test_bigbird_is_block_structured(causal):
    """The support is constant along each key block, except where the mask cuts it."""
    sup = support_of("bigbird", BB, causal=causal, m=64, n=64)[0, 0]
    allowed = causal_mask(64, 64) if causal else torch.ones(64, 64, dtype=torch.bool)
    blocks = (sup | ~allowed).unflatten(-1, (16, 4))
    assert (blocks.all(-1) | ~sup.unflatten(-1, (16, 4)).any(-1)).all()


def test_bigbird_keeps_the_window_the_global_blocks_and_r_random_ones():
    """The three families, counted: w window + g global + r random blocks per row."""
    meth = ab.build("bigbird", **BB)
    sel = meth.block_pattern(64, causal=False)                 # [16, 16]
    nb, half = 16, 1
    for pb in range(1, nb):                                    # pb = 0 is a global row
        kb = torch.arange(nb)
        window = (kb - pb).abs() <= half
        glob = kb < BB["global_blocks"]
        rand = sel[pb] & ~window & ~glob
        assert int(rand.sum()) == BB["random_blocks"], (pb, int(rand.sum()))
        assert (sel[pb] | ~(window | glob)).all(), f"block {pb} lost its window"
    assert sel[0].all(), "the global row is not dense"


@pytest.mark.parametrize("random_blocks", [0, 1, 3])
def test_bigbird_random_budget_is_what_was_asked_for(random_blocks):
    sel = ab.build("bigbird", **{**BB, "random_blocks": random_blocks}) \
        .block_pattern(64, causal=False)
    kb = torch.arange(16)
    extra = [int((sel[pb] & ((kb - pb).abs() > 1) & (kb >= 1)).sum()) for pb in range(2, 16)]
    assert extra == [random_blocks] * len(extra), extra


def test_bigbird_random_blocks_stay_in_the_past_when_causal():
    """The draw is taken from the causally admissible candidates, not masked afterwards."""
    meth = ab.build("bigbird", **BB)
    sel = meth.block_pattern(64, causal=True)
    for pb in range(BB["global_blocks"], 16):
        chosen = sel[pb].nonzero().flatten()
        assert int(chosen.max()) <= pb + 1, pb          # own block, or its window neighbour
        far = chosen[(chosen - pb).abs() > 1]
        far = far[far >= BB["global_blocks"]]
        assert (far < pb).all(), f"block {pb} drew a future random block {far.tolist()}"


def test_bigbird_random_blocks_make_the_support_reach_further_than_a_window():
    """What the random family buys: mass outside the band, at the same block budget."""
    cfg = {**BB, "global_blocks": 0}       # the global columns would reach back anyway
    far = support_of("bigbird", cfg, causal=True, m=64, n=64)
    band = support_of("bigbird", {**cfg, "random_blocks": 0}, causal=True, m=64, n=64)
    assert (far | band == far).all(), "r = 0 is not a sub-pattern"
    off = offsets(far[0, 0], 64, 64)
    assert int(off.max()) > int(offsets(band[0, 0], 64, 64).max())


def test_bigbird_global_blocks_are_symmetric():
    sup = support_of("bigbird", {**BB, "global_blocks": 2}, causal=True, m=64, n=64)
    allowed = causal_mask(64, 64)
    assert (sup[..., :8] | ~allowed[:, :8]).all(), \
        "a global column is not attended wherever it is visible"
    assert sup[0, 0, :8].sum(-1).tolist() == list(range(1, 9)), "global rows are not dense"


# ---------------------------------------------------------------------------
# strided: the Sparse Transformer's two factorized patterns
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stride", [2, 4, 7])
def test_strided_pattern_is_local_plus_a_residue_class(stride):
    """A_i^(1) = {i-l, ..., i} and A_i^(2) = {j : (i-j) mod l = 0}, exactly."""
    sup = support_of("strided", {"stride": stride, "pattern": "strided"},
                     causal=True, m=48, n=48)[0, 0]
    j = torch.arange(48)
    for p in (3, 20, 47):
        local = (p - j).abs() <= stride
        periodic = (p - j) % stride == 0
        assert torch.equal(sup[p], (local | periodic) & (j <= p)), (p, stride)
    off = offsets(sup, 48, 48)
    assert ((off <= stride) | (off % stride == 0)).all(), stride


@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("stride", [4, 8])
def test_fixed_pattern_is_the_own_block_plus_the_summarising_columns(c, stride):
    sup = support_of("strided", {"stride": stride, "pattern": "fixed", "c": c},
                     causal=True, m=48, n=48)[0, 0]
    j = torch.arange(48)
    for p in (10, 30, 47):
        row = sup[p]
        own = (j // stride) == (p // stride)
        cols = (j % stride) >= (stride - c)
        assert torch.equal(row, (own | cols) & (j <= p)), (p, stride, c)


@pytest.mark.parametrize("pattern", ["strided", "fixed"])
def test_stride_one_is_dense_for_both_patterns(pattern):
    q, k, v = qkv()
    for causal in (False, True):
        out = ab.build("strided", stride=1, pattern=pattern)(q, k, v, causal=causal)
        ref = ab.build("softmax")(q, k, v, causal=causal)
        assert torch.allclose(out.y, ref.y, atol=1e-13), pattern


def test_the_strided_budget_is_order_l_plus_n_over_l():
    """sqrt(n) is the stride that minimises the retained count -- the paper's choice."""
    n = 64
    kept = {l: int(support_of("strided", {"stride": l}, causal=True, m=n, n=n)[0, 0, -1].sum())
            for l in (2, 4, 8, 16, 32)}
    assert min(kept, key=kept.get) == 8 == int(n ** 0.5), kept


def test_an_unknown_pattern_is_rejected():
    with pytest.raises(ValueError, match="pattern must be one of"):
        ab.build("strided", pattern="diagonalish")


# ---------------------------------------------------------------------------
# shifted_window: block-diagonal, displaced
# ---------------------------------------------------------------------------
def test_shift_zero_is_block_diagonal():
    w = 4
    sup = support_of("shifted_window", {"w": w, "shift": 0}, causal=False, m=24, n=24)[0, 0]
    j = torch.arange(24)
    for p in range(24):
        assert torch.equal(sup[p], (j // w) == (p // w)), p


def test_the_windows_partition_the_sequence():
    """Any two rows see either the same keys or none in common -- unlike a band."""
    sup = support_of("shifted_window", {"w": 5}, causal=False, m=24, n=24)[0, 0]
    for a in range(24):
        for b in range(24):
            shared = sup[a] & sup[b]
            assert torch.equal(sup[a], sup[b]) or not shared.any(), (a, b)
    assert torch.equal(sup.any(0), torch.ones(24, dtype=torch.bool)), "a key is unreachable"


def test_the_shift_moves_the_window_boundaries():
    """Swin's displaced partition connects the pairs the regular one separates."""
    w = 4
    reg = support_of("shifted_window", {"w": w, "shift": 0}, causal=False, m=24, n=24)[0, 0]
    shd = support_of("shifted_window", {"w": w, "shift": w // 2}, causal=False,
                     m=24, n=24)[0, 0]
    assert not reg[3, 4] and shd[3, 4], "the boundary pair 3-4 was not connected"
    assert reg[3, 0] and not shd[3, 0], "the shifted window still starts at 0"
    assert ab.build("shifted_window", w=w).resolved_shift() == w // 2   # the Swin default
    assert int(shd[0].sum()) == w - w // 2, "the leading short window is wrong"


@pytest.mark.parametrize("w", [2, 3, 6])
def test_the_offset_within_the_window_is_reported_and_bounded(w):
    q, k, v = qkv(m=24, n=24)
    out = ab.build("shifted_window", w=w)(q, k, v)
    off = out.extras["window_offset"]
    assert ((off >= 0) & (off < w)).all(), off
    assert int(out.extras["n_windows"]) == (24 + out.extras["shift"] + w - 1) // w


def test_a_block_support_is_weaker_than_a_band_at_the_same_width():
    """The point of keeping both: a query at a block edge keeps only one key."""
    w = 4
    blk = support_of("shifted_window", {"w": w, "shift": 0}, causal=True, m=24, n=24)[0, 0]
    bnd = support_of("window", {"w": w}, causal=True, m=24, n=24)[0, 0]
    assert (blk.sum(-1) <= bnd.sum(-1)).all()
    assert int(blk[4].sum()) == 1 and int(bnd[4].sum()) == w
    q, k, v = qkv(m=24, n=24)
    ref = ab.build("softmax")(q, k, v, causal=True)
    e_blk = (ref.y - ab.build("shifted_window", w=w, shift=0)(
        q, k, v, causal=True).y).norm(dim=-1)
    e_bnd = (ref.y - ab.build("window", w=w)(q, k, v, causal=True).y).norm(dim=-1)
    assert float(e_bnd.mean()) < float(e_blk.mean())


# ---------------------------------------------------------------------------
# all four, at their limit, from both directions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", PATTERNS)
@pytest.mark.parametrize("causal", [False, True])
def test_widening_the_pattern_is_monotone_in_the_omitted_mass(name, causal):
    """A pattern that grows can only omit less: the sanity check behind the anchor."""
    grids = {"longformer": [{"w": 2}, {"w": 6}, {"w": 12}, {"w": None}],
             "bigbird": [{"window_blocks": 1}, {"window_blocks": 3},
                         {"window_blocks": 5}, {"window_blocks": None}],
             # strides are not nested (different residue classes); c is
             "strided": [{"pattern": "fixed", "stride": 4, "c": 1},
                         {"pattern": "fixed", "stride": 4, "c": 2},
                         {"pattern": "fixed", "stride": 4, "c": 3},
                         {"pattern": "fixed", "stride": 4, "c": 4}],
             "shifted_window": [{"w": 2}, {"w": 6}, {"w": 12}, {"w": None}]}
    q, k, v = qkv()
    eps = []
    for cfg in grids[name]:
        cfg = dict(cfg)
        if name == "bigbird":
            cfg["random_blocks"] = 0        # widen one family at a time
        eps.append(float(ab.build(name, **cfg)(q, k, v, causal=causal).extras["eps"].mean()))
    assert all(a >= b - 1e-12 for a, b in zip(eps, eps[1:])), (name, eps)
    assert eps[-1] == pytest.approx(0.0, abs=1e-15)
