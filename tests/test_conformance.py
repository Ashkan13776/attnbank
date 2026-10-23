"""The conformance suite of DESIGN.md Sec. 4, parametrized over every registered method.

Eleven properties, no exemptions outside the class attributes a method declares
(``supports_causal``, ``expresses_weights``, ``row_stochastic``, ``head_independent``,
``objective``).  A method that cannot pass one of these must say so in an attribute,
not in a skip.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import pytest
import torch

import attnbank as ab
from attnbank.backends import DISPATCHABLE
from attnbank.core import Objective

# The library ships its own bibliography (``attnbank/references.bib``) so that check 11
# works in the standalone repository; inside the monograph's tree the full
# ``references.bib`` sits two levels up and is used as well, so a key may live in either.
BIBS = [Path(__file__).resolve().parents[1] / "attnbank" / "references.bib",
        Path(__file__).resolve().parents[2] / "references.bib"]
METHODS = ab.available()

B, H, M, N, D, DV = 2, 4, 12, 24, 8, 6
SQ = 16           # square shape for the exact-limit anchor (m == n)
ATOL32, ATOL64 = 1e-5, 1e-10


def qkv(b=B, h=H, m=M, n=N, d=D, dv=DV, seed=0, dtype=torch.float32, scale=1.0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: (scale * torch.randn(*s, generator=g)).to(dtype=dtype, device=device)
    return mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)


def run(name, cfg=None, causal=False, **kw):
    meth = ab.build(name, **(cfg or {}))
    q, k, v = qkv(**kw)
    if causal and not meth.supports_causal:
        pytest.skip(f"{name} declares supports_causal = False")
    return meth, (q, k, v), meth(q, k, v, causal=causal, return_weights=True)


def softmax_ref(q, k, v, causal):
    return ab.build("softmax")(q, k, v, causal=causal, return_weights=True)


# ---------------------------------------------------------------------------
# 1. shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("causal", [False, True])
def test_shapes(name, causal):
    meth, (q, k, v), out = run(name, causal=causal)
    assert out.y.shape == (B, H, M, DV)
    if out.weights is not None:
        assert out.weights.shape == (B, H, M, N)
    if out.v_hat is not None:
        assert out.v_hat.shape == (B, H, N, DV)
    assert isinstance(out.extras, dict)


# ---------------------------------------------------------------------------
# 2. Invariant W:  y == weights @ (v_hat or v)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("causal", [False, True])
def test_invariant_w(name, causal):
    meth, (q, k, v), out = run(name, causal=causal, dtype=torch.float64)
    if not meth.expresses_weights:
        assert out.weights is None, f"{name} declares expresses_weights=False but returned weights"
        return
    assert out.weights is not None, f"{name} declares expresses_weights=True but returned None"
    recon = out.weights @ (out.v_hat if out.v_hat is not None else v)
    assert torch.allclose(out.y, recon, atol=1e-9, rtol=0), \
        f"{name}: max |y - W v| = {(out.y - recon).abs().max():.3e}"


# ---------------------------------------------------------------------------
# 3. row stochasticity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("causal", [False, True])
def test_row_stochastic(name, causal):
    meth, (q, k, v), out = run(name, causal=causal, dtype=torch.float64)
    if not meth.row_stochastic or out.weights is None:
        return
    assert (out.weights >= -1e-12).all(), f"{name}: negative weight {out.weights.min():.3e}"
    s = out.weights.sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-9), \
        f"{name}: row sums in [{s.min():.6f}, {s.max():.6f}]"


# ---------------------------------------------------------------------------
# 4. causal leakage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
def test_causal_leakage(name):
    meth = ab.build(name)
    if not meth.supports_causal:
        pytest.skip(f"{name} declares supports_causal = False")
    q, k, v = qkv(dtype=torch.float64)
    base = meth(q, k, v, causal=True).y
    for t in (1, 5, 13, N - 1):
        k2, v2 = k.clone(), v.clone()
        g = torch.Generator().manual_seed(100 + t)
        k2[..., t:, :] += torch.randn(k[..., t:, :].shape, generator=g, dtype=k.dtype)
        v2[..., t:, :] += torch.randn(v[..., t:, :].shape, generator=g, dtype=v.dtype)
        pert = meth(q, k2, v2, causal=True).y
        # query i sees keys 0..i+(N-M); the last unaffected query has i+(N-M) < t
        n_safe = max(0, t - (N - M))
        if n_safe == 0:
            continue
        assert torch.equal(base[..., :n_safe, :], pert[..., :n_safe, :]), \
            f"{name}: leak at t={t}, max diff " \
            f"{(base[..., :n_safe, :] - pert[..., :n_safe, :]).abs().max():.3e}"


# ---------------------------------------------------------------------------
# 5. batch and head independence
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("axis", ["batch", "head"])
def test_independence(name, axis):
    meth = ab.build(name)
    if axis == "head" and not meth.head_independent:
        pytest.skip(f"{name} declares head_independent = False (shares K/V across heads)")
    q, k, v = qkv(dtype=torch.float64)
    base = meth(q, k, v).y
    g = torch.Generator().manual_seed(7)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    sl = (1, slice(None)) if axis == "batch" else (slice(None), 1)
    for t, t2 in ((q, q2), (k, k2), (v, v2)):
        t2[sl] += torch.randn(t[sl].shape, generator=g, dtype=t.dtype)
    pert = meth(q2, k2, v2).y
    keep = [i for i in range(B if axis == "batch" else H) if i != 1]
    bs = base[keep] if axis == "batch" else base[:, keep]
    ps = pert[keep] if axis == "batch" else pert[:, keep]
    assert torch.equal(bs, ps), f"{name}: {axis} coupling, max diff {(bs - ps).abs().max():.3e}"


# ---------------------------------------------------------------------------
# 6. exact limit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("causal", [False, True])
def test_exact_limit(name, causal):
    cls = ab.get_class(name)
    cfg = cls.limit_config()
    if cfg is None:
        pytest.skip(f"{name} has no exact limit")
    meth = ab.build(name, **cfg)
    if causal and not meth.supports_causal:
        pytest.skip(f"{name} declares supports_causal = False")
    q, k, v = qkv(m=SQ, n=SQ)       # square: Nystrom's limit needs landmarks = all tokens
    out = meth(q, k, v, causal=causal, return_weights=True)
    ref = softmax_ref(q, k, v, causal)
    assert torch.allclose(out.y, ref.y, atol=ATOL32), \
        f"{name}{cfg}: max |y - y_softmax| = {(out.y - ref.y).abs().max():.3e}"
    if out.weights is not None:
        assert torch.allclose(out.weights, ref.weights, atol=1e-6), \
            f"{name}{cfg}: max |pi - pi_softmax| = {(out.weights - ref.weights).abs().max():.3e}"


@pytest.mark.parametrize("name", METHODS)
def test_has_an_anchor(name):
    """Every method that targets softmax must declare one of the two anchors."""
    cls = ab.get_class(name)
    if cls.objective is Objective.MODIFY:
        return                      # a different operator: neither anchor is meaningful
    assert cls.limit_config() is not None or cls.convergence_configs() is not None, \
        f"{name} is objective={cls.objective.value} but declares no limit_config " \
        f"and no convergence_configs"


# ---------------------------------------------------------------------------
# 7. convergence (Monte-Carlo methods)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
def test_convergence(name):
    cls = ab.get_class(name)
    cfgs = cls.convergence_configs()
    if cfgs is None:
        pytest.skip(f"{name} has no convergence_configs")
    q, k, v = qkv(m=SQ, n=SQ, scale=0.5, dtype=torch.float64)
    ref = softmax_ref(q, k, v, False).y
    errs = []
    for cfg in cfgs:
        e = []
        for seed in range(8):
            out = ab.build(name, seed=seed, **cfg)(q, k, v)
            e.append((out.y - ref).norm(dim=-1).mean().item())
        errs.append(sum(e) / len(e))
    assert all(a > b for a, b in zip(errs, errs[1:])), \
        f"{name}: mean error not monotone along convergence_configs: {errs}"


# ---------------------------------------------------------------------------
# 8. dtype and device
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
def test_dtypes_agree(name):
    meth = ab.build(name)
    y32 = meth(*qkv(dtype=torch.float32)).y
    y64 = meth(*qkv(dtype=torch.float64)).y
    assert y64.dtype == torch.float64
    assert torch.allclose(y32.double(), y64, atol=1e-4), \
        f"{name}: fp32 vs fp64 differ by {(y32.double() - y64).abs().max():.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("name", METHODS)
def test_cuda_matches_cpu(name):
    meth = ab.build(name)
    q, k, v = qkv()
    y_cpu = meth(q, k, v).y
    y_gpu = meth(q.cuda(), k.cuda(), v.cuda()).y.cpu()
    assert torch.allclose(y_cpu, y_gpu, atol=1e-4)


# ---------------------------------------------------------------------------
# 8b. backend equivalence -- every declared backend equals `reference`
# ---------------------------------------------------------------------------
#
# Declaring a backend is a claim, and this is where the claim is checked. DESIGN.md
# Sec. 6 says kernels "must match the reference within tolerance"; these are the
# tolerances, and they are deliberately tight -- a fused kernel reorders a summation,
# it does not change the operator.
#
#   float64   1e-10 * max(1, ||y_ref||_inf)
#   float32   1e-5  * max(1, ||y_ref||_inf)
#
# The scaling by the output magnitude is what makes the tolerance a statement about
# precision rather than about the test inputs: the `large-logits` case multiplies q, k
# AND v by 10, so y grows with it while the relative error of a reassociated reduction
# does not. Unscaled, these are exactly the suite's ATOL32 / ATOL64.
#
# The awkward shapes are the point of the parametrization: `m < n` catches a backend
# that aligns the causal mask at the top left instead of the bottom right (SDPA's
# `is_causal=True` does exactly that), `n == 1` catches single-element support, and
# `scale=10` catches a kernel that exponentiates an unshifted score.
BACKEND_ATOL = {torch.float32: ATOL32, torch.float64: ATOL64}

BACKEND_SHAPES = [
    pytest.param(dict(m=M, n=N), id="m<n"),
    pytest.param(dict(m=SQ, n=SQ), id="square"),
    pytest.param(dict(m=1, n=1), id="single-key"),
    pytest.param(dict(m=1, n=N), id="decode"),
    pytest.param(dict(m=M, n=N, scale=10.0), id="large-logits"),
]

BACKEND_CASES = [(name, backend)
                 for name in METHODS
                 for backend in ab.get_class(name).backends
                 if backend != "reference"]


@pytest.mark.parametrize("name,backend", BACKEND_CASES,
                         ids=[f"{n}-{b}" for n, b in BACKEND_CASES])
@pytest.mark.parametrize("shape", BACKEND_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64],
                         ids=["fp32", "fp64"])
@pytest.mark.parametrize("causal", [False, True])
def test_backend_equivalence(name, backend, shape, dtype, causal):
    """Every declared backend reproduces the reference, bit-close, on every shape."""
    if not ab.is_backend_available(backend):
        pytest.skip(f"{backend} is not available here: {ab.backend_info(backend).reason}")
    ref_method = ab.build(name, backend="reference")
    if causal and not ref_method.supports_causal:
        pytest.skip(f"{name} declares supports_causal = False")
    q, k, v = qkv(dtype=dtype, **shape)
    y_ref = ref_method(q, k, v, causal=causal).y
    with warnings.catch_warnings():
        warnings.simplefilter("error", ab.BackendFallbackWarning)
        y_backend = ab.build(name, backend=backend)(q, k, v, causal=causal).y
    atol = BACKEND_ATOL[dtype] * max(1.0, float(y_ref.abs().max()))
    assert y_backend.shape == y_ref.shape
    assert torch.allclose(y_backend, y_ref, atol=atol, rtol=0), (
        f"{name} backend={backend} dtype={dtype} causal={causal} {shape}: "
        f"max |y - y_reference| = {(y_backend - y_ref).abs().max():.3e} > {atol:.3e}")


@pytest.mark.parametrize("name", METHODS)
def test_backends_are_declared_honestly(name):
    """A declared backend must exist, be dispatchable, and have an implementation."""
    cls = ab.get_class(name)
    assert cls.backends, f"{name} declares no backends at all"
    assert cls.backends[0] == "reference", \
        f"{name}: 'reference' must come first; it is the anchor everything is tested against"
    assert len(set(cls.backends)) == len(cls.backends), f"{name}: duplicate backend"
    for backend in cls.backends:
        assert backend in ab.backends().names, f"{name}: unknown backend {backend!r}"
        if backend == "reference":
            continue
        assert backend in DISPATCHABLE, \
            f"{name} declares {backend!r}, which the library does not dispatch"
        assert hasattr(cls, f"_forward_{backend}"), \
            f"{name} declares backend {backend!r} but has no _forward_{backend}"


@pytest.mark.parametrize("name,backend", BACKEND_CASES,
                         ids=[f"{n}-{b}" for n, b in BACKEND_CASES])
def test_backend_falls_back_loudly_for_weights(name, backend):
    """No fused kernel materialises pi; asking for it must say so, not return None."""
    if not ab.is_backend_available(backend):
        pytest.skip(f"{backend} is not available here")
    meth = ab.build(name, backend=backend)
    q, k, v = qkv(dtype=torch.float64)
    with pytest.warns(ab.BackendFallbackWarning, match="falls back"):
        out = meth(q, k, v, return_weights=True)
    assert out.weights is not None, f"{name}: fell back but still returned no weights"
    ref = ab.build(name, backend="reference")(q, k, v, return_weights=True)
    assert torch.allclose(out.weights, ref.weights, atol=ATOL64, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("name,backend", BACKEND_CASES,
                         ids=[f"{n}-{b}" for n, b in BACKEND_CASES])
@pytest.mark.parametrize("causal", [False, True])
def test_backend_equivalence_on_cuda(name, backend, causal):
    """On CUDA `sdpa` dispatches to a real fused kernel; it must still equal reference."""
    if not ab.is_backend_available(backend):
        pytest.skip(f"{backend} is not available here")
    ref_method = ab.build(name, backend="reference")
    if causal and not ref_method.supports_causal:
        pytest.skip(f"{name} declares supports_causal = False")
    q, k, v = qkv(device="cuda")
    y_ref = ref_method(q, k, v, causal=causal).y
    y_backend = ab.build(name, backend=backend)(q, k, v, causal=causal).y
    atol = ATOL32 * max(1.0, float(y_ref.abs().max()))
    assert torch.allclose(y_backend, y_ref, atol=atol, rtol=0), (
        f"{name} backend={backend} on cuda, causal={causal}: "
        f"max |y - y_reference| = {(y_backend - y_ref).abs().max():.3e} > {atol:.3e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("kernel", ["math", "mem_efficient", "flash"])
def test_sdpa_sub_backends_agree(kernel):
    """`sdpa_kernel=` pins a sub-backend; all of them compute the same operator."""
    q, k, v = qkv(m=SQ, n=SQ, d=64, dv=64, dtype=torch.float16, device="cuda")
    ref = ab.build("softmax", backend="reference")(q.float(), k.float(), v.float()).y
    try:
        got = ab.build("softmax", backend="sdpa", sdpa_kernel=kernel)(q, k, v).y
    except RuntimeError as e:                 # a sub-backend this GPU cannot run
        pytest.skip(f"sdpa_kernel={kernel!r} unavailable: {e}")
    assert torch.allclose(got.float(), ref, atol=2e-2, rtol=0), \
        f"sdpa_kernel={kernel}: max diff {(got.float() - ref).abs().max():.3e}"


def test_an_unsupported_backend_raises_rather_than_downgrading():
    with pytest.raises(ab.BackendNotSupportedError, match="does not support backend"):
        ab.build("nmf_oracle", backend="sdpa")


def test_the_default_backend_is_reference():
    """Nothing may change a published number by changing the default execution path."""
    assert ab.get_default_backend() == "reference"
    assert all(ab.build(n).backend == "reference" for n in METHODS)


# ---------------------------------------------------------------------------
# 9. cost model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
def test_cost_model(name):
    c = ab.build(name).cost(M, N, D, DV)
    assert set(c) == {"flops", "state_bytes", "bytes_moved"}, f"{name}: cost keys {set(c)}"
    for key, val in c.items():
        assert math.isfinite(val) and val > 0, f"{name}: cost[{key}] = {val}"


# Methods whose point is a smaller KV footprint (tab:comparative column "Saves" = kv).
CAPACITY_SAVERS = {"quantkv", "lowrank_kv", "evict_h2o", "evict_snapkv", "merge_kv", "gqa",
                   "mla", "linformer", "lowprec", "performer", "trigrff", "linear_elu",
                   "cosformer", "nystrom", "latent_stochastic", "clustered",
                   "linear_rnn", "decay_rnn", "delta_rnn", "lowrank_ot"}


@pytest.mark.parametrize("name", METHODS)
def test_capacity_saving_is_real(name):
    """A method that claims to save cache capacity must have state_bytes <= dense.

    Evaluated at a long context (n = 4096), the regime the claim is about: a rank-r
    surrogate saves nothing when r exceeds n.
    """
    if name not in CAPACITY_SAVERS:
        return
    big = dict(m=1, n=4096, d=128, dv=128)
    dense = ab.build("softmax").cost(**big)["state_bytes"]
    mine = ab.build(name).cost(**big)["state_bytes"]
    assert mine <= dense, \
        f"{name} claims to save capacity but state_bytes {mine} > dense {dense}"


# ---------------------------------------------------------------------------
# 10. no NaN / no Inf, including degenerate supports and large logits
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("scale", [1.0, 10.0])
@pytest.mark.parametrize("causal", [False, True])
def test_finite(name, scale, causal):
    meth, (q, k, v), out = run(name, causal=causal, scale=scale)
    for key, t in (("y", out.y), ("weights", out.weights), ("v_hat", out.v_hat)):
        if t is not None:
            assert torch.isfinite(t).all(), f"{name} (scale={scale}): non-finite {key}"


@pytest.mark.parametrize("name", METHODS)
def test_finite_single_key(name):
    """n = 1 and the causal first row: supports of size one must not produce nan."""
    meth = ab.build(name)
    out = meth(*qkv(m=1, n=1), return_weights=True)
    assert torch.isfinite(out.y).all(), f"{name}: non-finite y at n=1"
    if meth.supports_causal:
        q, k, v = qkv(m=N, n=N)
        out = meth(q, k, v, causal=True, return_weights=True)
        assert torch.isfinite(out.y).all(), f"{name}: non-finite y on the causal first row"


@pytest.mark.parametrize("name", METHODS)
def test_no_input_mutation(name):
    meth = ab.build(name)
    q, k, v = qkv(dtype=torch.float64)
    q0, k0, v0 = q.clone(), k.clone(), v.clone()
    meth(q, k, v, return_weights=True)
    if meth.supports_causal:
        meth(q, k, v, causal=True, return_weights=True)
    assert torch.equal(q, q0) and torch.equal(k, k0) and torch.equal(v, v0), \
        f"{name} mutated its inputs"


@pytest.mark.parametrize("name", METHODS)
def test_is_deterministic(name):
    """Randomness goes through a seeded Generator, so two builds agree bit for bit."""
    q, k, v = qkv(dtype=torch.float64)
    y1 = ab.build(name)(q, k, v).y
    y2 = ab.build(name)(q, k, v).y
    assert torch.equal(y1, y2), f"{name} is not reproducible across builds"


# ---------------------------------------------------------------------------
# 11. references exist in references.bib
# ---------------------------------------------------------------------------
def _bib_keys():
    keys = set()
    for bib in BIBS:
        if bib.exists():
            keys |= set(re.findall(r"^@\w+\{([^,]+),", bib.read_text(errors="ignore"),
                                   flags=re.M))
    assert keys, f"no bibliography found; looked in {[str(b) for b in BIBS]}"
    return keys


def test_shipped_bibliography_covers_every_citation():
    """The packaged ``attnbank/references.bib`` must stand alone, outside the paper repo."""
    shipped = BIBS[0]
    assert shipped.exists(), f"{shipped} is missing; check 11 would not survive the split"
    keys = set(re.findall(r"^@\w+\{([^,]+),", shipped.read_text(errors="ignore"),
                          flags=re.M))
    cited = {r for name in METHODS for r in ab.get_class(name).references}
    assert not cited - keys, f"not packaged: {sorted(cited - keys)}"


@pytest.mark.parametrize("name", METHODS)
def test_references_exist(name):
    cls = ab.get_class(name)
    assert cls.references, f"{name} declares no references"
    missing = set(cls.references) - _bib_keys()
    assert not missing, f"{name}: keys absent from references.bib: {sorted(missing)}"


@pytest.mark.parametrize("name", METHODS)
def test_declares_paper_section(name):
    cls = ab.get_class(name)
    assert cls.paper_section.startswith("sec:"), \
        f"{name}: paper_section = {cls.paper_section!r}"


def test_method_table_is_complete():
    rows = ab.method_table()
    assert {r["name"] for r in rows} == set(METHODS)
    for r in rows:
        assert set(r) >= {"name", "category", "objective", "causal", "expresses_weights",
                          "row_stochastic", "trainable", "paper_section", "references"}
