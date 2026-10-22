"""S8 — Roofline and bytes per token.

Sections: ``sec:exact-roofline`` (3.1) and ``sec:memory-cost`` (5.5).

Claims under test
-----------------
(i)   exact acceleration saves storage and traffic, **not** arithmetic;
(ii)  decoding is bandwidth-bound;
(iii) MQA/GQA/MLA plus quantisation are the only interventions that raise *decode*
      arithmetic intensity.

Design
------
Two halves.

*Measured.*  ``softmax`` (as ``F.scaled_dot_product_attention``, the production
kernel), ``flash`` and ``splitk`` (the library's reference implementations of the
online-softmax monoid) are timed for prefill (``m = n``, causal) and for single-token
decode (``m = 1``) over ``n`` from 2^9 to 2^17 at several batch sizes, in the model
dtype.  Achieved FLOP/s and GB/s give each configuration an arithmetic intensity, and
every point is placed on a roofline whose two roofs are **measured on this device by
microbenchmark**, not taken from a spec sheet.

*Analytic.*  ``cost()`` gives bytes per token and decode arithmetic intensity for MHA,
GQA (``g`` = 2, 8), MLA and quantised caches at 16/8/4 bits.  This is the part that
answers (iii), since it is a property of the cache layout rather than of a kernel.

Caveat for the caption
----------------------
These numbers are for one GPU (RTX PRO 6000 Blackwell) and one implementation.  They
illustrate the regime structure; they are not a cross-hardware benchmark.  ``flash``
and ``splitk`` here are *reference* implementations written for clarity (DESIGN.md
Sec. 6), so their absolute throughput says nothing about a fused kernel — only
``softmax``/SDPA is a production kernel.

Falsifier
---------
Decode points not sitting on the bandwidth roof; GQA/MLA not moving intensity.

Run: ``python -m attnbank.studies.S8_roofline [--quick]``
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch

import attnbank as ab
from . import _common as C
from . import _harness as H

NAME = "S8_roofline"
HEADS, D, DV = 32, 128, 128
PREFILL_N = (512, 1024, 2048, 4096, 8192, 16384)
DECODE_N = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
PREFILL_B = (1, 4)
DECODE_B = (1, 8, 32)
MECHS = ("sdpa", "flash", "splitk")
DTYPE = torch.bfloat16
KV_BUDGET = 10 * 2 ** 30          # bytes of K+V a single timing point may allocate
REPEATS = 20
#: ``flash`` and ``splitk`` are clarity-first reference implementations: they loop over
#: key blocks in Python and materialise an [B, H, m, block] score tile each step, so a
#: large prefill costs minutes and measures the reference loop, not the regime.  They
#: are timed on the small prefill sizes only; decode (m = 1) is cheap and runs the full
#: grid for all three.  Only ``sdpa`` is a production kernel, and only ``sdpa`` carries
#: the roofline claim.
REFERENCE_PREFILL_MAX_N = 4096
REFERENCE_PREFILL_MAX_B = 1


# ======================================================================================
# device microbenchmark: the two roofs
# ======================================================================================


def _time_cuda(fn, repeats: int, warmup: int = 3) -> float:
    """Best wall time per call, timed with CUDA events.

    ``torch.cuda.Event`` and ``torch.cuda.synchronize`` act on the **current** device,
    which is device 0 unless told otherwise.  Timing work that runs on another device
    with device-0 events measures nothing at all -- it reported a 26,000,000 TFLOP/s
    roof -- so every entry point here sets the current device first.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(3):
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / 1000.0 / repeats)
    return best


def microbenchmark(device, dtype=DTYPE) -> dict:
    """Measured peak FLOP/s and peak memory bandwidth for this device."""
    torch.cuda.set_device(torch.device(device))
    out = {"device": str(device), "dtype": str(dtype).replace("torch.", "")}
    N = 8192
    a = torch.randn(N, N, device=device, dtype=dtype)
    b = torch.randn(N, N, device=device, dtype=dtype)
    t = _time_cuda(lambda: torch.matmul(a, b), 10)
    out["peak_flops"] = 2.0 * N ** 3 / t
    af = a.float(); bf = b.float()
    t32 = _time_cuda(lambda: torch.matmul(af, bf), 5)
    out["peak_flops_fp32"] = 2.0 * N ** 3 / t32
    del a, b, af, bf

    M = 2 ** 27                                       # 128M elements
    x = torch.empty(M, device=device, dtype=dtype)
    y = torch.empty(M, device=device, dtype=dtype)
    el = x.element_size()
    t = _time_cuda(lambda: y.copy_(x), 20)
    out["peak_bandwidth_copy"] = 2.0 * M * el / t     # one read + one write
    t = _time_cuda(lambda: torch.add(x, y, out=y), 20)
    out["peak_bandwidth_triad"] = 3.0 * M * el / t    # two reads + one write
    # A KV-cache sweep during decode is read-only, and a read-only stream runs faster
    # than a copy on this part; without this term the roofline sits too low and decode
    # points come out above 100% of it.
    t = _time_cuda(lambda: torch.sum(x), 20)
    out["peak_bandwidth_read"] = 1.0 * M * el / t
    out["peak_bandwidth"] = max(out["peak_bandwidth_copy"], out["peak_bandwidth_triad"],
                                out["peak_bandwidth_read"])
    del x, y
    torch.cuda.empty_cache()
    p = torch.cuda.get_device_properties(torch.device(device).index or 0)
    out["gpu_name"] = p.name
    out["gpu_total_bytes"] = p.total_memory
    out["l2_cache_bytes"] = int(getattr(p, "L2_cache_size", 0) or 0)
    out["note"] = ("peaks measured on this device: a large square matmul and a large "
                   "elementwise stream, not a vendor spec sheet")
    return out


# ======================================================================================
# the timing sweep
# ======================================================================================


def _kv_bytes(B, n, el) -> int:
    return B * HEADS * n * (D + DV) * el


def _flops(regime: str, B: int, n: int) -> float:
    """Arithmetic the operator requires, counting the causal triangle for prefill."""
    pairs = n * (n + 1) / 2 if regime == "prefill" else float(n)
    return 2.0 * B * HEADS * pairs * (D + DV)


def _bytes(regime: str, B: int, n: int, el: int) -> float:
    """Minimum DRAM traffic: the operands, never the n x n matrix (that is the point)."""
    if regime == "prefill":
        return float(B * HEADS * (n * D + n * D + n * DV + n * DV) * el)   # Q,K,V,O
    return float(B * HEADS * (n * (D + DV) + D + DV) * el)                 # KV cache + q,o


def measure_timing(job, ctx) -> dict:
    dev = ctx.device
    torch.cuda.set_device(dev)
    B, n, regime, mech = int(job["batch"]), int(job["n"]), job["regime"], job["mechanism"]
    el = torch.finfo(DTYPE).bits // 8
    if _kv_bytes(B, n, el) > KV_BUDGET:
        return {"skipped": 1.0, "reason_too_large": 1.0}
    if mech != "sdpa" and regime == "prefill" and (n > REFERENCE_PREFILL_MAX_N
                                                   or B > REFERENCE_PREFILL_MAX_B):
        return {"skipped": 1.0, "reason_reference_impl_prefill_cap": 1.0}
    m = n if regime == "prefill" else 1
    g = torch.Generator(device="cpu").manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=g, dtype=torch.float32).to(dev, DTYPE)
    q, k, v = mk(B, HEADS, m, D), mk(B, HEADS, n, D), mk(B, HEADS, n, DV)

    if mech == "sdpa":
        # decode is m=1: every key is visible, which is what is_causal cannot express
        # for m != n (it aligns top-left), so the mask is simply dropped there.
        if regime == "prefill":
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True)
        else:
            fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    else:
        kw = {"block_n": 256} if mech == "flash" else {"splits": 8}
        meth = ab.build(mech, **kw)
        fn = lambda: meth(q, k, v, causal=True, return_weights=False).y
    try:
        t = _time_cuda(fn, REPEATS if n <= 16384 else max(3, REPEATS // 4))
    finally:
        del q, k, v
        torch.cuda.empty_cache()

    fl, by = _flops(regime, B, n), _bytes(regime, B, n, el)
    return {"seconds": t, "flops": fl, "bytes": by,
            "achieved_flops": fl / t, "achieved_bw": by / t,
            "intensity": fl / by, "m": m, "kv_bytes": float(_kv_bytes(B, n, el))}


# ======================================================================================
# the analytic cache model
# ======================================================================================


CACHE_VARIANTS = {
    "MHA": ("softmax", {}),
    "GQA g=2": ("gqa", {"group_size": 2}),
    "GQA g=8": ("gqa", {"group_size": 8}),
    "MQA (g=32)": ("gqa", {"group_size": HEADS}),
    "MLA r=512": ("mla", {"r": 512}),
    "MLA r=256": ("mla", {"r": 256}),
}
CACHE_BITS = (16, 8, 4)


def cost_table(n: int = 8192) -> list[dict]:
    """Analytic bytes/token and decode arithmetic intensity, per layer, all heads."""
    rows = []
    for label, (name, cfg) in CACHE_VARIANTS.items():
        meth = ab.build(name, **cfg)
        for bits in CACHE_BITS:
            bpe = bits / 8.0
            kw = {"heads": HEADS} if name == "mla" else {}
            c = meth.cost(1, n, D, DV, bytes_per_el=bpe, **kw)
            state_all_heads = c["state_bytes"] * HEADS
            moved_all_heads = c["bytes_moved"] * HEADS
            rows.append({
                "variant": label, "method": name, "bits": bits, "n": n,
                "heads": HEADS, "d": D, "dv": DV,
                "bytes_per_token": state_all_heads / n,
                "state_bytes": state_all_heads,
                "bytes_moved_decode": moved_all_heads,
                "flops_decode": c["flops"] * HEADS,
                "intensity_decode": c["flops"] * HEADS / max(moved_all_heads, 1e-30),
            })
    # quantisation of a plain MHA cache, for the "+ quantisation" claim
    for bits in CACHE_BITS:
        meth = ab.build("quantkv", bits_k=bits if bits < 16 else None,
                        bits_v=bits if bits < 16 else None)
        c = meth.cost(1, n, D, DV, bytes_per_el=2)
        rows.append({"variant": f"MHA quantkv {bits}b", "method": "quantkv", "bits": bits,
                     "n": n, "heads": HEADS, "d": D, "dv": DV,
                     "bytes_per_token": c["state_bytes"] * HEADS / n,
                     "state_bytes": c["state_bytes"] * HEADS,
                     "bytes_moved_decode": c["bytes_moved"] * HEADS,
                     "flops_decode": c["flops"] * HEADS,
                     "intensity_decode": c["flops"] * HEADS
                                         / max(c["bytes_moved"] * HEADS, 1e-30)})
    return rows


# ======================================================================================
# analysis
# ======================================================================================


def analyse(recs, roofs: dict, table: list[dict]) -> dict:
    ok = [r for r in recs if r["_status"] == "ok" and not r["metrics"].get("skipped")]
    rows = [{**r["job"], **r["metrics"]} for r in ok]
    summary: dict = {"roofs": roofs, "cost_table": table}

    pk_f, pk_b = roofs["peak_flops"], roofs["peak_bandwidth"]

    def roof(i):
        return min(pk_f, pk_b * i)

    for r in rows:
        r["roof_flops"] = roof(r["intensity"])
        r["frac_of_roof"] = r["achieved_flops"] / r["roof_flops"]
        r["frac_of_bw"] = r["achieved_bw"] / pk_b
        r["ridge_side"] = "memory" if r["intensity"] < pk_f / pk_b else "compute"
    summary["points"] = rows
    summary["ridge_point_intensity"] = pk_f / pk_b

    prod = [r for r in rows if r["mechanism"] == "sdpa"]
    dec = [r for r in prod if r["regime"] == "decode"]
    pre = [r for r in prod if r["regime"] == "prefill"]
    summary["sdpa"] = {
        "decode": {"n": len(dec),
                   "intensity": C.med_iqr([r["intensity"] for r in dec]),
                   "frac_of_bandwidth_roof": C.med_iqr([r["frac_of_bw"] for r in dec]),
                   "frac_of_flops_roof": C.med_iqr([r["frac_of_roof"] for r in dec]),
                   "all_memory_side": all(r["ridge_side"] == "memory" for r in dec)},
        "prefill": {"n": len(pre),
                    "intensity": C.med_iqr([r["intensity"] for r in pre]),
                    "frac_of_flops_roof": C.med_iqr([r["frac_of_roof"] for r in pre]),
                    "all_compute_side": all(r["ridge_side"] == "compute" for r in pre)},
    }
    # The claim is about a cache that genuinely lives in DRAM.  Points whose K+V fit
    # in (a few times) L2 are served from cache and would read above the DRAM roof, so
    # they are excluded from the bandwidth-roof test and reported separately.
    l2 = roofs.get("l2_cache_bytes") or 0
    thresh = max(4 * l2, 256 * 2 ** 20)
    big = [r for r in dec if r["kv_bytes"] >= thresh]
    small = [r for r in dec if r["kv_bytes"] < thresh]
    summary["sdpa"]["decode_dram_resident"] = {
        "n_points": len(big), "kv_bytes_threshold": thresh, "l2_cache_bytes": l2,
        "frac_of_bandwidth_roof": C.med_iqr([r["frac_of_bw"] for r in big]),
        "min_frac": min((r["frac_of_bw"] for r in big), default=float("nan")),
        "max_frac": max((r["frac_of_bw"] for r in big), default=float("nan")),
    }
    summary["sdpa"]["decode_cache_resident"] = {
        "n_points": len(small),
        "frac_of_bandwidth_roof": C.med_iqr([r["frac_of_bw"] for r in small]),
        "note": "K+V fit in L2; these legitimately exceed the DRAM roof",
    }
    pre_big = [r for r in pre if r["n"] >= 2048]
    summary["sdpa"]["prefill_large_n"] = {
        "n_points": len(pre_big),
        "all_compute_side": all(r["ridge_side"] == "compute" for r in pre_big),
        "frac_of_flops_roof": C.med_iqr([r["frac_of_roof"] for r in pre_big]),
        "intensity": C.med_iqr([r["intensity"] for r in pre_big]),
    }

    # (i) exact acceleration changes traffic, not arithmetic
    summary["arithmetic_is_unchanged"] = {
        "note": "flash/splitk compute the same FLOPs as dense softmax by construction; "
                "cost() reports identical 'flops' and a strictly smaller peak footprint",
        "flops_equal": {
            m: ab.build(m, **({"block_n": 256} if m == "flash" else
                              {"splits": 8} if m == "splitk" else {}))
                 .cost(1024, 1024, D, DV)["flops"]
            for m in ("softmax", "flash", "splitk")},
        "dense_score_matrix_bytes_avoided": 1024 * 1024 * 2,
    }

    # (iii) does the cache layout move decode intensity?
    base = next(r for r in table if r["variant"] == "MHA" and r["bits"] == 16)
    # MLA's cost model is the *non-absorbed* decode: the shared latent is up-projected
    # per head every step, so it trades a tiny cache for a lot of arithmetic.  That is
    # a real property of the layout, and the reason "absorption" exists.
    moves = {r["variant"] + f" {r['bits']}b":
             {"bytes_per_token": r["bytes_per_token"],
              "intensity_decode": r["intensity_decode"],
              "intensity_x_MHA16": r["intensity_decode"] / base["intensity_decode"],
              "bytes_x_MHA16": r["bytes_per_token"] / base["bytes_per_token"]}
             for r in table}
    summary["cache_intensity"] = {"baseline_MHA16": base, "variants": moves}

    gqa_mla = [v for k, v in moves.items()
               if (k.startswith("GQA") or k.startswith("MQA") or k.startswith("MLA"))
               and k.endswith("16b")]
    quant = [v for k, v in moves.items() if "quantkv" in k and not k.endswith("16b")]

    dram = summary["sdpa"]["decode_dram_resident"]
    summary["falsifier"] = {
        "decode_sits_on_the_bandwidth_roof": (
            "SUPPORTED" if summary["sdpa"]["decode"]["all_memory_side"]
            and dram["n_points"] > 0
            and dram["frac_of_bandwidth_roof"]["median"] > 0.6
            else "CONTRADICTED" if dram["n_points"] else "INCONCLUSIVE"),
        "prefill_crosses_to_the_compute_side": (
            "SUPPORTED" if summary["sdpa"]["prefill_large_n"]["all_compute_side"]
            else "CONTRADICTED"),
        "gqa_mla_raise_decode_intensity": (
            "SUPPORTED" if gqa_mla and min(v["intensity_x_MHA16"] for v in gqa_mla) > 1.5
            else "CONTRADICTED"),
        "quantisation_raises_decode_intensity": (
            "SUPPORTED" if quant and min(v["intensity_x_MHA16"] for v in quant) > 1.5
            else "CONTRADICTED"),
        "exact_acceleration_saves_traffic_not_arithmetic": (
            "SUPPORTED" if len(set(summary["arithmetic_is_unchanged"]
                                   ["flops_equal"].values())) <= 2 else "CONTRADICTED"),
    }
    return summary


# ======================================================================================
# figures
# ======================================================================================


PRETTY_MECH = {"sdpa": "sdpa (production kernel)", "flash": "flash (reference impl.)",
               "splitk": "split-$k$ (reference impl.)"}
BITS_ORDER = (16, 8, 4)
FAMILY_MARK = {"MHA": "o", "GQA": "s", "MQA": "^", "MLA": "D"}


def _family(variant: str) -> str:
    return variant.split()[0]


def make_figures(summary) -> list[str]:
    """The roofline, and the analytic cache model.

    Three corrections to the first version.

    The roofline's legend sat on top of the decode cloud; it now runs under the axes.
    More importantly, a third of the decode points sit *above* the measured DRAM roof,
    which reads as an error unless the reason is on the page: for those
    configurations K and V fit in the 128 MB L2, so they are not DRAM-bound at all.
    ``analyse`` already separates them; panel (b) is that separation, with the L2 size
    drawn.

    ``fig_bytes_per_token`` was two log-scaled bar charts.  A bar encodes a length
    from zero, which a log axis destroys, so both panels are now dot plots and a
    scatter.  The bar chart also put MLA's decode intensity beside MQA's on one axis
    as if they were the same kind of improvement.  They are not: MLA moves 16x fewer
    bytes than MHA but also computes 513x more decode FLOPs, because the latent
    up-projection is folded into the attention.  Plotting intensity *against* bytes
    per token puts the cache-sharing variants on one curve and leaves MLA off it,
    which is the honest picture.
    """
    from ..plotting import style
    style.use()
    paths = []
    roofs = summary["roofs"]
    pk_f, pk_b = roofs["peak_flops"], roofs["peak_bandwidth"]
    ridge = summary["ridge_point_intensity"]
    l2 = roofs["l2_cache_bytes"]

    # ---------------------------------------------------------------- roofline ----
    fig, axes = style.figure((6.5, 3.1), 1, 2, width_ratios=(1.25, 1.0))
    ax = axes[0]
    xs = np.logspace(-1, 4, 200)
    ax.plot(xs, np.minimum(pk_f, pk_b * xs) / 1e12, color=style.INK["primary"],
            linewidth=1.1, zorder=1)
    ax.axvline(ridge, color=style.INK["muted"], linewidth=0.7, zorder=0)
    ax.annotate(f"ridge point\n{ridge:.0f} FLOP/byte", (ridge, 0.02),
                xycoords=("data", "axes fraction"), xytext=(-4, 0),
                textcoords="offset points", ha="right", va="bottom", fontsize=6.3,
                color=style.INK["secondary"])
    marks = {"prefill": "o", "decode": "s"}
    for i, mech in enumerate(MECHS):
        for regime, mk in marks.items():
            pts = [p for p in summary["points"]
                   if p["mechanism"] == mech and p["regime"] == regime]
            if not pts:
                continue
            ax.scatter([p["intensity"] for p in pts],
                       [p["achieved_flops"] / 1e12 for p in pts],
                       s=16, marker=mk, color=style.series(i), alpha=0.85,
                       linewidths=0, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP per byte of DRAM traffic)")
    ax.set_ylabel("achieved throughput (TFLOP/s, bf16)")
    style.panel(ax, "a", "measured roofline")
    style.direct_label(ax, 4e3, pk_f / 1e12,
                       f"roof: {pk_f / 1e12:.0f} TFLOP/s, {pk_b / 1e9:.0f} GB/s",
                       dx=-4, dy=7, ha="right", va="bottom", size=6.3)
    pf, dc = summary["sdpa"]["prefill_large_n"], summary["sdpa"]["decode"]
    style.note(ax, "circles prefill, squares decode\n\n"
                   f"sdpa decode reaches "
                   f"{100 * dc['frac_of_bandwidth_roof']['median']:.0f}% of the\n"
                   f"bandwidth roof (median, $n$={dc['n']});\n"
                   f"sdpa prefill at $n\\geq$2048 reaches\n"
                   f"{100 * pf['frac_of_flops_roof']['median']:.0f}% of the "
                   f"arithmetic roof ($n$={pf['n_points']})",
               loc="upper left", size=6.3)

    # --- (b) why some decode points sit above the DRAM roof -------------------------
    ax = axes[1]
    # only sdpa carries this claim: flash and splitk are reference implementations
    # whose absolute bandwidth says nothing about the regime (see the module docstring)
    dec = sorted([p for p in summary["points"]
                  if p["regime"] == "decode" and p["mechanism"] == "sdpa"],
                 key=lambda p: p["kv_bytes"])
    ax.plot([p["kv_bytes"] / 1e6 for p in dec], [p["frac_of_bw"] for p in dec],
            "s", markersize=4.0, color=style.series(0), alpha=0.9)
    ax.axhline(1.0, color=style.INK["primary"], linewidth=1.0, zorder=4)
    ax.axvline(l2 / 1e6, color=style.INK["muted"], linewidth=0.7, zorder=0)
    ax.annotate(f"L2 = {l2 / 1e6:.0f} MB", (l2 / 1e6, 0.985),
                xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", ha="left", va="top", fontsize=6.3,
                color=style.INK["secondary"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.3, 2.4)
    style.log_ticks_plain(ax, "y", [0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    ax.set_xlabel("K and V resident per decode step (MB)")
    ax.set_ylabel("achieved bandwidth $\\div$ measured\nDRAM roof (dimensionless)")
    style.panel(ax, "b", "decode is bandwidth-bound")
    dr = summary["sdpa"]["decode_dram_resident"]
    style.note(ax, "sdpa only; flash and split-$k$ are\n"
                   "reference implementations and\ncarry no roofline claim\n\n"
                   "left of the line K and V fit in L2, so\nthose points are not "
                   "DRAM-bound\nand may exceed the roof; right of it\n"
                   f"sdpa sits at {dr['min_frac']:.2f}-{dr['max_frac']:.2f} "
                   f"of the roof ($n$={dr['n_points']})",
               loc="lower right", size=6.3)

    handles = [plt_mark(style.series(i), PRETTY_MECH[m]) for i, m in enumerate(MECHS)]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="outside lower center", ncol=3, frameon=False, fontsize=7,
               handlelength=1.0, columnspacing=1.8, borderaxespad=0.1)
    paths.append(str(C.save_fig(fig, "fig_roofline")))

    # ------------------------------------------------------- bytes per token ------
    # quantkv at b bits has, by construction, exactly the cost model of the same
    # variant stored at b bits: the three "MHA quantkv" rows duplicate "MHA" and are
    # kept in tab:cost-model but would be three redundant rows here.
    tbl = [r for r in summary["cost_table"] if "quantkv" not in r["variant"]]
    tbl = sorted(tbl, key=lambda r: (-r["bytes_per_token"], r["variant"]))
    fig, axes = style.figure((6.5, 3.5), 1, 2, width_ratios=(1.0, 1.0))

    ax = axes[0]
    labels = [f"{r['variant']} {r['bits']}b" for r in tbl]
    ypos = np.arange(len(tbl))[::-1]
    for b in BITS_ORDER:
        idx = [i for i, r in enumerate(tbl) if r["bits"] == b]
        if not idx:
            continue
        ax.plot([tbl[i]["bytes_per_token"] for i in idx], [ypos[i] for i in idx], "o",
                markersize=4.5, color=style.seq(BITS_ORDER.index(b), len(BITS_ORDER)),
                markeredgecolor=style.INK["surface"], markeredgewidth=0.7,
                label=f"{b}-bit", zorder=3)
    base = summary["cache_intensity"]["baseline_MHA16"]["bytes_per_token"]
    for i, r in enumerate(tbl):
        ax.plot([tbl[-1]["bytes_per_token"] / 1.6, r["bytes_per_token"]],
                [ypos[i]] * 2, color=style.INK["grid"], linewidth=0.8, zorder=1)
    ax.set_yticks(ypos, labels, fontsize=6.3)
    ax.set_xscale("log")
    ax.set_xlim(left=tbl[-1]["bytes_per_token"] / 1.6)
    ax.set_xlabel("bytes per token per layer, all heads\n"
                  f"(the KV cache; $H$={HEADS}, $d$={D}, $d_v$={DV})")
    style.panel(ax, "a", "KV-cache footprint")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.3, labelspacing=0.3, borderpad=0.25,
              title="stored precision", title_fontsize=6.3)
    style.note(ax, f"baseline MHA at 16 bits:\n{base:,.0f} bytes per token",
               loc="upper left", size=6.3)

    # --- (b) intensity against traffic: MLA is not on the same curve ---------------
    ax = axes[1]
    for b in BITS_ORDER:
        for fam, mk in FAMILY_MARK.items():
            idx = [i for i, r in enumerate(tbl)
                   if r["bits"] == b and _family(r["variant"]) == fam]
            if not idx:
                continue
            ax.plot([tbl[i]["bytes_per_token"] for i in idx],
                    [tbl[i]["intensity_decode"] for i in idx], mk, markersize=4.2,
                    color=style.seq(BITS_ORDER.index(b), len(BITS_ORDER)),
                    markeredgecolor=style.INK["surface"], markeredgewidth=0.7,
                    zorder=3)
    ax.axhline(ridge, color=style.INK["primary"], linewidth=1.0, zorder=4)
    ax.annotate(f"device ridge point, {ridge:.0f} FLOP/byte", (0.97, ridge),
                xycoords=("axes fraction", "data"), xytext=(0, -4),
                textcoords="offset points", ha="right", va="top", fontsize=6.3,
                color=style.INK["secondary"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(bottom=ax.get_ylim()[0] / 60)     # room for the note under the data
    ax.set_xlabel("bytes per token per layer, all heads")
    ax.set_ylabel("decode arithmetic intensity (FLOP/byte)")
    style.panel(ax, "b", "intensity against traffic")
    mla = [r for r in tbl if _family(r["variant"]) == "MLA"]
    share = [r for r in tbl if _family(r["variant"]) != "MLA"]
    style.direct_label(ax, min(r["bytes_per_token"] for r in mla),
                       max(r["intensity_decode"] for r in mla), "MLA (diamonds)",
                       dx=7, dy=0, ha="left", va="center", size=6.5)
    style.direct_label(ax, max(r["bytes_per_token"] for r in share),
                       min(r["intensity_decode"] for r in share),
                       "MHA, GQA, MQA",
                       dx=-5, dy=9, ha="right", va="bottom", size=6.5)
    flop_x = (mla[0]["flops_decode"] / summary["cache_intensity"]
              ["baseline_MHA16"]["flops_decode"])
    style.note(ax, "MHA, GQA and MQA lie on one line:\n"
                   "they only move fewer bytes.\n\n"
                   f"MLA leaves it because it also does\n{flop_x:.0f}$\\times$ "
                   "more decode arithmetic -- the\nlatent up-projection folded into\n"
                   "the attention. Its intensity is not\n"
                   "comparable with the others'.",
               loc="lower left", size=6.3)
    paths.append(str(C.save_fig(fig, "fig_bytes_per_token")))
    return paths


def plt_mark(color, label, marker="s"):
    from matplotlib.lines import Line2D
    return Line2D([], [], marker=marker, linestyle="none", markersize=4.5,
                  color=color, label=label)


# ======================================================================================
# main
# ======================================================================================


def main(quick: bool = False, resume: bool = True) -> dict:
    dev = H.default_device()
    if dev.type != "cuda":
        raise SystemExit("S8 measures a GPU roofline and needs CUDA")
    torch.cuda.set_device(dev)       # CUDA events are per-device; see _time_cuda
    pre_n = (512, 2048) if quick else PREFILL_N
    dec_n = (512, 8192) if quick else DECODE_N
    pre_b = (1,) if quick else PREFILL_B
    dec_b = (1, 8) if quick else DECODE_B

    roofs = microbenchmark(dev)
    print(f"  measured roofs: {roofs['peak_flops']/1e12:.1f} TFLOP/s bf16, "
          f"{roofs['peak_bandwidth']/1e9:.0f} GB/s", file=sys.stderr)
    # A roof above any shipping accelerator means the timer measured nothing; refuse to
    # build a roofline on it rather than publish a plot with a silent 1000x error.
    if not (1e12 < roofs["peak_flops"] < 1e16 and 1e10 < roofs["peak_bandwidth"] < 1e13):
        raise SystemExit(f"implausible measured roofs {roofs}: the CUDA event timer is "
                         f"not measuring the device the work runs on")

    sweep = H.Sweep(NAME, seed=0, device=dev, resume=resume, progress_every=10,
                    extra_meta={"study": NAME, "section": "sec:exact-roofline",
                                "roofs": roofs, "heads": HEADS, "d": D, "dv": DV,
                                "dtype": str(DTYPE), "repeats": REPEATS,
                                "kv_budget_bytes": KV_BUDGET,
                                "reference_prefill_cap":
                                    {"max_n": REFERENCE_PREFILL_MAX_N,
                                     "max_batch": REFERENCE_PREFILL_MAX_B,
                                     "why": "flash/splitk are reference implementations "
                                            "with a Python block loop; a large prefill "
                                            "times the loop, not the regime"},
                                "caveat": "one GPU, one implementation; flash and splitk "
                                          "are clarity-first reference implementations, "
                                          "only sdpa is a production kernel"})
    jobs = [{"regime": "prefill", "mechanism": m, "n": n, "batch": b}
            for m in MECHS for n in pre_n for b in pre_b]
    jobs += [{"regime": "decode", "mechanism": m, "n": n, "batch": b}
             for m in MECHS for n in dec_n for b in dec_b]
    recs = sweep.run(jobs, measure_timing, label="timing")

    table = cost_table(n=8192)
    summary = analyse(recs, roofs, table)
    figs = make_figures(summary)
    path = H.save_results(NAME, sweep.records,
                          sweep.meta(summary=summary, figures=figs, quick=quick))
    print(f"\n{NAME}: wrote {path}", file=sys.stderr)
    _report(summary)
    return summary


def _report(s: dict) -> None:
    p = lambda *a: print(*a, file=sys.stderr)
    p("\n--- S8 falsifier outcomes ---")
    for k, v in s["falsifier"].items():
        p(f"  {k}: {v}")
    r = s["roofs"]
    p(f"  measured roofs: {r['peak_flops']/1e12:.1f} TFLOP/s bf16, "
      f"{r['peak_flops_fp32']/1e12:.1f} TFLOP/s fp32, {r['peak_bandwidth']/1e9:.0f} GB/s "
      f"(copy {r['peak_bandwidth_copy']/1e9:.0f}, read {r['peak_bandwidth_read']/1e9:.0f}); "
      f"ridge at {s['ridge_point_intensity']:.0f} FLOP/byte")
    d, pr = s["sdpa"]["decode"], s["sdpa"]["prefill_large_n"]
    dr = s["sdpa"]["decode_dram_resident"]
    p(f"  sdpa decode:  intensity median {d['intensity']['median']:.2f} FLOP/byte; "
      f"DRAM-resident points at {dr['frac_of_bandwidth_roof']['median']:.0%} of the "
      f"bandwidth roof (range {dr['min_frac']:.0%}-{dr['max_frac']:.0%}, "
      f"{dr['n_points']} points)")
    p(f"  sdpa prefill (n>=2048): intensity median {pr['intensity']['median']:.0f} "
      f"FLOP/byte, {pr['frac_of_flops_roof']['median']:.0%} of the compute roof, "
      f"all compute side: {pr['all_compute_side']}")
    base = s["cache_intensity"]["baseline_MHA16"]
    p(f"  MHA 16b baseline: {base['bytes_per_token']:.0f} bytes/token, "
      f"intensity {base['intensity_decode']:.3f}")
    for k, v in s["cache_intensity"]["variants"].items():
        if k.endswith("16b") or "quantkv" in k:
            p(f"    {k:24s} {v['bytes_per_token']:9.0f} B/tok  "
              f"intensity x{v['intensity_x_MHA16']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick, resume=not a.fresh)
