"""S5 — Factorizing through latents: the nonnegative-rank gap.

Sections: ``sec:factor`` (3.5) and ``sec:memory-lowrank`` (5.2).

Claims under test
-----------------
(i)  ``prop:factor-closure``: a stochastic factorization through ``r`` latents stays a
     probability measure.
(ii) nonnegative rank >= rank, so a *stochastic* factorization through ``r`` latents
     cannot reach the fidelity of a *signed* rank-``r`` one.

Design
------
For real heads: ``svd_oracle`` (the best signed rank-``r`` approximation of ``P``),
``nmf_oracle`` (best nonnegative rank-``r``, multiplicative updates with restarts),
``latent_stochastic``, ``clustered`` and ``nystrom``, at ``r`` from 1 to 256.
``TV(pi, pi_hat)``, the Frobenius residual and the output error are measured, and for
each method the fraction of rows whose weights leave ``[0, 1]`` or fail to sum to one.

**The gap is measured in the Frobenius norm, not in TV.**  Both oracles fit ``P`` in
the Frobenius norm, and Eckart-Young makes ``svd_oracle`` optimal over *all* rank-``r``
matrices there, so ``||P - WH||_F >= ||P - P_r||_F`` is a theorem and the excess is
exactly the price of the nonnegativity constraint.  In TV the comparison is not
apples-to-apples — neither method optimises an ``L1`` objective, and the nonnegative
fit can and does come out *ahead* in TV on real heads, which says nothing about
nonnegative rank.  TV and output error are reported as the downstream consequences.

Two measurement details matter.

* The signed/nonnegative comparison uses ``nmf_oracle``'s ``extras["raw"]`` — the
  actual ``WH`` — **not** its ``weights``, which are row-renormalized into a
  probability measure and would hide exactly the gap being measured.  ``svd_oracle``'s
  weights are already the raw signed approximation.
* Every method in this category declares ``supports_causal = False``, so the whole
  study is run non-causally against a non-causal softmax reference.  That is stated in
  the results file and belongs in the caption.

Falsifier
---------
``nmf_oracle`` matching ``svd_oracle`` at all ``r``.

Run: ``python -m attnbank.studies.S5_latent_factorization [--quick]``
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

import attnbank as ab
from attnbank import metrics
from . import _common as C
from . import _harness as H

NAME = "S5_latent_factorization"
R_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256)
METHODS = ("svd_oracle", "nmf_oracle", "latent_stochastic", "clustered", "nystrom")
SHAPE = dict(n_tokens=256, m_queries=256)      # square P so rank can reach 256
TV_TARGET = 0.05
#: at r = min(m, n) every method reconstructs P exactly and all residuals sit at
#: float64 roundoff, where ratios and the Eckart-Young comparison are 0/0.  Cells whose
#: SVD residual is below this are excluded from both.
EXACT_RECON = 1e-9

_SLICES: dict[str, C.SliceSet] = {}


def measure(job, ctx) -> dict:
    b = _SLICES["A"].get(job)
    r = int(job["r"])
    ref = C.reference(b, causal=False)
    pi, y = ref["pi"], ref["y"]
    name = job["method"]
    kw = {"r": r}
    if name == "nmf_oracle":
        kw.update(iters=int(job.get("iters", 200)), restarts=int(job.get("restarts", 3)),
                  seed=0)
    out = ab.build(name, **kw)(b.q, b.k, b.v, causal=False, return_weights=True)

    # the raw factorization, before any renormalization into a measure
    raw = out.extras.get("raw", out.weights)
    tv_raw = metrics.total_variation(pi, raw)                  # [1,H,m]
    tv_norm = metrics.total_variation(pi, out.weights)
    err = metrics.output_error(y, out.y)
    rel = metrics.output_error(y, out.y, relative=True)

    fro_res = (pi - raw).flatten(-2).norm(dim=-1)              # [1,H]
    p_fro = pi.flatten(-2).norm(dim=-1).clamp_min(1e-300)

    tol = 1e-9
    below = (raw < -tol)
    above = (raw > 1 + tol)
    rowsum = raw.sum(-1)
    leaves = (below | above).any(-1)                            # [1,H,m]
    res = {
        "tv_raw_head": C.per_head(tv_raw), "tv_norm_head": C.per_head(tv_norm),
        "out_err_head": C.per_head(err), "rel_err_head": C.per_head(rel),
        "fro_head": [float(x) for x in fro_res.double().flatten()],
        "rel_fro_head": [float(x) for x in (fro_res / p_fro).double().flatten()],
        "fro_median": float(fro_res.median()),
        "tv_raw_median": float(tv_raw.median()), "tv_norm_median": float(tv_norm.median()),
        "out_err_median": float(err.median()),
        "frac_rows_outside_01": float(leaves.double().mean()),
        "frac_heads_outside_01": float(leaves.any(-1).double().mean()),
        "min_weight": float(raw.min()), "max_weight": float(raw.max()),
        "max_rowsum_deviation": float((rowsum - 1).abs().max()),
        "negative_mass": float(raw.clamp_max(0).abs().sum(-1).median()),
        "r": r, "rank_cap": min(b.m, b.n),
    }
    if "fro_residual" in out.extras:
        res["fro_residual_median"] = float(out.extras["fro_residual"].median())
    if "singular_values" in out.extras:
        sv = out.extras["singular_values"].double()
        tot = (sv ** 2).sum(-1, keepdim=True).clamp_min(1e-300)
        kept = (sv[..., :r] ** 2).sum(-1, keepdim=True)
        res["energy_kept_median"] = float((kept / tot).median())
    return res


# ======================================================================================
# analysis
# ======================================================================================


def _reach(curve: dict, target: float) -> float:
    """Smallest r (log-interpolated) at which the median TV falls to ``target``."""
    rs = sorted(curve)
    for i, r in enumerate(rs):
        if curve[r] <= target:
            if i == 0:
                return float(r)
            r0, r1 = rs[i - 1], r
            y0, y1 = curve[r0], curve[r]
            # an exact factorization (r at full rank) gives y1 == 0, where the log
            # interpolation is undefined; the budget is then just r1
            if y1 <= 0 or y0 <= 0 or y1 == y0:
                return float(r1)
            t = (math.log(target) - math.log(y0)) / (math.log(y1) - math.log(y0))
            return float(math.exp(math.log(r0) + t * (math.log(r1) - math.log(r0))))
    return float("inf")


def analyse(recs) -> dict:
    rows = C.explode(recs, ["tv_raw_head", "tv_norm_head", "out_err_head", "rel_err_head",
                            "fro_head", "rel_fro_head"], "head")
    summary: dict = {"causal": False,
                     "note": "every factorize method declares supports_causal=False; the "
                             "reference is non-causal softmax"}
    summary["tv_by_method_r"] = C.group_rows(rows, ["source", "method", "r"], "tv_raw_head")
    summary["fro_by_method_r"] = C.group_rows(rows, ["source", "method", "r"], "fro_head")
    summary["error_by_method_r"] = C.group_rows(rows, ["source", "method", "r"],
                                                "out_err_head")

    gap, reach, tv_gap, ey = {}, {}, {}, {}
    for src in sorted({r["source"] for r in rows}):
        sub = [r for r in rows if r["source"] == src]
        fro_c, tv_c = {}, {}
        for meth in METHODS:
            mm = [r for r in sub if r["method"] == meth]
            fro_c[meth] = {int(x["r"]): x["median"]
                           for x in C.group_rows(mm, ["r"], "fro_head")}
            tv_c[meth] = {int(x["r"]): x["median"]
                          for x in C.group_rows(mm, ["r"], "tv_raw_head")}
        reach[src] = {m: _reach(c, TV_TARGET) for m, c in tv_c.items() if c}

        # Eckart-Young: svd must be the Frobenius minimiser over all rank-r matrices
        viol, skipped = [], 0
        for meth, c in fro_c.items():
            for r, val in c.items():
                sv_r = fro_c.get("svd_oracle", {}).get(r)
                if sv_r is None:
                    continue
                if sv_r <= EXACT_RECON:
                    skipped += 1          # exact reconstruction: 0 vs roundoff
                    continue
                if val < sv_r * (1 - 1e-9):
                    viol.append({"method": meth, "r": r, "fro": val, "svd_fro": sv_r})
        ey[src] = {"violations": viol, "n_violations": len(viol),
                   "cells_skipped_exact_reconstruction": skipped,
                   "threshold": EXACT_RECON}

        def _pair(curve_a, curve_b):
            sh = [r for r in sorted(set(curve_a) & set(curve_b))
                  if curve_a[r] > EXACT_RECON]
            if not sh:
                return {"by_r": {}, "median_ratio": float("nan"), "max_ratio": float("nan"),
                        "matches_everywhere": False}
            return {
                "by_r": {str(r): {"svd": curve_a[r], "nmf": curve_b[r],
                                  "difference": curve_b[r] - curve_a[r],
                                  "ratio": curve_b[r] / curve_a[r]}
                         for r in sh},
                "r_excluded_exact_reconstruction":
                    [r for r in sorted(set(curve_a) & set(curve_b))
                     if curve_a[r] <= EXACT_RECON],
                "median_ratio": float(np.median([curve_b[r] / curve_a[r] for r in sh]
                                                or [float("nan")])),
                "max_ratio": max((curve_b[r] / curve_a[r] for r in sh),
                                 default=float("nan")),
                "matches_everywhere": all(curve_b[r] <= curve_a[r] * 1.02 for r in sh),
            }

        gap[src] = _pair(fro_c.get("svd_oracle", {}), fro_c.get("nmf_oracle", {}))
        gap[src]["r_multiplier_at_tv_target"] = (
            reach[src].get("nmf_oracle", float("inf"))
            / max(reach[src].get("svd_oracle", 1.0), 1e-9))
        tv_gap[src] = _pair(tv_c.get("svd_oracle", {}), tv_c.get("nmf_oracle", {}))
    summary["nonnegative_rank_gap"] = gap            # Frobenius: the theorem-backed one
    summary["tv_gap_secondary"] = {
        "note": "TV is NOT the metric either oracle optimises; the nonnegative fit can "
                "come out ahead here without contradicting nonneg-rank >= rank",
        "by_source": tv_gap}
    summary["eckart_young_check"] = {
        "note": "svd_oracle must minimise the Frobenius residual over all rank-r "
                "matrices; any violation is a library bug, not a result",
        "by_source": ey}
    summary["r_to_reach_tv_target"] = {"target": TV_TARGET, "by_source": reach}

    # (i) closure: which methods stay probability measures?
    closure = {}
    for meth in METHODS:
        mm = [r for r in recs if r["_status"] == "ok" and r["job"]["method"] == meth]
        if not mm:
            continue
        cls = ab.registry.get_class(meth)
        closure[meth] = {
            "declared_row_stochastic": bool(cls.row_stochastic),
            "max_frac_rows_outside_01": max(r["metrics"]["frac_rows_outside_01"] for r in mm),
            "mean_frac_rows_outside_01": float(np.mean(
                [r["metrics"]["frac_rows_outside_01"] for r in mm])),
            "min_weight": min(r["metrics"]["min_weight"] for r in mm),
            "max_rowsum_deviation": max(r["metrics"]["max_rowsum_deviation"] for r in mm),
        }
    summary["factor_closure"] = {
        "note": "measured on the RAW factorization (extras['raw'] for nmf), before any "
                "renormalization; prop:factor-closure predicts the stochastic "
                "constructions never leave [0,1] and keep unit row sums",
        "by_method": closure}

    stoch = [m for m in ("latent_stochastic", "clustered") if m in closure]
    closure_ok = all(closure[m]["max_frac_rows_outside_01"] == 0.0
                     and closure[m]["max_rowsum_deviation"] < 1e-9 for m in stoch)
    real = [s for s in gap if s != "synthetic"]
    gap_ok = bool(real) and all(not gap[s]["matches_everywhere"] for s in real)
    n_ey = sum(v["n_violations"] for v in ey.values())
    summary["falsifier"] = {
        "prop_factor_closure": ("SUPPORTED" if stoch and closure_ok
                                else "CONTRADICTED" if stoch else "INCONCLUSIVE "
                                "(no stochastic factorization in this run)"),
        "nonnegative_rank_gap_exists_in_frobenius": (
            "SUPPORTED" if gap_ok else "CONTRADICTED (nmf matches svd at all r)"),
        "eckart_young_holds": (
            "SUPPORTED" if n_ey == 0
            else f"CONTRADICTED ({n_ey} violations above the exact-reconstruction "
                 f"threshold - this would be a library bug)"),
        "median_nmf_over_svd_frobenius_ratio": {s: gap[s]["median_ratio"] for s in gap},
        "median_nmf_over_svd_TV_ratio_secondary":
            {s: tv_gap[s]["median_ratio"] for s in tv_gap},
        "r_multiplier_nmf_over_svd_at_TV_target":
            {s: gap[s]["r_multiplier_at_tv_target"] for s in gap},
    }
    return summary


# ======================================================================================
# figure
# ======================================================================================


SOURCE = {"gpt2/wikitext103": "GPT-2 small", "qwen2.5-0.5b/wikitext103": "Qwen2.5-0.5B",
          "synthetic": "Gaussian control"}
SOURCE_ORDER = ("gpt2/wikitext103", "qwen2.5-0.5b/wikitext103", "synthetic")
PRETTY = {"svd_oracle": "SVD oracle", "nmf_oracle": "NMF oracle",
          "latent_stochastic": "latent stochastic", "clustered": "clustered",
          "nystrom": "Nyström"}
#: the figure's method -> slot map.  Slots (0, 3, 4, 5, 6) are the only five-subset of
#: CATEGORICAL that clears the validator all-pairs (worst dE 13.0 protan, 16.3 normal);
#: the contiguous 0-4 fails the normal floor at 12.9.  Inside that set the two oracles
#: take blue and amber, the most separated pair available (37.5 normal, 31.5 CVD),
#: because the whole figure is about the distance between those two curves.
METHOD_SLOT = {"svd_oracle": 0, "nmf_oracle": 3, "nystrom": 4,
               "clustered": 5, "latent_stochastic": 6}


def make_figure(recs, summary) -> list[str]:
    """The nonnegative-rank penalty, and the metric in which it does not appear.

    Two things were wrong with the previous version.  The context here is 256 tokens
    and ``P`` is square, so at ``r = 256`` every method reconstructs ``P`` exactly and
    every curve falls to float64 roundoff: twelve of the fourteen decades on the y
    axis were that single artefact, and the gap the figure exists to show was a
    hairline.  ``analyse`` already excludes ``r = 256`` from the gap statistics for
    exactly this reason, so the figure now excludes it too and says so.

    Second, the penalty was summarised as a median ratio over ``r``, which hides the
    shape: the ratio is 1.00 at ``r = 1`` and grows to 10.8 at ``r = 128``.  Panel (b)
    plots the ratio against ``r`` so that growth is the result.  Panel (c) reports the
    negative half honestly -- in total variation, which is *not* the objective either
    oracle minimises, the penalty is near 1 over most of the range.
    """
    from ..plotting import style
    style.use()
    rows = C.explode(recs, ["tv_raw_head", "rel_fro_head"], "head")
    gap = summary["nonnegative_rank_gap"]
    excluded = sorted({int(r) for src in gap
                       for r in gap[src].get("r_excluded_exact_reconstruction", [])})
    keep = [r for r in R_GRID if r not in excluded]
    fig, axes = style.figure("full-3", 1, 3)

    # --- (a) all five factorizations on the primary source --------------------------
    ax = axes[0]
    src0 = SOURCE_ORDER[0]
    sub = [r for r in rows if r["source"] == src0 and r["r"] in keep]
    curves, ends = {}, []
    for meth in METHODS:
        g = sorted(C.group_rows([r for r in sub if r["method"] == meth], ["r"],
                                "rel_fro_head"), key=lambda x: x["r"])
        if not g:
            continue
        rs = [x["r"] for x in g]
        med = [x["median"] for x in g]
        curves[meth] = dict(zip(rs, med))
        emph = meth in ("svd_oracle", "nmf_oracle")
        col = style.series(METHOD_SLOT[meth])
        ax.plot(rs, med, "-o" if emph else "-", color=col, markersize=3.0,
                linewidth=1.5 if emph else 1.0, alpha=1.0 if emph else 0.8)
        ends.append((rs[-1], med[-1], PRETTY[meth]))
    sv, nm = curves.get("svd_oracle", {}), curves.get("nmf_oracle", {})
    shared = sorted(set(sv) & set(nm))
    if shared:
        ax.fill_between(shared, [sv[r] for r in shared], [nm[r] for r in shared],
                        color=style.series(METHOD_SLOT["nmf_oracle"]), alpha=0.20,
                        linewidth=0,
                        label="nonnegative-rank penalty")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    style.log2_ticks(ax, "x", 0, 7, every=2)
    ax.set_xlim(2 ** -0.4, 2 ** 11.6)
    ax.set_ylim(bottom=ax.get_ylim()[0] / 22)      # room for the note under the data
    ax.set_xlabel("latent dimension $r$")
    ax.set_ylabel(r"$\|P-\widehat P\|_F\,/\,\|P\|_F$"
                  "\n(median over heads, dimensionless)")
    style.panel(ax, "a", SOURCE[src0])
    style.label_ends(ax, ends, size=7.5)
    rt = summary["r_to_reach_tv_target"]["by_source"][src0]
    style.note(ax, "shaded: signed vs. nonneg.\n"
                   f"TV = {summary['r_to_reach_tv_target']['target']:g}: SVD at "
                   f"$r$={rt['svd_oracle']:.0f},\n"
                   f"NMF never below $r$={rt['nmf_oracle']:.0f}\n"
                   f"$r$={', '.join(str(e) for e in excluded)} omitted: exact there",
               loc="lower left", size=7.5)

    # --- (b)/(c) the penalty as a ratio, in each metric ------------------------------
    for pan, (letter, block, title, ylab) in enumerate((
            ("b", gap, "penalty in Frobenius",
             r"NMF oracle $\div$ SVD oracle" "\n" r"$\|P-\widehat P\|_F$ (dimensionless)"),
            ("c", summary["tv_gap_secondary"]["by_source"], "penalty in total variation",
             r"NMF oracle $\div$ SVD oracle" "\n" r"$\mathrm{TV}(\pi,\widehat\pi)$ (dimensionless)")),
            start=1):
        ax = axes[pan]
        ends = []
        for i, src in enumerate(SOURCE_ORDER):
            by_r = block.get(src, {}).get("by_r", {})
            rs = [r for r in keep if str(r) in by_r]
            if not rs:
                continue
            ys = [by_r[str(r)]["ratio"] for r in rs]
            ax.plot(rs, ys, "-o", color=style.series(i), markersize=3.0,
                    label=SOURCE[src])
            ends.append((rs[-1], ys[-1], SOURCE[src]))
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.axhline(1.0, color=style.INK["primary"], linewidth=1.0, zorder=4)
        style.log2_ticks(ax, "x", 0, 7, every=2)
        ax.set_xlim(2 ** -0.4, 2 ** 11.6)
        style.log_ticks_plain(ax, "y")
        ax.set_xlabel("latent dimension $r$")
        ax.set_ylabel(ylab)
        style.panel(ax, letter, title)
        style.label_ends(ax, ends, size=7.5)
        meds = {src: block.get(src, {}).get("median_ratio", float("nan"))
                for src in SOURCE_ORDER}
        txt = ["median over $r$:"] + [f"  {SOURCE[s]}  {meds[s]:.2f}" for s in SOURCE_ORDER]
        if letter == "c":
            txt += ["", "TV is not either oracle's", "objective"]
        style.note(ax, "\n".join(txt), loc="upper left", size=7.5)
    return [str(C.save_fig(fig, "fig_latent_factorization"))]


# ======================================================================================
# main
# ======================================================================================


def main(quick: bool = False, resume: bool = True) -> dict:
    dev = H.default_device()
    n_layers, n_pass = (1, 1) if quick else (4, 3)
    r_grid = (4, 32) if quick else R_GRID
    meths = METHODS[:2] if quick else METHODS

    _SLICES["A"] = C.study_slices(n_layers=n_layers, n_passages=n_pass, device=dev, **SHAPE,
                                  synthetic=dict(n=SHAPE["n_tokens"], passages=n_pass,
                                                 m_queries=SHAPE["m_queries"], device=dev))
    sweep = H.Sweep(NAME, seed=0, device=dev, resume=resume, progress_every=25,
                    extra_meta={"study": NAME, "section": "sec:factor",
                                "data": _SLICES["A"].describe(), "r_grid": list(r_grid),
                                "methods": list(meths), "causal": False,
                                "tv_target": TV_TARGET,
                                "nmf_config": {"iters": 200, "restarts": 3},
                                "note": "TV is measured on the RAW factorization "
                                        "(extras['raw'] for nmf_oracle), not on the "
                                        "renormalized weights"})
    jobs = [{**s, "method": meth, "r": r, **_dims(s)}
            for s in _SLICES["A"].jobs() for meth in meths
            for r in r_grid if r <= SHAPE["n_tokens"]]
    recs = sweep.run(jobs, measure, label="factorization rank sweep")

    summary = analyse(recs)
    figs = make_figure(recs, summary)
    path = H.save_results(NAME, sweep.records,
                          sweep.meta(summary=summary, figures=figs, quick=quick))
    print(f"\n{NAME}: wrote {path}", file=sys.stderr)
    _report(summary)
    return summary


def _dims(s: dict) -> dict:
    b = _SLICES["A"].get(s)
    return {"n": b.n, "m": b.m}


def _report(s: dict) -> None:
    p = lambda *a: print(*a, file=sys.stderr)
    p("\n--- S5 falsifier outcomes ---")
    for k, v in s["falsifier"].items():
        p(f"  {k}: {v}")
    for src, g in s["nonnegative_rank_gap"].items():
        p(f"  {src}: Frobenius residual nmf/svd median {g['median_ratio']:.3f}, "
          f"max {g['max_ratio']:.3f}")
        p("     by r: " + "  ".join(f"r={r}:{d['ratio']:.2f}" for r, d in g["by_r"].items()))
    for src, g in s["tv_gap_secondary"]["by_source"].items():
        p(f"  {src}: (secondary) TV nmf/svd median {g['median_ratio']:.3f}")
    for src, d in s["r_to_reach_tv_target"]["by_source"].items():
        p(f"  {src}: r to reach TV={s['r_to_reach_tv_target']['target']}: "
          + "  ".join(f"{m}={v:.0f}" for m, v in d.items()))
    for meth, c in s["factor_closure"]["by_method"].items():
        p(f"  {meth}: declared row-stochastic={c['declared_row_stochastic']}, "
          f"rows outside [0,1] max {c['max_frac_rows_outside_01']:.1%}, "
          f"min weight {c['min_weight']:.3e}, max |rowsum-1| "
          f"{c['max_rowsum_deviation']:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick, resume=not a.fresh)
