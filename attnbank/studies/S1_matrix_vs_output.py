"""S1 — Attention-matrix error is not output error.

Section: ``sec:output_error`` (4), and the thesis of ``CONVENTIONS.md``.

Claim under test
----------------
The matrix norms used throughout the literature — ``||P - P_hat||_F``, the spectral
norm, the max entry — are poor proxies for ``||y - y_hat||``, and
``||(P - P_hat) V||`` is the right one.

Design
------
A grid of *mechanisms* at a grid of *strengths*, applied to every (layer, head,
passage) triple.  One point is one (mechanism, strength, triple).  For each point all
matrix norms (``metrics.attention_matrix_errors``) and the true output error are
measured in float64.

The pre-registered statistic is the Spearman correlation between each norm and the
output error, computed **within** a mechanism and **pooled across** mechanisms.

Two refinements are needed to read it honestly, and both are reported.

*Dynamic range.*  Pooling over strengths spans five or six decades of error, and over
that range almost any monotone quantity correlates well.  The decision the literature
actually makes is "method A has a smaller matrix error than B, therefore A is the
better approximation", which is a question about *pairs at comparable scale*.  So the
study also reports (i) the spread of output error at a matched Frobenius error and
(ii) the **inversion rate**: the fraction of point pairs the norm orders the opposite
way to the output error, split into pairs from the same mechanism and pairs from
different mechanisms.

*A tautology to avoid.*  For any method that does not touch the values
(``v_hat is None``), ``||(P-P_hat)V||_F`` **equals** the output error identically, so a
correlation of 1 is arithmetic, not evidence.  That identity is verified rather than
reported as a finding, and ``AV_fro``'s real content is measured where it can fail: on
the value-compressing mechanisms, where the omitted ``P_hat(V - V_hat)`` term is the
difference.

Two maskings are swept.  ``causal=True`` is the primary setting — it is the operator
the model actually computes — and covers every mechanism that declares
``supports_causal``.  ``causal=False`` is a second, wider sweep that admits the eleven
methods which cannot express a causal mask (``sinkhorn`` and ``nystrom`` among them,
both named in the study design), so the pooled statistic is also available over the
full taxonomy.  Both pooled correlations are reported.

Falsifier
---------
If the pooled Spearman rho for the Frobenius norm is >~ 0.9, the paper's claim is too
strong and Sec. 4's framing must soften.

Run: ``python -m attnbank.studies.S1_matrix_vs_output [--quick]``
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import attnbank as ab
from attnbank import metrics
from . import _common as C
from . import _harness as H

NAME = "S1_matrix_vs_output"
SHAPE = dict(n_tokens=512, m_queries=128)

#: mechanism -> (method name, config key, strengths).  Causal-capable set first.
CAUSAL_MECHS = {
    "topk": ("topk", "k", (4, 8, 16, 32, 64, 128, 256)),
    "window": ("window", "w", (4, 8, 16, 32, 64, 128, 256)),
    "blocktopk": ("blocktopk", "n_blocks", (1, 2, 4, 8, 16, 32)),
    "performer": ("performer", "r", (16, 32, 64, 128, 256, 512, 1024)),
    "trigrff": ("trigrff", "r", (16, 32, 64, 128, 256, 512, 1024)),
    "quantkv": ("quantkv", "bits", (2, 3, 4, 8)),
    "lsh_sample": ("lsh_sample", "n_samples", (8, 16, 32, 64, 128, 256)),
    "evict_h2o": ("evict_h2o", "budget", (8, 16, 32, 64, 128, 256)),
}
#: mechanisms that cannot express a causal mask; swept non-causally
NONCAUSAL_EXTRA = {
    "sinkhorn": ("sinkhorn", "n_iter", (1, 2, 4, 8, 16, 32)),
    "nystrom": ("nystrom", "r", (4, 8, 16, 32, 64, 128)),
    "merge_kv": ("merge_kv", "n_clusters", (8, 16, 32, 64, 128, 256)),
    "linformer": ("linformer", "r", (8, 16, 32, 64, 128, 256)),
    "lowrank_kv": ("lowrank_kv", "rank", (2, 4, 8, 16, 32, 64)),
    "svd_oracle": ("svd_oracle", "r", (1, 2, 4, 8, 16, 32, 64)),
}
#: the six mechanisms the study design names, for the figure
FIGURE_MECHS = ("topk", "performer", "quantkv", "window", "sinkhorn", "nystrom")
NORMS = ("fro", "spectral", "max_entry", "inf_to_inf", "AV_fro")
#: points where the mechanism has reached its exact limit (window wider than the
#: context, every block selected, ...) sit at float64 roundoff.  They are a tight,
#: far-separated cluster that every norm ranks correctly, so leaving them in inflates
#: the pooled correlation without saying anything about the proxy.  Dropped, and
#: counted.
EXACT_LIMIT_REL = 1e-10

_SLICES: dict[str, C.SliceSet] = {}


def _build(mech: str, strength) -> ab.AttentionMethod:
    name, key, _ = {**CAUSAL_MECHS, **NONCAUSAL_EXTRA}[mech]
    if mech == "quantkv":
        return ab.build(name, bits_k=strength, bits_v=strength, axis="token", group=128)
    if mech == "blocktopk":
        return ab.build(name, block=16, n_blocks=strength, score="bound")
    if mech == "lowrank_kv":
        return ab.build(name, rank_k=strength, rank_v=strength)
    return ab.build(name, **{key: strength})


def measure(job, ctx) -> dict:
    b = _SLICES["A"].get(job)
    causal = bool(job["causal"])
    ref = C.reference(b, causal=causal)
    pi, y = ref["pi"], ref["y"]
    meth = _build(job["mechanism"], job["strength"])
    out = meth(b.q, b.k, b.v, causal=causal, return_weights=True)
    if out.weights is None:
        return {"skipped_no_weights": 1.0}

    mat = metrics.attention_matrix_errors(pi, out.weights, b.v)   # per (batch, head)
    dy = y - out.y                                                # [1,H,m,dv]
    out_fro = dy.flatten(-2).norm(dim=-1)                         # [1,H]
    y_fro = y.flatten(-2).norm(dim=-1).clamp_min(1e-300)
    flat = lambda t: [float(x) for x in t.double().flatten()]
    res = {f"{k}_head": flat(v) for k, v in mat.items()}
    res.update({
        "out_err_head": flat(out_fro),
        "rel_err_head": flat(out_fro / y_fro),
        "tv_head": flat(metrics.total_variation(pi, out.weights).median(dim=-1).values),
        "out_err_median": float(out_fro.median()),
        "n_heads": int(out_fro.numel()),
        "row_stochastic": float(ab.registry.get_class(meth.name).row_stochastic),
        # declared, not inferred: a tiny numerical difference between ||(P-P_hat)V|| and
        # ||Y-Y_hat|| is cancellation, not value compression
        "value_compressing": float(out.v_hat is not None),
    })
    return res


# ======================================================================================
# analysis
# ======================================================================================


def _spearman(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    xs = torch.tensor(x, dtype=torch.float64)
    ys = torch.tensor(y, dtype=torch.float64)
    ok = torch.isfinite(xs) & torch.isfinite(ys)
    if int(ok.sum()) < 3:
        return float("nan")
    return float(metrics.spearman(xs[ok], ys[ok]))


def analyse(recs) -> dict:
    rows = C.explode(recs, [f"{n}_head" for n in NORMS] + ["out_err_head", "rel_err_head"],
                     "head")
    rows = [r for r in rows if r.get("skipped_no_weights") is None]
    # Keep only slices that carry the complete (mechanism, strength) grid for their
    # masking, so the pooled statistics are not weighted by a half-finished slice.
    import collections as _c
    grid = _c.defaultdict(set)
    for r in rows:
        grid[(bool(r["causal"]), r["source"], r["layer"], r["passage"])].add(
            (r["mechanism"], r["strength"]))
    full = {}
    for (causal, *sl), g in grid.items():
        full[causal] = max(full.get(causal, 0), len(g))
    keep = {k for k, g in grid.items() if len(g) == full[k[0]]}
    n_slices_dropped = len(grid) - len(keep)
    rows = [r for r in rows
            if (bool(r["causal"]), r["source"], r["layer"], r["passage"]) in keep]
    n_all = len(rows)
    rows = [r for r in rows if r.get("rel_err_head", 0.0) > EXACT_LIMIT_REL]
    summary: dict = {"n_points": len(rows), "n_points_before_filter": n_all,
                     "n_dropped_at_exact_limit": n_all - len(rows),
                     "exact_limit_threshold_rel_err": EXACT_LIMIT_REL,
                     "filter_note": "configurations that have reached the mechanism's "
                                    "exact limit sit at float64 roundoff and are "
                                    "excluded from every correlation",
                     "n_partial_slices_dropped": n_slices_dropped,
                     "slices_used": {str(c): sum(1 for k in keep if k[0] == c)
                                     for c in (True, False)}}

    corr: dict = {}
    for causal in (True, False):
        sub_all = [r for r in rows if bool(r["causal"]) is causal]
        if not sub_all:
            continue
        tag = "causal" if causal else "noncausal"
        entry: dict = {"n_points": len(sub_all),
                       "mechanisms": sorted({r["mechanism"] for r in sub_all}),
                       "pooled": {}, "within": {}, "by_source": {}}
        for norm in NORMS:
            entry["pooled"][norm] = _spearman([r[f"{norm}_head"] for r in sub_all],
                                              [r["out_err_head"] for r in sub_all])
            per = {}
            for mech in sorted({r["mechanism"] for r in sub_all}):
                mm = [r for r in sub_all if r["mechanism"] == mech]
                per[mech] = _spearman([r[f"{norm}_head"] for r in mm],
                                      [r["out_err_head"] for r in mm])
            entry["within"][norm] = per
            vals = [v for v in per.values() if np.isfinite(v)]
            entry["within"][norm + "_median"] = float(np.median(vals)) if vals else float("nan")
        for src in sorted({r["source"] for r in sub_all}):
            ss = [r for r in sub_all if r["source"] == src]
            entry["by_source"][src] = {
                n: _spearman([r[f"{n}_head"] for r in ss], [r["out_err_head"] for r in ss])
                for n in NORMS}
        corr[tag] = entry
    summary["spearman"] = corr

    primary = corr.get("causal") or next(iter(corr.values()))
    summary["gap_pooled_vs_within"] = {
        n: {"pooled": primary["pooled"][n],
            "within_median": primary["within"][n + "_median"],
            "gap": primary["within"][n + "_median"] - primary["pooled"][n]}
        for n in NORMS}

    # ---- is AV_fro an identity or a measurement? -------------------------------------
    av = {}
    for mech in sorted({r["mechanism"] for r in rows}):
        mm = [r for r in rows if r["mechanism"] == mech and r["out_err_head"] > 0]
        if not mm:
            continue
        rel = [abs(r["AV_fro_head"] - r["out_err_head"]) / r["out_err_head"] for r in mm]
        av[mech] = {"n": len(mm), "max_relative_difference": float(np.max(rel)),
                    "median_relative_difference": float(np.median(rel)),
                    "value_preserving": not bool(mm[0].get("value_compressing", 0.0))}
    summary["AV_fro_identity"] = {
        "note": "for v_hat = v the output error IS ||(P-P_hat)V||_F, so rho = 1 is "
                "arithmetic; the informative cells are the value-compressing ones",
        "by_mechanism": av,
        "value_compressing": [m for m, d in av.items() if not d["value_preserving"]],
    }

    # ---- inversion rate: how often does the norm order a pair the wrong way? ---------
    def _inversions(sub, norm, cross_only, seed=0, n_pairs=200_000):
        if len(sub) < 2:
            return {"n_pairs": 0, "inversion_rate": float("nan")}
        g = np.random.default_rng(seed)
        xv = np.array([r[f"{norm}_head"] for r in sub])
        yv = np.array([r["out_err_head"] for r in sub])
        mv = np.array([r["mechanism"] for r in sub])
        i = g.integers(0, len(sub), n_pairs)
        j = g.integers(0, len(sub), n_pairs)
        keep = (mv[i] != mv[j]) if cross_only else (mv[i] == mv[j])
        keep &= (xv[i] != xv[j]) & (yv[i] != yv[j])
        if not keep.any():
            return {"n_pairs": 0, "inversion_rate": float("nan")}
        i, j = i[keep], j[keep]
        disc = (np.sign(xv[i] - xv[j]) != np.sign(yv[i] - yv[j]))
        return {"n_pairs": int(disc.size), "inversion_rate": float(disc.mean())}

    prim_rows = [r for r in rows
                 if bool(r["causal"]) is (summary.get("_primary_causal", True))]
    prim_rows = prim_rows or rows
    summary["inversion_rate"] = {
        norm: {"cross_mechanism": _inversions(prim_rows, norm, True),
               "within_mechanism": _inversions(prim_rows, norm, False)}
        for norm in NORMS}

    # ---- de-scaled: pooled rho inside one decade of output error ---------------------
    oe_all = np.array([r["out_err_head"] for r in prim_rows if r["out_err_head"] > 0])
    bands = {}
    if oe_all.size:
        lo_all = np.log10(oe_all)
        for centre in (25, 50, 75):
            c = np.percentile(lo_all, centre)
            band = [r for r in prim_rows
                    if r["out_err_head"] > 0
                    and abs(np.log10(r["out_err_head"]) - c) <= 0.5]
            bands[f"p{centre}"] = {
                "n": len(band), "decade_centre_log10": float(c),
                **{n: _spearman([r[f"{n}_head"] for r in band],
                                [r["out_err_head"] for r in band]) for n in NORMS}}
    summary["rho_within_one_decade"] = {
        "note": "pooled rho restricted to points whose output error lies inside a single "
                "decade, which removes the dynamic-range effect that inflates the "
                "full-range pooled rho",
        "bands": bands}

    # how much the norms disagree with each other about which point is worse
    summary["norm_vs_norm"] = {
        f"{a}|{b}": _spearman([r[f"{a}_head"] for r in rows if bool(r["causal"])],
                              [r[f"{b}_head"] for r in rows if bool(r["causal"])])
        for a in ("fro", "spectral", "AV_fro") for b in ("fro", "spectral", "AV_fro")
        if a < b}

    # at a fixed Frobenius error, how wide is the spread of output errors?
    sub = [r for r in rows if bool(r["causal"]) and r["fro_head"] > 0]
    if sub:
        fro = np.array([r["fro_head"] for r in sub])
        oe = np.array([r["out_err_head"] for r in sub])
        lo, hi = np.percentile(fro, 45), np.percentile(fro, 55)
        band = oe[(fro >= lo) & (fro <= hi)]
        summary["output_spread_at_matched_frobenius"] = {
            "frobenius_band": [float(lo), float(hi)], "n": int(band.size),
            **C.med_iqr(band),
            "p05": float(np.percentile(band, 5)) if band.size else float("nan"),
            "p95": float(np.percentile(band, 95)) if band.size else float("nan"),
            "ratio_p95_p05": float(np.percentile(band, 95) / max(np.percentile(band, 5), 1e-300))
                             if band.size else float("nan"),
        }

    pooled_fro = primary["pooled"]["fro"]
    pooled_av = primary["pooled"]["AV_fro"]
    nc = corr.get("noncausal", {}).get("pooled", {})
    inv = summary["inversion_rate"]["fro"]
    dec = summary["rho_within_one_decade"]["bands"]
    dec_fro = [v["fro"] for v in dec.values() if np.isfinite(v.get("fro", float("nan")))]
    summary["falsifier"] = {
        "primary_setting": "causal" if "causal" in corr else "noncausal",
        "PRE_REGISTERED_frobenius_pooled_rho_below_0.9": (
            "SUPPORTED" if np.isfinite(pooled_fro) and pooled_fro < 0.9
            else f"CONTRADICTED (pooled rho = {pooled_fro:.3f} >= 0.9): on this evidence "
                 f"the blunt claim that Frobenius error is a poor proxy is too strong, "
                 f"and Sec. 4's framing must soften"),
        "pooled_rho_frobenius": pooled_fro,
        "pooled_rho_frobenius_noncausal": nc.get("fro"),
        "pooled_rho_spectral": primary["pooled"]["spectral"],
        "pooled_rho_max_entry": primary["pooled"]["max_entry"],
        "pooled_rho_AV_fro": pooled_av,
        "AV_fro_equals_output_error_for_value_preserving_methods":
            "IDENTITY, not a measurement: rho = 1 is arithmetic. See AV_fro_identity.",
        "within_mechanism_rho_frobenius_median": primary["within"]["fro_median"],
        "frobenius_cross_mechanism_inversion_rate": inv["cross_mechanism"]["inversion_rate"],
        "frobenius_within_mechanism_inversion_rate": inv["within_mechanism"]["inversion_rate"],
        "frobenius_rho_within_one_decade": (float(np.median(dec_fro)) if dec_fro
                                            else float("nan")),
        "refined_reading": (
            "the pooled rho is high because the sweep spans several decades of error; "
            "restricted to a single decade, and measured as a pairwise inversion rate, "
            "the Frobenius norm misorders a substantial fraction of cross-mechanism "
            "comparisons - which is the comparison the literature actually makes"),
    }
    return summary


# ======================================================================================
# figure
# ======================================================================================


#: the three mechanisms highlighted in the scatter panels.  Three, not six: the
#: palette is validated all-pairs only for slots 0-2, and six overlapping point clouds
#: at alpha 0.35 are a texture, not three readable series.  One per family, each with a
#: stable colour from ``METHOD_COLORS``: restriction, random features, compression.
HIGHLIGHT = ("topk", "performer", "quantkv")


def _panel_scatter(ax, rows, norm, summary, tag, style, *, highlight=HIGHLIGHT):
    """Output error against one matrix norm, log-log, three named mechanisms."""
    rest = [r for r in rows if r["mechanism"] not in highlight
            and r[f"{norm}_head"] > 0 and r["out_err_head"] > 0]
    ax.scatter([r[f"{norm}_head"] for r in rest], [r["out_err_head"] for r in rest],
               s=2.5, alpha=0.16, color=style.INK["muted"], linewidths=0, zorder=1,
               rasterized=True)
    for mech in highlight:
        sub = [r for r in rows if r["mechanism"] == mech
               and r[f"{norm}_head"] > 0 and r["out_err_head"] > 0]
        if not sub:
            continue
        ax.scatter([r[f"{norm}_head"] for r in sub], [r["out_err_head"] for r in sub],
                   s=3.5, alpha=0.40, color=style.method_color(mech), linewidths=0,
                   label=mech, zorder=2, rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")


def make_figure(recs, summary) -> list[str]:
    """Three panels, each making one claim, none of them an identity.

    (a) and (b) are the proxies the literature uses.  The statistic they carry is not
    the pooled Spearman rho -- pooling over six decades of error makes any monotone
    quantity look good -- but the *spread of output error at a matched matrix error*,
    which is the comparison a reader actually makes, so that is what the panel
    annotates and shades.

    (c) is the panel that has to be read carefully.  For every value-preserving
    method ``||(P-P_hat)V||_F`` IS the output error, so its rho = 1.00 is arithmetic.
    The panel therefore draws those mechanisms as the identity line they are, and
    plots as data only ``quantkv`` -- the one value-compressing mechanism in the
    primary causal sweep, where the omitted ``P_hat(V-V_hat)`` term can and does make
    the norm disagree with the output error.
    """
    from ..plotting import style
    style.use()
    rows = [r for r in C.explode(recs, [f"{n}_head" for n in NORMS]
                                 + ["out_err_head", "rel_err_head"], "head")
            if r.get("skipped_no_weights") is None
            and r.get("rel_err_head", 0.0) > EXACT_LIMIT_REL]
    tag = summary["falsifier"]["primary_setting"]
    want_causal = tag == "causal"
    rows = [r for r in rows if bool(r["causal"]) is want_causal]
    ident = summary["AV_fro_identity"]
    compressing = [m for m in ident["value_compressing"]
                   if any(r["mechanism"] == m for r in rows)]

    fig, axes = style.figure("full-3", 1, 3)

    # --- (a) Frobenius, (b) spectral ---------------------------------------------
    sp = summary["output_spread_at_matched_frobenius"]
    for ax, (norm, lab, letter, title) in zip(
            axes, [("fro", r"$\|P-\widehat P\|_F$", "a", "Frobenius norm"),
                   ("spectral", r"$\|P-\widehat P\|_2$", "b", "spectral norm")]):
        _panel_scatter(ax, rows, norm, summary, tag, style)
        if norm == "fro":
            # the headline: hold the matrix error fixed and the output error still
            # moves by two orders of magnitude
            b0, b1 = sp["frobenius_band"]
            ax.axvspan(b0, b1, color=style.INK["primary"], alpha=0.09, zorder=0,
                       linewidth=0)
            import matplotlib.patheffects as pe
            ax.annotate(f"{sp['ratio_p95_p05']:.0f}$\\times$ output-error spread\n"
                        f"in the shaded band",
                        xy=(np.sqrt(b0 * b1), 0.99), xycoords=("data", "axes fraction"),
                        xytext=(0, -1), textcoords="offset points",
                        ha="center", va="top", fontsize=7.5,
                        color=style.INK["secondary"], zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.2,
                                                    foreground=style.INK["surface"])])
        pooled = summary["spearman"][tag]["pooled"][norm]
        within = summary["spearman"][tag]["within"][norm + "_median"]
        dec = [b[norm] for b in summary["rho_within_one_decade"]["bands"].values()
               if np.isfinite(b.get(norm, float("nan")))]
        inv = summary["inversion_rate"][norm]["cross_mechanism"]["inversion_rate"]
        style.note(ax, f"pooled $\\rho$ {pooled:.2f}\n"
                       f"within mechanism {within:.2f}\n"
                       f"within one decade {np.median(dec):.2f}\n"
                       f"cross-mechanism inversions {100 * inv:.0f}%",
                   loc="lower right", size=7.5)
        ax.set_xlabel(lab)
        style.panel(ax, letter, title)
    axes[0].set_ylabel(r"output error $\|Y-\widehat Y\|_F$")
    axes[1].set_ylabel("")

    # --- (c) the value-weighted norm, with the identity marked as an identity -----
    ax = axes[2]
    pres = [r for r in rows if r["mechanism"] not in compressing
            and r["AV_fro_head"] > 0 and r["out_err_head"] > 0]
    ax.scatter([r["AV_fro_head"] for r in pres], [r["out_err_head"] for r in pres],
               s=2.5, alpha=0.18, color=style.INK["grid"], linewidths=0, zorder=1,
               rasterized=True)
    for i, mech in enumerate(compressing[:3]):
        sub = [r for r in rows if r["mechanism"] == mech
               and r["AV_fro_head"] > 0 and r["out_err_head"] > 0]
        ax.scatter([r["AV_fro_head"] for r in sub], [r["out_err_head"] for r in sub],
                   s=3.5, alpha=0.45, color=style.method_color(mech), linewidths=0,
                   label=mech, zorder=3, rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    av = [r["AV_fro_head"] for r in rows if r["AV_fro_head"] > 0]
    oe = [r["out_err_head"] for r in rows if r["AV_fro_head"] > 0]
    lim = (min(min(av), min(oe)) / 3, max(max(av), max(oe)) * 3)
    ax.set_xlim(lim); ax.set_ylim(lim)
    style.identity_line(ax, linewidth=0.8)
    style.direct_label(ax, lim[1] / 50, lim[1] / 50,
                       r"identity when $\widehat V=V$", dx=-4, dy=7, ha="right",
                       va="bottom", size=7.5)
    ax.set_xlabel(r"$\|(P-\widehat P)V\|_F$")
    style.panel(ax, "c", "value-weighted norm")
    md = ident["by_mechanism"]
    worst = max((md[m]["median_relative_difference"] for m in compressing), default=0.0)
    wmax = max((md[m]["max_relative_difference"] for m in compressing), default=0.0)
    n_pres = len({r["mechanism"] for r in rows}) - len(compressing)
    style.note(ax, f"grey: {n_pres} value-preserving\n"
                   f"mechanisms; $\\rho$ = 1.00 is\n"
                   f"arithmetic, not evidence\n\n"
                   f"colour: {', '.join(compressing)} compresses $V$\n"
                   f"departure median {100 * worst:.0f}%, max {100 * wmax:.0f}%",
               loc="lower right", size=7.5)

    handles = [plt_line(style.method_color(m), m) for m in HIGHLIGHT]
    handles += [plt_line(style.INK["muted"], "other mechanisms")]
    fig.legend(handles=handles, labels=[h.get_label() for h in handles],
               loc="outside lower center", ncol=4, frameon=False, fontsize=7.5,
               handlelength=1.0, columnspacing=1.6, borderaxespad=0.1)
    return [str(C.save_fig(fig, "fig_matrix_vs_output"))]


def plt_line(color, label):
    from matplotlib.lines import Line2D
    return Line2D([], [], marker="o", linestyle="none", markersize=4.5, color=color,
                  label=label)


# ======================================================================================
# main
# ======================================================================================


def main(quick: bool = False, resume: bool = True,
         analyse_only: bool = False) -> dict:
    dev = H.default_device()
    n_layers, n_pass = (1, 1) if quick else (4, 4)
    cm = dict(list(CAUSAL_MECHS.items())[:3]) if quick else CAUSAL_MECHS
    nm = dict(list(NONCAUSAL_EXTRA.items())[:2]) if quick else NONCAUSAL_EXTRA

    _SLICES["A"] = C.study_slices(n_layers=n_layers, n_passages=n_pass, device=dev, **SHAPE,
                                  synthetic=dict(n=SHAPE["n_tokens"], passages=n_pass,
                                                 m_queries=SHAPE["m_queries"], device=dev))
    sweep = H.Sweep(NAME, seed=0, device=dev, resume=resume, progress_every=100,
                    extra_meta={"study": NAME, "section": "sec:output_error",
                                "data": _SLICES["A"].describe(),
                                "causal_mechanisms": {k: list(v[2]) for k, v in cm.items()},
                                "noncausal_extra": {k: list(v[2]) for k, v in nm.items()},
                                "norms": list(NORMS),
                                "point": "(mechanism, strength, layer, head, passage)",
                                "note": "norms and output error are Frobenius over the "
                                        "query axis of one head, so they are comparable"})
    jobs = [{**s, "mechanism": mech, "strength": st, "causal": True, **_dims(s)}
            for s in _SLICES["A"].jobs() for mech, (_, _, sts) in cm.items() for st in sts]
    jobs += [{**s, "mechanism": mech, "strength": st, "causal": False, **_dims(s)}
             for s in _SLICES["A"].jobs()
             for mech, (_, _, sts) in {**cm, **nm}.items() for st in sts]
    if analyse_only:
        # Finalise from the resume cache. The causal pass is the primary setting and
        # completes first; the analysis keeps only slices with a complete grid, so a
        # partially finished secondary pass widens nothing and biases nothing.
        recs = list(sweep._records.values())
        print(f"  analysing {len(recs)} cached jobs without running new ones",
              file=sys.stderr)
    else:
        recs = sweep.run(jobs, measure, label="mechanism x strength")

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
    p("\n--- S1 falsifier outcomes ---")
    for k, v in s["falsifier"].items():
        p(f"  {k}: {v}" if isinstance(v, str) else f"  {k}: {v:.4f}"
          if isinstance(v, float) else f"  {k}: {v}")
    p(f"  points: {s['n_points']} (dropped {s['n_dropped_at_exact_limit']} at the "
      f"mechanism's exact limit, from {s['n_points_before_filter']}); slices used "
      f"{s['slices_used']}, {s['n_partial_slices_dropped']} partial slices dropped")
    for tag, e in s["spearman"].items():
        p(f"  [{tag}] {e['n_points']} points over {len(e['mechanisms'])} mechanisms")
        for n in NORMS:
            p(f"     {n:11s} pooled rho {e['pooled'][n]:+.3f} | within-mechanism median "
              f"{e['within'][n + '_median']:+.3f}")
    sp = s.get("output_spread_at_matched_frobenius")
    if sp:
        p(f"  at a matched Frobenius error the output error spans "
          f"{sp['ratio_p95_p05']:.0f}x (p05-p95, n={sp['n']})")
    p("  inversion rate (fraction of pairs ordered the wrong way):")
    for n in NORMS:
        v = s["inversion_rate"][n]
        p(f"     {n:11s} cross-mechanism {v['cross_mechanism']['inversion_rate']:.3f} | "
          f"within-mechanism {v['within_mechanism']['inversion_rate']:.3f}")
    p("  pooled rho inside one decade of output error:")
    for band, v in s["rho_within_one_decade"]["bands"].items():
        p(f"     {band} (n={v['n']}): " + "  ".join(f"{n}={v[n]:+.2f}" for n in NORMS))
    ident = s["AV_fro_identity"]
    p(f"  AV_fro is an identity for value-preserving methods; value-compressing "
      f"mechanisms here: {ident['value_compressing']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--analyse-only", action="store_true",
                    help="finalise from the resume cache without running new jobs")
    a = ap.parse_args()
    main(quick=a.quick, resume=not a.fresh, analyse_only=a.analyse_only)
