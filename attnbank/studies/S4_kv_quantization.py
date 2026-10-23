"""S4 — Keys are more sensitive than values.

Sections: ``sec:compress`` (3.4), ``prop:compress-sensitivity``, ``eq:error_decomposition``.

Claim under test
----------------
A perturbation of the **keys** passes through the softmax nonlinearity and produces a
*distributional* error governed by a ``tanh(delta/2)``-type bound; the same
perturbation of the **values** produces only an attention-weighted *value* error.  So
at equal bit width, key quantisation costs more.

Design
------
``quantkv`` over the full ``bits_k x bits_v`` grid (``None`` = unquantised, then
2, 3, 4, 8, 16), which contains the K-only row (``bits_v = None``), the V-only column
and the exact corner.  A second sweep varies the grouping: ``axis`` in
{token, channel} and ``group`` in {32, 128, whole}.  Per-channel grouping pools
statistics across tokens and is therefore **not causal**; those cells are run with a
non-causal reference and are labelled as such in the results file.

For each cell the error is split with ``metrics.error_decomposition`` into

    y - y_hat = (pi - pi_hat) v   +   pi_hat (v - v_hat)
                distributional         value

and the realised logit error ``delta_q = max_j |s_hat_j - s_j|`` is compared against
the realised ``TV(pi, pi_hat)``.  Two bounds are checked: the one as written in Sec. 4,
``TV <= tanh(delta/2)``, and the sharp shift-invariant form ``TV <= tanh(osc/4)`` with
``osc`` the oscillation of the logit error (a common shift of all logits is absorbed by
the normaliser).  Tightness, not just validity, is reported.

Falsifier
---------
Value quantisation costing as much as key quantisation at equal bits.  Or realised TV
exceeding ``tanh(delta/2)`` — which would be a *theorem* error, and is far more likely
to be a wrong ``delta`` or a quantiser that is not doing what we think, so the study
also checks the quantiser's own levels and the exact corner.

Run: ``python -m attnbank.studies.S4_kv_quantization [--quick]``
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

NAME = "S4_kv_quantization"
BITS = (None, 2, 3, 4, 8, 16)
GRID_BITS = (2, 3, 4, 8, 16)
GROUPS = (32, 128, None)
AXES = ("token", "channel")
MODES = ("both", "k_only", "v_only")
SHAPE = dict(n_tokens=1024, m_queries=256)

_SLICES: dict[str, C.SliceSet] = {}


def _cfg(mode: str, bits: int | None) -> dict:
    if mode == "both":
        return {"bits_k": bits, "bits_v": bits}
    if mode == "k_only":
        return {"bits_k": bits, "bits_v": None}
    return {"bits_k": None, "bits_v": bits}


def measure(job, ctx) -> dict:
    b = _SLICES["A"].get(job)
    causal = bool(job["causal"])
    bits_k = job["bits_k"]
    bits_v = job["bits_v"]
    cfg = dict(bits_k=bits_k, bits_v=bits_v, axis=job["axis"], group=job["group"])
    ref = C.reference(b, causal=causal)
    pi, y = ref["pi"], ref["y"]
    out = ab.build("quantkv", **cfg)(b.q, b.k, b.v, causal=causal, return_weights=True)
    pi_hat, v_hat = out.weights, out.v_hat

    dec = metrics.error_decomposition(pi, pi_hat, b.v, v_hat)
    err = metrics.output_error(y, out.y)
    rel = metrics.output_error(y, out.y, relative=True)
    tv = metrics.total_variation(pi, pi_hat)

    s, s_hat = out.extras["s"], out.extras["s_hat"]
    bnd = metrics.tanh_tv_bound(s, s_hat, b.v)
    delta = bnd["delta"]                                   # max_j |ds_j|
    tanh_delta = torch.tanh(delta / 2)                     # Sec. 4 as written
    tanh_sharp = bnd["tv_bound"]                           # tanh(osc/4), shift-invariant
    viol_d = (tv > tanh_delta * (1 + 1e-9))
    viol_s = (tv > tanh_sharp * (1 + 1e-9))
    tight_d = tv / tanh_delta.clamp_min(1e-300)
    tight_s = tv / tanh_sharp.clamp_min(1e-300)
    live = delta > 0

    yn = y.norm(dim=-1).clamp_min(1e-300)
    return {
        "out_err_head": C.per_head(err), "rel_err_head": C.per_head(rel),
        "dist_head": C.per_head(dec["distributional"]),
        "value_head": C.per_head(dec["value"]),
        "tv_head": C.per_head(tv), "delta_head": C.per_head(delta),
        "out_err_median": float(err.median()), "rel_err_median": float(rel.median()),
        "dist_median": float(dec["distributional"].median()),
        "value_median": float(dec["value"].median()),
        "decomposition_residual": float(
            (dec["total"] - err).abs().max() / yn.median()),
        "tv_median": float(tv.median()), "delta_median": float(delta.median()),
        "delta_max": float(delta.max()),
        "osc_median": float(bnd["delta_osc"].median() * 2),
        "tanh_delta_median": float(tanh_delta.median()),
        "tanh_sharp_median": float(tanh_sharp.median()),
        "tightness_delta_median": float(tight_d[live].median()) if bool(live.any()) else 0.0,
        "tightness_sharp_median": float(tight_s[live].median()) if bool(live.any()) else 0.0,
        "tightness_sharp_max": float(tight_s[live].max()) if bool(live.any()) else 0.0,
        "violations_delta": int(viol_d.sum()), "violations_sharp": int(viol_s.sum()),
        "n_rows": int(tv.numel()),
        "dk_median": float(out.extras["delta_k"].median()),
        "dv_median": float(out.extras["delta_v"].median()),
        "k_rel_err": float((out.extras["delta_k"] / b.k.norm(dim=-1)).median()),
        "v_rel_err": float((out.extras["delta_v"] / b.v.norm(dim=-1)).median()),
    }


# ======================================================================================
# analysis
# ======================================================================================


def analyse(recs) -> dict:
    ok = [r for r in recs if r["_status"] == "ok"]
    rows = C.explode(recs, ["out_err_head", "rel_err_head", "dist_head", "value_head",
                            "tv_head", "delta_head"], "head")
    summary: dict = {}

    # ---- the 5x5 grid for tab:quant-grid -------------------------------------------
    grid_rows = [r for r in rows if r["part"] == "grid"]
    summary["grid"] = {
        "config": "axis=token, group=128, causal; median relative output error over units",
        "cells": C.group_rows(grid_rows, ["source", "bits_k", "bits_v"], "rel_err_head"),
        "table": {},
    }
    for src in sorted({r["source"] for r in grid_rows}):
        tbl = {}
        for row in C.group_rows([r for r in grid_rows if r["source"] == src],
                                ["bits_k", "bits_v"], "rel_err_head"):
            tbl[f"{row['bits_k']}|{row['bits_v']}"] = {
                "median": row["median"], "q1": row["q1"], "q3": row["q3"], "n": row["n"]}
        summary["grid"]["table"][src] = tbl

    # ---- key vs value asymmetry -----------------------------------------------------
    asym = {}
    for src in sorted({r["source"] for r in grid_rows}):
        sub = [r for r in grid_rows if r["source"] == src]
        per = {}
        for bits in GRID_BITS:
            konly = [r for r in sub if r["bits_k"] == bits and r["bits_v"] is None]
            vonly = [r for r in sub if r["bits_v"] == bits and r["bits_k"] is None]
            if not konly or not vonly:
                continue
            # paired by unit so the ratio is not an artefact of pooling
            kmap = {(r["layer"], r["passage"], r["head"]): r for r in konly}
            vmap = {(r["layer"], r["passage"], r["head"]): r for r in vonly}
            keys = sorted(set(kmap) & set(vmap))
            kerr = np.array([kmap[k]["out_err_head"] for k in keys])
            verr = np.array([vmap[k]["out_err_head"] for k in keys])
            ratio = kerr / np.clip(verr, 1e-300, None)
            per[str(bits)] = {
                "n_pairs": len(keys),
                "k_only": C.med_iqr(kerr), "v_only": C.med_iqr(verr),
                "ratio_median": float(np.median(ratio)),
                "ratio_q1": float(np.percentile(ratio, 25)),
                "ratio_q3": float(np.percentile(ratio, 75)),
                "key_worse_frac": float(np.mean(kerr > verr)),
                "k_dist_median": float(np.median([kmap[k]["dist_head"] for k in keys])),
                "k_value_median": float(np.median([kmap[k]["value_head"] for k in keys])),
                "v_dist_median": float(np.median([vmap[k]["dist_head"] for k in keys])),
                "v_value_median": float(np.median([vmap[k]["value_head"] for k in keys])),
            }
        asym[src] = per
    summary["key_vs_value"] = asym

    # ---- bound validity and tightness ------------------------------------------------
    viol_d = sum(r["metrics"]["violations_delta"] for r in ok)
    viol_s = sum(r["metrics"]["violations_sharp"] for r in ok)
    nrows = sum(r["metrics"]["n_rows"] for r in ok)
    tight_d = [r["metrics"]["tightness_delta_median"] for r in ok
               if r["metrics"]["delta_median"] > 0]
    tight_s = [r["metrics"]["tightness_sharp_median"] for r in ok
               if r["metrics"]["delta_median"] > 0]
    summary["tanh_bound"] = {
        "n_query_rows": nrows,
        "violations_tanh_delta_over_2": viol_d,
        "violations_tanh_osc_over_4": viol_s,
        "tightness_tanh_delta": C.med_iqr(tight_d),
        "tightness_tanh_osc": C.med_iqr(tight_s),
        "max_tightness_sharp": max((r["metrics"]["tightness_sharp_max"] for r in ok),
                                   default=float("nan")),
        "by_bits_k": C.group_rows(
            [{"bits_k": r["job"]["bits_k"], "t": r["metrics"]["tightness_delta_median"],
              "source": r["job"]["source"]}
             for r in ok if r["metrics"]["delta_median"] > 0],
            ["source", "bits_k"], "t"),
        "note": "TV/bound; <= 1 means the bound holds. tanh(osc/4) is the sharp form.",
    }

    # ---- decomposition is exact ------------------------------------------------------
    summary["decomposition_check"] = {
        "max_residual_over_y": max((r["metrics"]["decomposition_residual"] for r in ok),
                                   default=float("nan")),
        "note": "||dist + value|| vs ||y - y_hat||, float64",
    }

    # ---- the exact corner ------------------------------------------------------------
    exact = [r for r in ok if r["job"]["bits_k"] is None and r["job"]["bits_v"] is None]
    summary["exact_corner"] = {
        "n": len(exact),
        "max_rel_err": max((r["metrics"]["rel_err_median"] for r in exact), default=0.0),
        "note": "bits_k = bits_v = None must reproduce softmax exactly",
    }

    # ---- grouping --------------------------------------------------------------------
    grp_rows = [r for r in rows if r["part"] == "grouping"]
    summary["grouping"] = C.group_rows(
        grp_rows, ["source", "mode", "bits", "axis", "group"], "out_err_head")

    # ---- falsifier -------------------------------------------------------------------
    real = [s for s in asym if s != "synthetic"]
    ratios = [asym[s][b]["ratio_median"] for s in real for b in asym[s]]
    fracs = [asym[s][b]["key_worse_frac"] for s in real for b in asym[s]]
    summary["falsifier"] = {
        "keys_cost_more_than_values": (
            "SUPPORTED" if ratios and min(ratios) > 1.0 and min(fracs) > 0.5
            else "CONTRADICTED" if ratios else "INCONCLUSIVE"),
        "tanh_bound_holds": ("SUPPORTED" if viol_d == 0 else
                             f"CONTRADICTED ({viol_d}/{nrows} rows) - check delta first"),
        "sharp_tanh_bound_holds": ("SUPPORTED" if viol_s == 0 else
                                   f"CONTRADICTED ({viol_s}/{nrows} rows)"),
        "decomposition_exact": (
            "SUPPORTED" if summary["decomposition_check"]["max_residual_over_y"] < 1e-12
            else "CONTRADICTED (library bug)"),
        "key_value_error_ratio_median": float(np.median(ratios)) if ratios else float("nan"),
    }
    return summary


# ======================================================================================
# figure
# ======================================================================================


SOURCE = {"gpt2/wikitext103": "GPT-2 small", "qwen2.5-0.5b/wikitext103": "Qwen2.5-0.5B",
          "synthetic": "Gaussian control"}
SOURCE_ORDER = ("gpt2/wikitext103", "qwen2.5-0.5b/wikitext103", "synthetic")
#: the grid cell every panel is read at; the grouping sweep varies it separately
FIG_AXIS, FIG_GROUP = "token", 128


def make_figure(recs, summary) -> list[str]:
    """Three panels: the raw cost, the claim as a ratio, and the bound's tightness.

    ``prop:compress-sensitivity`` is a statement about keys *relative to* values at the
    same bit width, so panel (b) plots exactly that ratio, paired per (head, layer,
    passage), with 1.0 drawn as the claim.  Panel (a) keeps the raw magnitudes, where
    the three curves sit within a factor of three of each other and overlapping IQR
    bands would otherwise be the only thing visible.

    Layout.  This was a 1x3 strip, which at the monograph's 4.65in block gives each
    panel 1.55in -- less than the five-entry legend in (a) or the three end labels in
    (b) need, and the type was being scaled to 5.4pt to make it fit.  (a) now spans the
    full block on its own row and (b), (c) share the row beneath it, which is as much
    width as the 2.7in caption leaves room for.  (b)'s sources are labelled on the
    curves rather than past their right-hand ends, where at half a block they cost a
    third of the panel.
    """
    from ..plotting import style
    style.use()
    fig, axd = style.mosaic("fig_kv_quantization", [["a", "a"], ["b", "c"]])
    axes = [axd["a"], axd["b"], axd["c"]]
    rows = [r for r in C.explode(recs, ["out_err_head", "dist_head", "value_head",
                                        "tv_head", "delta_head"], "head")
            if r["part"] == "grid" and r["axis"] == FIG_AXIS
            and r["group"] == FIG_GROUP]
    real = [r for r in rows if r["source"] != "synthetic"] or rows
    for r in rows:
        r["bits"] = r["bits_k"] if r["bits_k"] is not None else r["bits_v"]

    # --- (a) raw cost of quantising keys, values, or both ---------------------------
    ax = axes[0]
    series = {"keys only": lambda r: r["bits_v"] is None and r["bits_k"] is not None,
              "values only": lambda r: r["bits_k"] is None and r["bits_v"] is not None,
              "both": lambda r: r["bits_k"] == r["bits_v"] and r["bits_k"] is not None}
    ends = []
    for i, (lab, pred) in enumerate(series.items()):
        sub = [r for r in real if pred(r)]
        if not sub:
            continue
        g = sorted(C.group_rows(sub, ["bits"], "out_err_head"), key=lambda x: x["bits"])
        bs = [x["bits"] for x in g]
        med = [x["median"] for x in g]
        ax.plot(bs, med, "-o", color=style.series(i), markersize=3.2, label=lab)
        ax.fill_between(bs, [x["q1"] for x in g], [x["q3"] for x in g],
                        color=style.series(i), alpha=0.13, linewidth=0)
        if lab == "both":
            # Only "both" has two live terms: for keys-only the value term is
            # identically zero and for values-only the distributional term is, so
            # decomposing those would draw either a duplicate line or nothing at all.
            for key, ls, nm in (("dist_head", (0, (4, 2)), "distributional part"),
                                ("value_head", (0, (1, 2)), "value part")):
                gg = {x["bits"]: x["median"] for x in C.group_rows(sub, ["bits"], key)}
                # the dashed/dotted pair is keyed in the note, not the legend: two more
                # legend rows cost more of a 1.5in-tall panel than the words do
                ax.plot(bs, [gg[b] for b in bs], linestyle=ls, color=style.series(i),
                        linewidth=1.0)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    style.log_ticks_plain(ax, "x", [2, 3, 4, 8, 16])
    ax.set_xlim(2 ** 0.85, 2 ** 4.45)
    ax.set_ylim(ax.get_ylim()[0] / 3, ax.get_ylim()[1] * 2.5)
    ax.set_xlabel("bit width (bits per stored scalar)")
    ax.set_ylabel(r"output error $\|y-\widehat y\|$"
                  "\n(median, model units)")
    style.panel(ax, "a", "cost of quantising K, V, both")
    # the five curves end within half a decade of each other, so end labels would have
    # to be displaced further than the gaps they describe; a legend is the honest
    # choice here.  The curves run top-left to bottom-right, so it goes upper right,
    # in two columns -- one column of five entries is a third of the panel's height.
    ax.legend(loc="upper right", fontsize=7.5, borderpad=0.2, ncol=3,
              columnspacing=1.4, handlelength=1.4)
    dc = summary["decomposition_check"]
    style.note(ax, "real models pooled; band is IQR\n"
                   "dashed: distributional part, dotted: value part;\n"
                   f"they sum to 'both' to {dc['max_residual_over_y']:.0e} of "
                   r"$\|y\|$",
               loc="lower left", size=7.5)

    # --- (b) the claim: keys cost more than values at the same bit width ------------
    ax = axes[1]
    ends = []
    for i, src in enumerate(SOURCE_ORDER):
        kv = summary["key_vs_value"].get(src)
        if not kv:
            continue
        bs = sorted((int(b) for b in kv), key=int)
        med = [kv[str(b)]["ratio_median"] for b in bs]
        ax.plot(bs, med, "-o", color=style.series(i), markersize=3.2,
                label=SOURCE.get(src, src))
        ax.fill_between(bs, [kv[str(b)]["ratio_q1"] for b in bs],
                        [kv[str(b)]["ratio_q3"] for b in bs],
                        color=style.series(i), alpha=0.13, linewidth=0)
        ends.append((bs[-1], med[-1], SOURCE.get(src, src)))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.axhline(1.0, color=style.INK["primary"], linewidth=1.0, zorder=4)
    style.log_ticks_plain(ax, "x", [2, 3, 4, 8, 16])
    ax.set_xlim(2 ** 0.85, 2 ** 4.45)
    ax.set_ylim(0.55, 60.0)
    style.log_ticks_plain(ax, "y", [1, 2, 5, 10, 20])
    ax.set_xlabel("bit width (bits per stored scalar)")
    ax.set_ylabel("key-only $\\div$ value-only error\n"
                  "at the same bit width")
    style.panel(ax, "b", "keys are the sensitive side")
    # Half a block is 2.3in; three end labels past the right-hand edge would take a
    # third of it, so each curve is named where it runs instead.  The three are flat
    # and a factor apart, so a label sitting just above its own line is unambiguous.
    for xe, ye, lab in ends:
        below = ye < 1.2       # the control sits on the 1.0 line; its label goes under
        style.direct_label(ax, 2.05, ye, lab, dy=-4 if below else 4, ha="left",
                           va="top" if below else "bottom", size=7.5)
    worse = summary["key_vs_value"][SOURCE_ORDER[0]]["4"]
    style.note(ax, "band: IQR of the paired ratio\n"
                   f"at 4 bits keys worse on "
                   f"{100 * worse['key_worse_frac']:.0f}%\n"
                   f"of {worse['n_pairs']} GPT-2 pairs\n"
                   f"pooled median ratio "
                   f"{summary['falsifier']['key_value_error_ratio_median']:.2f}",
               loc="upper left", size=7.5)

    # --- (c) how tight is the tanh bound? -------------------------------------------
    # The old panel scattered TV against the bound on log-log axes, where every point
    # sits just under a 45-degree line and "how far under" -- the only quantity the
    # panel is for -- is unreadable.  The ratio is plotted directly instead, with 1.0
    # as the bound itself.
    ax = axes[2]
    cells = [{**r["job"], **r["metrics"]} for r in recs
             if r["_status"] == "ok" and r["job"]["part"] == "grid"
             and r["job"]["axis"] == FIG_AXIS and r["job"]["group"] == FIG_GROUP
             and r["metrics"]["delta_median"] > 0]
    for r in cells:
        r["bits"] = r["bits_k"] if r["bits_k"] is not None else r["bits_v"]
    ends = []
    for i, (key, lab) in enumerate(
            (("tightness_delta_median", r"$\tanh(\delta/2)$, Sec. 4"),
             ("tightness_sharp_median", r"$\tanh(\mathrm{osc}/4)$, sharp"))):
        g = sorted(C.group_rows(cells, ["bits"], key), key=lambda x: x["bits"])
        bs = [x["bits"] for x in g]
        med = [x["median"] for x in g]
        ax.plot(bs, med, "-o", color=style.series(i), markersize=3.2, label=lab)
        ax.fill_between(bs, [x["q1"] for x in g], [x["q3"] for x in g],
                        color=style.series(i), alpha=0.13, linewidth=0)
        ends.append((bs[-1], med[-1], lab))
    ax.set_xscale("log", base=2)
    ax.axhline(1.0, color=style.INK["primary"], linewidth=1.0, zorder=4)
    style.log_ticks_plain(ax, "x", [2, 3, 4, 8, 16])
    ax.set_xlim(2 ** 0.85, 2 ** 4.45)
    ax.set_ylim(-0.08, 1.52)
    ax.set_xlabel("key bit width (bits per scalar)")
    ax.set_ylabel(r"realised $\mathrm{TV}\div$ predicted bound"
                  "\n(median over heads)")
    style.panel(ax, "c", "tightness of the TV bound")
    t = summary["tanh_bound"]
    style.note(ax, f"{t['violations_tanh_delta_over_2']} violations in "
                   f"{t['tightness_tanh_delta']['n']:,} cells\n"
                   f"median {t['tightness_tanh_delta']['median']:.2f} for "
                   f"$\\tanh(\\delta/2)$,\n"
                   f"{t['tightness_tanh_osc']['median']:.2f} for the sharp form",
               loc="upper left", size=7.5)
    # the curves sit at 0.2 and the bound at 1.0; the empty strip is between them
    ax.legend(loc="center left", fontsize=7.5, labelspacing=0.3, borderpad=0.2)
    return [str(C.save_fig(fig, "fig_kv_quantization"))]


# ======================================================================================
# main
# ======================================================================================


def main(quick: bool = False, resume: bool = True) -> dict:
    dev = H.default_device()
    n_layers, n_pass = (1, 1) if quick else (4, 3)
    bits = (None, 2, 4) if quick else BITS
    groups = (128,) if quick else GROUPS

    _SLICES["A"] = C.study_slices(n_layers=n_layers, n_passages=n_pass, device=dev, **SHAPE,
                                  synthetic=dict(n=SHAPE["n_tokens"], passages=n_pass,
                                                 m_queries=SHAPE["m_queries"], device=dev))
    sweep = H.Sweep(NAME, seed=0, device=dev, resume=resume, progress_every=50,
                    extra_meta={"study": NAME, "section": "sec:compress",
                                "data": _SLICES["A"].describe(), "bits": list(bits),
                                "groups": [str(g) for g in groups], "axes": list(AXES),
                                "note": "axis='channel' pools across tokens and is run "
                                        "non-causally; those rows carry causal=false"})

    jobs = [{**s, "part": "grid", "bits_k": bk, "bits_v": bv, "axis": "token",
             "group": 128, "causal": True, **_dims(s)}
            for s in _SLICES["A"].jobs() for bk in bits for bv in bits]
    jobs += [{**s, "part": "grouping", "mode": mode, "bits": bb, "axis": ax,
              "group": g, "causal": ax == "token",
              **_cfg(mode, bb), **_dims(s)}
             for s in _SLICES["A"].jobs() for mode in MODES
             for bb in (b for b in bits if b is not None) for ax in AXES for g in groups]
    recs = sweep.run(jobs, measure, label="quantisation grid + grouping")

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
    p("\n--- S4 falsifier outcomes ---")
    for k, v in s["falsifier"].items():
        p(f"  {k}: {v}")
    for src, per in s["key_vs_value"].items():
        for bits, d in per.items():
            p(f"  {src} b={bits}: K-only {d['k_only']['median']:.3e} vs V-only "
              f"{d['v_only']['median']:.3e}  ratio {d['ratio_median']:.2f} "
              f"[{d['ratio_q1']:.2f},{d['ratio_q3']:.2f}], key worse in "
              f"{d['key_worse_frac']:.0%} of {d['n_pairs']} units")
    t = s["tanh_bound"]
    p(f"  tanh bound: {t['violations_tanh_delta_over_2']} violations of tanh(delta/2) and "
      f"{t['violations_tanh_osc_over_4']} of tanh(osc/4) in {t['n_query_rows']} rows; "
      f"median TV/bound {t['tightness_tanh_delta']['median']:.3g} "
      f"(sharp {t['tightness_tanh_osc']['median']:.3g})")
    p(f"  decomposition residual / ||y||: "
      f"{s['decomposition_check']['max_residual_over_y']:.2e}")
    p(f"  exact corner max relative error: {s['exact_corner']['max_rel_err']:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick, resume=not a.fresh)
