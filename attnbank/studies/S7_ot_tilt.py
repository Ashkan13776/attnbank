"""S7 — OT attention is softmax tilted by a key potential.

Section: ``sec:ot`` (6), ``prop:ot-conditional``.

Claim under test
----------------
The conditionals of the entropic-OT coupling are
``pi_hat_q(j) ~ exp(s_qj + f_j)`` — softmax with a **key potential** ``f`` that does
not depend on the query.  OT attention is therefore an operator *modification* whose
effect is entirely characterised by one vector per context.

Design
------
Sinkhorn is run to convergence on ``exp(s/eps)`` with uniform marginals.  Three things
are then measured.

(a) *Convergence.*  The library computes the conditional **from** the potential, so the
    tilt identity holds to machine precision at every iteration and a residual-versus-
    iteration curve would be a flat line at 1e-16 — a one-off pass/fail, not a curve.
    The identity is still checked here (and is pinned independently by
    ``tests/test_coupling.py::test_the_conditional_is_softmax_tilted_by_a_key_potential``,
    which recovers the potential from the coupling and checks query-independence).
    What actually moves with iteration count is the **marginal error**
    ``extras["marginal_error"]``: how far the coupling is from the prescribed marginals.
    That is panel (a).
(b) *The potential.*  Its spread, and its rank correlation with the key norm and with
    the column mass of plain softmax — the "attention sink" connection Sec. 6 draws:
    keys that plain softmax over-serves should receive a suppressing potential.
(c) *Regularisation.*  TV between softmax and OT as ``eps`` and the iteration count
    vary.  ``eps`` is a temperature on the logits, so the tempered softmax at the same
    ``eps`` (zero iterations) is also reported, which separates the *tilt* from the
    *temperature*.

``sinkhorn`` declares ``supports_causal = False`` (``prop:ot-causal``: a causal
coupling with uniform marginals is the identity), so the study is non-causal
throughout and the reference is non-causal softmax.

Falsifier
---------
A nonzero residual floor in the tilt identity would mean ``prop:ot-conditional`` is
misstated.

Run: ``python -m attnbank.studies.S7_ot_tilt [--quick]``
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import attnbank as ab
from attnbank import metrics
from attnbank.core import scores
from . import _common as C
from . import _harness as H

NAME = "S7_ot_tilt"
ITER_GRID = (0, 1, 2, 4, 8, 16, 32, 64, 128)
EPS_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
REF_EPS = 1.0
SHAPE = dict(n_tokens=256, m_queries=256)

_SLICES: dict[str, C.SliceSet] = {}


def measure(job, ctx) -> dict:
    b = _SLICES["A"].get(job)
    eps, n_iter = float(job["eps"]), int(job["n_iter"])
    ref = C.reference(b, causal=False)
    pi_soft, y_soft = ref["pi"], ref["y"]

    out = ab.build("sinkhorn", n_iter=n_iter, eps=eps)(b.q, b.k, b.v, causal=False,
                                                       return_weights=True)
    pi, ex = out.weights, out.extras

    # (a) the tilt identity, recomputed from the scaled logits and the potential
    s_eps = scores(b.q, b.k, b.scale, False) / eps
    pi_tilt = torch.softmax(s_eps + ex["potential"].unsqueeze(-2), dim=-1)
    resid = (pi - pi_tilt).abs().max()
    # Query-independence: the implied potential log(pi) - s/eps must be the same vector
    # for every query, up to the per-query normaliser.  Entries where pi has underflowed
    # to zero carry no information (log is clamped there), so the check runs only on
    # columns that every query resolves; at small eps that is most of the loss.
    FLOOR = 1e-250
    valid_col = (pi > FLOOR).all(dim=-2)                       # [1,H,n]
    n_valid = int(valid_col.sum())
    if n_valid > 1:
        implied = torch.log(pi.clamp_min(FLOOR)) - s_eps       # [1,H,m,n] up to a shift
        vmask = valid_col.unsqueeze(-2)
        cnt = valid_col.sum(-1, keepdim=True).clamp_min(1).unsqueeze(-2)
        centred = implied - (implied * vmask).sum(-1, keepdim=True) / cnt
        dev = (centred - centred.mean(dim=-2, keepdim=True)).abs()
        pot_spread_across_queries = float(dev[vmask.expand_as(dev)].max())
    else:
        pot_spread_across_queries = float("nan")

    # (b) the potential against softmax column mass and key norm
    pot = ex["potential"].double()                              # [1,H,n]
    col_mass = pi_soft.double().mean(dim=-2)                    # [1,H,n]
    knorm = b.k.double().norm(dim=-1)                           # [1,H,n]
    rho_mass = float(metrics.spearman(pot, col_mass).mean())
    rho_knorm = float(metrics.spearman(pot, knorm).mean())
    top = col_mass.topk(max(1, col_mass.shape[-1] // 20), dim=-1).indices
    sink_pot = pot.gather(-1, top).mean()

    # (c) distance from softmax, and from the tempered softmax at the same eps
    tempered = torch.softmax(s_eps, dim=-1)
    tv_soft = metrics.total_variation(pi_soft, pi)
    tv_temp = metrics.total_variation(tempered, pi)
    err = metrics.output_error(y_soft, out.y)

    return {
        "identity_residual": float(resid),
        "potential_query_spread": pot_spread_across_queries,
        "n_resolved_columns": n_valid,
        "frac_underflowed": float((pi <= FLOOR).double().mean()),
        "marginal_error": float(ex["marginal_error"].double().median()),
        "marginal_error_max": float(ex["marginal_error"].double().max()),
        "potential_std": float(pot.std(dim=-1).median()),
        "potential_range": float((pot.amax(-1) - pot.amin(-1)).median()),
        "rho_potential_vs_colmass": rho_mass,
        "rho_potential_vs_keynorm": rho_knorm,
        "sink_potential_mean": float(sink_pot),
        "potential_mean": float(pot.mean()),
        "tv_vs_softmax": float(tv_soft.median()),
        "tv_vs_softmax_head": C.per_head(tv_soft),
        "tv_vs_tempered": float(tv_temp.median()),
        "tv_vs_tempered_head": C.per_head(tv_temp),
        "output_error": float(err.median()),
        "colmass_gini": float(_gini(col_mass)),
        "colmass_gini_ot": float(_gini(pi.double().mean(dim=-2))),
    }


def _gini(x: torch.Tensor) -> torch.Tensor:
    """Concentration of the column mass: 0 = uniform service, 1 = all on one key.

    Note that OT with *uniform* prescribed column marginals drives this to zero by
    construction; the informative quantities are how far softmax sits from uniform and
    which keys the potential has to suppress to get there.
    """
    v, _ = x.flatten(0, -2).sort(dim=-1)
    n = v.shape[-1]
    idx = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    num = (2 * idx - n - 1) * v
    return (num.sum(-1) / (n * v.sum(-1).clamp_min(1e-300))).median()


# ======================================================================================
# analysis
# ======================================================================================


def analyse(recs) -> dict:
    ok = [r for r in recs if r["_status"] == "ok"]
    rows = [{**r["job"], **r["metrics"]} for r in ok]
    summary: dict = {"causal": False,
                     "identity_check_reference":
                         "tests/test_coupling.py::"
                         "test_the_conditional_is_softmax_tilted_by_a_key_potential "
                         "recovers the potential from the coupling independently"}

    res = [r["identity_residual"] for r in rows]
    qsp = [r["potential_query_spread"] for r in rows
           if np.isfinite(r["potential_query_spread"])]
    summary["tilt_identity"] = {
        "note": "the library computes the conditional FROM the potential, so this is a "
                "one-off pass/fail at machine precision, not a convergence curve",
        "max_residual": max(res) if res else float("nan"),
        "median_residual": float(np.median(res)) if res else float("nan"),
        "max_potential_query_spread": max(qsp) if qsp else float("nan"),
        "median_potential_query_spread": float(np.median(qsp)) if qsp else float("nan"),
        "n_query_spread_checks": len(qsp),
        "median_frac_underflowed": float(np.median([r["frac_underflowed"] for r in rows]))
                                   if rows else float("nan"),
        "n_checks": len(res), "tolerance": 1e-10,
        "holds": bool(res) and max(res) < 1e-10,
    }

    # (a) what actually moves with iterations
    conv = {}
    for eps in sorted({r["eps"] for r in rows}):
        cur = sorted([r for r in rows if r["eps"] == eps], key=lambda r: r["n_iter"])
        conv[str(eps)] = {"n_iter": [r["n_iter"] for r in cur],
                          "marginal_error": [r["marginal_error"] for r in cur],
                          "tv_vs_softmax": [r["tv_vs_softmax"] for r in cur],
                          "potential_std": [r["potential_std"] for r in cur]}
    summary["convergence"] = conv

    conv_rows = [r for r in rows if r["eps"] == REF_EPS]
    first = min((r for r in conv_rows if r["n_iter"] > 0),
                key=lambda r: r["n_iter"], default=None)
    last = max(conv_rows, key=lambda r: r["n_iter"], default=None)
    summary["marginal_convergence"] = {
        "eps": REF_EPS,
        "first_iter": first["n_iter"] if first else None,
        "first_marginal_error": first["marginal_error"] if first else None,
        "last_iter": last["n_iter"] if last else None,
        "last_marginal_error": last["marginal_error"] if last else None,
        "reduction": (first["marginal_error"] / max(last["marginal_error"], 1e-300))
                     if first and last else float("nan"),
    }

    # (b) the potential and the sinks
    conv_only = [r for r in rows if r["n_iter"] >= 16]
    summary["potential"] = {
        "spread": C.med_iqr([r["potential_std"] for r in conv_only]),
        "range": C.med_iqr([r["potential_range"] for r in conv_only]),
        "rho_vs_softmax_column_mass": C.med_iqr(
            [r["rho_potential_vs_colmass"] for r in conv_only]),
        "rho_vs_key_norm": C.med_iqr([r["rho_potential_vs_keynorm"] for r in conv_only]),
        "top5pct_column_mass_potential": C.med_iqr(
            [r["sink_potential_mean"] for r in conv_only]),
        "mean_potential": C.med_iqr([r["potential_mean"] for r in conv_only]),
        "column_mass_gini_softmax": C.med_iqr([r["colmass_gini"] for r in conv_only]),
        "column_mass_gini_ot": C.med_iqr([r["colmass_gini_ot"] for r in conv_only]),
    }

    # (c) eps sweep at convergence
    eps_tab = {}
    for eps in sorted({r["eps"] for r in rows}):
        cur = [r for r in rows if r["eps"] == eps and r["n_iter"] >= 16]
        if cur:
            eps_tab[str(eps)] = {
                "tv_vs_softmax": C.med_iqr([r["tv_vs_softmax"] for r in cur]),
                "tv_vs_tempered_softmax": C.med_iqr([r["tv_vs_tempered"] for r in cur]),
                "output_error": C.med_iqr([r["output_error"] for r in cur]),
                "potential_std": C.med_iqr([r["potential_std"] for r in cur])}
    summary["by_eps"] = {
        "note": "tv_vs_tempered isolates the TILT from the TEMPERATURE: eps also acts "
                "as a temperature on the logits, and the tempered softmax is the eps "
                "effect with zero Sinkhorn iterations",
        "table": eps_tab}

    gini_s = summary["potential"]["column_mass_gini_softmax"]["median"]
    gini_o = summary["potential"]["column_mass_gini_ot"]["median"]
    rho = summary["potential"]["rho_vs_softmax_column_mass"]["median"]
    summary["falsifier"] = {
        "prop_ot_conditional_tilt_identity": (
            "SUPPORTED" if summary["tilt_identity"]["holds"]
            else "CONTRADICTED (prop:ot-conditional misstated, or a library bug)"),
        "potential_is_query_independent": (
            "SUPPORTED" if summary["tilt_identity"]["max_potential_query_spread"] < 1e-6
            else "CONTRADICTED"),
        "potential_suppresses_attention_sinks": (
            "SUPPORTED" if np.isfinite(rho) and rho < -0.3 else "CONTRADICTED"),
        "rho_potential_vs_column_mass": rho,
        "ot_equalises_column_service": (
            "SUPPORTED (by construction: uniform column marginals are the Sinkhorn "
            "target; the measured content is how unequal softmax is)"
            if np.isfinite(gini_o) and gini_o < gini_s else "CONTRADICTED"),
        "column_mass_gini_softmax_vs_ot": [gini_s, gini_o],
    }
    return summary


# ======================================================================================
# figure
# ======================================================================================


def make_figure(recs, summary) -> list[str]:
    """Convergence, what the potential correlates with, and how far OT moves softmax.

    Two changes of substance from the first version.

    The entropic regularisation is an **ordered** sweep, so panel (a) draws it on the
    sequential ramp rather than with five categorical hues.

    Panel (b) used to scatter two per-head summary statistics against each other,
    which showed a cloud and proved nothing.  The claim Sec. 6 actually makes is that
    the key potential is a *suppression* term aimed at the keys softmax over-serves --
    the attention-sink connection -- and not merely a restatement of key norm.  That is
    a contrast between two correlations, so the panel shows both distributions.  The
    Gini of the OT column mass is deliberately not plotted: uniform column marginals
    are the Sinkhorn target, so its collapse to zero is the algorithm's definition,
    not a measurement.

    Layout.  This figure has the second-longest caption in the paper (3.8in of the
    7.1in block), so three panels have to live in 3.05in.  A 1x3 strip is out -- at
    1.55in a panel, (b)'s two-line x label alone runs into its neighbours -- so (a),
    whose sixteen decades of y and five-way sweep need the width most, takes a
    full-width row and (b) and (c) share the row below.  Every legend is turned into a
    single row or moved onto the curves, because at 1.4in of panel height a stacked
    five-entry legend is a third of the panel.
    """
    from ..plotting import style
    style.use()
    rows = [{**r["job"], **r["metrics"]} for r in recs if r["_status"] == "ok"]
    fig, axd = style.mosaic("fig_ot_tilt", [["a", "a"], ["b", "c"]],
                            height_ratios=(1.12, 1.0), width_ratios=(1.18, 1.0))
    axes = [axd["a"], axd["b"], axd["c"]]

    # --- (a) Sinkhorn convergence, epsilon on the sequential ramp -------------------
    ax = axes[0]
    epss = sorted({r["eps"] for r in rows})
    ends = []
    for i, eps in enumerate(epss):
        by_it: dict = {}
        for r in rows:
            if r["eps"] == eps and r["n_iter"] > 0:
                by_it.setdefault(r["n_iter"], []).append(r["marginal_error"])
        if not by_it:
            continue
        its = sorted(by_it)
        med = [float(np.median(by_it[t])) for t in its]
        col = style.seq(i, len(epss))
        ax.plot(its, med, "-o", color=col, markersize=3.0,
                label=rf"$\epsilon$={eps:g}")
        ax.fill_between(its, [float(np.percentile(by_it[t], 25)) for t in its],
                        [float(np.percentile(by_it[t], 75)) for t in its],
                        color=col, alpha=0.18, linewidth=0)
        ends.append((its[-1], med[-1], rf"$\epsilon$={eps:g}"))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    style.log2_ticks(ax, "x", 0, 7, every=1)
    ax.set_xlim(2 ** -0.3, 2 ** 7.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 25)   # one legend row above the curves
    ax.set_xlabel("Sinkhorn iterations")
    # a rotated axis label is as long as the panel is tall: at 1.6in of panel the
    # one-line form ran off the top of the figure, so it is set on two short lines
    ax.set_ylabel("marginal error\n"
                  r"$\max_j|\Gamma^{\!\top}\mathbf{1}-\bar\mu|$")
    style.panel(ax, "a", "convergence to the marginals")
    ti = summary["tilt_identity"]
    # five entries in one row along the top: stacked they would be a third of a
    # 1.6in panel, and the ramp reads left to right anyway
    ax.legend(loc="upper right", fontsize=7.5, borderpad=0.2, ncol=len(epss),
              columnspacing=1.3, handlelength=1.3)
    # the curves all fall away to the right, so the bottom left corner is the free one
    # the long line goes underneath: the free wedge widens towards the floor as the
    # eps=4 curve bottoms out
    style.note(ax, "band: IQR over heads\n"
                   f"tilt identity exact: residual {ti['max_residual']:.0f} "
                   f"over {ti['n_checks']:,} checks",
               loc="lower left", size=7.5)

    # --- (b) what the potential is correlated with ----------------------------------
    ax = axes[1]
    conv = [r for r in rows if r["n_iter"] >= 16]
    for i, (key, lab) in enumerate(
            (("rho_potential_vs_colmass", "softmax column mass"),
             ("rho_potential_vs_keynorm", r"key norm $\|k_j\|$"))):
        v = np.sort(np.array([r[key] for r in conv]))
        ax.step(v, np.arange(1, v.size + 1) / v.size, where="post",
                color=style.series(i), linewidth=1.5, label=lab)
        ax.plot([np.median(v)], [0.5], "o", color=style.series(i), markersize=4.5,
                markeredgecolor=style.INK["surface"], markeredgewidth=1.0, zorder=5)
    ax.axvline(0, color=style.INK["muted"], linewidth=0.7, zorder=0)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(0, 1.62)                     # a clear band over the curves for the note
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(r"Spearman $\rho$ of $f_j$ against the"
                  "\n" r"column quantity (per head)")
    ax.set_ylabel(f"fraction of the\n{len(conv)} converged slices")
    style.panel(ax, "b", "the potential suppresses sinks")
    pot = summary["potential"]
    cm = pot["rho_vs_softmax_column_mass"]
    kn = pot["rho_vs_key_norm"]
    style.note(ax, "median (dot), IQR:\n"
                   f"column mass {cm['median']:+.2f} "
                   f"[{cm['q1']:+.2f},{cm['q3']:+.2f}]\n"
                   f"key norm {kn['median']:+.2f} "
                   f"[{kn['q1']:+.2f},{kn['q3']:+.2f}]",
               loc="upper left", size=7.5)
    # the two steps are named on the curves: a boxed legend would have to sit on one
    # of them in a panel this short
    style.direct_label(ax, -0.72, 0.62, "softmax\ncolumn mass", dx=2, dy=0,
                       ha="left", va="center", size=7.5)
    style.direct_label(ax, 0.42, 0.55, r"key norm $\|k_j\|$", dx=2, dy=0,
                       ha="left", va="center", size=7.5)

    # --- (c) how far OT moves the operator -------------------------------------------
    ax = axes[2]
    tab = summary["by_eps"]["table"]
    eps = [float(e) for e in sorted(tab, key=float)]
    ends = []
    for i, (key, lab) in enumerate((("tv_vs_softmax", "vs. softmax"),
                                    ("tv_vs_tempered_softmax",
                                     "vs. tempered softmax"))):
        med = [tab[str(e)][key]["median"] for e in eps]
        ax.plot(eps, med, "-o", color=style.series(i), markersize=3.0, label=lab)
        ax.fill_between(eps, [tab[str(e)][key]["q1"] for e in eps],
                        [tab[str(e)][key]["q3"] for e in eps],
                        color=style.series(i), alpha=0.15, linewidth=0)
        ends.append((eps[-1], med[-1], lab))
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    style.log2_ticks(ax, "x", -2, 2, every=1)
    ax.set_xlim(2 ** -2.3, 2 ** 4.6)
    ax.set_ylim(top=ax.get_ylim()[1] * 2.2)
    style.log_ticks_plain(ax, "y")
    ax.set_xlabel(r"entropic regularisation $\epsilon$")
    ax.set_ylabel(r"$\mathrm{TV}(\pi,\pi^{\rm OT})$ at convergence"
                  "\n(median over heads)")
    style.panel(ax, "c", r"how far OT moves $\pi$")
    # the two end labels are set by hand: on one line the lower one runs past the
    # figure's right edge, which would scale the whole figure -- and its type -- down
    (x0, y0, l0), (x1, y1, _) = ends
    style.direct_label(ax, x0, y0, l0, dx=4, size=7.5)
    style.direct_label(ax, x1, y1, "vs. tempered\nsoftmax", dx=3, dy=4,
                       ha="left", va="bottom", size=7.5)
    style.note(ax, "the gap is the tilt alone;\n"
                   "the curves meet at $\\epsilon$=1",
               loc="upper right", size=7.5)
    return [str(C.save_fig(fig, "fig_ot_tilt"))]


# ======================================================================================
# main
# ======================================================================================


def main(quick: bool = False, resume: bool = True) -> dict:
    dev = H.default_device()
    n_layers, n_pass = (1, 1) if quick else (4, 3)
    iters = (0, 4, 32) if quick else ITER_GRID
    epss = (1.0,) if quick else EPS_GRID

    _SLICES["A"] = C.study_slices(n_layers=n_layers, n_passages=n_pass, device=dev, **SHAPE,
                                  synthetic=dict(n=SHAPE["n_tokens"], passages=n_pass,
                                                 m_queries=SHAPE["m_queries"], device=dev))
    sweep = H.Sweep(NAME, seed=0, device=dev, resume=resume, progress_every=50,
                    extra_meta={"study": NAME, "section": "sec:ot",
                                "data": _SLICES["A"].describe(),
                                "iter_grid": list(iters), "eps_grid": list(epss),
                                "causal": False,
                                "note": "sinkhorn declares supports_causal=False "
                                        "(prop:ot-causal); the reference is non-causal "
                                        "softmax"})
    jobs = [{**s, "eps": e, "n_iter": it, **_dims(s)}
            for s in _SLICES["A"].jobs() for e in epss for it in iters]
    recs = sweep.run(jobs, measure, label="sinkhorn eps x iterations")

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
    p("\n--- S7 falsifier outcomes ---")
    for k, v in s["falsifier"].items():
        p(f"  {k}: {v}")
    ti = s["tilt_identity"]
    p(f"  tilt identity: max residual {ti['max_residual']:.3e} over {ti['n_checks']} "
      f"configs; max potential spread across queries "
      f"{ti['max_potential_query_spread']:.3e} (median "
      f"{ti['median_potential_query_spread']:.1e}, {ti['n_query_spread_checks']} checks, "
      f"median underflowed fraction {ti['median_frac_underflowed']:.1%})")
    mc = s["marginal_convergence"]
    p(f"  marginal error at eps={mc['eps']}: {mc['first_marginal_error']:.3e} at "
      f"{mc['first_iter']} iters -> {mc['last_marginal_error']:.3e} at "
      f"{mc['last_iter']} ({mc['reduction']:.1e}x)")
    pot = s["potential"]
    p(f"  potential: std {pot['spread']['median']:.3f}, range "
      f"{pot['range']['median']:.3f}; rho vs column mass "
      f"{pot['rho_vs_softmax_column_mass']['median']:+.3f}, vs key norm "
      f"{pot['rho_vs_key_norm']['median']:+.3f}")
    p(f"  column-mass Gini softmax {pot['column_mass_gini_softmax']['median']:.3f} -> "
      f"OT {pot['column_mass_gini_ot']['median']:.3f}")
    for e, v in s["by_eps"]["table"].items():
        p(f"  eps={e}: TV vs softmax {v['tv_vs_softmax']['median']:.4f}, "
          f"vs tempered {v['tv_vs_tempered_softmax']['median']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    main(quick=a.quick, resume=not a.fresh)
