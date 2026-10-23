"""S9 — Does the collapse point scale with the state size?

Section: ``sec:memory-state`` (5), ``prop:memory-rank``.  Pre-registered in
``STUDIES.md`` before it was run.

Claim under test
----------------
S6 established that finite-state operators fail at recall where softmax does not, but
its collapse point is *flat* in state rank from 8 to 256: a 32x larger state bought no
extra associations, so the ``O(S)`` scaling of ``prop:memory-rank`` was nowhere
exhibited.  The round-3 review judged the cause to be S6's construction rather than the
proposition: with ``elu(k) + 1`` features on Gaussian keys the feature map, not the
state, is the binding constraint.  S9 reruns S6's construction with **two changes and
nothing else**:

* ``phi = identity`` in place of ``elu(k) + 1``;
* **orthonormal** keys in place of standard Gaussian keys.

Everything else -- the grid, the metrics, the readout, the seeds, the query count, the
``decay = 0.999`` convention, the separate decay sweep, float64, the reference backend
-- is S6's.

What the key construction is
----------------------------
``N`` unit-norm rows in ``R^{d_phi}`` that are orthonormal whenever that is possible.
For ``N <= d_phi`` exact orthonormality exists and is used; for ``N > d_phi`` it does
not exist for any construction, and the nearest thing is taken: the polar factor
``K = U V^T`` of a Gaussian ``G = U S V^T``, which is the orthogonal matrix closest to
``G`` in Frobenius norm and gives a *tight frame* (``K^T K = I_{d_phi}``), with the rows
renormalized to unit length.  The two regimes are one formula -- at ``N <= d_phi`` the
polar factor already has orthonormal rows and the renormalization is a no-op -- so
there is no discontinuity in the construction at ``N = d_phi`` that could manufacture a
collapse there.  ``key_coherence`` (the largest off-diagonal ``|<k_i, k_j>|``) is
recorded for every cell so the distance from orthonormality is in the results file.

The query gain is defined, as in S6, to hold the *matched softmax logit* at
``MATCHED_LOGIT``, so the control is never temperature-limited.  S6's keys were
Gaussian with ``||k||^2 ~ d``, giving ``gain = MATCHED_LOGIT / sqrt(d)``; these keys are
unit-norm, so the same invariant is ``gain = MATCHED_LOGIT * sqrt(d)``.  The invariant
is unchanged; only its arithmetic follows the key norm.

Measured
--------
The collapse point -- the smallest ``N`` at which median recall falls below 0.5 -- as a
function of ``d_phi``, and the load ``N/S`` at collapse.  Two estimators are reported
for the same curves: the pre-registered grid-valued one (smallest ``N`` *on the grid*),
and a log-interpolated crossing, which removes the octave-wide quantization of the grid
and is what the log--log slope is fitted on.  Both are in the results file.

A post-hoc extension
--------------------
On the pre-registered grid (``N <= 512``) the collapse point for ``d_phi`` of 128 and
256 lies off the top of the grid, so those two cells are *censored* and the slope could
be fitted over only four of the six state sizes.  ``N`` in {1024, 2048, 4096} is
therefore also run, as a separately labelled part (``part = "extended_N"``), for the
sole purpose of uncensoring those two cells.  Every pre-registered quantity is computed
on the pre-registered grid alone and reported as such; the extension is reported
beside it and is marked wherever it is plotted or tabulated.  It was added after the
pre-registered grid had already answered the falsifier, not in order to answer it.

A second post-hoc extension: the methods the expansion added
------------------------------------------------------------
The pre-registration sweeps "state size" by sweeping ``d``, because at the time every
recurrent method in the bank carried a state of shape ``d x d_v``.  The expansion broke
that identity, so the extension runs through :mod:`attnbank.studies._state`, which
reads each operator's state off its own ``cost()["state_bytes"]`` instead of assuming
the shape (see that module for why it is derived rather than tabulated).  The gated
family is swept by ``state_rank``, ``s4d``/``s5``/``lru`` by ``n_state`` -- at equal
``n_state`` the latter two hold ``d_v`` times fewer bytes than ``s4d``, which the
mapping reports rather than hides -- and ``mega`` by its attention ``chunk``.
``hyena`` has no bounded state at all and is run at one setting, with the mapping
recording that there is no size to put its recall against.

This part (``part = "extended_methods"``) is **not** part of the pre-registration.  It
is labelled exactly as the ``extended_N`` part is, no pre-registered quantity uses it,
and it is marked wherever it is plotted or tabulated.

Four of the added operators -- ``s4d``, ``s5``, ``lru``, ``hyena`` -- are linear and
time-invariant in the value field and never read ``q`` or ``k``, so on multi-query
associative recall they return **bit-identical** output whichever item is queried.
Their recall is therefore chance, and the study measures and records that identity
(``query_spread``, exactly ``0.0``) so the result is reported as the structural zero it
is and not as a measured capacity limit: ``prop:memory-rank``'s premise is inapplicable
to them, which is a different statement from the proposition being tested and passed.

Falsifier (as pre-registered)
-----------------------------
*The collapse point still fails to move with ``S`` under identity features and
orthonormal keys.*  If that happens, the flatness is not an artefact of the feature
map, S6's null stands, and Section 5 must say the predicted scaling was not observed
under two independent constructions.

Run: ``python -m attnbank.studies.S9_state_scaling [--quick]``
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import attnbank as ab

from . import _common as C
from . import _harness as H
from . import _state as ST

NAME = "S9_state_scaling"
N_GRID = (4, 8, 16, 32, 64, 128, 256, 512)      # the pre-registered grid, exactly S6's
N_EXT = (1024, 2048, 4096)                      # the labelled post-hoc extension
D_GRID = (8, 16, 32, 64, 128, 256)
DV = 64
METHODS = ("linear_rnn", "decay_rnn", "delta_rnn", "softmax")
MATCHED_LOGIT = 16.0        # the gain makes the matched softmax logit this big
N_SEEDS = 5
MAX_QUERIES = 64
FEATURE = "identity"        # S6 used "elu"; this is one of the two changes
REC_METHODS = tuple(m for m in METHODS if m != "softmax")
#: the second, separately labelled extension: the operators the expansion added
EXT_METHODS = ST.EXT_METHODS


# ======================================================================================
# the construction
# ======================================================================================


def orthonormal_keys(N: int, d: int, g: torch.Generator) -> torch.Tensor:
    """``N`` unit-norm keys in ``R^d``, orthonormal whenever ``N <= d``.

    The polar factor ``U V^T`` of a Gaussian ``G = U S V^T`` is the orthogonal matrix
    nearest ``G``.  For ``N <= d`` its rows are exactly orthonormal; for ``N > d`` no
    orthonormal set exists and the polar factor is a tight frame (``K^T K = I_d``) whose
    rows all have norm ``sqrt(d/N)``, so the renormalization below is a no-op in the
    first regime and the only change in the second.
    """
    G = torch.randn(N, d, generator=g, dtype=torch.float64)
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    K = U @ Vh
    return K / K.norm(dim=-1, keepdim=True)


def _mqar_batch(N: int, d: int, dv: int, seed: int, device, n_queries: int):
    """S6's MQAR batch with orthonormal rather than Gaussian keys.

    One query per batch element with ``m = 1`` and ``n = N``: decode semantics, and the
    query tokens cannot write into the state.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = orthonormal_keys(N, d, g)
    v = torch.randn(N, dv, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True)
    qn = min(n_queries, N)
    idx = torch.randperm(N, generator=g)[:qn]
    # matched softmax logit q.k/sqrt(d) = gain*||k||^2/sqrt(d) = gain/sqrt(d) for unit k
    gain = MATCHED_LOGIT * (d ** 0.5)
    q = (gain * k[idx]).unsqueeze(1).unsqueeze(1)             # [Q,1,1,d]
    kk = k.unsqueeze(0).unsqueeze(0).expand(qn, 1, N, d)      # [Q,1,N,d]
    vv = v.unsqueeze(0).unsqueeze(0).expand(qn, 1, N, dv)
    return (q.to(device), kk.contiguous().to(device), vv.contiguous().to(device),
            v.to(device), idx.to(device), k)


def _coherence(k: torch.Tensor) -> float:
    """Largest off-diagonal ``|<k_i, k_j>|``: 0 iff the keys are exactly orthonormal."""
    gram = (k @ k.transpose(-1, -2)).abs()
    gram.fill_diagonal_(0.0)
    return float(gram.max()) if gram.numel() > 1 else 0.0


def measure_construction(job, ctx) -> dict:
    N, d, meth = int(job["N"]), int(job["d"]), job["method"]
    accs, margins, coh = [], [], []
    for seed in range(int(job["n_seeds"])):
        q, k, v, vbank, target, kcpu = _mqar_batch(N, d, DV, 1000 + seed, ctx.device,
                                                   MAX_QUERIES)
        kw = {"feature": FEATURE} if meth != "softmax" else {}
        if meth == "decay_rnn":
            kw["decay"] = float(job["decay"])
        out = ab.build(meth, **kw)(q, k, v, causal=True, return_weights=False)
        y = out.y[:, 0, 0, :]                                   # [Q, dv]
        logits = y @ vbank.transpose(-1, -2)                    # [Q, N]
        pred = logits.argmax(-1)
        accs.append(float((pred == target).double().mean()))
        best = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        other = logits.scatter(-1, target.unsqueeze(-1), float("-inf")).amax(-1)
        margins.append(float((best - other).median()))
        coh.append(_coherence(kcpu))
    return {"recall": float(np.median(accs)), "recall_q1": float(np.percentile(accs, 25)),
            "recall_q3": float(np.percentile(accs, 75)),
            "recall_seeds": accs, "margin_median": float(np.median(margins)),
            "key_coherence": float(np.median(coh)),
            "state_rank": d if meth != "softmax" else N,
            "state_numbers": (d * DV) if meth != "softmax" else N * (d + DV),
            "N_over_S": N / d if meth != "softmax" else float("nan")}


def measure_extended(job, ctx) -> dict:
    """The same cell, for an operator whose state is not ``d x d_v``.

    Identical construction, identical scoring; the two differences are that the
    configuration comes from :func:`attnbank.studies._state.ext_config` (so the knob
    that sizes *this* operator's state is the one being swept, and ``feature=`` is
    passed only to the operators that accept it), and that the spread of the output
    *across queries* is recorded.  A spread of exactly zero is the signature of an
    operator that never read ``q`` or ``k``.
    """
    N, d, meth = int(job["N"]), int(job["d"]), job["method"]
    cfg = ST.ext_config(job, feature=FEATURE)
    op = ab.build(meth, **cfg)
    accs, margins, spread = [], [], 0.0
    for seed in range(int(job["n_seeds"])):
        q, k, v, vbank, target, _ = _mqar_batch(N, d, DV, 1000 + seed, ctx.device,
                                                MAX_QUERIES)
        y = op(q, k, v, causal=True, return_weights=False).y[:, 0, 0, :]   # [Q, dv]
        spread = max(spread, float((y - y[:1]).abs().max()))
        logits = y @ vbank.transpose(-1, -2)
        accs.append(float((logits.argmax(-1) == target).double().mean()))
        best = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        other = logits.scatter(-1, target.unsqueeze(-1), float("-inf")).amax(-1)
        margins.append(float((best - other).median()))
    return ST.ext_metrics(meth, cfg, N=N, d=d, dv=DV, recalls=accs, margins=margins,
                          query_spread=spread)


# ======================================================================================
# analysis
# ======================================================================================


def _crossing(cur: list[dict]) -> tuple[int | None, float | None]:
    """``(smallest grid N with recall < 0.5, log-interpolated crossing of 0.5)``.

    The first is the pre-registered estimator and is quantized to the octave grid; the
    second interpolates ``recall`` linearly in ``log2 N`` between the last cell at or
    above 0.5 and the first below it, which is what the slope is fitted on.
    """
    half = next((r["N"] for r in cur if r["recall"] < 0.5), None)
    if half is None:
        return None, None
    below = next(r for r in cur if r["N"] == half)
    above = [r for r in cur if r["N"] < half]
    if not above:
        return half, float(half)
    a = above[-1]
    x0, x1 = np.log2(a["N"]), np.log2(below["N"])
    y0, y1 = a["recall"], below["recall"]
    t = 0.0 if y0 == y1 else (y0 - 0.5) / (y0 - y1)
    return half, float(2.0 ** (x0 + t * (x1 - x0)))


def _collapse_block(rows, methods, ds) -> dict:
    out = {}
    for meth in methods:
        per_d = {}
        for d in ds:
            cur = sorted([r for r in rows if r["method"] == meth and r["d"] == d],
                         key=lambda r: r["N"])
            if not cur:
                continue
            half, interp = _crossing(cur)
            per_d[str(d)] = {
                "recall_by_N": {str(r["N"]): r["recall"] for r in cur},
                "N_at_recall_0.5": half,
                "N_at_recall_0.5_interp": interp,
                "N_over_S_at_collapse": (half / d) if (half and meth != "softmax") else None,
                "N_over_S_at_collapse_interp": ((interp / d) if (interp and meth != "softmax")
                                                else None),
                "max_N_tested": max(r["N"] for r in cur),
                "censored": half is None,
            }
        out[meth] = per_d
    return out


def _slope_block(collapse, ds, key="N_at_recall_0.5_interp") -> dict:
    """Log--log slope of the collapse point on the state rank, per operator."""
    out = {}
    for meth in REC_METHODS:
        blk = collapse.get(meth, {})
        xs = [d for d in ds if blk.get(str(d), {}).get(key) is not None]
        ys = [blk[str(d)][key] for d in xs]
        fit = C.fit_loglog_slope(xs, ys)
        span = (max(ys) / min(ys)) if len(ys) >= 2 else float("nan")
        out[meth] = {**fit, "d_values": xs, "collapse_values": ys,
                     "collapse_span": span,
                     "d_span": (max(xs) / min(xs)) if len(xs) >= 2 else float("nan")}
    return out


def analyse(records) -> dict:  # noqa: C901
    allrows = [{**r["job"], **r["metrics"]} for r in records if r["_status"] == "ok"]
    rows = [r for r in allrows if r.get("part") == "construction"]
    ext_rows = [r for r in allrows if r.get("part") == "extended_N"]
    ext_m_rows = [r for r in allrows if r.get("part") == "extended_methods"]
    dec_rows = [r for r in allrows if r.get("part") == "decay_sweep"]
    ds = sorted({r["d"] for r in rows})
    Ns = sorted({r["N"] for r in rows})

    summary: dict = {
        "recall_grid": rows, "extended_grid": ext_rows, "decay_sweep": dec_rows,
        "extended_methods_grid": ext_m_rows,
        "dv": DV, "matched_logit": MATCHED_LOGIT, "feature": FEATURE,
        "key_geometry": "orthonormal (polar factor of a Gaussian; unit-norm rows)",
        "changed_from_S6": ["feature map elu(k)+1 -> identity",
                            "keys standard Gaussian -> orthonormal"],
    }

    if dec_rows:
        per = {}
        for dc in sorted({r["decay"] for r in dec_rows}):
            cur = sorted([r for r in dec_rows if r["decay"] == dc], key=lambda r: r["N"])
            half, interp = _crossing(cur)
            per[str(dc)] = {"recall_by_N": {str(r["N"]): r["recall"] for r in cur},
                            "N_at_recall_0.5": half,
                            "N_at_recall_0.5_interp": interp,
                            "forgetting_horizon": 1.0 / max(1e-9, 1.0 - dc)}
        summary["decay_effect"] = {
            "note": "at d=64 the rank is fixed, so any movement here is the forgetting "
                    "horizon 1/(1-decay), not state capacity",
            "by_decay": per}

    # -- the pre-registered measurement, on the pre-registered grid --------------------
    collapse = _collapse_block(rows, METHODS, ds)
    summary["collapse"] = collapse
    summary["scaling_fit"] = _slope_block(collapse, ds)
    summary["scaling_fit_grid"] = _slope_block(collapse, ds, key="N_at_recall_0.5")
    summary["censored"] = {m: [int(d) for d in ds
                               if collapse[m].get(str(d), {}).get("censored")]
                           for m in REC_METHODS}

    # -- the labelled extension, for the two censored state ranks ----------------------
    if ext_rows:
        both = rows + ext_rows
        collapse_x = _collapse_block(both, METHODS, ds)
        summary["collapse_extended"] = collapse_x
        summary["scaling_fit_extended"] = _slope_block(collapse_x, ds)
        summary["extension"] = {
            "note": "post-hoc, labelled: N in " + str(list(N_EXT)) + " was added only "
                    "to uncensor d_phi = " + str(summary["censored"].get("linear_rnn"))
                    + ", where the collapse point lies above the pre-registered grid. "
                      "Every pre-registered quantity is computed without it.",
            "N_values": list(N_EXT)}

    # -- the second labelled extension: the operators the expansion added --------------
    if ext_m_rows:
        summary["extension_methods"] = ST.extension_block(ext_m_rows, dv=DV)

    # -- the same summaries S6 reports, so the two studies compare term by term --------
    ratios = [v["N_over_S_at_collapse"] for m in REC_METHODS
              for v in collapse.get(m, {}).values()
              if v["N_over_S_at_collapse"] is not None]
    ratios_i = [v["N_over_S_at_collapse_interp"] for m in REC_METHODS
                for v in collapse.get(m, {}).values()
                if v["N_over_S_at_collapse_interp"] is not None]
    summary["collapse_ratio"] = {
        "note": "N/S at which median recall first falls below 0.5, S = feature rank; "
                "pre-registered grid only",
        **C.med_iqr(ratios), "values": ratios}
    summary["collapse_ratio_interp"] = {**C.med_iqr(ratios_i), "values": ratios_i}

    emb = {}
    for dd in ds:
        sm = [r for r in rows if r["method"] == "softmax" and r["d"] == dd]
        if sm:
            worst = min(sm, key=lambda r: r["recall"])
            emb[str(dd)] = {"min_recall": worst["recall"], "at_N": worst["N"],
                            "recall_by_N": {str(r["N"]): r["recall"]
                                            for r in sorted(sm, key=lambda x: x["N"])}}
    summary["embedding_limit"] = {
        "note": "softmax recall by key dimension; a value below 1 means the key code "
                "itself is saturated at that d, so no operator could do better",
        "by_d": emb,
        "d_where_embedding_is_clean": [int(k) for k, v in emb.items()
                                       if v["min_recall"] > 0.95]}
    clean_d = set(summary["embedding_limit"]["d_where_embedding_is_clean"])
    soft = [r["recall"] for r in rows if r["method"] == "softmax" and r["d"] in clean_d]
    soft_big = [r["recall"] for r in rows if r["method"] == "softmax"
                and r["N"] >= 256 and r["d"] in clean_d]
    summary["softmax_control"] = {
        "min_recall": min(soft) if soft else float("nan"),
        "min_recall_large_N": min(soft_big) if soft_big else float("nan"),
        **C.med_iqr(soft)}

    past = [r["recall"] for r in rows if r["method"] != "softmax" and r["N"] >= 4 * r["d"]]
    summary["recall_past_4S"] = {**C.med_iqr(past), "n": len(past)}
    at_s = [r["recall"] for r in rows if r["method"] != "softmax" and r["N"] == r["d"]]
    summary["recall_at_S"] = {**C.med_iqr(at_s), "n": len(at_s)}

    # -- does recall rescale?  If capacity is proportional to S, recall as a function
    #    of the load N/S is one curve, not six.  The spread of the six curves at a
    #    common load is the number that says so.
    rescale = {}
    for meth in REC_METHODS:
        spread, by_load = [], {}
        for load in (0.5, 1.0, 2.0, 4.0, 8.0):
            vals = [r["recall"] for r in rows
                    if r["method"] == meth and abs(r["N"] / r["d"] - load) < 1e-9]
            if len(vals) >= 3:
                by_load[str(load)] = {"n": len(vals), "median": float(np.median(vals)),
                                      "spread": float(max(vals) - min(vals))}
                spread.append(float(max(vals) - min(vals)))
        rescale[meth] = {"by_load": by_load,
                         "max_spread": max(spread) if spread else float("nan")}
    summary["rescaled_by_load"] = {
        "note": "recall at a fixed load N/S across the state ranks that reach it; a "
                "small spread means the recall curve is a function of N/S alone",
        "by_method": rescale}

    # -- the falsifier -----------------------------------------------------------------
    fits = summary["scaling_fit"]
    moves = {m: bool(fits[m]["n"] >= 3 and np.isfinite(fits[m]["ci95"][0])
                     and fits[m]["ci95"][0] > 0.0 and fits[m]["collapse_span"] >= 4.0)
             for m in REC_METHODS}
    all_move = all(moves.values())
    summary["falsifier"] = {
        "statement": "the collapse point still fails to move with S under identity "
                     "features and orthonormal keys",
        "verdict": "CONTRADICTED" if all_move else (
            "SUPPORTED" if not any(moves.values()) else "MIXED"),
        "criterion": "CONTRADICTED iff, for every recurrent operator, the log-log slope "
                     "of the collapse point on d_phi has a 95% CI strictly above 0 and "
                     "the collapse point moves by at least 4x across the sweep",
        "per_method_moves": moves,
        "slopes": {m: {"slope": fits[m]["slope"], "ci95": fits[m]["ci95"],
                       "r2": fits[m]["r2"], "n": fits[m]["n"],
                       "collapse_span": fits[m]["collapse_span"],
                       "d_span": fits[m]["d_span"]} for m in REC_METHODS},
        "median_N_over_S_at_collapse": float(np.median(ratios)) if ratios else float("nan"),
        "median_recall_past_4S": summary["recall_past_4S"]["median"],
        "softmax_min_recall_at_large_N": summary["softmax_control"]["min_recall_large_N"],
        "softmax_is_unaffected": ("SUPPORTED" if soft_big and min(soft_big) > 0.9
                                  else "CONTRADICTED" if soft_big else "INCONCLUSIVE"),
    }
    if "scaling_fit_extended" in summary:
        summary["falsifier"]["slopes_extended"] = {
            m: {"slope": summary["scaling_fit_extended"][m]["slope"],
                "ci95": summary["scaling_fit_extended"][m]["ci95"],
                "r2": summary["scaling_fit_extended"][m]["r2"],
                "n": summary["scaling_fit_extended"][m]["n"],
                "collapse_span": summary["scaling_fit_extended"][m]["collapse_span"]}
            for m in REC_METHODS}
    summary["N_grid"] = Ns
    return summary


# ======================================================================================
# figure
# ======================================================================================


PRETTY = {"linear_rnn": "linear RNN", "decay_rnn": "decay RNN",
          "delta_rnn": "delta RNN", "softmax": "softmax"}


def _pool(rows, **match) -> list[float]:
    """Every per-seed recall in the cells matching ``match`` (see S6's ``_pool``)."""
    out: list[float] = []
    for r in rows:
        if all(r.get(k) == v for k, v in match.items()):
            out.extend(r.get("recall_seeds") or [r["recall"]])
    return out


def group_collapse(ext: dict, methods) -> tuple:
    """``(x, median, lo, hi, n)`` of the collapse point over a group of operators.

    Keyed by the *knob rung* rather than by the state size, because the five gated
    operators put very slightly different byte counts on the same ``state_rank``
    (``mlstm`` carries a normaliser alongside the matrix memory), and plotting five
    curves that differ by 1.5% in ``x`` would be five curves on top of one another.
    ``x`` is the mean of the rung's state sizes in the rank-equivalent unit.
    """
    per, xs_by_rung = {}, {}
    for m in methods:
        for blk in ext["by_method"][m]["collapse"].values():
            if (blk["N_at_recall_0.5"] is None or blk["knob_value"] is None
                    or blk["state_rank_equivalent"] is None):
                continue
            per.setdefault(blk["knob_value"], []).append(blk["N_at_recall_0.5"])
            xs_by_rung.setdefault(blk["knob_value"], []).append(
                blk["state_rank_equivalent"])
    rungs = sorted(per)
    return ([float(np.mean(xs_by_rung[r])) for r in rungs],
            [float(np.median(per[r])) for r in rungs],
            [float(min(per[r])) for r in rungs], [float(max(per[r])) for r in rungs],
            [len(per[r]) for r in rungs])


def lti_recall(ext_rows, Ns) -> list[float]:
    """Median recall of the query-independent operators at each ``N``: the chance curve."""
    out = []
    for N in Ns:
        v = [r["recall"] for r in ext_rows
             if r["N"] == N and r["method"] in ST.EXT_LTI]
        out.append(float(np.median(v)) if v else float("nan"))
    return out


def make_figure(summary) -> list[str]:  # noqa: C901
    """The same three statements S6's figure makes, under the fixed construction.

    Panel for panel this is ``fig_finite_state_recall``, which is the point: (a) recall
    against the number of stored pairs for every operator, (b) the same sweep for one
    operator resolved by state rank, (c) the collapse point against the state rank on
    log--log axes with the ``N = d_phi`` line.  S6's three readings were "everything
    collapses at N of 4 to 8", "the curves do not separate" and "the collapse point is
    flat".  Reading the same three panels here is how a reader sees that the only thing
    that changed is the feature map and the key geometry.

    Panel (b) carries the extra line S6's did not need: the six curves are the *same*
    curve shifted by one octave per doubling of the rank, so the inset note gives the
    spread of recall at a fixed load ``N/S``, which is the quantitative form of "it
    rescales".
    """
    from ..plotting import style
    style.use()
    rows = summary["recall_grid"]
    ext = summary.get("extended_grid") or []
    extm = summary.get("extension_methods")
    ext_rows = summary.get("extended_methods_grid") or []
    ds = sorted({r["d"] for r in rows})
    Ns = sorted({r["N"] for r in rows})
    el = summary.get("embedding_limit", {})
    fig, axd = style.mosaic("fig_state_scaling",
                            [["a", "c"], ["b", "c"]], width_ratios=(1.1, 0.9))
    axes = [axd["a"], axd["b"], axd["c"]]
    axes[1].sharex(axes[0])
    axes[1].sharey(axes[0])

    # --- (a) every operator, pooled over state size ---------------------------------
    ax = axes[0]
    ends = []
    for i, meth in enumerate(REC_METHODS):
        med, q1, q3 = [], [], []
        for N in Ns:
            v = _pool(rows, method=meth, N=N)
            med.append(float(np.median(v)))
            q1.append(float(np.percentile(v, 25)))
            q3.append(float(np.percentile(v, 75)))
        ax.plot(Ns, med, "-o", color=style.series(i), markersize=3.2,
                linewidth=2.6 if i == 0 else 1.3, label=PRETTY[meth])
        ax.fill_between(Ns, q1, q3, color=style.series(i), alpha=0.13, linewidth=0)
        ends.append((Ns[-1], med[-1], PRETTY[meth]))
    soft = [float(np.median(_pool(rows, method="softmax", N=N))) for N in Ns]
    ax.plot(Ns, soft, "-", color=style.INK["primary"], linewidth=1.6)
    ends.append((Ns[-1], soft[-1], "softmax"))
    ax.set_xscale("log", base=2)
    style.log2_ticks(ax, "x", 2, 9, every=1)
    ax.set_xlim(2 ** 1.7, 2 ** 10.4)
    ax.set_ylim(-0.04, 1.58)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelbottom=False)
    ax.set_ylabel("recall (fraction correct)")
    # The four query-independent operators the expansion added.  They are drawn here
    # and not in (c) because they have no collapse point: their output does not depend
    # on which item is queried, so recall *is* chance and the curve is 1/N.  Showing it
    # beside the control is the only way a reader can see that the zero is structural.
    if extm:
        lti = lti_recall(ext_rows, Ns)
        ax.plot(Ns, lti, "--", color=style.INK["muted"], linewidth=1.2, zorder=2)
        style.direct_label(ax, Ns[0], 0.78, "s4d, s5, lru, hyena are\n"
                                            "query-independent, so\n"
                                            "recall is chance $1/N$",
                           dx=2, dy=0, ha="left", va="top", size=7.5)
    style.panel(ax, "a", "recall against load")
    style.label_ends(ax, ends, size=7.5, min_gap_frac=0.075)
    d8 = el.get("by_d", {}).get(str(min(ds)), {}).get("min_recall", float("nan"))
    style.note(ax, f"median and IQR over the {len(ds)} state sizes:\n"
                   f"the band is wide because the curves now\n"
                   f"move with the state (softmax = {d8:.2f} at every rank)",
               loc="upper left", size=7.5)

    # --- (b) does the state size matter?  Here it does. -----------------------------
    ax = axes[1]
    meth = REC_METHODS[0]
    for i, dd in enumerate(ds):
        med = [float(np.median(_pool(rows, method=meth, d=dd, N=N))) for N in Ns]
        ax.plot(Ns, med, "-o", color=style.seq(i, len(ds)), markersize=3.0,
                label=f"${dd}$")
    ax.set_xlabel("stored pairs $N$")
    style.panel(ax, "b", f"{PRETTY[meth]}, by state rank")
    ax.legend(loc="upper right", fontsize=7.5, title="state rank $d_\\varphi$",
              title_fontsize=7.5, ncol=len(ds) // 2, labelspacing=0.12,
              columnspacing=0.6, borderpad=0.2, handlelength=0.9,
              handletextpad=0.3)
    rs = summary["rescaled_by_load"]["by_method"][meth]["max_spread"]
    style.note(ax, "one octave of rank buys one\n"
                   "octave of $N$: at matched load\n"
                   f"$N/S$ the six curves agree\n"
                   f"to {rs:.2f} in recall",
               loc="lower left", size=7.5)

    # --- (c) the collapse point against the state rank ------------------------------
    ax = axes[2]
    cx = summary.get("collapse_extended") or summary["collapse"]
    ends = []
    for i, meth in enumerate(REC_METHODS):
        blk = summary["collapse"].get(meth, {})
        bx = cx.get(meth, {})
        xs = [d for d in ds if bx.get(str(d), {}).get("N_at_recall_0.5") is not None]
        ys = [bx[str(d)]["N_at_recall_0.5"] for d in xs]
        if not xs:
            continue
        ax.plot(xs, ys, "-", color=style.series(i),
                linewidth=2.6 if i == 0 else 1.3, label=PRETTY[meth])
        # filled = pre-registered grid; open = the labelled extension
        pre = [(x, y) for x, y in zip(xs, ys)
               if blk.get(str(x), {}).get("N_at_recall_0.5") is not None]
        post = [(x, y) for x, y in zip(xs, ys)
                if blk.get(str(x), {}).get("N_at_recall_0.5") is None]
        if pre:
            ax.plot(*zip(*pre), "o", color=style.series(i), markersize=3.4,
                    linestyle="none")
        if post:
            ax.plot(*zip(*post), "o", color=style.INK["surface"], markersize=3.4,
                    markeredgecolor=style.series(i), markeredgewidth=0.9,
                    linestyle="none")
        ends.append((xs[-1], ys[-1], PRETTY[meth].replace(" RNN", "")))
    # The labelled extension to the operators the expansion added.  The gated five are
    # one curve -- the median collapse point over the five, with their min-max range --
    # because they lie within half an octave of one another and five more categorical
    # hues would be unreadable; the per-operator cells are in \cref{tab:state-scaling}.
    if extm:
        shown = extm["groups_measured"]["gated"]
        gx, gm, glo, ghi, _ = group_collapse(extm, shown)
        if gx:
            ax.fill_between(gx, glo, ghi, color=style.series(3), alpha=0.18,
                            linewidth=0, zorder=2)
            ax.plot(gx, gm, "--s", color=style.series(3), markersize=3.0,
                    linewidth=1.3, zorder=3)
            ends.append((gx[-1], gm[-1],
                         f"gated ({len(shown)} of {len(extm['groups']['gated'])})"))
        hx, hm, _, _, _ = group_collapse(extm, extm["groups_measured"]["hybrid"])
        if hx:
            ax.plot(hx, hm, "--^", color=style.series(4), markersize=3.0,
                    linewidth=1.3, zorder=3)
            ends.append((hx[-1], hm[-1], "mega"))
    bnd = [min(ds), 2 ** 9.9]
    ax.plot(bnd, bnd, color=style.INK["primary"], linewidth=1.0, zorder=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    style.log2_ticks(ax, "x", 3, 9, every=1)
    style.log2_ticks(ax, "y", 3, 12, every=2)
    ax.set_xlim(2 ** 2.7, 2 ** 9.9)
    ax.set_ylim(2 ** 2.4, 2 ** 13.2)
    if ext:
        ax.axhspan(2 ** 9.5, 2 ** 13.2, color=style.INK["grid"], alpha=0.55, zorder=0,
                   linewidth=0)
        style.direct_label(ax, 2 ** 2.85, 2 ** 13.0,
                           "shaded: past the registered grid.\n"
                           "open markers are the labelled $N$\n"
                           "extension; dashed curves the\n"
                           "labelled method extension",
                           dx=0, dy=0, ha="left", va="top", size=7.5)
    style.direct_label(ax, ds[0], ds[0], r"bound $N=S$", dx=3, dy=-3,
                       ha="left", va="top", size=7.5)
    ax.set_xlabel(r"state size $S$ = bytes$/(d_v\cdot 8)$;" "\n"
                  r"$=d_\varphi$ for the additive recurrences")
    ax.set_ylabel("$N$ at which median\nrecall first drops below 0.5")
    style.panel(ax, "c", "collapse tracks the state")
    style.label_ends(ax, ends, size=7.5, min_gap_frac=0.045)
    fit = summary["scaling_fit"][REC_METHODS[0]]
    style.note(ax, f"slope {fit['slope']:.2f}\n"
                   f"[{fit['ci95'][0]:.2f}, {fit['ci95'][1]:.2f}]\n"
                   f"on the registered grid",
               loc="lower right", size=7.5)
    return [str(C.save_fig(fig, "fig_state_scaling"))]


# ======================================================================================
# main
# ======================================================================================


def _measure(job, ctx) -> dict:
    """Dispatch: the pre-registered cells and the labelled extension differ only here."""
    return (measure_extended if job.get("part") == "extended_methods"
            else measure_construction)(job, ctx)


def main(quick: bool = False, resume: bool = True, analyse_only: bool = False,
         extend: bool = True, ext_methods: bool = True) -> dict:
    dev = H.default_device()
    n_grid = (8, 64) if quick else N_GRID
    d_grid = (16, 64) if quick else D_GRID
    n_ext = (128,) if quick else N_EXT
    seeds = 2 if quick else N_SEEDS

    sweep = H.Sweep(NAME, seed=0, device=dev, resume=resume, progress_every=25,
                    extra_meta={
                        "study": NAME, "section": "sec:memory-state",
                        "preregistered": "STUDIES.md S9",
                        "N_grid": list(n_grid), "N_grid_extension": list(n_ext),
                        "d_grid": list(d_grid), "dv": DV, "methods": list(METHODS),
                        "n_seeds": seeds, "matched_logit": MATCHED_LOGIT,
                        "max_queries": MAX_QUERIES, "feature": FEATURE,
                        "state_definition": "S = feature rank d_phi; storage is "
                                            "d_phi * d_v numbers",
                        "key_embedding": "orthonormal: the polar factor U V^T of a "
                                         "Gaussian, rows renormalized to unit norm; "
                                         "exactly orthonormal for N <= d_phi and a "
                                         "unit-norm tight frame beyond it",
                        "relation_to_S6": "S6's construction with two changes and "
                                          "nothing else: feature map elu(k)+1 -> "
                                          "identity, keys Gaussian -> orthonormal",
                        "extension_note": "N > 512 is a post-hoc, separately labelled "
                                          "part run only to uncensor the two largest "
                                          "state ranks; no pre-registered quantity "
                                          "uses it",
                        "extended_methods": list(EXT_METHODS) if ext_methods else [],
                        "extended_methods_note":
                            "a second post-hoc, separately labelled part "
                            "(part='extended_methods'): the operators the library "
                            "expansion added, each swept through the knob that sizes "
                            "ITS OWN state.  The state size is read off "
                            "cost()['state_bytes'] by attnbank.studies._state rather "
                            "than assumed to be d x d_v, which is what the pre-"
                            "registered part assumes and what stopped being true: "
                            "s4d carries n_state x d_v, s5 and lru carry n_state "
                            "shared across channels (d_v times smaller at equal "
                            "n_state), hyena carries no bounded state at all, and "
                            "mega's is bounded only by its attention chunk.  No "
                            "pre-registered quantity uses any of it.",
                        "extended_methods_knob_ladder": {
                            k: list(v) for k, v in ST.KNOB_LADDER.items()},
                        "data": "constructed MQAR (not real activations); construction "
                                "only, nothing is trained"})

    jobs = [{"part": "construction", "method": m, "N": N, "d": d, "dv": DV,
             "n_seeds": seeds, "decay": 0.999}
            for m in METHODS for N in n_grid for d in d_grid]
    # the same decay sweep S6 runs: forgetting horizon is a different mechanism from rank
    jobs += [{"part": "decay_sweep", "method": "decay_rnn", "N": N, "d": 64,
              "dv": DV, "n_seeds": seeds, "decay": dc}
             for N in n_grid for dc in ((0.9, 0.999) if quick else (0.9, 0.98, 0.995, 0.999))]
    if extend:
        jobs += [{"part": "extended_N", "method": m, "N": N, "d": d, "dv": DV,
                  "n_seeds": seeds, "decay": 0.999}
                 for m in METHODS for N in n_ext for d in d_grid]
    if ext_methods:
        # The gated family and the chunked hybrid can collapse above the registered
        # grid for the same reason the pre-registered operators do, so they get the
        # same N range; the query-independent operators are at chance everywhere and
        # nothing above N = 512 would change that.
        ext_m = list(ST.EXT_GATED) + list(ST.EXT_HYBRID) if not quick else ["retnet"]
        jobs += ST.ext_jobs(ext_m, list(n_grid) + list(n_ext), d=64, dv=DV,
                            n_seeds=seeds)
        jobs += ST.ext_jobs(list(ST.EXT_LTI) if not quick else ["s5"], n_grid,
                            d=64, dv=DV, n_seeds=seeds)

    if analyse_only:
        records = [r for r in sweep.all_records
                   if r["job"].get("part") in ("construction", "decay_sweep",
                                               "extended_N", "extended_methods")]
        print(f"  analysing {len(records)} cached jobs without running new ones",
              file=sys.stderr)
    else:
        records = sweep.run(jobs, _measure, label="MQAR construction")

    summary = analyse(records)
    figs = make_figure(summary)
    path = H.save_results(NAME, records,
                          sweep.meta(summary=summary, figures=figs, quick=quick,
                                     analyse_only=analyse_only, extended=extend))
    print(f"\n{NAME}: wrote {path}", file=sys.stderr)
    _report(summary)
    return summary


def _report(s: dict) -> None:
    p = lambda *a: print(*a, file=sys.stderr)
    p("\n--- S9 falsifier outcome ---")
    p(f"  statement: {s['falsifier']['statement']}")
    p(f"  verdict:   {s['falsifier']['verdict']}")
    for m, f in s["falsifier"]["slopes"].items():
        p(f"    {m}: slope {f['slope']:+.3f} CI [{f['ci95'][0]:+.3f}, "
          f"{f['ci95'][1]:+.3f}] r2 {f['r2']:.3f} n={f['n']} "
          f"collapse span {f['collapse_span']:.1f}x over d span {f['d_span']:.0f}x")
    for m, f in s["falsifier"].get("slopes_extended", {}).items():
        p(f"    {m} (with extension): slope {f['slope']:+.3f} CI "
          f"[{f['ci95'][0]:+.3f}, {f['ci95'][1]:+.3f}] r2 {f['r2']:.3f} n={f['n']}")
    for meth, per_d in s["collapse"].items():
        for d, v in per_d.items():
            p(f"  {meth} S={d}: N at recall 0.5 = {v['N_at_recall_0.5']} "
              f"(interp {v['N_at_recall_0.5_interp']}), N/S = "
              f"{v['N_over_S_at_collapse']}")
    p(f"  N/S at collapse: median {s['collapse_ratio']['median']} "
      f"IQR {s['collapse_ratio']['q1']}-{s['collapse_ratio']['q3']} "
      f"(n={s['collapse_ratio']['n']})")
    p(f"  recall at N=S: median {s['recall_at_S']['median']:.3f} "
      f"(n={s['recall_at_S']['n']}); past N=4S: "
      f"{s['recall_past_4S']['median']:.3f} (n={s['recall_past_4S']['n']})")
    p(f"  softmax control: median {s['softmax_control']['median']:.3f}, min at large N "
      f"{s['softmax_control']['min_recall_large_N']:.3f}")
    for dd, v in s["embedding_limit"]["by_d"].items():
        p(f"     embedding d={dd}: softmax min recall {v['min_recall']:.3f} at N={v['at_N']}")
    for m, v in s["rescaled_by_load"]["by_method"].items():
        p(f"  {m}: max recall spread at matched load N/S = {v['max_spread']:.3f}")
    for dc, v in s.get("decay_effect", {}).get("by_decay", {}).items():
        p(f"  decay={dc} (horizon {v['forgetting_horizon']:.0f}): N at recall 0.5 = "
          f"{v['N_at_recall_0.5']}")
    ext = s.get("extension_methods")
    if ext:
        p("\n--- labelled extension: the operators the expansion added ---")
        p("  (post-hoc; no pre-registered quantity uses any of this)")
        for m, v in sorted(ext["by_method"].items()):
            c = v["classification"]
            p(f"  {m:8s} knob={str(c['knob']):11s} verdict={c['verdict']}")
            p(f"           state bytes {c['state_bytes'][:1]}..{c['state_bytes'][-1:]}"
              f"  max query spread {c['max_query_spread']:.3g}"
              f"  median recall {c['median_recall']:.3f}")
            if c["verdict"] == ST.MEASURED:
                f = v["scaling_fit"]
                p("           collapse: " + ", ".join(
                    f"S={b['state_rank_equivalent']:g}:{b['N_at_recall_0.5']}"
                    for b in v["collapse"].values())
                  + f"  slope {f['slope']:+.3f} CI [{f['ci95'][0]:+.3f}, "
                    f"{f['ci95'][1]:+.3f}] r2 {f['r2']:.3f} n={f['n']}")
        sz = ext["structural_zero"]
        p(f"  structural zero (query-independent): {sz['methods']}")
        p(f"    max |recall - chance| {sz['max_abs_recall_minus_chance']}")
        p(f"  unbounded state: {ext['unbounded_state']['methods']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--analyse-only", action="store_true",
                    help="finalise from the resume cache without running new jobs")
    ap.add_argument("--no-extend", action="store_true",
                    help="pre-registered grid only; skip the labelled N > 512 part")
    ap.add_argument("--no-ext-methods", action="store_true",
                    help="skip the labelled extension to the operators the library "
                         "expansion added")
    a = ap.parse_args()
    main(quick=a.quick, resume=not a.fresh, analyse_only=a.analyse_only,
         extend=not a.no_extend, ext_methods=not a.no_ext_methods)
