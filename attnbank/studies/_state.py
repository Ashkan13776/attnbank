"""What "state size" means for each operator, derived from ``cost()``.

Not a study.  S6 and S9 sweep the state of a recurrent operator against the number of
associations it has to serve (``prop:memory-rank``), and until the library's recurrent
corner was one family that was a single line of code: every method carried
``H in R^{d x d_v}``, so sweeping the key dimension ``d`` *was* sweeping the state.
That identity is gone.

* ``s4d`` carries one SISO state-space model **per value channel**: ``n_state x d_v``.
* ``s5`` and ``lru`` carry **one** diagonal state shared by every channel: ``n_state``.
  At equal ``n_state`` their state is ``d_v`` times smaller than ``s4d``'s, which is
  the MIMO simplification and not a bookkeeping detail
  (``tests/test_recurrent_ssm.py::test_s5_carries_one_state_for_all_channels_...``).
* ``hyena`` has **no** finite state: an arbitrary length-``n`` FIR filter needs
  ``Theta(n)`` memory, so there is nothing for ``prop:memory-rank`` to bound.
* ``mega`` is a hybrid: a bounded EMA state beside an attention branch that is bounded
  only because it is chunked.
* The gated family takes ``state_rank``, which is the ``r`` of ``H in R^{r x d_v}``.

Why this is derived and not tabulated
-------------------------------------
A hand-written ``{method: state_formula}`` table is exactly the object that rots: it
has to be edited every time a method is added, nothing fails when it is not, and the
repository has already removed three of them.  Every method in the bank already
implements ``cost()``, whose ``state_bytes`` entry the conformance suite requires and
checks (``tests/test_conformance.py``), so **bytes from ``cost()`` are the common
unit** and everything here is a probe of that function:

* :func:`state_bytes` is ``cost(...)["state_bytes"]`` and nothing else;
* :func:`is_bounded` asks whether that number moves when the context grows by ``2^10``;
* :func:`state_knobs` finds the configuration keys it responds to, by building the
  method twice and comparing --- so a method added tomorrow is handled without an edit
  here, and a method whose ``cost()`` is wrong is wrong in one place rather than two;
* :func:`reads_qk` is measured, not declared: the operator is run twice on the same
  values with different queries and keys, and if the output is *bit-identical* it did
  not read them.

Where a method has no bounded state the mapping says so --- :attr:`StateSpec.bounded`
is ``False`` and :attr:`StateSpec.state_bytes` is ``None`` --- rather than returning a
number that would be a function of the sweep's context length.

Use::

    from attnbank.studies import _state as ST

    ST.state_spec("s5", {"n_state": 16}, n=512, d=64, dv=64).state_bytes   # 256.0
    ST.state_spec("s4d", {"n_state": 16}, n=512, d=64, dv=64).state_bytes  # 16384.0
    ST.state_spec("hyena", {}, n=512, d=64, dv=64).bounded                 # False
"""
from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field

import torch

import attnbank as ab

__all__ = ["StateSpec", "BYTES_PER_EL", "state_bytes", "is_bounded", "state_knobs",
           "reads_qk", "state_spec", "sweep_knob", "describe_methods", "accepts",
           "ladder", "ext_jobs", "ext_config", "ext_metrics", "classify",
           "EXT_METHODS", "EXT_GATED", "EXT_LTI", "EXT_HYBRID", "KNOB_LADDER",
           "STRUCTURAL", "UNBOUNDED", "MEASURED", "NO_REGIME", "collapse_crossing",
           "ext_collapse", "extension_block"]

#: The studies measure in float64 (``_harness.DEFAULT_DTYPE``), so the honest byte
#: count for a state the sweep actually materialises is 8 bytes an element.  ``cost()``
#: defaults to 2 (bf16 deployment); passing it explicitly keeps the unit the study's.
BYTES_PER_EL = 8

#: Context lengths used to decide whether a state is bounded.  A *windowed* state --
#: ``mega``'s chunk of KV, and the eviction and window methods elsewhere in the bank --
#: is ``min(window, n)``, so it only looks flat once the shorter probe is past the
#: window; probing at 64 would report a 256-wide chunk as unbounded, which is the
#: opposite of the truth.  Both probes are therefore past any window the ladders below
#: configure, and 2^7 apart, so a state that grows at all grows by two orders of
#: magnitude between them while one that does not is flat to the last bit.
_SHORT_N, _LONG_N = 8192, 1 << 20

#: Values a candidate configuration key is probed at.  Two is enough to see a response
#: and small enough that no method refuses them.
_PROBE_VALUES = (4, 8)


@dataclass(frozen=True)
class StateSpec:
    """The state one configuration of one method carries, in bytes.

    ``state_bytes`` is ``None`` exactly when ``bounded`` is ``False``: a method whose
    state grows with the context has no state *size*, only a state size at a context
    length, and reporting the latter as the former is the error this class exists to
    prevent.
    """

    method: str
    config: dict = field(default_factory=dict)
    state_bytes: float | None = None
    bounded: bool = True
    knob: str | None = None
    knob_value: int | None = None
    rank_equivalent: float | None = None
    reads_qk: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ======================================================================================
# the three probes
# ======================================================================================


def state_bytes(method: str, config: dict | None = None, *, m: int = 1, n: int,
                d: int, dv: int, bytes_per_el: int = BYTES_PER_EL) -> float:
    """``cost()["state_bytes"]`` for one configuration at one shape.  The whole unit."""
    return float(ab.build(method, **(config or {})).cost(
        m=m, n=n, d=d, dv=dv, bytes_per_el=bytes_per_el)["state_bytes"])


def is_bounded(method: str, config: dict | None = None, *, d: int = 64, dv: int = 64,
               bytes_per_el: int = BYTES_PER_EL) -> tuple[bool, float, float]:
    """``(bounded, bytes at n=64, bytes at n=65536)``.

    The same test ``tests/test_recurrent_ssm.py`` applies to the state-space family,
    one window width apart (see :data:`_SHORT_N`), so a method that passes there is
    classified here the way that test classifies it.
    """
    kw = dict(d=d, dv=dv, bytes_per_el=bytes_per_el)
    short = state_bytes(method, config, n=_SHORT_N, **kw)
    long = state_bytes(method, config, n=_LONG_N, **kw)
    return (short == long, short, long)


def _int_parameters(method: str) -> list[str]:
    """Constructor keys that could plausibly size a state: ``int`` or ``int | None``."""
    klass = ab.get_class(method)
    try:
        sig = inspect.signature(klass.__init__)
    except (TypeError, ValueError):                       # pragma: no cover - builtins
        return []
    out = []
    for name, p in sig.parameters.items():
        if name in ("self", "kw", "kwargs") or p.kind is p.VAR_KEYWORD:
            continue
        if p.default is p.empty or isinstance(p.default, int) or p.default is None:
            if not isinstance(p.default, bool):
                out.append(name)
    return out


def state_knobs(method: str, *, d: int = 64, dv: int = 64,
                bytes_per_el: int = BYTES_PER_EL) -> list[str]:
    """Configuration keys this method's ``state_bytes`` responds to, discovered.

    Every integer-valued constructor key is tried at two values and kept if the state
    size moves.  When none does, the key dimension itself is tried and returned as
    ``"d"`` --- the case of ``linear_rnn``, ``decay_rnn`` and ``delta_rnn``, whose state
    is ``d x d_v`` and which is why the studies used to sweep ``d``.
    """
    shape = dict(n=_SHORT_N, d=d, dv=dv, bytes_per_el=bytes_per_el)
    found = []
    for key in _int_parameters(method):
        try:
            a = state_bytes(method, {key: _PROBE_VALUES[0]}, **shape)
            b = state_bytes(method, {key: _PROBE_VALUES[1]}, **shape)
        except Exception:                                 # a value the method refuses
            continue
        if a != b:
            found.append(key)
    if found:
        return found
    base = state_bytes(method, {}, **shape)
    wide = state_bytes(method, {}, **{**shape, "d": 2 * d})
    return ["d"] if base != wide else []


def reads_qk(method: str, config: dict | None = None, *, n: int = 16, d: int = 8,
             dv: int = 4, seed: int = 0) -> bool:
    """Measured, not declared: does the output change when ``q`` and ``k`` change?

    Two draws of ``(q, k)`` against one ``v``.  A linear time-invariant layer -- ``s4d``,
    ``s5``, ``lru``, ``hyena`` -- is a convolution of the values in the lag and has
    nowhere for ``q`` and ``k`` to enter, so it returns the *same bits* both times.  On
    associative recall that is a structural zero and not a measured capacity limit,
    which is a distinction the write-up has to make and therefore one the study has to
    record.
    """
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64)
    v = rnd(1, 1, n, dv)
    meth = ab.build(method, **(config or {}))
    y1 = meth(rnd(1, 1, 1, d), rnd(1, 1, n, d), v, causal=True).y
    y2 = meth(rnd(1, 1, 1, d), rnd(1, 1, n, d), v, causal=True).y
    return not bool(torch.equal(y1, y2))


# ======================================================================================
# the mapping the studies call
# ======================================================================================


def state_spec(method: str, config: dict | None = None, *, n: int, d: int, dv: int,
               bytes_per_el: int = BYTES_PER_EL, knob: str | None = None,
               check_reads_qk: bool = True) -> StateSpec:
    """The state one configuration carries, in bytes, or the reason there is no number.

    ``rank_equivalent`` is ``state_bytes / (d_v * bytes_per_el)``: the rank an additive
    ``r x d_v`` memory would need to hold the same bytes.  It is the axis S6 and S9 were
    written against, so it puts every method back on one comparable scale --- ``s4d`` at
    ``n_state = 16`` lands at 32 (a complex state is two reals) and ``s5`` at the same
    ``n_state`` lands at ``32 / d_v``, which is the ``d_v``-fold gap between them.
    """
    config = dict(config or {})
    bounded, short, long = is_bounded(method, config, d=d, dv=dv,
                                      bytes_per_el=bytes_per_el)
    knob = knob or (state_knobs(method, d=d, dv=dv, bytes_per_el=bytes_per_el) or [None])[0]
    kv = config.get(knob) if knob and knob != "d" else (d if knob == "d" else None)
    rq = reads_qk(method, config) if check_reads_qk else True
    if not bounded:
        return StateSpec(method=method, config=config, state_bytes=None, bounded=False,
                         knob=knob, knob_value=kv, rank_equivalent=None, reads_qk=rq,
                         note=f"state_bytes grows with the context ({short:g} B at "
                              f"n={_SHORT_N}, {long:g} B at n={_LONG_N}): the operator "
                              f"has no finite state, so prop:memory-rank has nothing "
                              f"to bound and no state size can be reported")
    b = state_bytes(method, config, n=n, d=d, dv=dv, bytes_per_el=bytes_per_el)
    return StateSpec(method=method, config=config, state_bytes=b, bounded=True,
                     knob=knob, knob_value=kv,
                     rank_equivalent=b / (dv * bytes_per_el), reads_qk=rq,
                     note=f"cost()['state_bytes'] at d={d}, dv={dv}, "
                          f"{bytes_per_el} B/element; flat in n")


def sweep_knob(method: str, value: int, *, d: int, knob: str | None = None) -> dict:
    """The config that puts ``method``'s state knob at ``value``.

    ``"d"`` is a *shape* and not a configuration key, so a method swept through ``d``
    gets an empty config and the caller varies the key dimension instead; this returns
    the dict to pass to ``ab.build`` either way, which is what lets the two studies run
    one code path over both families.
    """
    knob = knob or (state_knobs(method, d=d) or [None])[0]
    return {} if knob in (None, "d") else {knob: int(value)}


def describe_methods(methods, *, n: int, d: int, dv: int,
                     bytes_per_el: int = BYTES_PER_EL) -> dict:
    """One :class:`StateSpec` per method at its default configuration, as plain dicts.

    Written into the results file so the state definition a run used is in the run's own
    provenance rather than in a docstring that may have moved on.
    """
    out = {}
    for m in methods:
        out[m] = state_spec(m, {}, n=n, d=d, dv=dv, bytes_per_el=bytes_per_el).to_dict()
        out[m]["knobs_found"] = state_knobs(m, d=d, dv=dv, bytes_per_el=bytes_per_el)
    return out


# ======================================================================================
# applying the mapping: the extension sweep S6 and S9 share
# ======================================================================================

#: The recurrent methods the expansion added, grouped by the structure of their state.
#: Membership is the only hand-written thing in this module and it is a *roster*, not a
#: formula: what each one's state costs, which knob sizes it and whether it is bounded
#: are all read off ``cost()`` by the probes above.
EXT_GATED = ("retnet", "gla", "hgrn2", "rwkv", "mlstm")
EXT_LTI = ("s4d", "s5", "lru", "hyena")
EXT_HYBRID = ("mega",)
EXT_METHODS = EXT_GATED + EXT_LTI + EXT_HYBRID

#: Values each *knob* is swept over -- keyed by the knob :func:`state_knobs` finds and
#: not by the method name, so a method added later is swept by the ladder its own knob
#: implies rather than by an entry someone has to remember to add.  The three ladders
#: are chosen to cover one byte range: at ``d_v = 64`` in float64, ``state_rank`` of
#: 8-256 is 4 KB to 128 KB, ``n_state`` of 4-128 is 4 KB to 128 KB for ``s4d``'s
#: per-channel state and 64 B to 2 KB for ``s5``/``lru``'s shared one -- the ``d_v``-fold
#: gap, left visible rather than normalised away.
KNOB_LADDER = {
    "state_rank": (8, 16, 32, 64, 128, 256),
    "n_state": (4, 8, 16, 32, 64, 128),
    "chunk": (8, 16, 32, 64, 128, 256),
}

#: ``state_rank`` truncates the *key* dimension (``_gated.truncate``), so a key code
#: built in ``R^d`` and then cut to its first ``r`` coordinates is no longer the code the
#: construction built.  For that knob the study builds the keys in ``R^r`` instead, which
#: makes the truncation a no-op and keeps the construction identical to the one the
#: pre-registered operators run.  Every other knob leaves the key dimension alone.
KEY_DIM_FOLLOWS = ("state_rank",)


def accepts(method: str, key: str) -> bool:
    """Does ``method``'s constructor take this configuration key?

    ``build`` rejects an unknown key (``ConfigError``), so a study that passes
    ``feature=`` to every operator would crash on the ones that hard-code their feature
    map.  Asking is how the extension runs one code path over both.
    """
    return key in _int_parameters(method) or key in (
        inspect.signature(ab.get_class(method).__init__).parameters)


def ladder(method: str, *, d: int = 64, dv: int = 64) -> list[dict]:
    """The state ladder for one method: ``[{knob, knob_value, config, d}]``.

    One rung per state size, or a single rung with ``knob = None`` for a method whose
    state ``cost()`` does not respond to any configuration key -- or does not exist, in
    which case the rung is still run (the recall is still a measurement) and
    :func:`state_spec` records that there is no size to put it against.
    """
    knob = (state_knobs(method, d=d, dv=dv) or [None])[0]
    if knob in (None, "d") or knob not in KNOB_LADDER:
        return [{"knob": knob, "knob_value": None, "config": {}, "d": d}]
    return [{"knob": knob, "knob_value": int(v), "config": {knob: int(v)},
             "d": int(v) if knob in KEY_DIM_FOLLOWS else d}
            for v in KNOB_LADDER[knob]]


def ext_jobs(methods, n_grid, *, d: int = 64, dv: int = 64, **extra) -> list[dict]:
    """``part = "extended_methods"`` jobs: every method, every rung, every ``N``."""
    jobs = []
    for meth in methods:
        for rung in ladder(meth, d=d, dv=dv):
            for N in n_grid:
                jobs.append({"part": "extended_methods", "method": meth, "N": int(N),
                             "d": rung["d"], "dv": dv, "knob": rung["knob"],
                             "knob_value": rung["knob_value"], **extra})
    return jobs


def ext_config(job, *, feature: str | None = None) -> dict:
    """The ``ab.build`` config for one extension job, plus ``feature`` where it is taken."""
    kv = job.get("knob_value")
    cfg = {} if (kv is None or job.get("knob") in (None, "d")) else {job["knob"]: int(kv)}
    if feature is not None and accepts(job["method"], "feature"):
        cfg["feature"] = feature
    return cfg


def ext_metrics(method: str, config: dict, *, N: int, d: int, dv: int,
                recalls, margins, query_spread: float,
                bytes_per_el: int = BYTES_PER_EL) -> dict:
    """The metrics block every extension cell carries, state mapping included.

    ``query_spread`` is the largest absolute difference between the outputs two
    *different* queries produce on the same stored content.  Exactly ``0.0`` means the
    operator did not read the query: its recall is then whatever chance gives and is a
    structural zero, not a measured capacity limit.

    The state is sized at the *asymptotic* context (:data:`_SHORT_N`) and not at this
    cell's ``N``.  "The state this operator carries" is the steady-state number; a
    windowed state is ``min(window, n)``, so sizing it at ``N = 4`` would report a
    256-wide chunk as four positions wide and make the state size a function of the
    sweep's x axis.
    """
    import numpy as np
    spec = state_spec(method, config, n=_SHORT_N, d=d, dv=dv,
                      bytes_per_el=bytes_per_el, check_reads_qk=False)
    return {"recall": float(np.median(recalls)),
            "recall_q1": float(np.percentile(recalls, 25)),
            "recall_q3": float(np.percentile(recalls, 75)),
            "recall_seeds": list(recalls),
            "margin_median": float(np.median(margins)),
            "query_spread": float(query_spread),
            "query_independent": bool(query_spread == 0.0),
            "chance": 1.0 / N,
            "recall_minus_chance": float(np.median(recalls)) - 1.0 / N,
            "state_bytes": spec.state_bytes,
            "state_bounded": spec.bounded,
            "state_rank_equivalent": spec.rank_equivalent,
            "state_note": spec.note}


#: Verdicts :func:`classify` can return, in the order it tests them.
STRUCTURAL = "structural zero: query-independent"
UNBOUNDED = "no finite state"
NO_REGIME = "no working regime: below 0.5 at the smallest load"
MEASURED = "capacity measured"


def classify(rows, threshold: float = 0.5) -> dict:
    """What a method's extension rows are evidence *of*.

    Four outcomes, and keeping them apart is the point of classifying before reading
    any number off the rows.

    ``structural zero`` -- the operator returned bit-identical output for every query,
    so recall is chance by construction: ``prop:memory-rank``'s premise does not apply
    to it.  ``no finite state`` -- ``cost()`` says the state grows with the context, so
    there is nothing for the proposition to bound.  ``no working regime`` -- the
    operator read the query and has a bounded state, but recall is already below the
    threshold at the *smallest* ``N`` at every state size, so a "collapse point" would
    be the left edge of the grid rather than a measurement; the construction never
    worked for it.  ``capacity measured`` -- everything else: the collapse point is a
    measurement.
    """
    import numpy as np
    if not rows:
        return {"verdict": "no rows", "n_cells": 0}
    qi = all(r["query_independent"] for r in rows)
    bounded = all(r["state_bounded"] for r in rows)
    dev = max(abs(r["recall_minus_chance"]) for r in rows)
    nmin = min(r["N"] for r in rows)
    best_small = max((r["recall"] for r in rows if r["N"] == nmin), default=0.0)
    if qi:
        verdict = STRUCTURAL
    elif not bounded:
        verdict = UNBOUNDED
    elif best_small < threshold:
        verdict = NO_REGIME
    else:
        verdict = MEASURED
    return {"verdict": verdict, "n_cells": len(rows),
            "query_independent": qi, "state_bounded": bounded,
            "max_query_spread": max(r["query_spread"] for r in rows),
            "max_abs_recall_minus_chance": float(dev),
            "median_recall": float(np.median([r["recall"] for r in rows])),
            "best_recall_at_smallest_N": float(best_small),
            "smallest_N": int(nmin),
            "state_bytes": sorted({r["state_bytes"] for r in rows
                                   if r["state_bytes"] is not None}),
            "knob": rows[0].get("knob"),
            "knob_values": sorted({r["knob_value"] for r in rows
                                   if r.get("knob_value") is not None})}


def collapse_crossing(cells, threshold: float = 0.5) -> tuple:
    """``(smallest grid N below ``threshold``, log-interpolated crossing)``.

    The same estimator pair S9's pre-registered analysis uses, so the extension's
    collapse points are comparable with the pre-registered ones cell for cell.
    ``cells`` must be sorted by ``N``.
    """
    half = next((r["N"] for r in cells if r["recall"] < threshold), None)
    if half is None:
        return None, None
    below = next(r for r in cells if r["N"] == half)
    above = [r for r in cells if r["N"] < half]
    if not above:
        return half, float(half)
    a = above[-1]
    import math
    x0, x1 = math.log2(a["N"]), math.log2(below["N"])
    y0, y1 = a["recall"], below["recall"]
    t = 0.0 if y0 == y1 else (y0 - threshold) / (y0 - y1)
    return half, float(2.0 ** (x0 + t * (x1 - x0)))


def ext_collapse(mrows) -> dict:
    """``{knob value -> collapse block}`` for one extension operator.

    Keyed by the knob the mapping found for *that* operator, with the state size the
    knob buys carried in the block: bytes, and the rank an additive ``r x d_v`` memory
    would need to hold them.  An operator with no knob has one block under ``"None"``.
    """
    keys = sorted({r["knob_value"] for r in mrows if r.get("knob_value") is not None})
    out = {}
    for kv in (keys or [None]):
        cur = sorted([r for r in mrows if r.get("knob_value") == kv],
                     key=lambda r: r["N"])
        if not cur:
            continue
        half, interp = collapse_crossing(cur)
        rank = cur[0]["state_rank_equivalent"]
        out[str(kv)] = {
            "knob_value": kv, "state_bytes": cur[0]["state_bytes"],
            "state_rank_equivalent": rank,
            "recall_by_N": {str(r["N"]): r["recall"] for r in cur},
            "chance_by_N": {str(r["N"]): r["chance"] for r in cur},
            "N_at_recall_0.5": half, "N_at_recall_0.5_interp": interp,
            "N_over_S_at_collapse": (half / rank) if (half and rank) else None,
            "max_N_tested": max(r["N"] for r in cur), "censored": half is None,
            "max_query_spread": max(r["query_spread"] for r in cur)}
    return out


def extension_block(rows, *, dv: int) -> dict:
    """The labelled extension: what each added operator's rows are evidence of.

    Every operator gets a verdict from :func:`classify` *before* any number is read off
    it, because three different things produce a low recall here and only one of them
    is a capacity measurement: an operator that never read the query, an operator with
    no finite state to bound, and an operator whose bounded state ran out.
    """
    import numpy as np

    from . import _common as C
    per, fits = {}, {}
    for meth in sorted({r["method"] for r in rows}):
        mrows = [r for r in rows if r["method"] == meth]
        cls = classify(mrows)
        collapse = ext_collapse(mrows)
        pairs = [(v["state_rank_equivalent"], v["N_at_recall_0.5_interp"])
                 for v in collapse.values()
                 if v["N_at_recall_0.5_interp"] is not None
                 and v["state_rank_equivalent"]]
        fit = C.fit_loglog_slope([x for x, _ in pairs], [y for _, y in pairs])
        ys = [y for _, y in pairs]
        fit["collapse_span"] = (max(ys) / min(ys)) if len(ys) >= 2 else float("nan")
        fit["state_sizes"] = [x for x, _ in pairs]
        fit["collapse_values"] = ys
        per[meth] = {"classification": cls, "collapse": collapse, "scaling_fit": fit,
                     "state_knob": cls.get("knob")}
        if cls["verdict"] == MEASURED:
            fits[meth] = fit
    groups = {"gated": [m for m in EXT_GATED if m in per],
              "lti": [m for m in EXT_LTI if m in per],
              "hybrid": [m for m in EXT_HYBRID if m in per]}
    # Only an operator whose verdict is MEASURED may be drawn on a capacity axis.  A
    # structural zero or an operator with no working regime has a "collapse point" at
    # the left edge of the grid, and plotting that number beside a real one is exactly
    # the confusion this classification exists to prevent.
    measured_only = {g: [m for m in ms
                         if per[m]["classification"]["verdict"] == MEASURED]
                     for g, ms in groups.items()}
    structural = [m for m, v in per.items()
                  if v["classification"]["verdict"] == STRUCTURAL]
    unbounded = [m for m, v in per.items()
                 if not v["classification"]["state_bounded"]]
    loads = [b["N_over_S_at_collapse"] for m in fits
             for b in per[m]["collapse"].values()
             if b["N_over_S_at_collapse"] is not None]
    return {
        "note": "post-hoc and separately labelled, exactly as extended_N is: the "
                "operators the library expansion added, each swept through the knob "
                "that sizes ITS state.  studies/_state.py reads that size off "
                "cost()['state_bytes'] rather than assuming a d x d_v shape.  No "
                "pre-registered quantity uses any of it.",
        "by_method": per, "groups": groups, "groups_measured": measured_only,
        "structural_zero": {
            "methods": structural,
            "why": "linear and time-invariant in the value field: q and k are never "
                   "read, so the operator returns bit-identical output whichever item "
                   "is queried and recall is chance by construction.  That is "
                   "prop:memory-rank's premise being inapplicable, not a measured "
                   "capacity limit, and it does not move with the state size.",
            "max_query_spread": {m: per[m]["classification"]["max_query_spread"]
                                 for m in structural},
            "max_abs_recall_minus_chance": {
                m: per[m]["classification"]["max_abs_recall_minus_chance"]
                for m in structural}},
        "no_working_regime": {
            "methods": [m for m, v in per.items()
                        if v["classification"]["verdict"] == NO_REGIME],
            "why": "recall is below 0.5 at the smallest N on the grid at every state "
                   "size, so there is no collapse point to locate: the construction "
                   "never gave this operator a working regime to collapse out of.  "
                   "hgrn2 is the case that matters -- it has no key projection at all "
                   "(the key IS the forget gate, k_t = 1 - f_t), so a key code built "
                   "for orthogonality does not enter the operator as a key and the "
                   "premise of the construction, not of the proposition, fails.",
            "best_recall_at_smallest_N": {
                m: v["classification"]["best_recall_at_smallest_N"]
                for m, v in per.items()
                if v["classification"]["verdict"] == NO_REGIME}},
        "unbounded_state": {
            "methods": unbounded,
            "why": "cost()['state_bytes'] grows with the context, so there is no "
                   "state size to place the recall against and prop:memory-rank has "
                   "nothing to bound"},
        "measured": {
            "methods": sorted(fits),
            "slopes": {m: {"slope": f["slope"], "ci95": f["ci95"], "r2": f["r2"],
                           "n": f["n"], "collapse_span": f["collapse_span"]}
                       for m, f in fits.items()},
            "N_over_S_at_collapse": C.med_iqr(loads) | {"values": loads}},
        "state_map": describe_methods(sorted({r["method"] for r in rows}),
                                      n=max(r["N"] for r in rows), d=64, dv=dv),
    }
