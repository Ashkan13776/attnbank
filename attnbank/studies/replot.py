"""Re-derive every figure from the cached ``results/*.json`` — no sweep, no GPU.

Each study's ``make_figure``/``make_figures`` is a pure function of the records and the
summary that the study already wrote to ``results/<name>.json``.  This module feeds them
straight from that file, so the publication pass can iterate on the plots without
re-running a single measurement (``STUDIES.md``: the results file is the only place the
paper's numbers may come from).

Run::

    python -m attnbank.studies.replot            # all eight
    python -m attnbank.studies.replot S1 S5      # a subset
"""
from __future__ import annotations

import argparse
import importlib
import sys

from . import _harness as H

#: study tag -> (module, the records the figure function wants)
STUDIES = {
    "S1": "S1_matrix_vs_output",
    "S2": "S2_restriction",
    "S3": "S3_random_features",
    "S4": "S4_kv_quantization",
    "S5": "S5_latent_factorization",
    "S6": "S6_finite_state_recall",
    "S7": "S7_ot_tilt",
    "S8": "S8_roofline",
}


#: Studies whose saved records contain stale resume-cache slices.
#:
#: ``save_results`` writes ``sweep.records`` -- the resume cache -- not the job list
#: the run actually submitted, so a study whose layer grid changed between runs ships
#: records for slices its own ``analyse`` never saw.  S4 and S5 each carry one such
#: slice per real source (GPT-2 layer 6, Qwen layer 12), 21 and 4 records against 126
#: and 45 for a live slice.  That is enough to move a grid median in the third digit,
#: so a figure replotted without this filter would be drawn from one population and
#: annotated with statistics from another.  A live slice always carries the full job
#: grid for its source, which is the test below; it is the same completeness test S1's
#: ``analyse`` applies internally, so S1 is left to do its own.
#:
#: S1, S2, S3 and S7 were checked cell by cell against their summaries and need no
#: filter: S2's extra layer-6 slices are the oracle sub-study, which declares them.
STALE_SLICE_FILTER = ("S4", "S5")


def _drop_partial_slices(records: list[dict]) -> list[dict]:
    """Keep only slices carrying the full job grid for their source."""
    import collections
    count: dict[tuple, int] = collections.Counter()
    for r in records:
        j = r.get("job", {})
        if {"source", "layer", "passage"} <= set(j):
            count[(j["source"], j["layer"], j["passage"])] += 1
    full: dict[str, int] = {}
    for (src, _, _), n in count.items():
        full[src] = max(full.get(src, 0), n)
    keep = {k for k, n in count.items() if n == full[k[0]]}
    out = []
    for r in records:
        j = r.get("job", {})
        if not {"source", "layer", "passage"} <= set(j):
            out.append(r)
        elif (j["source"], j["layer"], j["passage"]) in keep:
            out.append(r)
    return out


def load(tag: str) -> tuple[object, list[dict], dict]:
    mod = importlib.import_module(f".{STUDIES[tag]}", package=__package__)
    payload = H.load_results(STUDIES[tag])
    recs = payload["records"]
    if tag in STALE_SLICE_FILTER:
        recs = _drop_partial_slices(recs)
        dropped = len(payload["records"]) - len(recs)
        if dropped:
            print(f"{tag}: dropped {dropped} stale cached records from slices that do "
                  f"not carry the full job grid", file=sys.stderr)
    return mod, recs, payload["meta"]["summary"]


def _part(records, **match) -> list[dict]:
    return [r for r in records
            if all(r.get("job", {}).get(k) == v for k, v in match.items())]


def replot(tag: str) -> list[str]:
    """Rebuild one study's figures from its results file."""
    mod, recs, summary = load(tag)
    if tag == "S2":
        return mod.make_figures(_part(recs, part="identity"),
                                _part(recs, part="selection"), summary)
    if tag == "S3":
        return mod.make_figure(_part(recs, part="scaling"),
                               _part(recs, part="logit_scale"), summary)
    if tag in ("S6", "S8"):
        return mod.make_figures(summary) if hasattr(mod, "make_figures") \
            else mod.make_figure(summary)
    return mod.make_figure(recs, summary)


def main(tags: list[str] | None = None) -> None:
    for tag in tags or list(STUDIES):
        try:
            for p in replot(tag):
                print(f"{tag}: {p}")
        except Exception as e:                                  # noqa: BLE001
            print(f"{tag}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tags", nargs="*", help="S1 ... S8; default all")
    main([t.upper() for t in ap.parse_args().tags])
