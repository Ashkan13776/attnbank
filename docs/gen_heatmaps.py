"""Compute the attention maps that ``docs/heatmaps.md`` draws, and write that page.

    python docs/gen_heatmaps.py

Why this page exists
--------------------
Every method in the bank satisfies invariant W, ``y == weights @ (v_hat or v)``, so
every one of them can hand you the measure it realises over context positions. The
taxonomy says each category changes exactly one object in ``y(q) = int v dpi_q``. An
``n x n`` image of ``weights`` is therefore a direct picture of the object the taxonomy
is about, and the signed difference against dense softmax is the distributional term of
the error decomposition (``attnbank.metrics.error_decomposition``) drawn position by
position.

What this generator does
------------------------
1. Captures real per-layer, per-head ``(Q, K, V)`` from GPT-2 with
   :func:`attnbank.data.get_activations`, which asserts at capture time that
   ``softmax(Q K^T / sqrt(d) + causal mask) V`` reproduces the model's own attention
   output. If the model or dataset cannot be obtained it falls back to
   :func:`attnbank.data.synthetic` and the page says so.
2. Runs every (method, config) in :data:`LADDERS` over that one ``(layer, head,
   passage)`` unit and keeps ``out.weights``.
3. Rejects any method whose realised weights are not a measure over positions
   (negative entries, or rows that do not sum to one). That is a declaration-level
   fact, not a failure: a method with ``row_stochastic = False`` may realise a signed
   or unnormalised kernel, which cannot be drawn on a scale that reads as probability
   mass. The rejected names are listed on the page.
4. Quantises to 8 bits and writes one greyscale PNG *atlas* per method -- all the
   frames of one config ladder side by side in a single image -- plus one atlas of the
   signed difference. The browser decodes the PNG and colour-maps it on a canvas, so
   one payload serves both themes.
5. Writes ``manifest.json`` with the codec parameters, the colour ramps, the ladders
   and, per frame, real numbers from :mod:`attnbank.metrics`.
6. Writes a pre-coloured static triptych (light and dark) for the no-JavaScript case,
   and writes ``docs/heatmaps.md``.

Codec
-----
``pi_hat`` is stored on a **logarithmic** scale, ``u = 255 * (log10(pi) + DECADES) /
DECADES`` clamped to ``[0, 255]``, with ``u = 0`` reserved for exact zero. Attention is
heavy-tailed -- the median non-zero entry of the captured head is ~3e-4 while the sink
column reaches ~1 -- so a linear 8-bit code would quantise the entire tail to the same
level. Four decades at 256 levels is 0.0157 decades per step, a 3.7% relative error,
below one step of any display ramp. The scale is **absolute and shared by every
method**, so two maps are directly comparable.

The difference is stored on a **symmetric logarithmic** scale spanning ``DIFF_DECADES``
below ``dmax``, with 128 the exact centre so zero lands on the neutral colour. It was
linear at first and that was wrong for the same reason a linear sequential scale is
wrong: with ``dmax`` shared across a ladder, one rung's single large cell sets the scale
and every small difference on every other rung collapses into one or two codes. The
measured consequence was a difference map for ``topk`` at ``k = 8`` that was visually
blank while its mean total variation was 0.14 -- a picture that contradicted the number
printed under it.

``dmax`` is shared across the whole config ladder of one method, which is what makes
stepping the knob show convergence: renormalising each frame would hide it. A frame
whose differences are a hundred times smaller than the ladder's worst occupies a third
of the scale and is visibly paler, which is the reading we want.

Storing both maps rather than deriving one from the other is deliberate. Deriving the
difference from a quantised ``pi_hat`` would turn the 3.7% codec error into visible
structure for methods whose true difference is zero -- ``flash`` reorganises the
arithmetic and changes nothing -- and a map that invents structure is worse than no map.

Determinism
-----------
Every method that takes a ``seed`` is left at its registered default, the capture is
seeded, floats in the manifest are rounded before serialisation, and PIL writes no
timestamp chunk. Two runs produce byte-identical output; ``--check`` asserts it.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import pathlib
import shutil
import sys
import warnings

import numpy as np
import torch
from PIL import Image

import attnbank as ab
from attnbank import data, metrics

HERE = pathlib.Path(__file__).resolve().parent
ASSETS = HERE / "assets" / "heatmaps"
PAGE = HERE / "heatmaps.md"

# --------------------------------------------------------------------------------------
# what is drawn
# --------------------------------------------------------------------------------------

#: Context length of the square map. 80 keeps a cell visible at the size the page draws
#: the panel (an 80-wide canvas upscaled ~3.5x with smoothing off, so a reader can pick
#: out and hover a single query row) and keeps the whole payload inside a few hundred kB.
#: The payload is dominated by n^2 x frames cells of genuinely high-entropy real data --
#: about 3.3 bits a cell after PNG, which is close to its actual entropy -- so n is the
#: one knob that moves the total, and 80 is where it stops being the thing to worry about.
N = 80

#: The unit the maps are computed on. Chosen by scoring every (layer, head, passage) of
#: the capture on three structures a reader should be able to see -- the attention sink
#: at position 0, the local band on the diagonal, and long-range mass away from both --
#: and taking a head that has all three. GPT-2 layer 5 head 3 has sink mass 0.35, band
#: mass 0.31 in the first four off-diagonals, and 0.22 further out.
MODEL, LAYER, HEAD, PASSAGE = "gpt2", 5, 3, 0

#: Decades of dynamic range kept by the sequential codec (see the module docstring).
DECADES = 4

#: Decades below ``dmax`` kept by the signed-difference codec. Three puts a difference
#: a thousandth of the ladder's worst cell at the first visible step away from zero,
#: which on this data is well below the quantisation error of the maps being compared.
DIFF_DECADES = 3

#: The method shown before anything is clicked, and the one drawn into the static
#: no-JavaScript triptych.
DEFAULT_METHOD, DEFAULT_FRAME = "topk", 3        # k = 8

#: A difference at or below this is the floating-point floor, not a difference. ``flash``
#: reassociates the arithmetic and lands 3e-16 from dense softmax; drawn on its own
#: symmetric scale that noise would fill the panel and read as a large error, which is
#: the one thing a difference map must never do. Such a method gets a blank panel and
#: the real number printed beside it instead.
ZERO_TOL = 1e-9

#: ``(method, knob, [values], fixed)`` per method, grouped by category. ``knob`` is the
#: name of the one configuration parameter the page steps; ``fixed`` pins the others
#: away from their defaults only where the default would make the map meaningless.
#:
#: Ladders marked "convergence" below are ``klass.convergence_configs()`` verbatim --
#: the library's own declared route to its limit -- so stepping the knob to the right
#: is the convergence anchor, drawn.
LADDERS: list[tuple[str, str | None, list, dict]] = [
    # exact: changes nothing but the order of the arithmetic
    ("softmax",        None,       [],                          {}),
    ("flash",          "block_n",  [8, 16, 32],                 {}),
    ("lowprec",        "bits",     [4, 5, 6, 8, 12],            {}),
    # restrict: changes the support of pi_q
    ("window",         "w",        [1, 2, 4, 8, 16, 32, 64],    {}),
    ("topk",           "k",        [1, 2, 4, 8, 16, 32, 64],    {}),
    ("sink_window",    "w",        [1, 2, 4, 8, 16, 32],        {"sinks": 4}),
    ("strided",        "stride",   [16, 8, 4, 2, 1],            {}),
    ("blocktopk",      "n_blocks", [1, 2, 4, 8, 16],            {"block": 4}),
    ("bigbird",        "window_blocks", [1, 3, 5, 7],           {"block": 4}),
    ("reformer",       "n_buckets", [16, 8, 4, 2, 1],           {}),
    ("lsh_sample",     "n_samples", [2, 4, 8, 14],              {}),   # convergence
    # interaction: changes the interaction g
    ("performer",      "r",        [8, 32, 128, 512],           {}),   # convergence
    ("based",          "degree",   [2, 4, 6, 8],                {}),   # convergence
    ("linear_elu",     None,       [],                          {}),
    ("cosformer",      None,       [],                          {}),
    # compress: changes the stored context (mu, k, v)
    ("quantkv",        "bits_k",   [1, 2, 3, 4, 8],             {}),
    ("kivi",           "bits_k",   [1, 2, 3, 4],                {}),
    ("qjl",            "sketch",   [8, 32, 128, 512],           {}),   # convergence
    ("evict_h2o",      "budget",   [4, 8, 16, 32, 64],          {}),
    ("evict_tova",     "budget",   [4, 8, 16, 32, 64],          {}),
    ("mla",            "r",        [4, 8, 16, 32, 64],          {}),
    ("merge_kv",       "n_clusters", [4, 8, 16, 32, 64],        {}),
    # factorize: changes the shape of pi_q
    ("abc",            "r",        [1, 2, 4, 8, 16, 32],        {}),
    ("latte",          "r",        [2, 4, 8, 16, 32],           {}),
    ("clustered",      "r",        [2, 4, 8, 16, 32],           {}),
    ("nmf_oracle",     "r",        [1, 2, 4, 8, 16],            {}),
    # coupling: changes the normaliser Z
    ("sinkhorn",       "n_iter",   [0, 1, 2, 5, 20],            {}),
    ("unbalanced_sinkhorn", "lam_col", [0.0, 0.3, 1.0, 3.0],    {}),
    ("dustbin_sinkhorn", "dustbin", [0.0, 0.5, 1.0, 2.0],       {}),
    # recurrent: changes the operator itself
    ("decay_rnn",      "decay",    [0.5, 0.8, 0.9, 0.95, 0.99, 1.0], {"normalize": True}),
    ("linear_rnn",     None,       [],                          {"normalize": True}),
    ("mega",           "chunk",    [2, 4, 8, 16, 32],           {}),
]

#: Methods probed at their registered defaults purely so the page can say, with the
#: check actually run rather than asserted, which parts of the bank a measure-valued
#: heat map cannot show. Every one of these declares ``row_stochastic = False`` and
#: realises a signed or unnormalised kernel: the recurrent gates can drive a weight
#: negative, and the trigonometric and Nystrom feature maps are not positivity
#: preserving. ``is_measure`` decides; nothing here is hand-classified.
PROBE = ["retnet", "rwkv", "gla", "hgrn2", "mlstm", "delta_rnn",
         "trigrff", "polysketch", "nystrom", "skyformer", "linformer", "svd_oracle"]

#: ``bits_k`` and ``bits_v`` are one knob on this page: a cache quantised to 4 bits is
#: quantised to 4 bits in both tensors, and sweeping them apart is a different study.
MIRROR = {"bits_k": "bits_v"}

#: The strip on the landing page. One panel per category, chosen for a mechanism that
#: is legible at a glance rather than for being representative: a band, a smear, a set
#: of low-rank stripes, a full square where every other panel is a triangle, a chunked
#: decay. Each is (method, frame), and the frame is a rung of that method's own ladder
#: mild enough that the picture is still an approximation of this head rather than a
#: degenerate one. ``tools/check_sync.py`` holds the landing page's markup to this list,
#: so a method leaving the bank cannot leave a dead panel behind.
TEASER = [
    ("softmax", 0),        # the reference: dense, with this head's sinks and bands
    ("linear_elu", 0),     # interaction -- the surrogate smears the measure
    ("window", 3),         # restrict -- w = 8, the support is a band
    ("abc", 3),            # factorize -- r = 8 caps the rank, and it stripes
    ("sinkhorn", 3),       # coupling -- no causal version exists, so it is a square
    ("mega", 3),           # recurrent -- chunk = 16, a different operator
]


CATEGORY_ORDER = ["exact", "interaction", "restrict", "compress",
                  "factorize", "coupling", "recurrent"]

# --------------------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------------------
#
# A heat map needs a *sequential* scale and none of the seven category hues is one: they
# are a nominal set, chosen to be told apart, not to be ordered. The ramp here is magma,
# which is perceptually uniform and monotonic in lightness, so a step in mass is the same
# visual step everywhere on the scale and the map survives greyscale and the common forms
# of colour blindness. It is oriented so that **no mass is the page ground and full mass
# is ink**: an attention matrix is mostly small, and a ramp that painted small values
# dark would turn every panel into a black rectangle. The bottom of the ramp is blended
# into the exact page colour over the first 8% so that zero disappears into the page
# instead of sitting on it as a pale yellow (light) or a near-black that is not quite the
# ground (dark).
MAGMA = [(0, 0, 4), (10, 8, 34), (29, 17, 71), (54, 16, 107), (81, 18, 124),
         (106, 28, 129), (131, 38, 129), (156, 46, 127), (183, 55, 121),
         (208, 65, 111), (231, 82, 99), (245, 107, 92), (252, 137, 97),
         (254, 167, 114), (254, 196, 136), (253, 226, 163), (252, 253, 191)]

#: Page grounds, from ``docs/stylesheets/tokens.css`` (``--md-default-bg-color``).
GROUND = {"light": (252, 252, 251), "dark": (18, 22, 29)}

#: The signed difference needs a *diverging* scale with a true zero, and zero must not be
#: a colour: a reader has to be able to see at a glance where a method left the measure
#: alone. Zero is therefore the page ground exactly, and the two arms are the house
#: diverging hues from ``attnbank.plotting.style.DIVERGING`` -- blue for mass the method
#: removed, red for mass it added -- lifted for the dark ground, same hue and same order,
#: which is the licence the dark theme already takes with the category hues.
DIVERGING_ARMS = {
    "light": {"neg": [(252, 252, 251), (134, 182, 239), (42, 120, 214), (13, 54, 107)],
              "pos": [(252, 252, 251), (240, 163, 162), (227, 73, 72), (143, 31, 31)]},
    "dark":  {"neg": [(18, 22, 29), (42, 120, 214), (104, 168, 242), (190, 220, 252)],
              "pos": [(18, 22, 29), (227, 73, 72), (240, 112, 111), (252, 200, 199)]},
}


def _ramp(stops: list[tuple[int, int, int]], n: int = 256) -> list[tuple[int, int, int]]:
    """Resample a list of RGB stops to ``n`` evenly spaced entries, linearly in sRGB."""
    out = []
    for i in range(n):
        t = i / (n - 1) * (len(stops) - 1)
        lo = min(int(t), len(stops) - 2)
        f = t - lo
        a, b = stops[lo], stops[lo + 1]
        out.append(tuple(int(round(a[c] + (b[c] - a[c]) * f)) for c in range(3)))
    return out


#: Fraction of the sequential scale over which the low end is blended into the page
#: ground, so that "no mass" is the page itself rather than magma's pale yellow (light)
#: or a near-black that is not quite the ground (dark).
BLEND = 0.08


def sequential_stops(theme: str) -> list[tuple[int, int, int]]:
    """Magma oriented low-to-ground, as stops, with the ground spliced onto the bottom."""
    stops = list(MAGMA) if theme == "dark" else list(reversed(MAGMA))
    inner = _ramp(stops, 256)[int(round(BLEND * 255))]
    keep = [s for i, s in enumerate(stops) if i / (len(stops) - 1) > BLEND]
    return [GROUND[theme], inner] + keep


def sequential_ramp(theme: str) -> list[tuple[int, int, int]]:
    """The 256-entry sequential ramp, resampled from :func:`sequential_stops`.

    The first two stops cover ``BLEND`` of the scale and the rest share what is left, so
    the piecewise-linear resample has to be told the stop positions rather than assuming
    they are evenly spaced.
    """
    stops = sequential_stops(theme)
    head = _ramp(stops[:2], int(round(BLEND * 255)) + 1)
    tail = _ramp(stops[1:], 256 - len(head) + 1)
    return head + tail[1:]


def diverging_ramp(theme: str) -> list[tuple[int, int, int]]:
    """256 entries, index 128 exactly the page ground, arms leaving it in both directions."""
    arms = DIVERGING_ARMS[theme]
    neg = _ramp(arms["neg"], 129)          # index 0 = ground .. 128 = full negative
    pos = _ramp(arms["pos"], 128)
    out = [None] * 256
    for i in range(129):
        out[128 - i] = neg[i]
    for i in range(128):
        out[128 + i] = pos[i]
    return [tuple(c) for c in out]


# --------------------------------------------------------------------------------------
# codec
# --------------------------------------------------------------------------------------

def encode_pi(w: np.ndarray) -> np.ndarray:
    """Quantise a non-negative measure to the shared logarithmic 8-bit scale."""
    floor = 10.0 ** -DECADES
    g = (np.log10(np.clip(w, floor, 1.0)) + DECADES) / DECADES
    u = np.round(np.clip(g, 0.0, 1.0) * 255.0)
    return np.where(w <= floor, 0, np.maximum(u, 1)).astype(np.uint8)


def encode_delta(d: np.ndarray, dmax: float) -> np.ndarray:
    """Quantise a signed difference to a symmetric logarithmic 8-bit scale.

    128 is exactly zero. Magnitudes are placed by their logarithm over
    ``DIFF_DECADES`` below ``dmax``; anything smaller than that floor rounds to 128 and
    is drawn as untouched, which is the honest reading -- it is below the resolution of
    the quantised maps being differenced.
    """
    if dmax <= 0:
        return np.full(d.shape, 128, dtype=np.uint8)
    floor = dmax * 10.0 ** -DIFF_DECADES
    mag = (np.log10(np.clip(np.abs(d), floor, dmax)) - np.log10(floor)) / DIFF_DECADES
    mag = np.where(np.abs(d) <= floor, 0.0, np.clip(mag, 0.0, 1.0))
    return np.round(128.0 + 127.0 * np.sign(d) * mag).astype(np.uint8)


def write_png(path: pathlib.Path, a: np.ndarray) -> int:
    """Write a greyscale PNG with no ancillary chunks, and return its size in bytes."""
    buf = io.BytesIO()
    Image.fromarray(a, "L").save(buf, format="PNG", optimize=True, compress_level=9)
    raw = buf.getvalue()
    path.write_bytes(raw)
    return len(raw)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def load_unit() -> tuple[object, dict]:
    """The one ``[1, 1, N, d]`` unit every map is computed on, and its provenance.

    Real activations if the model and dataset can be obtained, the seeded Gaussian
    control otherwise. The page states which, because a heat map of Gaussian noise and
    a heat map of GPT-2 are not the same picture and a caption that confused them would
    be the page's one invented fact.
    """
    try:
        batch = next(data.get_activations(MODEL, layers=[LAYER], n_tokens=256,
                                          n_passages=8, seed=0, verify=True))
        batch = batch.select(n_tokens=N)
        _, unit = next(iter(batch.units(heads=[HEAD], passages=[PASSAGE])))
        logit_max = float(((unit.q[0, 0] @ unit.k[0, 0].T) * unit.scale).max())
        import numpy as _np
        _w = ab.build("softmax")(unit.q, unit.k, unit.v, causal=True,
                                 return_weights=True).weights[0, 0].double().numpy()
        _nz = _w[_np.tril(_np.ones_like(_w)) > 0]
        prov = {"kind": "real", "logit_max": logit_max,
                "pi_max": float(_w.max()), "pi_median": float(_np.median(_nz)),
                "pi_decades": round(math.log10(float(_w.max())
                                               / float(_np.median(_nz)))),
                "dataset_name": str(batch.meta.get("dataset_name", "")),
                "model": MODEL, "repo": batch.meta.get("model_repo", ""),
                "dataset": batch.meta.get("dataset_path", ""),
                "layer": LAYER, "head": HEAD, "passage": PASSAGE,
                "n": N, "d": int(unit.q.shape[-1])}
        return unit, prov
    except Exception as exc:                                  # noqa: BLE001 - any failure
        warnings.warn(f"real capture unavailable ({type(exc).__name__}: {exc}); "
                      f"falling back to synthetic()", stacklevel=2)
        batch = data.synthetic(n=N, d=64, heads=1, passages=1, seed=0)
        _, unit = next(iter(batch.units()))
        logit_max = float(((unit.q[0, 0] @ unit.k[0, 0].T) * unit.scale).max())
        import numpy as _np
        _w = ab.build("softmax")(unit.q, unit.k, unit.v, causal=True,
                                 return_weights=True).weights[0, 0].double().numpy()
        _nz = _w[_np.tril(_np.ones_like(_w)) > 0]
        prov = {"kind": "synthetic", "logit_max": logit_max,
                "pi_max": float(_w.max()), "pi_median": float(_np.median(_nz)),
                "pi_decades": round(math.log10(float(_w.max())
                                               / float(_np.median(_nz)))),
                "dataset_name": "",
                "model": "", "repo": "", "dataset": "",
                "layer": 0, "head": 0, "passage": 0, "n": N, "d": 64}
        return unit, prov


def configs_for(knob: str | None, values: list, fixed: dict) -> list[dict]:
    """The config dict per frame, with ``MIRROR`` knobs moved together."""
    if knob is None:
        return [dict(fixed)]
    out = []
    for v in values:
        cfg = dict(fixed)
        cfg[knob] = v
        if knob in MIRROR:
            cfg[MIRROR[knob]] = v
        out.append(cfg)
    return out


def is_measure(w: np.ndarray, causal: bool) -> tuple[bool, str]:
    """Is ``w`` a probability measure over context positions, row by row?"""
    if not np.isfinite(w).all():
        return False, "non-finite entries"
    if w.min() < -1e-9:
        return False, f"negative entries (min {w.min():.3g})"
    rows = w.sum(axis=1)
    if np.abs(rows - 1.0).max() > 1e-6:
        return False, f"rows do not sum to one (range {rows.min():.4g}..{rows.max():.4g})"
    if causal and np.abs(np.triu(w, 1)).max() > 1e-12:
        return False, "mass above the diagonal under a causal mask"
    return True, ""


def _round(x: float, nd: int = 6) -> float:
    """Round for the manifest. Fixed precision keeps two runs byte-identical."""
    if not math.isfinite(x):
        return 0.0
    return float(f"{x:.{nd}g}")


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------

def build(out_dir: pathlib.Path) -> dict:
    """Compute every map, write every asset, and return the manifest."""
    unit, prov = load_unit()
    table = {r["name"]: r for r in ab.method_table()}
    q, k, v = unit.q, unit.k, unit.v

    refs = {}
    for mask, causal in (("causal", True), ("full", False)):
        out = ab.build("softmax")(q, k, v, causal=causal, return_weights=True)
        refs[mask] = {"pi": out.weights[0, 0].double().numpy(), "y": out.y[0, 0].double()}

    methods, rejected, sizes = [], [], {}
    for name, knob, values, fixed in LADDERS:
        row = table[name]
        causal = bool(row["causal"])
        mask = "causal" if causal else "full"
        pi = refs[mask]["pi"]
        y_ref = refs[mask]["y"]

        frames, mats, bad = [], [], None
        for cfg in configs_for(knob, values, fixed):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = ab.build(name, **cfg)(q, k, v, causal=causal, return_weights=True)
            w = out.weights[0, 0].double().numpy()
            ok, why = is_measure(w, causal)
            if not ok:
                bad = f"{why}, at {cfg or 'its registered default'}"
                break
            v_hat = out.v_hat[0, 0].double() if out.v_hat is not None else None
            dec = metrics.error_decomposition(
                torch.from_numpy(pi), torch.from_numpy(w), v[0, 0].double(), v_hat)
            tv = metrics.total_variation(torch.from_numpy(pi), torch.from_numpy(w))
            frames.append({
                "config": {kk: (float(vv) if isinstance(vv, float) else vv)
                           for kk, vv in cfg.items()},
                "label": ("" if knob is None else
                          (f"{values[len(frames)]:g}" if isinstance(values[len(frames)], float)
                           else str(values[len(frames)]))),
                "tv": _round(float(tv.mean())),
                "tv_rows": base64.b64encode(
                    np.round(np.clip(tv.numpy(), 0, 1) * 255).astype(np.uint8).tobytes()
                ).decode(),
                "dist": _round(float(dec["distributional"].mean())),
                "value": _round(float(dec["value"].mean())),
                "rel": _round(float(metrics.output_error(
                    y_ref, out.y[0, 0].double(), relative=True).mean())),
                "supp": _round(float(metrics.effective_support(
                    torch.from_numpy(w)).mean()), 4),
                "nnz": _round(float((w > 10.0 ** -DECADES).sum(axis=1).mean()), 4),
            })
            mats.append(w)

        if bad is not None:
            rejected.append({"name": name, "category": row["category"], "why": bad,
                             "declared": bool(row["row_stochastic"])})
            continue

        deltas = [m - pi for m in mats]
        dmax = max(float(np.abs(d).max()) for d in deltas)
        zero = dmax <= ZERO_TOL

        pi_atlas = np.concatenate([encode_pi(m) for m in mats], axis=1)
        sizes[f"m-{name}.png"] = write_png(out_dir / f"m-{name}.png", pi_atlas)
        if not zero:
            d_atlas = np.concatenate([encode_delta(d, dmax) for d in deltas], axis=1)
            sizes[f"d-{name}.png"] = write_png(out_dir / f"d-{name}.png", d_atlas)

        methods.append({
            "name": name, "category": row["category"], "objective": row["objective"],
            "causal": causal, "mask": mask, "knob": knob, "dmax": _round(dmax),
            "zero_delta": zero, "frames": frames,
            "summary": ab.describe(name).summary,
        })

    for name in PROBE:
        row = table[name]
        causal = bool(row["causal"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = ab.build(name)(q, k, v, causal=causal, return_weights=True)
            ok, why = is_measure(out.weights[0, 0].double().numpy(), causal)
        except Exception as exc:                              # noqa: BLE001
            ok, why = False, f"{type(exc).__name__}: {exc}"
        if ok:
            raise AssertionError(
                f"{name} is in PROBE but its weights are a measure; move it to LADDERS "
                f"rather than telling the reader it cannot be drawn")
        rejected.append({"name": name, "category": row["category"], "why": why,
                         "declared": bool(row["row_stochastic"])})

    for mask in ("causal", "full"):
        sizes[f"ref-{mask}.png"] = write_png(out_dir / f"ref-{mask}.png",
                                             encode_pi(refs[mask]["pi"]))

    static = write_static(out_dir, refs, methods)
    sizes.update(static)
    sizes.update(write_teaser(out_dir, methods))

    manifest = {
        "n": N, "decades": DECADES, "diff_decades": DIFF_DECADES, "provenance": prov,
        "default": {"method": DEFAULT_METHOD, "frame": DEFAULT_FRAME},
        "categories": CATEGORY_ORDER,
        # The stops, not the resampled 256-entry ramp: heatmap.js runs the same
        # linear sRGB resample, so the canvas and the static PNGs are the same colours
        # from the same numbers, at a twelfth of the bytes.
        "ramps": {
            theme: {"seq": [list(c) for c in sequential_stops(theme)],
                    "neg": [list(c) for c in DIVERGING_ARMS[theme]["neg"]],
                    "pos": [list(c) for c in DIVERGING_ARMS[theme]["pos"]]}
            for theme in ("light", "dark")
        },
        "methods": methods, "rejected": rejected,
        "bytes": dict(sorted(sizes.items())),
    }
    text = json.dumps(manifest, sort_keys=False, separators=(",", ":"))
    (out_dir / "manifest.json").write_text(text, encoding="utf-8")
    manifest["bytes"]["manifest.json"] = len(text.encode())
    return manifest


def write_teaser(out_dir: pathlib.Path, methods: list) -> dict:
    """One small pre-coloured panel per entry in ``TEASER``, for the landing page.

    Separate files rather than one strip, so the landing page can lay them out in a
    grid that wraps on a narrow screen and label each one in real text instead of
    baking words into an image. Each is the method's own atlas frame, so the panel is
    the same data the full page draws, at one device pixel per matrix cell.
    """
    by_name = {m["name"]: m for m in methods}
    sizes = {}
    for theme in ("light", "dark"):
        seq = np.array(sequential_ramp(theme), dtype=np.uint8)
        rule = np.array(GROUND["light" if theme == "dark" else "dark"], dtype=np.uint8)
        for name, frame in TEASER:
            if name not in by_name:                       # the bank changed under us
                raise SystemExit(f"TEASER names {name!r}, which is not drawn here")
            atlas = np.array(Image.open(out_dir / f"m-{name}.png"))
            cell = atlas[:, frame * N:(frame + 1) * N]
            panel = np.pad(seq[cell], ((1, 1), (1, 1), (0, 0)), constant_values=0)
            panel[0, :], panel[-1, :], panel[:, 0], panel[:, -1] = rule, rule, rule, rule
            buf = io.BytesIO()
            Image.fromarray(panel, "RGB").save(buf, format="PNG", optimize=True,
                                               compress_level=9)
            key = f"teaser-{theme}-{name}.png"
            (out_dir / key).write_bytes(buf.getvalue())
            sizes[key] = len(buf.getvalue())
    return sizes


def write_static(out_dir: pathlib.Path, refs: dict, methods: list) -> dict:
    """The pre-coloured triptych the page shows before -- and without -- JavaScript."""
    spec = next(m for m in methods if m["name"] == DEFAULT_METHOD)
    pi = refs[spec["mask"]]["pi"]
    atlas = np.array(Image.open(out_dir / f"m-{DEFAULT_METHOD}.png"))
    hat = atlas[:, DEFAULT_FRAME * N:(DEFAULT_FRAME + 1) * N]
    datlas = np.array(Image.open(out_dir / f"d-{DEFAULT_METHOD}.png"))
    delta = datlas[:, DEFAULT_FRAME * N:(DEFAULT_FRAME + 1) * N]

    sizes = {}
    for theme in ("light", "dark"):
        seq = np.array(sequential_ramp(theme), dtype=np.uint8)
        div = np.array(diverging_ramp(theme), dtype=np.uint8)
        rule = np.array(GROUND["light" if theme == "dark" else "dark"], dtype=np.uint8)
        gap = np.tile(np.array(GROUND[theme], dtype=np.uint8), (N + 2, 10, 1))
        panels = [seq[hat], seq[encode_pi(pi)], div[delta]]
        panels = [np.pad(p, ((1, 1), (1, 1), (0, 0)), constant_values=0) for p in panels]
        for p in panels:
            p[0, :], p[-1, :], p[:, 0], p[:, -1] = rule, rule, rule, rule
        img = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1)
        buf = io.BytesIO()
        Image.fromarray(img, "RGB").save(buf, format="PNG", optimize=True,
                                         compress_level=9)
        name = f"static-{theme}.png"
        (out_dir / name).write_bytes(buf.getvalue())
        sizes[name] = len(buf.getvalue())
    return sizes


# --------------------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------------------

def _source_sentence(prov: dict) -> str:
    if prov["kind"] != "real":
        return (
            "The numbers on this page come from "
            "`attnbank.data.synthetic`, the seeded Gaussian control "
            f"(`q, k ~ N(0, I_d)`, `v ~ N(0, I_dv)`, `n = {prov['n']}`, `d = {prov['d']}`): "
            "the pretrained model could not be obtained when this page was built. "
            "Real attention is heavy-tailed and structured and Gaussian scores are not, "
            "so read the shapes here as a control, not as what a language model does."
        )
    return (
        f"The numbers on this page are **real activations**: layer {prov['layer']}, "
        f"head {prov['head']} of **{prov['model']}** (`{prov['repo']}`) reading one "
        f"{prov['n']}-token passage of `{prov['dataset_name'] or prov['dataset']}`. "
        "`attnbank.data` takes `(Q, K, V)` at exactly the point the model "
        "computes attention and asserts at capture time that "
        "`softmax(QKᵀ/√d + causal mask) V` reproduces the model's own attention output, "
        "so the dense map below is the one this head actually used."
    )


def _rejected_sentence(rejected: list) -> str:
    if not rejected:
        return ""
    by = {}
    for r in rejected:
        by.setdefault(r["category"], []).append(r["name"])
    parts = _names([n for cat in CATEGORY_ORDER for n in sorted(by.get(cat, []))])
    example = min(rejected, key=lambda r: r["name"])
    return (
        f"{parts} are in the bank and are not drawn here. Each realises weights that are "
        "not a measure over context positions — `{}` has {} — so a sequential scale that "
        "reads as probability mass cannot honestly be put behind them, and a signed "
        "measure needs a different picture than this page draws. That is not a defect: "
        f"every one of the {len(rejected)} declares `row_stochastic = false` in the "
        "registry, and a method with objective `modify` is under no obligation to "
        "produce a distribution. The generator runs the check on every matrix and will "
        "not draw one that fails it."
    ).format(example["name"], example["why"])


def _names(xs: list[str]) -> str:
    xs = [f"`{x}`" for x in xs]
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def _convergence_sentences(methods: list, logit_max: float) -> str:
    """Say what each ladder actually does -- read off the measurements, not written down.

    This paragraph used to name the methods whose ladder is ``convergence_configs()``
    and claim that stepping it walks the map onto softmax. On this head that claim is
    false for the random-feature methods and the generator's own numbers say so, which
    is exactly the case "nothing is invented" exists to catch. So the sentences are
    classified from the measured total variation instead.
    """
    arrives, departs, shrinks, flat = [], [], [], []
    for m in methods:
        tvs = [f["tv"] for f in m["frames"]]
        if len(tvs) < 2 or m["zero_delta"]:
            continue                      # the exact category gets its own sentence
        if tvs[0] <= 0.01 < tvs[-1]:
            departs.append(m["name"])
        elif tvs[-1] <= 0.01:
            arrives.append(m["name"])
        elif tvs[-1] <= 0.5 * tvs[0]:
            shrinks.append(m["name"])
        elif tvs[-1] >= 0.9 * tvs[0]:
            flat.append((m["name"], tvs[0], tvs[-1]))
    out = []
    if arrives:
        out.append(
            f"**Step the knob and some ladders arrive.** {_names(arrives)} walk their "
            "map onto dense softmax: the difference panel fades to its zero colour and "
            "the mean total variation reaches zero at the right-hand end, because the "
            "last rung of each of those ladders *is* the dense operator — all of the "
            "support, every bucket, full precision. That is the convergence anchor, "
            "drawn.")
    if departs:
        out.append(
            f"**Two ladders run the other way.** {_names(departs)} *start* at dense "
            "softmax — the first rung is each method's declared `limit_config`, zero "
            "Sinkhorn iterations and no column penalty — and walk away from it as the "
            "knob turns. These are `coupling` methods with objective `modify`: they do "
            "not approximate softmax, they replace its normaliser, and the difference "
            "map is a picture of what balancing the columns does to the measure rather "
            "than of an error.")
    if shrinks:
        out.append(
            f"**Some get closer without arriving.** {_names(shrinks)} more than halve "
            "their total variation across the ladder but do not reach zero within the "
            "range this page can draw.")
    if flat:
        names = _names([n for n, _, _ in flat])
        worst = max(flat, key=lambda t: t[2])
        out.append(
            f"**And some do not move at all.** {names} end the ladder no better than "
            f"they started — `{worst[0]}` goes from {worst[1]:.2f} to {worst[2]:.2f} "
            "total variation. For the random-feature and polynomial methods among them "
            "that is not a bug and not a bad ladder: the largest logit on this head is "
            f"{logit_max:.1f}, the variance of the estimator grows with it, and the "
            "largest rung drawn here is nowhere near enough to resolve `exp` at that "
            "scale. A page that showed these converging would be showing you something "
            "that did not happen.")
    return "\n\n".join(out)


def _exact_sentence(manifest: dict) -> str:
    """The exact category, named and quantified from the measurements."""
    rows = [m for m in manifest["methods"]
            if m["category"] == "exact" and m["zero_delta"]]
    biggest = max((m["dmax"] for m in rows), default=0.0)
    names = _names([m["name"] for m in rows])
    return (
        f"{names} are in the `exact` category: they change the order of the arithmetic "
        "and nothing else. Across every configuration drawn here the largest difference "
        f"from dense softmax anywhere in the matrix is {biggest:.1e} — the floating-point "
        "floor — so their difference panel is blank by construction rather than by "
        "rounding. Painting that noise on a scale normalised to its own maximum would "
        "make the panel look like a large error, which is the one thing a difference map "
        "must never do.")


def page(manifest: dict) -> str:
    prov = manifest["provenance"]
    n = manifest["n"]
    # the teaser panels belong to the landing page, not this one, so they are
    # not part of what this page costs a reader
    total = sum(v for k, v in manifest["bytes"].items()
                if not k.startswith("teaser-"))
    spec = next(m for m in manifest["methods"] if m["name"] == DEFAULT_METHOD)
    frame = spec["frames"][DEFAULT_FRAME]
    k = frame["config"]["k"]
    n_methods = len(manifest["methods"])
    n_frames = sum(len(m["frames"]) for m in manifest["methods"])

    opts = []
    for cat in CATEGORY_ORDER:
        rows = [m for m in manifest["methods"] if m["category"] == cat]
        if not rows:
            continue
        inner = "".join(
            f'<option value="{m["name"]}"'
            f'{" selected" if m["name"] == DEFAULT_METHOD else ""}>{m["name"]}</option>'
            for m in rows)
        opts.append(f'<optgroup label="{cat}">{inner}</optgroup>')
    select = "".join(opts)

    chips = "".join(
        f'<button type="button" class="ab-hm-cat" data-cat="{c}" data-value="{c}" '
        f'aria-pressed="false">{c}</button>' for c in CATEGORY_ORDER)

    return f"""---
hide:
  - toc
---
<!-- Generated by docs/gen_heatmaps.py. Do not edit; change the generator. -->

# Attention maps

$$y(q) = \\int v\\,\\mathrm{{d}}\\pi_q$$

Every method in the bank satisfies invariant **W**, `y == weights @ (v_hat or v)`, so
every one of them can hand you the measure $\\hat\\pi_q$ it realises over context
positions. The taxonomy says each category changes exactly one object in that integral.
An $n\\times n$ image of `weights` is a picture of the object the taxonomy is about, and
the signed difference against dense softmax is the **distributional term of the error
decomposition**, drawn position by position: it shows where the mass went instead.

{_source_sentence(prov)}

<div class="ab-hm" id="ab-heatmap"
     data-src="../assets/heatmaps/"
     data-manifest="../assets/heatmaps/manifest.json">

<figure class="ab-hm-static">
  <img src="../assets/heatmaps/static-light.png" class="ab-hm-static-light"
       width="{3 * (n + 2) + 20}" height="{n + 2}"
       alt="Three {n} by {n} attention maps side by side. Left: the measure realised by
            topk at k={k}, a sparse scatter of bright cells below the diagonal. Centre:
            dense softmax on the same head, showing a bright first column, a bright band
            along the diagonal and faint structure elsewhere. Right: the signed
            difference, blue everywhere topk removed mass and red on the few cells it
            concentrated mass onto.">
  <img src="../assets/heatmaps/static-dark.png" class="ab-hm-static-dark"
       width="{3 * (n + 2) + 20}" height="{n + 2}" alt="" aria-hidden="true">
  <figcaption>
    <b>`{DEFAULT_METHOD}`, <code>k = {k}</code></b> against dense softmax, on the head
    named above. Left, the measure <i>&#960;&#770;<sub>q</sub></i> the method realises;
    centre, dense softmax <i>&#960;<sub>q</sub></i>; right, the signed difference
    <i>&#960;&#770; &#8722; &#960;</i>, blue where mass was removed and red where it was
    added. Rows are queries, columns are keys, and the upper triangle is empty because
    the head is causal. Mean total variation
    <b>{frame['tv']:.3f}</b>; relative output error <b>{frame['rel']:.3f}</b>;
    largest single-cell difference <b>{spec['dmax']:.3f}</b>.
  </figcaption>
</figure>

<div class="ab-hm-app" hidden>
  <div class="ab-hm-controls">
    <div class="ab-hm-cats" role="group" aria-label="Filter methods by category">
      {chips}
    </div>
    <div class="ab-hm-pick">
      <label for="ab-hm-method">Method</label>
      <select id="ab-hm-method">{select}</select>
      <p class="ab-hm-summary" id="ab-hm-summary"></p>
    </div>
    <div class="ab-hm-knob">
      <label for="ab-hm-config" id="ab-hm-knoblabel">Configuration</label>
      <input type="range" id="ab-hm-config" min="0" max="0" step="1" value="0">
      <output for="ab-hm-config" id="ab-hm-value"></output>
    </div>
  </div>

  <div class="ab-hm-maps" id="ab-hm-maps"></div>
  <div class="ab-hm-profile" id="ab-hm-profile"></div>
  <table class="ab-hm-stats" id="ab-hm-stats"></table>
</div>
</div>

## How to read it

**Rows are queries, columns are keys.** Row $i$ is the measure $\\hat\\pi_{{q_i}}$ that
query $i$ puts over the {n} context positions; it sums to one. Under a causal mask the
upper triangle is empty, so every map is a lower triangle.

**The sequential scale is logarithmic over four decades**, shared by every map on the
page, so two methods are directly comparable and a cell that is bright here is bright
there. Attention is heavy-tailed: on this head the largest weight is
{prov["pi_max"]:.2f} and the median weight that is not structurally zero is
{prov["pi_median"]:.1e}, {prov["pi_decades"]:.0f} orders of magnitude below it, so a
linear 8-bit scale would flatten the entire tail to one shade. The ramp is magma: perceptually uniform, monotonic in lightness, and
deliberately *not* one of the seven category hues, which are a nominal set for telling
categories apart and carry no order.

**The difference scale is diverging with a true zero.** Zero is the page ground, not a
colour, so untouched mass is invisible and only what a method changed is drawn. Blue is
mass the method removed, red is mass it added. Each arm is logarithmic over
{DIFF_DECADES} decades below $\\mathrm{{d}}_{{\\max}}$, the largest single-cell difference
anywhere in that method's ladder, which is printed under the panel and is **shared
across the whole ladder** — otherwise every frame would renormalise against itself and
convergence would be invisible. A linear arm was tried first and could not resolve a
difference of 0.005 against a ladder whose worst cell is 0.9, which made maps look
blank that the numbers beside them said were not.

{_convergence_sentences(manifest["methods"], prov["logit_max"])}

**And one category never had anywhere to go.** {_exact_sentence(manifest)}

**Hover or arrow through a row** to see that query's profile: $\\hat\\pi$ against $\\pi$
over all {n} key positions, on the same logarithmic scale, with the row's total
variation. The strip beside the difference map is the per-row total variation down the
whole context, so you can see which queries a method hurts.

## What is not here

{_rejected_sentence(manifest['rejected'])}

Nothing on this page is smoothed, interpolated or illustrative. Every cell is a number
`attnbank` computed from the activations named above, quantised to eight bits and
colour-mapped in your browser. {n_methods} methods and {n_frames} configurations are
drawn, in {total // 1024}&nbsp;kB of images and manifest all told; the page loads one
method's maps at a time.
"""


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------

def _digest(d: pathlib.Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.iterdir()) if p.is_file()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="build twice into temporary directories and assert the output "
                         "is byte-identical, without touching docs/")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if args.check:
        digests = []
        for i in range(2):
            tmp = ASSETS.parent / f".heatmaps-check-{i}"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            pages = page(build(tmp))
            digests.append((_digest(tmp), hashlib.sha256(pages.encode()).hexdigest()))
            shutil.rmtree(tmp)
        same = digests[0] == digests[1]
        print("byte-identical" if same else "DIFFERS")
        if not same:
            a, b = digests[0][0], digests[1][0]
            for key in sorted(set(a) | set(b)):
                if a.get(key) != b.get(key):
                    print("  differs:", key)
            if digests[0][1] != digests[1][1]:
                print("  differs: heatmaps.md")
        return 0 if same else 1

    shutil.rmtree(ASSETS, ignore_errors=True)
    ASSETS.mkdir(parents=True)
    manifest = build(ASSETS)
    PAGE.write_text(page(manifest), encoding="utf-8")

    # the teaser panels belong to the landing page, not this one, so they are
    # not part of what this page costs a reader
    total = sum(v for k, v in manifest["bytes"].items()
                if not k.startswith("teaser-"))
    biggest = sorted(manifest["bytes"].items(), key=lambda kv: -kv[1])[:5]
    initial = (manifest["bytes"]["manifest.json"]
               + manifest["bytes"][f"m-{DEFAULT_METHOD}.png"]
               + manifest["bytes"][f"d-{DEFAULT_METHOD}.png"]
               + manifest["bytes"]["static-light.png"]
               + manifest["bytes"]["static-dark.png"])
    print(f"{len(manifest['methods'])} methods, "
          f"{sum(len(m['frames']) for m in manifest['methods'])} frames, "
          f"{len(manifest['rejected'])} rejected")
    print(f"payload {total / 1024:.0f} kB total, {initial / 1024:.0f} kB on first paint")
    print("largest: " + ", ".join(f"{k} {v / 1024:.1f}kB" for k, v in biggest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
