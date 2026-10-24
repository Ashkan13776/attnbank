"""Compute the site's figures from the library and write them as inline SVG.

    python docs/gen_figures.py

Four measurements, each one the library making a claim it can check:

    error_split      where an approximation's error comes from, split exactly
                     into a distributional and a value term (``invariant.md``)
    selection_gap    ``topk`` against ``value_topk`` at a matched budget: the
                     largest weights are not the best ones (``invariant.md``)
    convergence      Monte-Carlo estimators approaching dense softmax as ``r``
                     grows, and a fixed feature map that does not
                     (``conformance.md``, check 7)
    state_capacity   ``cost()["state_bytes"]`` against context length for all
                     78 methods, coloured by category (``conformance.md``,
                     check 9)

Nothing here is read from disk, downloaded or copied from anywhere: every
number is computed in this process from ``attnbank``, on a seeded
``attnbank.data.synthetic()`` control or from the analytic cost models.  Two
runs produce byte-identical files, which ``--check`` asserts.

Why not real activations
------------------------
``attnbank.data.get_activations`` captures real per-layer, per-head (Q, K, V)
from GPT-2 and Qwen, and for two of these figures that would be a sharper
measurement.  It would also mean the site's figures could only be regenerated
by someone willing to download a model and a dataset, which is the opposite of
what a public, offline-reproducible repository should ask.  The control is
``synthetic()``, every caption says so, and every shape and seed is printed in
the caption so a reader can re-run it.

Theme, and why these are not PNGs
---------------------------------
The SVG carries no colours of its own.  Ink is ``var(--ab-ink*)`` and the
seven category hues are ``var(--c-*)``, both from ``docs/stylesheets/tokens.css``,
so a figure repaints itself when the page switches between light and dark and a
category is the same colour in a figure as it is everywhere else on the site.
Each ``var()`` carries the light-theme value as its fallback -- read out of
``tokens.css`` at generation time, so it cannot drift -- which is what the file
renders as when it is opened on its own, outside the page.

The pages inline these files with ``pymdownx.snippets``; a ``<img src=...>``
would be a separate document and the page's custom properties would not reach
it.  That also keeps the text as text: selectable, searchable, and crisp at any
zoom.  Hence no ``<style>`` element anywhere below -- inlined into an HTML page,
an SVG's stylesheet is the page's stylesheet.  Everything is presentation
attributes; ``docs/stylesheets/figure.css`` styles the wrapper, not the marks.

Two widths, not one
-------------------
Type inside an SVG scales with the SVG.  The site's figure box runs from about
256 px on a 320 px phone to 726 px at the viewport width where Material has
folded both sidebars away, and one layout stretched across that range renders
its tick labels at 4.6 px at the small end.  So each figure is emitted twice --
``<name>.svg`` laid out at 700 units and ``<name>.narrow.svg`` at 360 -- and
``docs/stylesheets/figure.css`` shows one and hides the other at 40em.
Measured in the built pages, the smallest type actually rendered is 10.3 px for
the wide variant and 9.2 px for the narrow one.
"""
from __future__ import annotations

import argparse
import html
import math
import pathlib
import re
import sys
from dataclasses import dataclass

import torch

LIB = pathlib.Path(__file__).resolve().parents[1]
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

import attnbank as ab  # noqa: E402
from attnbank import metrics  # noqa: E402
from attnbank.data import synthetic  # noqa: E402
from attnbank.plotting import style  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "figures"
TOKENS = pathlib.Path(__file__).resolve().parent / "stylesheets" / "tokens.css"

#: Category -> slot in ``style.CATEGORICAL``.  This is the site's mapping, the one
#: ``tokens.css`` publishes as ``--c-*``; :func:`palette` asserts the two agree, so
#: the figures cannot drift away from the rest of the site.  Note that it is not
#: ``style.METHOD_COLORS``: in the monograph a hue tracks a *method* across the
#: argument it belongs to, while on this site a hue means a *category* and nothing
#: else.  Where a figure needs to separate two methods of one category it does it
#: with a dash pattern and a direct label, never with a second hue.
CATEGORY_SLOT = {"interaction": 0, "restrict": 1, "compress": 2,
                 "coupling": 3, "factorize": 6, "recurrent": 7}
CATEGORY_ORDER = ["exact", "interaction", "restrict", "compress",
                  "factorize", "coupling", "recurrent"]


# ======================================================================================
# tokens
# ======================================================================================

def palette() -> dict[str, str]:
    """The light-theme value of every token the figures reference.

    Read out of ``tokens.css`` rather than written down here, so the fallback a
    standalone ``.svg`` renders with is always the value the site would have given
    it.  Agent 1 owns that file and may change any value in it; this function is
    how the figures follow.
    """
    src = TOKENS.read_text(encoding="utf-8")
    root = src.split(":root {", 1)[1].split("\n}", 1)[0]
    want = ("--ab-ink", "--ab-ink-soft", "--ab-ink-muted", "--ab-faint",
            "--ab-rule", "--ab-rule-soft", "--ab-sunk", "--ab-accent",
            *(f"--c-{c}" for c in CATEGORY_ORDER))
    out = {}
    for tok in want:
        m = re.search(rf"^\s*{re.escape(tok)}:\s*(#[0-9a-fA-F]{{3,8}})\s*;", root, re.M)
        if not m:
            raise SystemExit(f"gen_figures: tokens.css has no literal value for {tok}")
        out[tok] = m.group(1).lower()

    # The seven hues are CATEGORICAL verbatim; SITE_DESIGN.md forbids changing them
    # and this is the check that says so out loud.
    for cat, slot in CATEGORY_SLOT.items():
        if out[f"--c-{cat}"] != style.CATEGORICAL[slot].lower():
            raise SystemExit(
                f"gen_figures: --c-{cat} is {out[f'--c-{cat}']} but "
                f"style.CATEGORICAL[{slot}] is {style.CATEGORICAL[slot]}")
    return out


PAL = palette()


def tok(name: str) -> str:
    """``var(--token, <its light value>)`` -- themed in the page, legible on its own."""
    return f"var({name},{PAL[name]})"


def hue(category: str) -> str:
    return tok(f"--c-{category}")


INK = tok("--ab-ink")
INK_SOFT = tok("--ab-ink-soft")
INK_MUTED = tok("--ab-ink-muted")
FAINT = tok("--ab-faint")
RULE = tok("--ab-rule")
RULE_SOFT = tok("--ab-rule-soft")

SANS = ("system-ui,-apple-system,'Segoe UI',Roboto,'Helvetica Neue',Arial,"
        "'Nimbus Sans','Liberation Sans',sans-serif")
MONO = ("ui-monospace,'SF Mono','JetBrains Mono',Menlo,Consolas,"
        "'DejaVu Sans Mono','Liberation Mono',monospace")


# ======================================================================================
# a very small SVG kernel
# ======================================================================================

def f(x: float, nd: int = 2) -> str:
    """A coordinate, rounded so that two runs of this file agree byte for byte."""
    s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0", "-") else s


def esc(s: str) -> str:
    return html.escape(str(s), quote=True)


@dataclass(frozen=True)
class Geom:
    """Everything that differs between the wide and the narrow rendering."""
    key: str
    w: float
    pad: float
    fs_tick: float
    fs_lab: float
    fs_key: float
    fs_note: float

    @property
    def narrow(self) -> bool:
        return self.key == "narrow"


#: The two layouts, and the arithmetic behind the numbers in them.  Rendered type
#: size is nominal x (rendered width / viewBox width), so each variant's smallest
#: type has to clear 9 px at the *narrow* end of the column range it serves.
#:
#:   wide    serves a 574-760 px figure (viewport >= 40em).  12.5 / 700 x 574
#:           = 10.3 px at the worst end, 13.0 px at the widest.
#:   narrow  serves a 256-400 px figure (viewport < 40em, capped at 20rem by
#:           figure.css).  13 / 360 x 256 = 9.2 px on a 320 px phone, 14.4 px
#:           at the cap.
#:
#: Both were measured in the built pages rather than derived, because the column
#: is not monotone in the viewport: it is widest at about 900 px, where Material
#: has folded both sidebars away but not yet hit its 61rem grid.
WIDE = Geom(key="wide", w=700, pad=4, fs_tick=12.5, fs_lab=13, fs_key=13.5, fs_note=12.5)
NARROW = Geom(key="narrow", w=360, pad=3, fs_tick=13, fs_lab=13, fs_key=13,
              fs_note=12.5)
GEOMS = (WIDE, NARROW)


class Canvas:
    """An SVG document under construction.

    ``uid`` prefixes every id in the file.  Both widths of a figure are inlined into
    the same page, so an unprefixed ``id="title"`` would appear twice in one document
    and ``aria-labelledby`` would resolve to whichever came first.
    """

    def __init__(self, uid: str, w: float, h: float, title: str, desc: str):
        self.uid, self.w, self.h = uid, w, h
        self.title, self.desc = title, desc
        self.body: list[str] = []

    def add(self, s: str) -> None:
        self.body.append(s)

    # -- marks -------------------------------------------------------------------
    def line(self, x1, y1, x2, y2, stroke, width=1, dash=None, opacity=None, cap=None,
             rule=False):
        a = [f'x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}"',
             f'stroke="{stroke}"']
        if width != 1:
            a.append(f'stroke-width="{f(width)}"')
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'stroke-opacity="{f(opacity)}"')
        if cap:
            a.append(f'stroke-linecap="{cap}"')
        if rule:
            # Grid and axis hairlines stay one device pixel whichever of the two
            # widths is on screen; data strokes are left to scale with the figure.
            a.append('vector-effect="non-scaling-stroke"')
        self.add(f'<line {" ".join(a)}/>')

    def grid(self, x1, y1, x2, y2):
        self.line(x1, y1, x2, y2, RULE_SOFT, 1, rule=True)

    def axis(self, x1, y1, x2, y2):
        self.line(x1, y1, x2, y2, FAINT, 1, rule=True)

    def rect(self, x, y, w, h, fill, opacity=None, rx=None, stroke=None, sw=1):
        a = [f'x="{f(x)}" y="{f(y)}" width="{f(max(w, 0))}" height="{f(max(h, 0))}"',
             f'fill="{fill}"']
        if opacity is not None:
            a.append(f'fill-opacity="{f(opacity)}"')
        if rx:
            a.append(f'rx="{f(rx)}"')
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{f(sw)}"')
        self.add(f'<rect {" ".join(a)}/>')

    def path(self, d, stroke=None, width=1.4, dash=None, fill="none", opacity=None,
             cap="round", join="round"):
        a = [f'd="{d}"', f'fill="{fill}"']
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{f(width)}"')
            a.append(f'stroke-linecap="{cap}" stroke-linejoin="{join}"')
        if dash:
            a.append(f'stroke-dasharray="{dash}"')
        if opacity is not None:
            a.append(f'{"fill" if fill != "none" else "stroke"}-opacity="{f(opacity)}"')
        self.add(f'<path {" ".join(a)}/>')

    def text(self, x, y, s, size, fill=None, anchor="start", mono=False, weight=None,
             opacity=None, baseline=None):
        a = [f'x="{f(x)}" y="{f(y)}"', f'font-size="{f(size)}"']
        if mono:
            a.append(f'font-family="{MONO}"')
        if anchor != "start":
            a.append(f'text-anchor="{anchor}"')
        if fill:
            a.append(f'fill="{fill}"')
        if weight:
            a.append(f'font-weight="{weight}"')
        if opacity is not None:
            a.append(f'fill-opacity="{f(opacity)}"')
        if baseline:
            a.append(f'dominant-baseline="{baseline}"')
        self.add(f'<text {" ".join(a)}>{esc(s)}</text>')

    def dot(self, x, y, r, fill, stroke=None, sw=1.6):
        a = [f'cx="{f(x)}" cy="{f(y)}" r="{f(r)}"', f'fill="{fill}"']
        if stroke:
            a.append(f'stroke="{stroke}" stroke-width="{f(sw)}"')
        self.add(f'<circle {" ".join(a)}/>')

    # -- output ------------------------------------------------------------------
    def render(self, variant: str) -> str:
        """The file, as one line per element and not a blank line anywhere.

        pymdownx.snippets splices this straight into the Markdown, where a blank
        line ends the raw-HTML block the ``<figure>`` opened and the rest of the
        SVG would be handed to the Markdown parser as prose.
        """
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'viewBox="0 0 {f(self.w)} {f(self.h)}" '
                f'role="img" aria-labelledby="{self.uid}-t {self.uid}-d" '
                f'class="ab-figsvg ab-fig-{variant}" font-family="{SANS}" '
                f'fill="{INK}">')
        parts = [head,
                 f'<title id="{self.uid}-t">{esc(self.title)}</title>',
                 f'<desc id="{self.uid}-d">{esc(self.desc)}</desc>',
                 *self.body,
                 "</svg>"]
        return "\n".join(p for p in parts if p.strip()) + "\n"


# -- scales --------------------------------------------------------------------------

def linmap(d0, d1, r0, r1):
    span = (d1 - d0) or 1.0
    return lambda v: r0 + (v - d0) / span * (r1 - r0)


def logmap(d0, d1, r0, r1):
    a, b = math.log10(d0), math.log10(d1)
    span = (b - a) or 1.0
    return lambda v: r0 + (math.log10(max(v, 1e-300)) - a) / span * (r1 - r0)


def polyline(xs, ys, sx, sy) -> str:
    return "M" + "L".join(f"{f(sx(x))} {f(sy(y))}" for x, y in zip(xs, ys))


#: Superscript digits, so an axis tick can read ``2¹⁷`` without MathJax reaching
#: inside an SVG -- which it does not, and should not.
_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sup_pow2(k: int) -> str:
    return "1" if k == 0 else ("2" if k == 1 else "2" + str(k).translate(_SUP))


def si_bytes(b: float) -> str:
    for unit, div in (("MiB", 2 ** 20), ("KiB", 2 ** 10)):
        if b >= div:
            v = b / div
            return f"{v:.0f} {unit}" if v >= 10 or v == int(v) else f"{v:.1f} {unit}"
    return f"{b:.0f} B"


def sci(v: float) -> str:
    """``1.33e+00`` -- the same shape the examples print, so the two can be compared."""
    return f"{v:.2e}"


def log_bounds(values, pad: float = 1.35) -> tuple[float, float]:
    """A log range that fits the data, not the nearest decades.

    Rounding out to whole powers of ten is the default everywhere and it is why so
    many log panels put their data in a band across the middle and leave half the
    panel empty.  These sweeps span a decade and a half; a padded range uses the
    height.
    """
    return min(values) / pad, max(values) * pad


def log_ticks(lo: float, hi: float, most: int = 6) -> list[float]:
    """Labelled values inside a short log range -- 0.3 and 2, not just ``10⁰``.

    ``style.log_ticks_plain`` makes the same point for matplotlib: a panel spanning
    less than about 1.5 decades gets one ``10⁰`` from the default locator and
    nothing else, which leaves a reader unable to put a number on anything.
    """
    out = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 3, 5):
            v = m * 10.0 ** e
            if lo <= v <= hi:
                out.append(v)
    while len(out) > most:
        out = out[::2]
    return out


def num(v: float) -> str:
    """A tick value as a reader would write it: ``0.3``, ``2``, ``10³``."""
    if 1e-3 <= v < 1e4:
        return f"{v:g}"
    return _pow10(int(round(math.log10(v))))


# ======================================================================================
# measurement 1 -- where the error comes from
# ======================================================================================

#: Nine configurations chosen to span the decomposition rather than to rank methods:
#: four that move mass and keep every value exactly, one of each kind of
#: cache compression, and -- the point of the figure -- the same method,
#: ``quantkv``, at three bit allocations that put the error in three different
#: places.  Four categories appear, so the hues in this figure mean what they mean
#: everywhere else on the site.
ERROR_SPLIT_CASES = [
    ("window", {"w": 32}),
    ("topk", {"k": 16}),
    ("performer", {"r": 64, "seed": 0}),
    ("quantkv", {"bits_k": 2, "bits_v": 16}),
    ("lowprec", {}),
    ("mla", {"r": 16, "seed": 0}),
    ("kivi", {"bits_k": 3, "bits_v": 3}),
    ("quantkv", {"bits_k": 3, "bits_v": 3}),
    ("quantkv", {"bits_k": 16, "bits_v": 2}),
]
ERROR_SPLIT_SHAPE = dict(n=256, d=64, dv=64, heads=4, passages=2, seed=0)


def label_of(name: str, cfg: dict) -> str:
    shown = {k: v for k, v in cfg.items() if k != "seed"}
    if not shown:
        return name
    return name + " " + " ".join(f"{k}={v}" for k, v in shown.items())


def data_error_split() -> dict:
    b = synthetic(**ERROR_SPLIT_SHAPE, dtype=torch.float64).expand_kv()
    q, k, v = b.q, b.k, b.v
    ref = ab.build("softmax")(q, k, v, causal=True, return_weights=True)
    cats = {r["name"]: r["category"] for r in ab.method_table()}

    rows = []
    for name, cfg in ERROR_SPLIT_CASES:
        out = ab.build(name, **cfg)(q, k, v, causal=True, return_weights=True)
        d = metrics.error_decomposition(ref.weights, out.weights, v, out.v_hat)
        dist = float(d["distributional"].median())
        val = float(d["value"].median())
        rows.append({"label": label_of(name, cfg), "category": cats[name],
                     "dist": dist, "value": val,
                     "total": float(d["total"].median()),
                     "share": dist / (dist + val) if (dist + val) else 0.0})
    rows.sort(key=lambda r: (-r["share"], r["label"]))
    return {"rows": rows, "shape": tuple(q.shape),
            "seed": ERROR_SPLIT_SHAPE["seed"]}


def draw_error_split(D: dict, g: Geom) -> Canvas:
    """A hundred-per-cent bar per configuration: the two terms, as shares.

    The split is what the figure is for, so the split is what the geometry encodes.
    Totals span three orders of magnitude and would flatten nine of the ten bars to
    nothing if the bars were drawn to scale; the total is printed instead, which is
    also how a reader gets a number they can check against ``examples/03``.

    Colour stays category colour: the whole bar is the method's hue and the two
    terms are told apart by weight, solid against a wash of the same hue. A second
    hue here would be the one thing SITE_DESIGN.md says a hue may never mean.
    """
    rows = D["rows"]
    c = Canvas(f"abfig-split-{g.key}", g.w, 10,
               "Where each method's output error comes from",
               "One horizontal bar per configuration, each drawn to the full width "
               "and divided at the share of the output error that the "
               "distributional term carries; the remainder is the value term. The "
               "median total error is printed beside each bar. Bars are coloured by "
               "the method's taxonomy category.")
    seen = [k for k in CATEGORY_ORDER if any(r["category"] == k for r in rows)]

    if g.narrow:
        # Two lines per row: the name and its total above, the bar beneath. A
        # 26-character method label and a readable bar do not both fit in 380 units
        # side by side, and shrinking the label is how a figure stops being readable
        # on the device most people will open it on.
        head = 26
        rh = 37
        bar_h = 12
        x0, x1 = g.pad, g.w - g.pad
        c.text(x0, head - 11, "distributional", g.fs_lab, INK_SOFT, weight="600")
        c.text(x1, head - 11, "value", g.fs_lab, INK_SOFT, anchor="end", weight="600")
        for i, r in enumerate(rows):
            y = head + i * rh
            paint = hue(r["category"])
            c.rect(x0, y, 3, g.fs_key + bar_h + 4, paint)
            c.text(x0 + 9, y + g.fs_key - 2, r["label"], g.fs_key, INK, mono=True)
            c.text(x1, y + g.fs_key - 2, sci(r["total"]), g.fs_tick, INK_MUTED,
                   anchor="end", mono=True)
            by = y + g.fs_key + 4
            split = x0 + r["share"] * (x1 - x0)
            c.rect(x0, by, x1 - x0, bar_h, paint, opacity=0.26)
            c.rect(x0, by, split - x0, bar_h, paint)
            if 0.015 < r["share"] < 0.985:
                c.line(split, by, split, by + bar_h, tok("--ab-sunk"), 1.4)
        ky = head + len(rows) * rh + 8
        kx = g.pad
    else:
        head = 36
        rh = 25
        gap = 5.5
        lab_w, num_w = 238, 64
        x0, x1 = lab_w, g.w - num_w - g.pad
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = x0 + frac * (x1 - x0)
            c.grid(x, head - 7, x, head + len(rows) * rh - gap)
        c.line(x0, head - 7, x1, head - 7, RULE, 1, rule=True)
        c.text(x0, head - 14, "distributional  (π − π̂) v", g.fs_lab, INK_SOFT,
               weight="600")
        c.text(x1, head - 14, "value  π̂ (v − v̂)", g.fs_lab, INK_SOFT, anchor="end",
               weight="600")
        c.text(g.w - g.pad, head - 14, "‖y − ŷ‖", g.fs_tick, FAINT, anchor="end")
        for i, r in enumerate(rows):
            y = head + i * rh
            paint = hue(r["category"])
            bh = rh - gap
            c.rect(g.pad, y, 3, bh, paint)
            c.text(g.pad + 9, y + bh * 0.5, r["label"], g.fs_key, INK, mono=True,
                   baseline="central")
            split = x0 + r["share"] * (x1 - x0)
            c.rect(x0, y, x1 - x0, bh, paint, opacity=0.26)
            c.rect(x0, y, split - x0, bh, paint)
            if 0.015 < r["share"] < 0.985:
                c.line(split, y, split, y + bh, tok("--ab-sunk"), 1.4)
            c.text(g.w - g.pad, y + bh * 0.5, sci(r["total"]), g.fs_tick, INK_MUTED,
                   anchor="end", mono=True, baseline="central")
        ky = head + len(rows) * rh + 12
        kx = g.pad

    for j, cat in enumerate(seen):
        if g.narrow and j and j % 3 == 0:
            ky += 17
            kx = g.pad
        c.rect(kx, ky - 7, 9, 9, hue(cat), rx=2)
        c.text(kx + 13, ky, cat, g.fs_tick, INK_MUTED)
        kx += 13 + len(cat) * g.fs_tick * 0.56 + 14
    c.h = ky + 8
    return c


# ======================================================================================
# measurement 2 -- the largest weights are not the best ones
# ======================================================================================

SELECTION_BUDGETS = (2, 4, 8, 16, 32)
SELECTION_SHAPE = dict(n=128, d=32, dv=32, heads=2, passages=2, seed=0)


def data_selection_gap() -> dict:
    b = synthetic(**SELECTION_SHAPE, dtype=torch.float64).expand_kv()
    q, k, v = b.q, b.k, b.v
    pts = []
    for budget in SELECTION_BUDGETS:
        rows = {r["name"]: r for r in ab.compare(
            ["topk", "value_topk"], q, k, v, causal=True,
            configs={"topk": {"k": budget},
                     "value_topk": {"k": budget, "chunk": 16}})}
        a = float(rows["topk"]["output_error"])
        o = float(rows["value_topk"]["output_error"])
        pts.append({"k": budget, "topk": a, "value_topk": o, "ratio": a / o})
    return {"pts": pts, "shape": tuple(q.shape), "seed": SELECTION_SHAPE["seed"],
            "lo": min(p["ratio"] for p in pts), "hi": max(p["ratio"] for p in pts)}


def _pow10(e: int) -> str:
    return "1" if e == 0 else "10" + str(e).translate(_SUP)


def draw_selection_gap(D: dict, g: Geom) -> Canvas:
    pts = D["pts"]
    paint = hue("restrict")
    left = 52 if not g.narrow else 46
    right = g.w - (112 if not g.narrow else 12)
    top = 32 if not g.narrow else 28
    ph = 214 if not g.narrow else 184
    bottom = top + ph
    h = bottom + 44 + (24 if g.narrow else 0)

    ks = [p["k"] for p in pts]
    ys = [p["topk"] for p in pts] + [p["value_topk"] for p in pts]
    lo, hi = log_bounds(ys)
    sx = logmap(ks[0], ks[-1], left, right)
    sy = logmap(lo, hi, bottom, top)

    c = Canvas(f"abfig-sel-{g.key}", g.w, h,
               "Top-k selection against output-optimal selection at a matched budget",
               "Two curves of output error against the number of keys kept per "
               "query, on logarithmic axes. The upper curve is topk, which keeps the "
               "largest attention weights; the lower is value_topk, which keeps the "
               "keys that minimise output error. The shaded band between them is the "
               "cost of selecting on scores alone, and it does not close as the "
               "budget grows.")

    for t in log_ticks(lo, hi):
        y = sy(t)
        c.grid(left, y, right, y)
        c.text(left - 7, y, num(t), g.fs_tick, INK_MUTED, anchor="end",
               baseline="central")
    c.axis(left, bottom, right, bottom)
    for p in pts:
        x = sx(p["k"])
        c.axis(x, bottom, x, bottom + 4)
        c.text(x, bottom + 17, str(p["k"]), g.fs_tick, INK_MUTED, anchor="middle")
    c.text((left + right) / 2, bottom + 34, "keys kept per query  k", g.fs_lab,
           INK_SOFT, anchor="middle")
    c.text(left - 7, top - 12, "output error", g.fs_lab, INK_SOFT)

    band = ("M" + "L".join(f"{f(sx(p['k']))} {f(sy(p['topk']))}" for p in pts)
            + "L" + "L".join(f"{f(sx(p['k']))} {f(sy(p['value_topk']))}"
                             for p in reversed(pts)) + "Z")
    c.path(band, fill=paint, opacity=0.2)

    c.path(polyline(ks, [p["topk"] for p in pts], sx, sy), stroke=INK, width=1.9)
    c.path(polyline(ks, [p["value_topk"] for p in pts], sx, sy), stroke=paint,
           width=1.9)
    for p in pts:
        c.dot(sx(p["k"]), sy(p["topk"]), 3.2, tok("--ab-sunk"), INK, 1.7)
        c.dot(sx(p["k"]), sy(p["value_topk"]), 3.2, tok("--ab-sunk"), paint, 1.7)

    last = pts[-1]
    note = f"{D['lo']:.2f}–{D['hi']:.2f}× the error at every budget"
    if g.narrow:
        # The two curves converge towards the right-hand edge, so at 360 units an
        # end label lands on the other series or on the axis. A key below the plot
        # costs two lines and collides with nothing.
        ly = bottom + 50
        for txt, fill in (("topk", INK), ("value_topk", paint)):
            c.line(g.pad, ly - 4, g.pad + 20, ly - 4, fill, 1.9)
            c.dot(g.pad + 10, ly - 4, 3.2, tok("--ab-sunk"), fill, 1.7)
            c.text(g.pad + 26, ly, txt, g.fs_tick, INK_SOFT, mono=True)
            ly += 16
        c.text(g.pad, ly + 4, note, g.fs_note, INK_SOFT)
        c.h = ly + 12
    else:
        c.text(right + 9, sy(last["topk"]), "topk", g.fs_key, INK, mono=True,
               baseline="central", weight="600")
        c.text(right + 9, sy(last["value_topk"]), "value_topk", g.fs_key, paint,
               mono=True, baseline="central", weight="600")
        # the band's own statistic, in the empty lower-left corner rather than on
        # top of the two curves it describes
        c.text(left + 6, bottom - 10, note, g.fs_note, INK_SOFT)
    return c


# ======================================================================================
# measurement 3 -- what converges, and what does not
# ======================================================================================

CONVERGENCE_R = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
CONVERGENCE_SEEDS = 8
CONVERGENCE_SHAPE = dict(n=128, d=32, dv=32, heads=2, passages=1, seed=0)
CONVERGENCE_SCALE = 0.5


def data_convergence() -> dict:
    b = synthetic(**CONVERGENCE_SHAPE, dtype=torch.float64).expand_kv()
    q, k, v = (CONVERGENCE_SCALE * t for t in (b.q, b.k, b.v))
    ref = ab.build("softmax")(q, k, v).y

    def err(y):
        return float((y - ref).norm(dim=-1).mean())

    series = {}
    for name in ("performer", "trigrff"):
        means, spread = [], []
        for r in CONVERGENCE_R:
            e = [err(ab.build(name, r=r, seed=s)(q, k, v).y)
                 for s in range(CONVERGENCE_SEEDS)]
            means.append(sum(e) / len(e))
            spread.append((min(e), max(e)))
        # the slope over the last decade, which is the one a rate claim is about
        tail = math.log(means[-1] / means[-4]) / math.log(
            CONVERGENCE_R[-1] / CONVERGENCE_R[-4])
        series[name] = {"mean": means, "spread": spread, "slope": tail}
    return {"series": series, "fixed": err(ab.build("linear_elu")(q, k, v).y),
            "shape": tuple(q.shape), "seeds": CONVERGENCE_SEEDS,
            "scale": CONVERGENCE_SCALE}


def draw_convergence(D: dict, g: Geom) -> Canvas:
    paint = hue("interaction")
    left = 54 if not g.narrow else 48
    right = g.w - (104 if not g.narrow else 12)
    top = 32 if not g.narrow else 28
    ph = 222 if not g.narrow else 190
    bottom = top + ph

    rs = list(CONVERGENCE_R)
    allv = [v for s in D["series"].values() for v in s["mean"]]
    allv += [b for s in D["series"].values() for p in s["spread"] for b in p]
    allv.append(D["fixed"])
    lo, hi = log_bounds(allv)
    sx = logmap(rs[0], rs[-1], left, right)
    sy = logmap(lo, hi, bottom, top)

    c = Canvas(f"abfig-conv-{g.key}", g.w, 10,
               "Random features converging on dense softmax, and a fixed map that "
               "does not",
               "Mean output error against the number of random features r, on "
               "logarithmic axes, with a shaded band over eight seeds. performer and "
               "trigrff both fall along a slope close to r to the minus one half. "
               "linear_elu uses a fixed feature map and declares a different "
               "objective; it is a flat line that no value of r improves.")

    for t in log_ticks(lo, hi):
        y = sy(t)
        c.grid(left, y, right, y)
        c.text(left - 7, y, num(t), g.fs_tick, INK_MUTED, anchor="end",
               baseline="central")
    c.axis(left, bottom, right, bottom)
    step = 1 if not g.narrow else 2
    for i, r in enumerate(rs):
        x = sx(r)
        c.axis(x, bottom, x, bottom + 4)
        if i % step == 0:
            c.text(x, bottom + 17, sup_pow2(int(math.log2(r))), g.fs_tick, INK_MUTED,
                   anchor="middle")
    c.text((left + right) / 2, bottom + 34, "random features  r", g.fs_lab, INK_SOFT,
           anchor="middle")
    c.text(left - 7, top - 12, "mean output error", g.fs_lab, INK_SOFT)

    # The rate the theory predicts, anchored on performer's first point. It runs
    # between the two curves for the whole sweep, so there is nowhere beside it to
    # put a label; the slope is named by a detached gauge in the empty corner
    # underneath instead -- the convention a physics paper uses for exactly this.
    p0 = D["series"]["performer"]["mean"][0]
    gy = [p0 * (rs[0] / r) ** 0.5 for r in rs]
    c.path(polyline(rs, gy, sx, sy), stroke=FAINT, width=1, dash="2 3")

    # Its *lower* end is pinned a fixed fraction up the panel, not a fixed factor
    # above the axis: pinning the upper end put the label on the tick labels on
    # the short narrow panel, where the gauge's own half-decade drop is a much
    # larger share of the height.
    gx0, gx1 = rs[0] * 1.25, rs[0] * 5.0
    gy1 = lo * (hi / lo) ** 0.15
    gy0 = gy1 * (gx1 / gx0) ** 0.5
    c.path(f"M{f(sx(gx0))} {f(sy(gy0))}L{f(sx(gx1))} {f(sy(gy1))}", stroke=FAINT,
           width=1.3, dash="2 3")
    c.text(sx(gx1) + 6, sy(gy1) + 1, "slope −1/2", g.fs_note, FAINT,
           baseline="central")

    for name, dash in (("performer", None), ("trigrff", "5 3")):
        s = D["series"][name]
        area = ("M" + "L".join(f"{f(sx(r))} {f(sy(p[0]))}"
                               for r, p in zip(rs, s["spread"]))
                + "L" + "L".join(f"{f(sx(r))} {f(sy(p[1]))}"
                                 for r, p in zip(reversed(rs), reversed(s["spread"])))
                + "Z")
        c.path(area, fill=paint, opacity=0.11)
    for name, dash in (("performer", None), ("trigrff", "5 3")):
        s = D["series"][name]
        c.path(polyline(rs, s["mean"], sx, sy), stroke=paint, width=1.9, dash=dash)
        for r, m in zip(rs, s["mean"]):
            c.dot(sx(r), sy(m), 2.6, paint)

    c.line(left, sy(D["fixed"]), right, sy(D["fixed"]), INK, 1.7, dash="1 4",
           cap="round")

    pm = D["series"]["performer"]["mean"][-1]
    tm = D["series"]["trigrff"]["mean"][-1]
    if not g.narrow:
        for y, txt, fill in ((sy(pm), "performer", paint),
                             (sy(tm), "trigrff", paint),
                             (sy(D["fixed"]), "linear_elu", INK)):
            c.text(right + 9, y, txt, g.fs_key, fill, mono=True, baseline="central",
                   weight="600")
        c.h = bottom + 46
    else:
        ly = bottom + 52
        for txt, fill, dash in (("performer", paint, None),
                                ("trigrff", paint, "5 3"),
                                ("linear_elu", INK, "1 4")):
            c.line(g.pad, ly - 4, g.pad + 20, ly - 4, fill, 1.9, dash=dash)
            c.text(g.pad + 26, ly, txt, g.fs_tick, INK_SOFT, mono=True)
            ly += 16
        c.h = ly + 2
    return c


# ======================================================================================
# measurement 4 -- whose state is bounded
# ======================================================================================

STATE_EXP = tuple(range(7, 18))
STATE_D = 128


def data_state_capacity() -> dict:
    ns = [2 ** e for e in STATE_EXP]
    cats = {r["name"]: r["category"] for r in ab.method_table()}
    curves = []
    for name in sorted(ab.available()):
        m = ab.build(name)
        ys = [float(m.cost(1, n, STATE_D, STATE_D)["state_bytes"]) for n in ns]
        # "bounded" is read off the model rather than declared: the last doubling of
        # the context has to leave the resident state essentially where it was.
        curves.append({"name": name, "category": cats[name], "y": ys,
                       "grows": ys[-1] / max(ys[-2], 1.0) > 1.2})
    return {"ns": ns, "curves": curves,
            "dense": next(c for c in curves if c["name"] == "softmax"),
            "n_grow": sum(c["grows"] for c in curves),
            "n_bounded": sum(not c["grows"] for c in curves)}


def draw_state_capacity(D: dict, g: Geom) -> Canvas:
    """Two panels on one scale, because that is the claim.

    Drawn as a single panel this is 78 lines in a thicket, and the one thing it has
    to show -- that the bank splits cleanly into state that tracks the context and
    state that does not -- has to be pointed at with an annotation. Faceted on the
    property itself, the reader sees two slopes and needs no annotation at all. The
    y scale is shared, so the panels stay comparable; dense softmax is drawn in both,
    as the reference each bundle is measured against.
    """
    ns = D["ns"]
    left = 64 if not g.narrow else 58
    right = g.w - (100 if not g.narrow else 10)
    ph = 136 if not g.narrow else 116
    title_h = 22
    gap = 22
    top_a = 30
    bot_a = top_a + ph
    top_b = bot_a + gap + title_h
    bot_b = top_b + ph

    allv = [v for cu in D["curves"] for v in cu["y"]]
    lo = 2 ** math.floor(math.log2(min(allv)))
    hi = 2 ** math.ceil(math.log2(max(allv)))
    sx = logmap(ns[0], ns[-1], left, right)

    c = Canvas(f"abfig-state-{g.key}", g.w, 10,
               "Resident state against context length, for every registered method",
               "Two panels sharing one logarithmic scale, with one line per "
               "registered method coloured by taxonomy category. The upper panel "
               "holds the methods whose resident key/value state grows in proportion "
               "to the context; the lower holds those whose state is bounded. Dense "
               "softmax is drawn in both panels as the reference.")

    def panel(y_top, y_bot, label, want_grow, xticks):
        sy = logmap(lo, hi, y_bot, y_top)
        c.text(g.pad, y_top - 9, label, g.fs_lab, INK_SOFT, weight="600")
        for e in range(int(math.log2(lo)), int(math.log2(hi)) + 1, 4):
            y = sy(2.0 ** e)
            c.grid(left, y, right, y)
            c.text(left - 7, y, si_bytes(2.0 ** e), g.fs_tick, INK_MUTED,
                   anchor="end", baseline="central")
        c.axis(left, y_bot, right, y_bot)
        # Seventy-eight curves, grouped by category so the stroke declaration is
        # written once per hue instead of once per method. Same picture, a third
        # of the bytes, and a third of the attributes for the browser to resolve.
        for cat in CATEGORY_ORDER:
            ds = [polyline(ns, cu["y"], sx, sy) for cu in D["curves"]
                  if cu["category"] == cat and cu["grows"] == want_grow
                  and cu["name"] != "softmax"]
            if not ds:
                continue
            c.add(f'<g fill="none" stroke="{hue(cat)}" stroke-width="1.15" '
                  f'stroke-opacity="0.6" stroke-linecap="round" '
                  f'stroke-linejoin="round">'
                  + "".join(f'<path d="{d}"/>' for d in ds) + "</g>")
        # dense softmax is the reference in both panels: the line the upper bundle
        # tracks, and the line the lower bundle walks away from.
        c.path(polyline(ns, D["dense"]["y"], sx, sy), stroke=INK,
               width=2.1 if want_grow else 1.5, dash=None if want_grow else "4 3")
        if not g.narrow:
            c.text(right + 8, sy(D["dense"]["y"][-1]), "dense softmax", g.fs_note,
                   INK, baseline="central", weight="600")
        step = 2 if not g.narrow else 4
        for e in STATE_EXP:
            x = sx(2 ** e)
            c.axis(x, y_bot, x, y_bot + 4)
            if xticks and (e - STATE_EXP[0]) % step == 0:
                c.text(x, y_bot + 17, sup_pow2(e), g.fs_tick, INK_MUTED,
                       anchor="middle")

    c.text(g.pad, top_a - 26, "resident key/value state, fp16", g.fs_tick, FAINT)
    panel(top_a, bot_a, f"(a) state grows with n — {D['n_grow']} methods", True,
          False)
    panel(top_b, bot_b, f"(b) state is bounded — {D['n_bounded']} methods", False,
          True)
    c.text((left + right) / 2, bot_b + 34, "context length  n", g.fs_lab, INK_SOFT,
           anchor="middle")

    ky = bot_b + 56
    if g.narrow:
        c.line(g.pad, ky - 4, g.pad + 20, ky - 4, INK, 1.9)
        c.text(g.pad + 26, ky, "dense softmax", g.fs_tick, INK_SOFT)
        ky += 18
    kx = g.pad
    per_row = 7 if not g.narrow else 3
    for j, cat in enumerate(CATEGORY_ORDER):
        if j and j % per_row == 0:
            ky += 17
            kx = g.pad
        c.rect(kx, ky - 7, 9, 9, hue(cat), rx=2)
        c.text(kx + 13, ky, cat, g.fs_tick, INK_MUTED)
        kx += 13 + len(cat) * g.fs_tick * 0.56 + (12 if not g.narrow else 14)
    c.h = ky + 8
    return c


# ======================================================================================
# driver
# ======================================================================================

FIGURES = [
    ("error_split", data_error_split, draw_error_split),
    ("selection_gap", data_selection_gap, draw_selection_gap),
    ("convergence", data_convergence, draw_convergence),
    ("state_capacity", data_state_capacity, draw_state_capacity),
]


def build() -> dict[str, str]:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    files: dict[str, str] = {}
    for name, compute, draw in FIGURES:
        D = compute()
        for g in GEOMS:
            suffix = "" if g is WIDE else ".narrow"
            files[f"{name}{suffix}.svg"] = draw(D, g).render(g.key)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="rebuild and fail if anything on disk would change")
    args = ap.parse_args()

    files = build()
    OUT.mkdir(parents=True, exist_ok=True)
    changed = []
    for rel, text in sorted(files.items()):
        p = OUT / rel
        old = p.read_text(encoding="utf-8") if p.exists() else None
        if old != text:
            changed.append(rel)
            if not args.check:
                p.write_text(text, encoding="utf-8", newline="\n")
    stale = sorted(q.name for q in OUT.glob("*.svg") if q.name not in files)
    for rel in stale:
        if not args.check:
            (OUT / rel).unlink()

    if args.check:
        if changed or stale:
            for rel in changed:
                print(f"would rewrite  docs/figures/{rel}", file=sys.stderr)
            for rel in stale:
                print(f"would remove   docs/figures/{rel}", file=sys.stderr)
            return 1
        print(f"{len(files)} figures up to date")
        return 0

    for rel in sorted(files):
        print(f"wrote docs/figures/{rel}  "
              f"({len(files[rel].encode('utf-8')) / 1024:.1f} kB)")
    for rel in stale:
        print(f"removed docs/figures/{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
