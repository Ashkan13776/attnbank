"""House style for every figure in the paper.

Use:
    from attnbank.plotting import style
    style.use()
    fig, ax = style.figure("column")      # or "full", "full-tall", (w, h)
    ax.plot(x, y, color=style.series(0))  # categorical slots in FIXED order
    style.save(fig, "fig_restriction_error")  # -> figures/<name>.pdf (+ .png preview)

Rules (from the dataviz method; do not override per figure):
  * Categorical colors are assigned in fixed slot order, by entity, never by rank.
    The same method keeps the same color in every figure: register it in METHOD_COLORS.
  * Scatter / small-multiple plots: at most 3 categorical series per panel
    (palette validated all-pairs only for slots 1-3); otherwise facet.  Line panels may
    use up to 6, which the validator clears on the adjacent pairlist, and only with
    direct labels.
  * Sequential = one hue (blue ramp); diverging = blue <-> red with gray midpoint.
    ORDERED sweeps (bit width, rank r, state size, iteration count) take ``seq()``,
    never ``series()``.
  * One y-axis per panel. Never twin axes.
  * Lines 1.4pt, markers >= 4pt, recessive grid, no top/right spines.
  * No text in a figure is set below 7.5pt -- the legend/tick size -- and that includes
    in-panel notes and direct labels.  At the 6.5in text width these render 1:1, so a
    6.5pt note is 6.5pt in print.  When a note will not fit at 7.5pt, cut words: the
    long-form explanation is in the caption (``sections/tables/cap_fig_*.tex``), which
    renders directly beneath the figure, so a note only has to carry the number and
    its name.  7.0pt is the absolute floor, for a panel where trimming has run out.
  * >= 2 series need a legend; <= 4 series should also be direct-labelled.
  * Panel titles are ONE short line set by ``panel()`` and never repeat the axis label;
    the panel's statistic goes inside the axes via ``note()``.  A two-line title on a
    1x3 strip overruns its own panel and collides with the neighbour's.
  * Legends never sit on the data: pass ``loc`` explicitly to an empty corner, or use
    ``fig.legend`` below the axes, or direct-label and drop the box.
  * Vector PDF output; fonts embedded (Type 42); sizes match the 6.5in text width.

The palette claims above are computed, not eyeballed: the dataviz validator
(``scripts/validate_palette.js``; a Python twin was used here because the box has no
node) reports slots 0-2 all-pairs worst dE 9.2 protan / 24.0 normal = pass, slots 0-3
all-pairs 13.7 normal = FAIL, all 8 on the adjacent pairlist 9.1 protan / 19.6
normal = pass, and ``ordinal_steps(n<=5)`` pass for n up to 5.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Validated categorical palette (light mode; print surface).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING = ["#0d366b", "#2a78d6", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#8f1f1f"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#8a8984", "grid": "#e4e3df",
       "surface": "#ffffff"}

# Stable method -> slot assignment shared by all figures (extend; never reorder).
# Families share hue so a reader can track a family across the paper.
#
# Only three categorical slots are legal in one scatter panel, so the assignment below
# gives slots 0-2 to the three mechanism *families* that carry the paper's argument --
# restriction (blue), random features (orange), compression (green) -- and the
# remaining slots to the members that only ever appear alongside their own family.
# Verified with the dataviz validator: slots 0-2 pass all-pairs
# (worst dE 9.2 protan, 24.0 normal); slot 3 against slot 1 is dE 13.7 normal, below
# the 15 floor, which is why `series()` past index 2 is for line charts (adjacent
# pairlist) with direct labels only.
METHOD_COLORS: dict[str, str] = {
    "softmax": INK["primary"],          # the exact operator is always ink, never a hue
    "topk": CATEGORICAL[0],             # restriction, the paper's running baseline
    "performer": CATEGORICAL[1],        # random features
    "performer_orth": CATEGORICAL[1],
    "quantkv": CATEGORICAL[2],          # compression
    "trigrff": CATEGORICAL[6],
    "window": CATEGORICAL[7],
}

#: Per-study validated sets.  Slots are reused across studies whose methods never
#: appear in the same panel; what must not change is a method's colour *within* the
#: argument it belongs to.  Each set below was checked with the validator on the
#: pairlist its panel actually needs.
#:
#:   S1 scatter   slots 0,1,2       all-pairs  dE  9.2 protan / 24.0 normal   pass
#:   S2 ratio     slots 0,3,4       all-pairs  dE 13.0 protan / 19.6 normal   pass
#:   S2/S5 lines  slots 0,3,4,5,6   all-pairs  dE 13.0 protan / 16.3 normal   pass
#:   sources      slots 0,1,2       all-pairs  as S1
#:
#: The contiguous five-subset 0-4 FAILS the normal-vision floor (12.9), which is why
#: the five-series panels skip slots 1 and 2.
FACTORIZE_SLOT = {"svd_oracle": 0, "nmf_oracle": 3, "nystrom": 4,
                  "clustered": 5, "latent_stochastic": 6}

#: S2's five selectors, as one all-pairs-validated set.  Slots (0, 3, 4, 5, 6) are the
#: best five-subset of CATEGORICAL under the validator: worst all-pairs dE 13.0
#: (protan) and 16.3 (normal), both above their gates, where the contiguous slots
#: 0-4 fail the normal floor at 13.7.  ``topk`` is the denominator of every comparison
#: in that study, so like ``softmax`` it is drawn in ink rather than given a hue.
SELECTOR_COLORS: dict[str, str] = {
    "topk": INK["primary"],
    "forward_greedy": CATEGORICAL[0],
    "value_topk": CATEGORICAL[3],
    "value_oracle": CATEGORICAL[4],
    "blocktopk16": CATEGORICAL[5],
    "blocktopk64": CATEGORICAL[6],
}

WIDTHS = {"column": 3.25, "full": 6.5}
#: ``full-3`` is the 1x3 strip: a third of 6.5in is 2.17in wide, so the panel needs a
#: little more height than ``full`` to stay near a readable aspect once the axis band
#: and the one-line panel title are accounted for.
HEIGHTS = {"column": 2.3, "full": 2.6, "full-3": 2.9, "full-tall": 4.6,
           "full-stack": 5.4}

seq_cmap = LinearSegmentedColormap.from_list("ab_seq", SEQ_BLUE)
div_cmap = LinearSegmentedColormap.from_list("ab_div", DIVERGING)

FIG_DIR = Path(__file__).resolve().parents[3] / "figures"   # repo-root figures/


def series(i: int) -> str:
    if i >= len(CATEGORICAL):
        raise IndexError("more than 8 series: fold into 'Other' or facet into small multiples")
    return CATEGORICAL[i]


def method_color(name: str) -> str:
    if name not in METHOD_COLORS:
        METHOD_COLORS[name] = CATEGORICAL[(len(METHOD_COLORS) - 1) % len(CATEGORICAL)]
    return METHOD_COLORS[name]


def use():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": INK["secondary"], "axes.labelcolor": INK["primary"],
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "xtick.color": INK["secondary"], "ytick.color": INK["secondary"],
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.4, "lines.markersize": 4, "lines.solid_capstyle": "round",
        "patch.linewidth": 0, "legend.frameon": False, "legend.handlelength": 1.6,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "image.cmap": "ab_seq",
    })
    try:
        mpl.colormaps.register(seq_cmap)
        mpl.colormaps.register(div_cmap)
    except ValueError:
        pass


def figure(size="full", nrows=1, ncols=1, **kw):
    if isinstance(size, tuple):
        w, h = size
    else:
        w, h = WIDTHS.get(size.split("-")[0], 6.5), HEIGHTS[size]
    return plt.subplots(nrows, ncols, figsize=(w, h), constrained_layout=True, **kw)


def seq(i: int, n: int) -> tuple:
    """Sequential ramp step ``i`` of ``n`` -- for ORDERED variables only.

    Ordered sweeps (bit width, state size, rank r, iteration count) get the one-hue
    ramp, never categorical hues.  The ramp starts at ``SEQ_BLUE[3]`` because the three
    palest steps fall below the validator's 2:1 light-end contrast floor on a white
    print surface; the five-step sample over ``[3, 12]`` passes ``--ordinal``.
    """
    lo, hi = 3, len(SEQ_BLUE) - 1
    t = 0.0 if n <= 1 else i / (n - 1)
    return SEQ_BLUE[int(round(lo + t * (hi - lo)))]


def ordinal_steps(n: int) -> list[str]:
    """``n`` validated steps of the sequential ramp, light -> dark."""
    return [seq(i, n) for i in range(n)]


def panel(ax, letter: str, title: str = "") -> None:
    """A short, left-aligned panel heading.

    Panel titles are one line and name the *panel*, never repeat the axis label and
    never carry a statistic -- a two-line title in a 1x3 full-width figure runs past
    its own panel and collides with its neighbour.  Statistics go in :func:`note`.
    """
    ax.set_title(f"({letter}) {title}" if title else f"({letter})",
                 loc="left", pad=4, color=INK["primary"])


def note(ax, text: str, loc: str = "upper left", pad: float = 0.03,
         color: str | None = None, size: float = 7.5,
         ypad: float | None = None) -> None:
    """An in-axes annotation, in ink, for the statistic a panel is making.

    ``loc`` is one of the four corners; the text is placed in axes coordinates so it
    cannot leave the panel, which is what a title does when it is too long.
    ``ypad`` overrides the vertical inset alone, for the case where the horizontal
    corner is the only free one and the note has to clear a legend above it.
    """
    import matplotlib.patheffects as pe
    va = "top" if "upper" in loc else ("center" if "center" in loc else "bottom")
    ha = "left" if "left" in loc else "right"
    vpad = pad if ypad is None else ypad
    x = pad if ha == "left" else 1 - pad
    y = {"top": 1 - vpad, "center": 0.5, "bottom": vpad}[va]
    t = ax.text(x, y, text, transform=ax.transAxes, va=va, ha=ha, fontsize=size,
                color=color or INK["secondary"], linespacing=1.35, zorder=5)
    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground=INK["surface"])])
    return t


def direct_label(ax, x, y, text, color=None, dx=3, dy=0, size: float = 7.5,
                 ha: str = "left", va: str = "center", halo: bool = True):
    """Label a series at its end point; text in ink, never the series color.

    ``halo`` paints a surface-coloured stroke behind the glyphs so a label that lands
    on a gridline or another series stays readable without a legend box.
    """
    import matplotlib.patheffects as pe
    t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                    va=va, ha=ha, fontsize=size, color=INK["secondary"], zorder=6,
                    annotation_clip=False)
    if halo:
        t.set_path_effects([pe.withStroke(linewidth=2.0, foreground=INK["surface"])])
    return t


def label_ends(ax, ends, size: float = 7.5, min_gap_frac: float = 0.055, dx: float = 3):
    """Direct-label a set of series at their end points, pushed apart so none collide.

    ``ends`` is ``[(x, y, text)]`` in data coordinates.  Two curves that converge --
    an oracle and the method that approaches it, two variants of one estimator -- would
    otherwise print their labels on top of each other, which is how a direct label
    becomes less readable than the legend it replaced.  Nudging happens in log space
    when the axis is logarithmic, so the gap is uniform on screen.
    """
    import math
    if not ends:
        return
    log = ax.get_yscale() == "log"
    y0, y1 = ax.get_ylim()
    f = (lambda v: math.log10(max(v, 1e-300))) if log else (lambda v: v)
    g = (lambda v: 10.0 ** v) if log else (lambda v: v)
    span = abs(f(y1) - f(y0)) or 1.0
    prev = -float("inf")
    for x, y, text in sorted(ends, key=lambda e: f(e[1])):
        yy = max(f(y), prev + min_gap_frac * span)
        direct_label(ax, x, g(yy), text, size=size, dx=dx)
        prev = yy


def log2_ticks(ax, axis: str = "x", lo: int | None = None, hi: int | None = None,
               every: int = 1) -> None:
    """Powers-of-two ticks labelled $2^k$, at a density that does not crowd.

    Matplotlib's default log-base-2 locator drops labels unpredictably on a short
    axis, which is how a sweep over ``2^0..2^9`` ends up labelled ``2^1, 2^3, 2^9``.
    """
    import math
    a = ax.xaxis if axis == "x" else ax.yaxis
    v0, v1 = (ax.get_xlim() if axis == "x" else ax.get_ylim())
    lo = int(math.floor(math.log2(v0))) if lo is None else lo
    hi = int(math.ceil(math.log2(v1))) if hi is None else hi
    ks = list(range(lo, hi + 1, every))
    a.set_major_locator(mpl.ticker.FixedLocator([2.0 ** k for k in ks]))
    a.set_minor_locator(mpl.ticker.NullLocator())
    a.set_major_formatter(mpl.ticker.FixedFormatter(
        [f"$2^{{{k}}}$" if k not in (0, 1) else ("1" if k == 0 else "2") for k in ks]))


def log_ticks_plain(ax, axis: str = "y", values=None) -> None:
    """Plain decimal labels on a log axis that spans less than ~1.5 decades.

    Matplotlib's log locator draws a single ``10^0`` across a 0.3-3 range and leaves
    every other value unreadable; a ratio panel whose whole point is "is it above or
    below 1" has to show 0.6 and 2 as numbers.
    """
    a = ax.yaxis if axis == "y" else ax.xaxis
    v0, v1 = (ax.get_ylim() if axis == "y" else ax.get_xlim())
    if values is None:
        cand = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 50.0]
        values = [v for v in cand if v0 <= v <= v1]
    a.set_major_locator(mpl.ticker.FixedLocator(values))
    a.set_minor_locator(mpl.ticker.NullLocator())
    a.set_major_formatter(mpl.ticker.FixedFormatter(
        [f"{v:g}" for v in values]))


def identity_line(ax, color=None, **kw):
    """A thin y = x guide across the current limits, drawn under the data."""
    lo = max(min(ax.get_xlim()[0], ax.get_ylim()[0]), 1e-300)
    hi = min(ax.get_xlim()[1], ax.get_ylim()[1])
    kw.setdefault("linewidth", 0.7)
    return ax.plot([lo, hi], [lo, hi], color=color or INK["muted"], zorder=0, **kw)


def save(fig, name: str, out_dir: Path | None = None):
    out = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)
    return out / f"{name}.pdf"
