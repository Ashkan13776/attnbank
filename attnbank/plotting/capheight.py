"""Measure a generated figure caption's typeset height, and the budget it leaves.

Not imported by any figure.  This is the tool behind :data:`style.FIG_HEIGHT`: the
caption is set *inside* the float, so a figure's height budget is ``\\textheight``
minus its own caption, and that caption is generated (``attnbank.studies.tables``),
so it changes whenever the caption text does.  Without a way to re-measure, the
budgets go stale silently and the next pass draws every figure against the wrong
height.

It typesets each ``sections/tables/cap_fig_*.tex`` caption in the real class
(``fnt/now.cls``) at the real ``\\textwidth``, in a ``\\vbox`` whose ``\\ht+\\dp``
is reported, and prints the budget :data:`style.FIG_HEIGHT` should carry.  Labels
are taken from ``fnt_main.aux`` so a ``\\cref`` sets at its real width rather than
as ``??``.

``fnt_main.aux`` must exist and be current, or every ``\\cref`` sets as ``??`` and
the measurement comes out short by a few characters a reference.  Tectonic keeps
its intermediates in a temp directory, so build the monograph once with::

    tectonic -X compile fnt_main.tex --keep-intermediates

Run from the repo root::

    python -m attnbank.plotting.capheight               # every caption
    python -m attnbank.plotting.capheight cap_fig_ot_tilt

Needs ``tectonic`` on PATH or at ``~/.local/bin/tectonic``.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

#: ``\textheight`` of ``fnt/now.cls`` at the monograph's settings, in inches.
#: 512.15pt of TeX points, which are 72.27 to the inch -- not 72.  The nine budgets
#: that predate this module were derived with the bp conversion (7.11in); the 0.02in
#: difference is inside the slack below, so they stand, but re-measure with this.
TEXTHEIGHT = 512.15 / 72.27

#: ``\abovecaptionskip`` (caption package default) and the slack held back, in inches.
#: The measured box already contains the skip, so it is reported separately rather
#: than subtracted twice.
ABOVECAPTIONSKIP = 10.0 / 72.27
SLACK = 8.0 / 72.27

PREAMBLE = r"""
\makeatletter
\def\input@path{{fnt/}}
\makeatother
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\PassOptionsToPackage{numbers,sort&compress}{natbib}
\PassOptionsToPackage{hidelinks}{hyperref}
\documentclass[openany]{now}
\usepackage{mathtools}
\usepackage[expansion=false]{microtype}
\usepackage{booktabs,multirow,array,xcolor,rotating}
\usepackage{graphicx}
\setlength{\headheight}{14.4pt}
\addtolength{\topmargin}{-2.4pt}
\newcommand{\R}{\mathbb R}
\newcommand{\dd}{\mathrm d}
\newcommand{\KL}{\operatorname{KL}}
\newcommand{\TV}{\operatorname{TV}}
\newcommand{\E}{\mathbb E}
\newcommand{\ind}{\mathbf 1}
\DeclareMathOperator{\softmax}{softmax}
\DeclareMathOperator{\rank}{rank}
\usepackage[capitalise,noabbrev]{cleveref}
\crefname{chapter}{Chapter}{Chapters}
\Crefname{chapter}{Chapter}{Chapters}
\makeatletter
\InputIfFileExists{@LABELS@}{}{}
\newsavebox{\capb}
\newenvironment{capmeasure}{%
  \setbox\capb=\vbox\bgroup\hsize=\textwidth \@parboxrestore \noindent
}{%
  \egroup
  \typeout{CAPHEIGHT \the\ht\capb\space + \the\dp\capb}%
}
\makeatother
"""


def _tectonic() -> str:
    local = Path.home() / ".local/bin/tectonic"
    return str(local) if local.exists() else (shutil.which("tectonic") or "tectonic")


def caption_body(path: Path) -> str:
    r"""The argument of the single ``\caption{...}`` in a generated cap_fig file."""
    tex = path.read_text()
    i = tex.index(r"\caption{")
    j, depth = i + len(r"\caption{"), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(tex[j], 0)
        j += 1
    return tex[i + len(r"\caption{"):j - 1]


def measure(names: list[str]) -> dict[str, float]:
    """Typeset each caption and return its ``\\ht+\\dp`` in inches (skip included)."""
    aux = ROOT / "fnt_main.aux"
    if not aux.exists():
        print("warning: no fnt_main.aux; every \\cref will set as '??' and every "
              "caption will measure short.  Build once with --keep-intermediates.",
              file=sys.stderr)
    labels = "\n".join(l for l in aux.read_text(errors="replace").splitlines()
                       if l.startswith(r"\newlabel")) if aux.exists() else ""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "labels.tex").write_text(labels)
        body = []
        for n in names:
            cap = caption_body(ROOT / "sections" / "tables" / f"{n}.tex")
            body.append(r"\typeout{CAPNAME " + n + "}")
            body.append(r"\begin{capmeasure}\captionof{figure}{" + cap
                        + r"}\end{capmeasure}")
        src = (PREAMBLE.replace("@LABELS@", str(td / "labels.tex"))
               + "\n\\begin{document}\n\\mainmatter\n\\setcounter{chapter}{5}\n"
               + "\n".join(body) + "\nx\n\\end{document}\n")
        # now.cls is found relative to the source file, so the scratch source has to
        # sit at the repo root next to fnt/; it is removed again either way
        srcf = ROOT / "_capheight_scratch.tex"
        srcf.write_text(src)
        try:
            p = subprocess.run([_tectonic(), "-X", "compile", "--outdir", str(td),
                                "--keep-logs", "--print", str(srcf)],
                               capture_output=True, text=True, cwd=ROOT)
        finally:
            srcf.unlink(missing_ok=True)
        log = td / "_capheight_scratch.log"
        text = p.stdout + p.stderr + (log.read_text(errors="replace")
                                      if log.exists() else "")
    out: dict[str, float] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"CAPNAME (\S+)", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"CAPHEIGHT ([\d.]+)pt \+ ([\d.]+)pt", line)
        if m and cur:
            out[cur] = (float(m.group(1)) + float(m.group(2))) / 72.27
            cur = None
    if not out:
        print(text[-3000:], file=sys.stderr)
        raise RuntimeError("no caption measured; see the log above")
    return out


def budget(measured: float) -> float:
    """``FIG_HEIGHT`` for a caption box of ``measured`` inches, rounded down to 0.05."""
    import math
    return math.floor((TEXTHEIGHT - measured - SLACK) * 20) / 20


def main(names: list[str] | None = None) -> None:
    names = names or sorted(p.stem for p in
                            (ROOT / "sections" / "tables").glob("cap_fig_*.tex"))
    res = measure(names)
    print(f"{'caption':34s} {'box':>6s} {'caption':>8s} {'FIG_HEIGHT':>11s}")
    for n in names:
        m = res[n]
        print(f"{n:34s} {m:6.3f} {m - ABOVECAPTIONSKIP:8.2f} "
              f"{budget(m):11.2f}   # {n[4:]}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
