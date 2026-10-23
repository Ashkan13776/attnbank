"""Regenerate the examples gallery pages from the scripts in ``examples/``.

Each script's module docstring supplies the title and the prose; the code is *included*
rather than copied (``--8<--``), so a page can never drift from the script. The include
starts after the docstring, because the docstring already appears on the page as prose.
The captured output is stored beside it and is the only part that can go stale -- re-run
this script after changing an example::

    python docs/gen_gallery.py

The index is a gallery of cards in reading order. A card shows the example's number,
its title, the first paragraph of its docstring, and a short excerpt of its *captured
stdout* -- the result is the thumbnail. Nothing on a card is written here: the sentence
comes from the script, the excerpt is a verbatim slice of what the script printed, and
``SNIPS`` below guards each slice with the prefix its first line must carry, so a script
whose output moves fails this generator instead of shipping a stale excerpt.

``tests/test_examples.py`` executes every script in CI, so a broken example is a failing
build rather than a stale page.
"""
from __future__ import annotations

import ast
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PAGES = ROOT / "docs" / "examples"
OUTPUT = PAGES / "output"
REPO = "https://github.com/Ashkan13776/attnbank"

# The reading order, stated as structure: which card sits under which heading. The
# blurbs describe the *genre* of each group; every claim about the library itself
# lives in the scripts' own docstrings.
GROUPS: list[tuple[str, str, list[str]]] = [
    ("Start here", "",
     ["01_quickstart"]),
    ("The foundation", "The invariant every method satisfies, and the error split it"
     " makes possible. Everything else in the gallery measures with these two.",
     ["02_the_invariant", "03_error_decomposition"]),
    ("The results", "Not tutorials: each of these is a claim, with the run that backs"
     " it printed underneath.",
     ["04_topk_is_not_output_optimal", "05_random_features_converge",
      "06_matrix_error_is_not_output_error"]),
    ("The bank in practice", "Finding a method, and reading what it costs.",
     ["07_browse_the_bank", "08_cost_and_capacity"]),
    ("The contract", "What the declarations buy: a new method is tested against what"
     " it declares, and changing the backend cannot move a result.",
     ["09_add_your_own_method", "10_backends"]),
]

# Card excerpts: a 1-indexed inclusive line slice of the captured stdout, plus the
# prefix the first line must start with. The slice is shown verbatim.
SNIPS: dict[str, tuple[int, int, str]] = {
    "01_quickstart": (5, 10, "method "),
    "02_the_invariant": (1, 9, "method "),
    "03_error_decomposition": (1, 6, "method "),
    "04_topk_is_not_output_optimal": (1, 5, " budget k"),
    "05_random_features_converge": (1, 6, "     r "),
    "06_matrix_error_is_not_output_error": (9, 10, "ranked by Frobenius"),
    "07_browse_the_bank": (1, 9, "THE TAXONOMY"),
    "08_cost_and_capacity": (1, 9, "decode step"),
    "09_add_your_own_method": (19, 24, "example_sink_topk: "),
    "10_backends": (3, 9, "attnbank backends"),
}


def rst_to_md(text: str) -> str:
    """The docstrings are written in the package's RST-ish style; the site is Markdown."""
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    text = re.sub(r":class:`~?([\w.]+)`", r"`\1`", text)
    text = re.sub(r"::[ \t]*$", ":", text, flags=re.M)
    return _fence_indented_blocks(text)


def _fence_indented_blocks(text: str) -> str:
    """Wrap RST literal blocks (runs of indented lines) in Markdown code fences."""
    out, block = [], []
    for line in text.split("\n") + [""]:
        if line.startswith("    ") or (block and not line.strip()):
            block.append(line[4:] if line.startswith("    ") else line)
            continue
        if block:
            while block and not block[-1].strip():
                block.pop()
            out += ["```text", *block, "```", ""]
            block = []
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def split_docstring(text: str) -> tuple[str, str, int]:
    """``(title, prose, body_start)``: the parts of the ``Title\\n=====\\n\\nprose``
    module docstring, and the first code line after it (1-indexed)."""
    tree = ast.parse(text)
    doc = ast.get_docstring(tree) or ""
    lines = doc.strip().split("\n")
    title = lines[0].strip()
    body = "\n".join(lines[2:]).strip()
    start = tree.body[0].end_lineno + 1
    source = text.split("\n")
    while start <= len(source) and not source[start - 1].strip():
        start += 1
    return title, rst_to_md(body), start


def first_paragraph(prose: str) -> str:
    """The card sentence: the docstring's own first paragraph, on one line."""
    para = prose.split("\n\n")[0].replace("\n", " ").strip()
    return para[:-1] + "." if para.endswith(":") else para


def excerpt(stem: str, stdout: str) -> str:
    """A verbatim, guarded slice of the captured stdout, escaped for a raw ``<pre>``."""
    lo, hi, guard = SNIPS[stem]
    lines = stdout.split("\n")
    if len(lines) < hi or not lines[lo - 1].startswith(guard):
        raise SystemExit(f"{stem}: output moved; line {lo} no longer starts with"
                         f" {guard!r} -- update SNIPS in {__file__}")
    return html.escape("\n".join(lines[lo - 1:hi]))


def card(number: int, stem: str, title: str, para: str, snip: str) -> list[str]:
    return ['<div class="gal-card" markdown>',
            '<div class="gal-card-text" markdown>', "",
            f'### <span class="gal-num">{number:02d}</span> [{title}]({stem}.md)', "",
            para, "",
            "</div>", "",
            f'<pre class="gal-snip"><code>{snip}</code></pre>', "",
            "</div>", ""]


def page(script: Path, title: str, prose: str, body_start: int, n_lines: int) -> str:
    name = script.name
    lines = [
        f"# {title}", "",
        f'<p class="gal-meta"><code>examples/{name}</code> &middot;'
        f' <a href="{REPO}/blob/main/examples/{name}">source on GitHub</a></p>', "",
        prose, "",
        "## The script", "",
        f'```{{.python title="examples/{name}"}}',
        f'--8<-- "examples/{name}:{body_start}:{n_lines}"',
        "```", "",
        "## Output", "",
        '<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code>'
        ' ran the script and stored what it printed, and'
        ' <code>tests/test_examples.py</code> runs it again in CI.</p>', "",
        f'```{{.text .gal-stdout title="$ python examples/{name}"}}',
        f'--8<-- "docs/examples/output/{script.stem}.txt"',
        "```", ""]
    return "\n".join(lines)


def main() -> int:
    PAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scripts = {p.stem: p for p in sorted(EXAMPLES.glob("*.py"))}
    order = [stem for _, _, stems in GROUPS for stem in stems]
    if sorted(order) != sorted(scripts):
        raise SystemExit(f"GROUPS and examples/ disagree: "
                         f"{sorted(set(order) ^ set(scripts))}")

    total_lines = sum(len(p.read_text().splitlines()) for p in scripts.values())
    cards: dict[str, list[str]] = {}
    for number, stem in enumerate(order, 1):
        script = scripts[stem]
        source = script.read_text()
        title, prose, body_start = split_docstring(source)
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, cwd=ROOT, timeout=900)
        if proc.returncode != 0:
            print(f"FAILED {script.name}\n{proc.stderr}", file=sys.stderr)
            return 1
        (OUTPUT / f"{stem}.txt").write_text(proc.stdout)
        n_lines = len(source.splitlines())
        (PAGES / f"{stem}.md").write_text(page(script, title, prose, body_start, n_lines))
        cards[stem] = card(number, stem, title, first_paragraph(prose),
                           excerpt(stem, proc.stdout))
        print(f"wrote docs/examples/{stem}.md")

    n = len(order)
    index = [
        "# Examples", "",
        f'<p class="gal-lead">Short, self-contained scripts -- and the output under'
        f' each one is real. <code>docs/gen_gallery.py</code> runs every script and'
        f' stores its stdout next to the code, and CI runs all {n} again on each'
        f' build, so a broken example is a failing build rather than a stale'
        f' page.</p>', "",
        f'<p class="gal-facts"><span>{n} scripts</span><span>{total_lines} lines of'
        f' Python</span><span>0 outputs written by hand</span></p>', "",
        "```bash",
        f"git clone {REPO}",
        f"python attnbank/examples/{scripts[order[0]].name}",
        "```", ""]
    for heading, blurb, stems in GROUPS:
        index += [f"## {heading}", ""]
        if blurb:
            index += [f'<p class="gal-blurb">{blurb}</p>', ""]
        index += ['<div class="gal-grid" markdown>', ""]
        for stem in stems:
            index += cards[stem]
        index += ["</div>", ""]
    index += ["## Run them yourself", "",
              "Every example runs from a clone, with no arguments. The whole gallery:", "",
              "```bash",
              "pytest tests/test_examples.py -q   # execute every script, as CI does",
              "python docs/gen_gallery.py         # re-run them and regenerate these pages",
              "```", ""]
    (PAGES / "index.md").write_text("\n".join(index))
    print("wrote docs/examples/index.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
