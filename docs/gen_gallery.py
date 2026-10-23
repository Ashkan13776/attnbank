"""Regenerate the examples gallery pages from the scripts in ``examples/``.

Each script's module docstring supplies the title and the prose; the code is *included*
rather than copied (``--8<--``), so a page can never drift from the script. The captured
output is stored beside it and is the only part that can go stale -- re-run this script
after changing an example::

    python docs/gen_gallery.py

``tests/test_examples.py`` executes every script in CI, so a broken example is a failing
build rather than a stale page.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PAGES = ROOT / "docs" / "examples"
OUTPUT = PAGES / "output"


def rst_to_md(text: str) -> str:
    """The docstrings are written in the package's RST-ish style; the site is Markdown."""
    text = re.sub(r"``([^`]+)``", r"`\1`", text)
    text = re.sub(r":class:`~?([\w.]+)`", r"`\1`", text)
    text = re.sub(r"::\s*$", ":", text, flags=re.M)
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


def split_docstring(text: str) -> tuple[str, str]:
    """``(title, prose)`` from the ``Title\\n=====\\n\\nprose`` module docstring."""
    doc = ast.get_docstring(ast.parse(text)) or ""
    lines = doc.strip().split("\n")
    title = lines[0].strip()
    body = "\n".join(lines[2:]).strip()
    return title, rst_to_md(body)


def main() -> int:
    PAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scripts = sorted(EXAMPLES.glob("*.py"))
    index = ["# Examples gallery", "",
             "Short, self-contained scripts. Every one of them runs as written, and",
             "`tests/test_examples.py` executes all of them in CI, so a broken example",
             "is a failing build rather than a stale page.", "",
             "```bash", "git clone https://github.com/Ashkan13776/attnbank",
             "python attnbank/examples/01_quickstart.py", "```", "",
             "| | example | what it shows |", "|---|---|---|"]

    for i, script in enumerate(scripts, 1):
        title, prose = split_docstring(script.read_text())
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, cwd=ROOT, timeout=900)
        if proc.returncode != 0:
            print(f"FAILED {script.name}\n{proc.stderr}", file=sys.stderr)
            return 1
        (OUTPUT / f"{script.stem}.txt").write_text(proc.stdout)

        page = [f"# {title}", "", prose, "",
                "## The script", "",
                f"```python title=\"examples/{script.name}\"",
                f"--8<-- \"examples/{script.name}\"", "```", "",
                "## Output", "",
                "```text",
                f"--8<-- \"docs/examples/output/{script.stem}.txt\"", "```", ""]
        (PAGES / f"{script.stem}.md").write_text("\n".join(page))
        first = prose.split("\n\n")[0].replace("\n", " ")
        index.append(f"| {i} | [{title}]({script.stem}.md) | {first} |")
        print(f"wrote docs/examples/{script.stem}.md")

    index += ["", "## Running them all", "",
              "```bash", "pytest tests/test_examples.py -q", "```", "",
              "Or regenerate these pages, output included:", "",
              "```bash", "python docs/gen_gallery.py", "```", ""]
    (PAGES / "index.md").write_text("\n".join(index))
    print("wrote docs/examples/index.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
