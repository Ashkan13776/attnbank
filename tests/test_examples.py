"""Every script in ``examples/`` is executed, in a subprocess, and must exit 0.

The gallery doubles as documentation, so a broken example is a broken page. Scripts
run in a subprocess for two reasons: the registry is process-global (``09`` registers
a method), and a hung example should not take the suite with it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


def test_the_gallery_is_not_empty():
    assert SCRIPTS, f"no examples found in {EXAMPLES_DIR}"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_runs(script: Path):
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, \
        f"{script.name} exited {proc.returncode}\n--- stderr ---\n{proc.stderr[-4000:]}"
    assert proc.stdout.strip(), f"{script.name} printed nothing; an example should show something"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_has_a_gallery_header(script: Path):
    """Each script opens with a title and a paragraph -- the gallery page is generated from it."""
    lines = script.read_text().splitlines()
    assert lines[0] == '"""', f"{script.name} must open with a module docstring"
    assert lines[1].strip(), f"{script.name} needs a title on the second line"
    assert set(lines[2].strip()) == {"="}, \
        f"{script.name}: underline the title with '=' so the gallery can render it"
