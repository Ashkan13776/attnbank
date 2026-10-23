"""Fail the build when the library, its documentation and the monograph disagree.

Three artefacts state facts about each other: the registry, the documentation
site, and the monograph that cites them. Nothing enforced agreement, and the
first expansion wave broke every hardcoded count at once -- doctests asserting
"28 reference-only methods", a README saying "32 methods", a study note claiming
no factorize method is causal. Each was true when written.

The fix is not to update the literals. It is to stop writing them down, and to
fail the build where that was not possible.

    python tools/check_sync.py            # all applicable checks
    python tools/check_sync.py --verbose  # list every item checked

Checks that need the monograph (``sections/``, ``references.bib`` one level up)
are skipped with a note when it is absent, because the library is public and
standalone while the monograph is not.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]
PAPER = LIB.parent                      # the monograph repo, if this is a submodule


class Report:
    def __init__(self, verbose: bool) -> None:
        self.fail: list[str] = []
        self.skip: list[str] = []
        self.ok: list[str] = []
        self.verbose = verbose

    def check(self, name: str, problems: list[str]) -> None:
        (self.fail.extend(f"{name}: {p}" for p in problems) if problems
         else self.ok.append(name))

    def skipped(self, name: str, why: str) -> None:
        self.skip.append(f"{name} ({why})")

    def exit_code(self) -> int:
        for s in self.skip:
            print(f"skip  {s}")
        if self.verbose:
            for o in self.ok:
                print(f"ok    {o}")
        for f in self.fail:
            print(f"FAIL  {f}", file=sys.stderr)
        print(f"\n{len(self.ok)} ok, {len(self.skip)} skipped, {len(self.fail)} failed")
        return 1 if self.fail else 0


def bib_keys(path: Path) -> set[str]:
    return set(re.findall(r"^@\w+\s*\{\s*([^,\s]+)", path.read_text(), re.M))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true")
    r = Report(ap.parse_args().verbose)

    sys.path.insert(0, str(LIB))
    import attnbank as ab

    table = ab.method_table()
    names = [row["name"] for row in table]

    # ---- 1. every declared citation key resolves -------------------------------
    keys = bib_keys(LIB / "attnbank" / "references.bib")
    missing = []
    for row in table:
        for k in (row["references"] or "").split(","):
            if k.strip() and k.strip() not in keys:
                missing.append(f"{row['name']} cites {k.strip()}, which is not in references.bib")
    r.check("citation keys resolve", missing)

    # ---- 2. the generated bibliography is current ------------------------------
    try:
        from merge_bib import build  # type: ignore
        text, conflicts = build()
        current = (LIB / "attnbank" / "references.bib").read_text()
        r.check("references.bib is up to date",
                conflicts + ([] if current == text
                             else ["stale; run tools/merge_bib.py"]))
    except Exception as exc:                                   # pragma: no cover
        r.skipped("references.bib is up to date", f"{type(exc).__name__}: {exc}")

    # ---- 3. no duplicate or malformed registry names ---------------------------
    dupes = [n for n in set(names) if names.count(n) > 1]
    bad = [n for n in names if not re.fullmatch(r"[a-z0-9_]+", n)]
    r.check("registry names are unique and well formed",
            [f"duplicate: {n}" for n in dupes] + [f"malformed: {n}" for n in bad])

    # ---- 4. counts written into prose match the registry -----------------------
    n_methods = len(table)
    stale = []
    for rel in ("README.md", "docs/index.md", "docs/taxonomy.md",
                "docs/methods/index.md", "CHANGELOG.md"):
        p = LIB / rel
        if not p.exists():
            continue
        text = p.read_text()
        cats = {row["category"] for row in table}
        # "78 methods", and also "all 78" / "every one of the 78", which is how the
        # count gets written when the noun is already in the heading. The second
        # form is the one that went stale unnoticed: the home page said "held by
        # all 32" for the whole first expansion.
        for m in re.finditer(
                r"\b(\d{2,4})\s+methods\b|\ball\s+(\d{2,4})\b", text):
            got = int(m.group(1) or m.group(2))
            if got == n_methods:
                continue
            # a per-category count is not a claim about the library; the generated
            # taxonomy page legitimately says "18 methods" of compress.
            ctx = text[max(0, m.start() - 160):m.end() + 160]
            if any(re.search(rf"\b{c}\b", ctx) for c in cats):
                continue
            stale.append(f"{rel} says {got} methods ({m.group(0)!r}), "
                         f"registry has {n_methods}")
    r.check("method counts in prose", stale)

    # ---- 5. test counts written into prose match the suite ---------------------
    # The site advertises the suite's size on the home page, the conformance page
    # and in the README. Three separate expansions moved it and none of the three
    # followed; the figure read 2838 for a suite that had grown to 3334. Collecting
    # costs about a second, which is worth paying to stop the number being
    # decoration. Tags are stripped first because the home page sets the digits in
    # <b> and the word beside them in <span>, which no plain-text pattern sees.
    import subprocess

    def collect(target: str) -> int | None:
        out = subprocess.run([sys.executable, "-m", "pytest", target, "-q",
                              "--collect-only"], cwd=LIB, capture_output=True,
                             text=True).stdout
        m = re.search(r"(\d+) tests? collected", out)
        return int(m.group(1)) if m else None

    total = collect("tests")
    if total is None:
        r.skipped("test counts in prose", "could not collect the suite")
    else:
        # A page may also quote one file's own count; "N tests from this file"
        # is checked against that file rather than against the whole suite.
        per_file = {"this file": collect("tests/test_conformance.py")}
        bad = []
        for rel in ("README.md", "docs/index.md", "docs/conformance.md",
                    "docs/invariant.md", "docs/quickstart.md", "SITE_DESIGN.md"):
            path = LIB / rel
            if not path.exists():
                continue
            text = re.sub(r"<[^>]+>", "", path.read_text())      # drop tags
            for pa, sk in re.findall(
                    r"\b(\d{3,5})\s*passed[\s,\u00b7/-]+(\d{2,5})\s*skipped", text):
                if int(pa) + int(sk) != total:
                    bad.append(f"{rel}: {pa} passed + {sk} skipped != {total} collected")
            for m in re.finditer(r"\b(\d{3,5})\s*\+?\s*tests\b", text):
                want, tail = total, text[m.end():m.end() + 40]
                for phrase, n in per_file.items():
                    if n and phrase in tail:
                        want = n
                if int(m.group(1)) != want:
                    bad.append(f"{rel}: {m.group(0)!r} != {want}")
        r.check("test counts in prose", bad)

    # ---- 6. paper_section labels exist in the monograph ------------------------
    sections = PAPER / "sections"
    if not sections.is_dir():
        r.skipped("paper_section labels exist", "monograph not present")
    else:
        src = "\n".join(p.read_text() for p in sections.glob("*.tex"))
        labels = set(re.findall(r"\\label\{([^}]+)\}", src))
        bad_lbl = []
        for row in table:
            lbl = row.get("paper_section")
            if lbl and lbl not in labels:
                bad_lbl.append(f"{row['name']} declares {lbl}, which no section defines")
        r.check("paper_section labels exist", bad_lbl)

    # ---- 7. the monograph's own library counts ---------------------------------
    if not sections.is_dir():
        r.skipped("monograph counts match the library", "monograph not present")
    else:
        words = {"thirty-two": 32, "thirty two": 32}
        src = "\n".join(p.read_text() for p in sections.glob("*.tex"))
        wrong = []
        for w, v in words.items():
            if re.search(rf"{w}\s+(methods|of them)", src, re.I) and v != n_methods:
                wrong.append(f"sections/ says '{w}' methods, registry has {n_methods}")
        for m in re.finditer(r"\b(\d{2,4})\s+methods\b", src):
            # Appendix C counts surveyed methods, not implemented ones; only flag
            # claims that are explicitly about this library.
            ctx = src[max(0, m.start() - 120):m.start()]
            if re.search(r"attnbank|the library|companion library", ctx, re.I) \
                    and int(m.group(1)) != n_methods:
                wrong.append(f"sections/ claims {m.group(1)} library methods, "
                             f"registry has {n_methods}")
        r.check("monograph counts match the library", wrong)

    return r.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
