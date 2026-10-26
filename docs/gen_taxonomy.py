"""Regenerate ``docs/taxonomy.md`` from the live registry, so the page cannot drift.

    python docs/gen_taxonomy.py

The page is a tree: the operator at the top and seven branches hanging off a trunk,
one per object of ``y(q) = int v dpi_q`` that a method can change. All 78 leaves are
always visible, hanging off their branch as linked chips; each chip carries the
method's ``objective`` as a glyph (= accelerate, ~ approximate, != modify), so both
axes of the taxonomy are on screen at once. Opening a branch reveals what the
category changes, preserves and costs, and lights the symbol of the operator it
reaches for. All of that is progressive enhancement -- ``docs/javascripts/taxonomy.js``
collapses the detail panels that are open in the HTML -- and the complete table of
every method follows underneath, so with scripting off the page is the whole taxonomy
with nothing to click and nothing missing.

The prose for each category is here, one entry per *category* and never per method;
everything else -- the method lists, the counts, the declaration columns -- comes from
``ab.list_methods`` and ``ab.describe``.
"""
from __future__ import annotations

import html
import pathlib
import re

from gen_methods import oracle_names

_COUNT_WORD = {1: "One method —", 2: "Two methods —", 3: "Three methods —",
               4: "Four methods —", 5: "Five methods —", 6: "Six methods —"}


def _oracle_sentence() -> str:
    """Name the oracles from the registry, so the prose cannot fall behind it.

    This sentence used to list three method names by hand. The expansion added a
    fourth and the generator's own consistency guard caught it -- which is the
    guard working, but the sentence should not have been hand-written at all.
    """
    names = sorted(oracle_names())
    links = [f"[`{n}`](methods/{n}.md)" for n in names]
    lead = _COUNT_WORD.get(len(names), f"{len(names)} methods —")
    if len(links) == 1:
        return f"{lead} {links[0]} —"
    return f"{lead} {', '.join(links[:-1])} and {links[-1]} —"

import attnbank as ab

#: The phrase that completes "this category changes ___", for the branch header. Kept
#: shorter than ``ab.categories()``, which is written for a terminal.
#: Hand-set rather than written as $math$: this phrase lives inside the branch
#: header, which is the one line of a closed branch, and MathJax does not run with
#: scripting off.
SHORT = {
 "exact": "nothing but the order of the arithmetic",
 "interaction": "the interaction <i>g</i>",
 "restrict": "the support of &#960;<sub><i>q</i></sub>",
 "compress": "the stored context (&#956;,&#8201;<i>k</i>,&#8201;<i>v</i>)",
 "factorize": "the shape of &#960;<sub><i>q</i></sub>",
 "coupling": "the normaliser <i>Z</i>",
 "recurrent": "the operator itself",
}

#: The one symbol of the operator that names the branch, set large on the branch
#: header so the seven categories are told apart by *what they touch* before a word
#: is read. It is the same claim ``SYMS`` makes to the operator plate, reduced to the
#: single glyph a reader can hold: ``exact`` lights nothing and so carries the empty
#: set, ``recurrent`` replaces the operator and so carries the operator's own ``y``.
#: Mnemonic only -- every one of them is said again in words in ``SHORT`` on the same
#: line, which is why the span is ``aria-hidden``.
GLYPH = {
 "exact": "&#8709;",                      # none: the arithmetic, not the formula
 "interaction": "<i>g</i>",
 "restrict": "&#8747;",                   # the domain the integral runs over
 "compress": "&#956;",
 "factorize": "&#960;",
 "coupling": "<i>Z</i>",
 "recurrent": "<i>y</i>",                 # the operator itself
}

#: Which symbols of the operator the category reaches for. ``exact`` reaches for none
#: and asks for the whole operator to be ruled off instead; ``recurrent`` replaces all
#: of them and shows the recurrence underneath.
SYMS = {
 "exact": ("", "frame"),
 "interaction": ("g", ""),
 "restrict": ("int", ""),
 "compress": ("mu v", ""),
 "factorize": ("pi", ""),
 "coupling": ("Z", ""),
 "recurrent": ("y int v pi g mu Z", "replace"),
}

#: What survives the change. The complement of ``SHORT``, and the half of the story a
#: table of methods never tells.
PRESERVES = {
 "exact": "everything. The measure, the values and the normaliser are the reference "
          "operator's, bit for bit where the arithmetic allows it.",
 "interaction": "the support, the values, and per-query normalisation — "
                "&#960;&#770;<sub><i>q</i></sub> still lives on all <i>n</i> positions.",
 "restrict": "the values and the scores of everything retained. Nothing is discarded "
             "from the context, so a later query may select what this one dropped.",
 "compress": "the operator and the per-query normalisation. What is no longer "
             "preserved is the context itself, which is the point.",
 "factorize": "row-stochasticity, when both factors are Markov kernels — a "
              "composition of two of them is still a probability measure.",
 "coupling": "the values, and &#960;&#770;<sub><i>q</i></sub> as a probability "
             "measure. What goes is the independence of the queries, and with it "
             "causal masking.",
 "recurrent": "linearity in <i>v</i>. That is the whole reason the implied weights can be "
              "recovered at all, and why Invariant W still holds here.",
}

#: Which term of the error decomposition the category incurs. The decomposition is
#: exact, so this is a statement about the mechanism, not a rule of thumb.
ERROR = {
 "exact": "**None**, up to floating point — which is a claim the conformance suite "
          "checks rather than a promise. `lowprec` is the exception, and locating the "
          "exception is the purpose of the category.",
 "interaction": "**Distributional only.** <i>v</i>&#770; = <i>v</i>, so every bit of the error "
                "is in the measure. For random features it shrinks with the feature "
                "count; for a fixed feature map it does not shrink at all, because "
                "there is nothing it is converging to.",
 "restrict": "**Distributional only**, and in closed form: the omitted mass "
             "&#949;<sub><i>q</i></sub> times the contrast between the omitted and the retained "
             "conditional means. Minimising &#949;<sub><i>q</i></sub> is not the same as "
             "minimising the error, which is what `value_topk` exists to show.",
 "compress": "**Both.** A distributional term through the keys and a value term "
             "through <i>v</i>&#770; &#8800; <i>v</i> — the only category that routinely "
             "produces the second. Key error passes through the softmax; value error "
             "is only reweighted, so keys cost more per bit.",
 "factorize": "**Distributional, with a floor set by rank.** No rank-<i>r</i> factorisation "
              "beats the truncated SVD, and a nonnegative one cannot reach it. "
              "`svd_oracle` is that lower bound and `nmf_oracle` measures the gap.",
 "coupling": "**Not an approximation error at all.** The result is exactly softmax "
             "tilted by one potential per key — a deterministic change of operator, "
             "measured against softmax only to say how far it has moved.",
 "recurrent": "**Not error but capacity.** A fixed-size state must answer every future "
              "query, which bounds recall however the model is trained. Fidelity to "
              "softmax is not the goal, so quoting an output error against it answers "
              "a question nobody asked.",
}

#: The longer note on each category. Maths is hand-set rather than written as
#: $...$ for the same reason SHORT is: the branch is readable before MathJax runs.
BLURB = {
 "exact": ("Nothing in the formula changes — only how it is evaluated. Tiling, "
           "chunking, reduced precision. The interesting question in this category is "
           "*where exactness stops*: `flash` and `splitk` are exact up to floating "
           "point, `lowprec` is not, and the boundary between them is the point."),
 "interaction": ("The interaction <i>g</i>(<i>q</i>,&#8201;<i>x</i>) is replaced by a "
                 "cheaper surrogate <i>g&#770;</i>. Random features estimate "
                 "exp&#8202;&#10216;<i>q</i>,&#8201;<i>k</i>&#10217;; fixed feature "
                 "maps replace it outright. The first is an approximation whose error "
                 "shrinks with the feature count; the second is a different operator "
                 "that happens to be cheap. The `objective` column is where that "
                 "difference is recorded."),
 "restrict": ("The support of &#960;<sub><i>q</i></sub> shrinks to a subset "
              "<i>S</i><sub><i>q</i></sub>, and the retained mass is renormalised. The "
              "whole category is one operator — `prop:sparse-identity` — differing "
              "only in how <i>S</i><sub><i>q</i></sub> is chosen. Error is purely "
              "*distributional*: the values are untouched."),
 "compress": ("The stored context (&#956;,&#8201;<i>k</i>,&#8201;<i>v</i>) is replaced "
              "by something smaller: quantised, low-rank, evicted, merged, or shared "
              "across heads. This is the only category that routinely produces *value* "
              "error, because it is the only one that changes what is aggregated."),
 "factorize": ("&#960;<sub><i>q</i></sub> is routed through <i>r</i> latents, "
               "&#960;<sub><i>q</i></sub> = <i>a</i>(<i>q</i>,&#8201;&#183;)&#8202;"
               "<i>b</i>(&#183;,&#8201;<i>x</i>). Whether the factorisation can stay "
               "row-stochastic is the question: `latent_stochastic` is stochastic by "
               "construction, `svd_oracle` is the signed lower bound it cannot reach, "
               "and `nmf_oracle` measures the gap."),
 "coupling": ("Normalisation stops being per-query. Sinkhorn balances the whole "
              "matrix to prescribed marginals, which makes the result softmax *tilted "
              "by a key potential* rather than softmax. Queries are no longer "
              "independent, so causal masking is not available."),
 "recurrent": ("The operator becomes a finite-state recurrence: a fixed-size state "
               "<i>H</i><sub><i>t</i></sub> absorbs the context. Linear in the values, "
               "so Invariant W still holds and the implied weights can still be "
               "written down — they are simply not a probability measure."),
}

#: The key to the leaves, one line between the operator and the tree. Each leaf
#: carries its method's ``objective`` as a glyph, and the three relations a method
#: can claim to the reference operator happen to *be* the three relation signs:
#: the same operator (=), an approximation of it (&#8776;), a different operator
#: (&#8800;). The glyph itself is set from ``data-obj`` by the stylesheet; the word
#: is on the chip's title and in the table below, so the glyph is never the only
#: carrier of the fact. The meanings here are the same three sentences as the
#: `objective` table further down, shortened.
KEY = (
    '<p class="tx-key">Each leaf is a method. Its glyph is the <code>objective</code>'
    ' it declares: <span class="tx-g">=</span>&#8201;accelerate, the same operator'
    ' &#183; <span class="tx-g">&#8776;</span>&#8201;approximate, judged by output'
    ' error &#183; <span class="tx-g">&#8800;</span>&#8201;modify, a different'
    ' operator. A <span class="tx-g-oracle">dashed leaf</span> is an oracle bound,'
    ' not a practical method.</p>'
)

HEADER = r"""---
hide:
  - toc
---

# The taxonomy

<div class="ab-measure" markdown="1">
Attention is a query-conditioned expectation. Efficiency has to come from somewhere,
and there are only so many places it can come from: a method either evaluates the
expectation differently, or it changes one of the objects in it. **Seven categories,
because there are seven objects to change.**

Every method in the bank hangs off one of the seven branches below. Open a branch to
see what that category changes, what it leaves alone, and which term of the error it
incurs — and which symbol of the operator it reaches for.
</div>
"""

AXIS = r"""
## The second axis

Cutting across all seven is the question that decides how a method should be judged at
all:

| | `objective` | meaning |
|---|---|---|
| <span class="tx-g">=</span> | `accelerate` | the same operator, exact up to floating point |
| <span class="tx-g">&#8776;</span> | `approximate` | targets dense softmax; judged by output error |
| <span class="tx-g">&#8800;</span> | `modify` | a *different operator*; fidelity to softmax is not the goal |

The glyph on every leaf of the tree above is this declaration.

Conflating `approximate` and `modify` is the most persistent confusion in this
literature. Asking how closely `linear_elu` approximates softmax is asking the wrong
question — it is not trying to. Every method here declares which it is, and the
conformance suite uses the declaration: an `approximate` method must supply a
correctness anchor, a `modify` method is exempt and the suite *asserts the exemption*
rather than skipping it.
"""

FOOTER = r"""
## Reading the columns

**`declares`** is three fixed slots, in this order, and a letter
<span class="tx-struck">struck through</span> is the method declaring *no* — which on
this page is never an omission and always a claim.

**<span class="tx-dec">w</span>, weights** — can the method name its measure
$\widehat\pi_q$ on the *original* $n$ positions? A struck-through `w` means it
genuinely cannot, and the method is then excluded from the error-decomposition studies
**by declaration, not by accident**. See [the invariant](invariant.md).

**<span class="tx-dec">s</span>, row-stochastic** — are those weights nonnegative and
summing to one? `nystrom`, `linformer` and the unnormalised recurrences say no, and
say it in an attribute rather than in a footnote.

**<span class="tx-dec">c</span>, causal** — a struck-through `c` is a statement about
the operator. Sinkhorn normalises across queries; there is no causal version of that,
and asking for one raises an error that says so.

**`backends`** — the execution paths this method claims are equal to the reference.
Most of the bank is `reference` only, which is the correct answer for an operator that
is not a dense softmax over a mask. See [Backends](backends.md).

## Oracles

{_oracle_sentence()} are not practical. They are
**upper bounds on what a mechanism can achieve**, which is exactly what several of the
monograph's arguments need: you cannot show that top-$k$ selection is suboptimal
without computing what the optimal selection would have been.

!!! info "This page is generated"
    Everything but the category prose comes from the live registry, so it cannot drift
    from the code. Regenerate with `python docs/gen_taxonomy.py`; `ab.list_methods()`
    prints the same rows in a terminal. To search and filter the whole bank at once —
    by category, objective, causal support, backend — use
    [**Browse the bank**](methods/index.md).
"""


#: The three declaration booleans, as the ``declares`` column renders them: a letter,
#: and what that letter is saying either way. One column instead of three is what lets
#: the table fit the page it is on; the slots are *fixed*, so a row that declares
#: nothing still shows three letters and a reader scanning down the column is reading
#: positions, not looking things up.
#: ``letter -> (registry field, what the letter says, what the struck letter says)``
DECLARES = {
    "c": ("causal", "causal", "not causal"),
    "w": ("expresses_weights",
          "names its weights on the original positions",
          "cannot name its weights on the original positions"),
    "s": ("row_stochastic", "row-stochastic", "not row-stochastic"),
}


def declares(row: dict) -> str:
    """The three booleans of one method as three glyphs, in a fixed order."""
    out = []
    for letter, (field, yes, no) in DECLARES.items():
        on = row[field]
        out.append(f'<abbr class="tx-dec"{"" if on else " data-no"} '
                   f'title="{yes if on else no}">{letter}</abbr>')
    return "".join(out)


#: The key, set in the same glyphs as the column so the two read as one thing. It sits
#: immediately above the table rather than in "Reading the columns" below it: a legend
#: a reader has to scroll to find is a legend that has not been written.
LEGEND = (
    '<p class="tx-legend" markdown="1">**`declares`** &mdash; '
    '<abbr class="tx-dec" title="causal">c</abbr>&#8201;causal, '
    '<abbr class="tx-dec" title="names its weights on the original positions">w'
    '</abbr>&#8201;names its weights on the original positions, '
    '<abbr class="tx-dec" title="row-stochastic">s</abbr>&#8201;row-stochastic. '
    'A letter <span class="tx-struck">struck through</span> is the method '
    'declaring it does <em>not</em>.</p>'
)


#: ASCII maths as it is written in the registry, mapped to the same hand-set HTML the
#: operator above uses. This is a renderer, not a second source of prose: whatever
#: ``ab.categories()`` says is what the page says, and a token with no rule here simply
#: passes through unchanged.
_TYPESET = [
    ("pi_q", "&#960;<sub><i>q</i></sub>"),
    ("S_q", "<i>S</i><sub><i>q</i></sub>"),
    ("H_t", "<i>H</i><sub><i>t</i></sub>"),
    ("g(q, x)", "<i>g</i>(<i>q</i>,&#8201;<i>x</i>)"),
    ("a(q, .) b(., x)", "<i>a</i>(<i>q</i>,&#8201;&#183;)&#8202;<i>b</i>(&#183;,&#8201;<i>x</i>)"),
    ("(mu, k, v)", "(&#956;,&#8201;<i>k</i>,&#8201;<i>v</i>)"),
    (" r latents", " <i>r</i> latents"),
]


def _sentence(text: str) -> str:
    """Capitalise, unless typesetting has already turned the first word into markup.

    ``pi_q is factored through ...`` must not become ``Pi_q``; once ``typeset`` has
    made it an entity there is nothing to capitalise.
    """
    return text[:1].upper() + text[1:] if text[:1].isascii() and text[:1].isalpha() \
        else text


def typeset(text: str) -> str:
    """The registry's ASCII maths, set the way the operator at the top of the page is."""
    for ascii_form, html in _TYPESET:
        text = text.replace(ascii_form, html)
    return text


def leaf(row: dict, oracles: tuple) -> str:
    """One method as one leaf: a link, its objective, and its summary on hover."""
    name = row["name"]
    tip = html.escape(row["summary"], quote=True)
    if name in oracles:
        tip += " &mdash; an oracle: an upper bound, not a practical method"
    # `../`: this page builds to taxonomy/index.html, so a bare "methods/x/"
    # resolves to taxonomy/methods/x/ and 404s. mkdocs rewrites markdown links
    # for you and leaves raw HTML hrefs alone, which is why the markdown links
    # elsewhere in this file are fine and these 78 were not.
    return (f'<li><a href="../methods/{name}/" data-obj="{row["objective"]}"'
            + (' data-oracle=""' if name in oracles else "")
            + f' title="{tip}">{name}</a></li>')


def tree(oracles: tuple) -> list[str]:
    """The seven branches. The leaves are always visible; only the detail folds.

    The leaf list sits *outside* ``.tx-body`` so that the collapsed tree is still a
    tree — seven branches with all 78 leaves hanging off them — rather than seven
    closed headings. What folds is the scholarly detail.
    """
    lines = ['<div class="tx-tree" markdown="1">\n',
             # inside the tree, so the trunk runs up past it to the plate unbroken
             KEY + "\n"]
    for cat, blurb in BLURB.items():
        rows = ab.list_methods(cat)
        syms, mode = SYMS[cat]
        attrs = f'data-cat="{cat}" data-syms="{syms}"' + (f" data-{mode}" if mode else "")
        chips = "".join(leaf(r, oracles) for r in rows)
        count = f'{len(rows)} method{"s" if len(rows) != 1 else ""}'
        lines += [
            f'<section class="tx-node" {attrs} markdown="1">\n',
            f'## <span class="tx-glyph" aria-hidden="true">{GLYPH[cat]}</span>'
            f'`{cat}` <span class="tx-changes">changes {SHORT[cat]}</span>'
            f'<span class="tx-n">{count}</span>'
            f' {{: #{cat} data-toc-label="{cat}" }}\n',
            f'<ul class="tx-leaves">{chips}</ul>\n',
            '<div class="tx-body" markdown="1">\n',
            '<div class="tx-facts" markdown="1">',
            f"Changes\n:   {_sentence(typeset(ab.categories()[cat]))}.\n",
            f"Preserves\n:   {PRESERVES[cat][0].upper()}{PRESERVES[cat][1:]}\n",
            f"Error term\n:   {ERROR[cat]}\n",
            "</div>\n",
            blurb + "\n",
            "</div>\n",
            "</section>\n",
        ]
    lines.append("</div>\n")
    return lines


def table(oracles: tuple) -> list[str]:
    """Every method, in one table, underneath the tree.

    The tree is the view; this is the index. It is plain Markdown so it is in the
    search index, linkable row by row, and present whether or not anything runs.
    """
    order = list(BLURB)
    rows = [r for cat in order for r in ab.list_methods(cat)]
    out = [
        "\n## Every method, in one table\n",
        f"All {len(rows)} of them, grouped by the object they change. This is the same "
        "content as the branches above, flattened — regenerated from "
        "`ab.list_methods()`, and the view that works whatever your browser is "
        "doing.\n",
        LEGEND + "\n",
        "<!-- Not indexed: the same 78 methods are indexed on their own\n"
        "     pages, and a second copy here matched nearly every query. -->\n",
        "<div class=\"tx-all-wrap\" data-search-exclude markdown=\"1\">\n",
        "| method | what it is | category | objective | declares | backends |",
        "|---|---|---|---|---|---|",
    ]
    for cat in order:
        for r in ab.list_methods(cat):
            name = (f"[`{r['name']}`](methods/{r['name']}.md)"
                    + (" **(oracle)**" if r["name"] in oracles else ""))
            # the sentence sits second, next to the name; the three booleans are one
            # column, not three, because three headers of their own were what pushed
            # the table past the width of the page it is on
            out.append(
                f"| {name} | {r['summary']} | "
                f"<span data-cat=\"{cat}\">`{cat}`</span> | "
                f"{r['objective']} | {declares(r)} | "
                f"{', '.join(r['backends'])} |")
    out.append("\n</div>\n")
    return out


def main() -> None:
    oracles = oracle_names()
    lines = [HEADER,
             '\n<div class="op" data-op markdown="1">\n',
             '<div class="op-plate tx-sticky" data-plate="the operator">',
             '--8<-- "docs/includes/operator.html"',
             "</div>\n",
             *tree(oracles),
             "</div>\n",
             AXIS,
             *table(oracles),
             FOOTER]
    text = re.sub(r"``([^`]+)``", r"`\1`", "\n".join(lines))
    out = pathlib.Path(__file__).resolve().parent / "taxonomy.md"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
