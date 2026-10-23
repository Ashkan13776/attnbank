"""Regenerate ``docs/taxonomy.md`` from the live registry, so the page cannot drift.

    python docs/gen_taxonomy.py

The prose for each category is here; the tables come from ``ab.list_methods``.
"""
from __future__ import annotations

import pathlib
import re

import attnbank as ab

BLURB = {
 "exact": ("Nothing in the formula changes — only how it is evaluated. Tiling, "
           "chunking, reduced precision. The interesting question in this category is "
           "*where exactness stops*: `flash` and `splitk` are exact up to floating "
           "point, `lowprec` is not, and the boundary between them is the point."),
 "interaction": ("The interaction $g(q, x)$ is replaced by a cheaper surrogate "
                 "$\\widehat g$. Random features estimate $\\exp\\langle q,k\\rangle$; "
                 "fixed feature maps replace it outright. The first is an "
                 "approximation whose error shrinks with the feature count; the second "
                 "is a different operator that happens to be cheap. The `objective` "
                 "column is where that difference is recorded."),
 "restrict": ("The support of $\\pi_q$ shrinks to a subset $S_q$, and the retained mass "
              "is renormalised. The whole category is one operator — "
              "`prop:sparse-identity` — differing only in how $S_q$ is chosen. Error is "
              "purely *distributional*: the values are untouched."),
 "compress": ("The stored context $(\\mu, k, v)$ is replaced by something smaller: "
              "quantised, low-rank, evicted, merged, or shared across heads. This is "
              "the only category that routinely produces *value* error, because it is "
              "the only one that changes what is aggregated."),
 "factorize": ("$\\pi_q$ is routed through $r$ latents, $\\pi_q = a(q,\\cdot) b(\\cdot,x)$. "
               "Whether the factorisation can stay row-stochastic is the question: "
               "`latent_stochastic` is stochastic by construction, `svd_oracle` is the "
               "signed lower bound it cannot reach, and `nmf_oracle` measures the gap."),
 "coupling": ("Normalisation stops being per-query. Sinkhorn balances the whole "
              "matrix to prescribed marginals, which makes the result softmax *tilted "
              "by a key potential* rather than softmax. Queries are no longer "
              "independent, so causal masking is not available."),
 "recurrent": ("The operator becomes a finite-state recurrence: a fixed-size state "
               "$H_t$ absorbs the context. Linear in the values, so Invariant W still "
               "holds and the implied weights can still be written down — they are "
               "simply not a probability measure."),
}

HEADER = r"""# The taxonomy

Attention is a query-conditioned expectation,

$$y(q) \;=\; \int v \,\mathrm{d}\pi_q, \qquad \pi_q \;=\; \frac{g(q,\cdot)\,\mu}{Z(q)} .$$

Efficiency has to come from somewhere, and there are only so many places it can come
from. A method either evaluates this expression differently, or it changes one of the
objects in it. **Seven categories, because there are seven objects to change.**

Cutting across them is a second question, the one that decides how a method should be
judged at all:

| `objective` | meaning |
|---|---|
| `accelerate` | the same operator, exact up to floating point |
| `approximate` | targets dense softmax; judged by output error |
| `modify` | a *different operator*; fidelity to softmax is not the goal |

Conflating `approximate` and `modify` is the most persistent confusion in this
literature. Asking how closely `linear_elu` approximates softmax is asking the wrong
question — it is not trying to. Every method here declares which it is, and the
conformance suite uses the declaration: an `approximate` method must supply a
correctness anchor, a `modify` method is exempt and the suite *asserts the exemption*
rather than skipping it.

!!! info "This page is generated"
    The tables below come from the live registry, so they cannot drift from the code.
    Regenerate with `python docs/gen_taxonomy.py`; `ab.list_methods()` prints the same
    thing in a terminal.
"""

FOOTER = r"""
## Reading the columns

**`weights`** — can the method name its measure $\widehat\pi_q$ on the *original* $n$
positions? A `—` means it genuinely cannot, and the method is then excluded from the
error-decomposition studies **by declaration, not by accident**. See
[the invariant](invariant.md).

**`stochastic`** — are those weights nonnegative and summing to one? `nystrom`,
`linformer` and the unnormalised recurrences say no, and say it in an attribute rather
than in a footnote.

**`causal`** — a `—` is a statement about the operator. Sinkhorn normalises across
queries; there is no causal version of that, and asking for one raises an error that
says so.

**`backends`** — the execution paths this method claims are equal to the reference.
Most of the bank is `reference` only, which is the correct answer for an operator that
is not a dense softmax over a mask. See [Backends](backends.md).

## Oracles

Three methods — `value_topk`, `nmf_oracle`, `svd_oracle` — are not practical. They are
**upper bounds on what a mechanism can achieve**, which is exactly what several of the
monograph's arguments need: you cannot show that top-$k$ selection is suboptimal
without computing what the optimal selection would have been.
"""


def main() -> None:
    lines = [HEADER]
    for cat, blurb in BLURB.items():
        lines += [f"\n## `{cat}`\n",
                  f"**Changes:** {ab.categories()[cat]}.\n",
                  blurb + "\n",
                  "| method | objective | causal | weights | stochastic | backends | what it is |",
                  "|---|---|---|---|---|---|---|"]
        for r in ab.list_methods(cat):
            flag = lambda b: "yes" if b else "—"       # noqa: E731
            lines.append(
                f"| `{r['name']}` | {r['objective']} | {flag(r['causal'])} | "
                f"{flag(r['expresses_weights'])} | {flag(r['row_stochastic'])} | "
                f"{', '.join(r['backends'])} | {r['summary']} |")
        lines.append("")
    lines.append(FOOTER)
    text = re.sub(r"``([^`]+)``", r"`\1`", "\n".join(lines))
    out = pathlib.Path(__file__).resolve().parent / "taxonomy.md"
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
