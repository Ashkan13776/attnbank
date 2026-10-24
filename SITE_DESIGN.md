# Site design contract

Ten agents are working on this site at once. This file is what stops them
building ten sites. If an agent's taste and this file disagree, this file wins —
raise the disagreement in your report instead of acting on it.

The author's brief: the current site is *dull*, and the direction is **a
combination of three** — the precision of a scientific instrument, the energy of
a modern developer tool, and the memorability of something genuinely visual.
Those pull against each other if applied everywhere at once. They do not if each
owns a different part of the site.

---

## The zone doctrine

**Entry surfaces — developer tool.** `index.md`, `quickstart.md`, the examples
index. Confident hero, strong type, an obvious first action, visible momentum.
This is where someone decides whether to keep reading. Energy is appropriate
here and nowhere else.

**Reference surfaces — scientific instrument.** Method pages, the API reference,
the comparison view, the browse table. Dense, typographic, quiet, trustworthy.
Someone here already decided; do not sell to them. Information density is a
feature. No hero, no motion that is not functional.

**Signature moments — visual.** The operator plate and the taxonomy tree. Two
places, deliberately. They are what someone remembers and links to. Spend the
boldness here; it buys the restraint everywhere else.

A page belongs to exactly one zone. If you find yourself wanting dev-tool energy
on a method page, the answer is no — that is what the front page is for.

---

## What makes this subject distinctive

Every efficient attention method changes **one object in one formula**:

$$y(q) = \int v\,\mathrm{d}\pi_q, \qquad \pi_q = g(q,\cdot)\,\mu / Z(q)$$

Seven categories, each defined by which symbol it touches. That is the site's
one genuinely unusual asset, and no other ML library has it. Design from it.
Generic ML-library chrome — gradient hero, floating cards, a row of feature
icons — is what everyone else already has and is the thing to avoid.

---

## Token contract

Agent 1 owns `docs/stylesheets/tokens.css`. Nobody else edits it, and nobody
declares a raw colour, face, size or duration anywhere else.

These tokens **exist today** and are referenced across the other eleven
stylesheets. Agent 1 may change any **value**; agent 1 may **not rename** any of
them, because a rename means editing nine other agents' files at once:

```
--ab-serif  --ab-mono                           faces (system stacks; no webfont fetch)
--ab-plate  --ab-sunk                           surfaces
--ab-rule   --ab-rule-soft  --ab-faint          rules and recessive ink
--c-exact --c-interaction --c-restrict --c-compress
--c-factorize --c-coupling --c-recurrent        the seven category hues
--hl                                            the current category's hue, set by [data-cat]
```

The site is thin on tokens, which is part of why it reads as unconsidered:
there is no type scale, no spacing scale, no named ink levels, no motion
durations, and no accent that is not a category hue. **Agent 1 adds them**, in
`tokens.css`, under the `--ab-` prefix, and reports the list so the other nine
can adopt it. Suggested shape, not a mandate:

```
--ab-ink  --ab-ink-soft  --ab-ink-muted         text, three levels
--ab-accent  --ab-accent-soft                   one accent, distinct from the seven hues
--ab-display                                    a display face, if the identity wants one
--ab-step--1 … --ab-step-5                      type scale
--ab-space-1 … --ab-space-8                     spacing scale
--ab-radius  --ab-rule-width
--ab-motion-fast  --ab-motion-slow  --ab-ease
```

Until agent 1 publishes those, build with what exists and with Material's own
`--md-*` tokens; do not hardcode. If you need a token that is not there, ask
agent 1 through the coordinator rather than inventing one locally.

**Category hues are `attnbank.plotting.style.CATEGORICAL` verbatim and may not
be changed in hue.** A category is the same colour on this site, in every figure
of the monograph, and in the paper's taxonomy figure. That consistency is
earned; do not spend it. The dark-theme block lifts them for the dark ground —
same hue, same order — and that is the only licence.

Colour carries meaning only through category and state. It is never decoration,
and category hues never sit behind text or become text: the amber and the green
clear 4.5:1 on neither ground. Rules, ticks, and underlines.

**Themes.** Material stamps `data-md-color-scheme` and `data-md-color-accent`
on **`<body>`**, not on `<html>`. This matters more than it sounds: a custom
property resolves from the nearest ancestor that sets it, so Material's
body-level palette beats anything `tokens.css` declares on `:root`, whatever
the specificity. Declaring the `--md-*` bridge only on `:root` is how content
links came out Material indigo in light and instrument cyan in dark in the same
build — the dark bridge escaped only because its selector happens to be
body-level too. Declare every colour at body level, in both schemes.

## Non-negotiable

These are not style preferences. Each was expensive to establish and is checked.

1. **Zero third-party requests.** No CDN, no webfont fetch, no API call, no
   remote image. MathJax is vendored at 2 MB to earn this. I verify by rendering
   with every host except localhost blocked at the resolver.
2. **`navigation.instant` is on.** Scripts initialise through
   `document$.subscribe(...)` with a fallback — copy the pattern at the foot of
   `docs/javascripts/bank.js`. A feature that works on reload and dies on the
   first click is this site's known failure mode.
3. **Works with JavaScript off.** Content present, no dead controls. Interactive
   enhancements are progressive, never load-bearing.
4. **Generated pages stay generated.** `docs/methods/*`, `docs/taxonomy.md` and
   the nav come from `docs/gen_methods.py` and `docs/gen_taxonomy.py`. Change the
   generator, never its output; both must stay byte-identical on re-run.
5. **Nothing is invented.** Every fact on a page comes from the registry, a
   module docstring, or `references.bib`. No illustrative numbers, no example
   output that was not executed, no claim about a method that the method does
   not declare.
6. **`mkdocs build --strict` clean; `pytest lib/tests` at 2967 passed, 367
   skipped.**
7. **Respect `prefers-reduced-motion`,** keep contrast legible in both themes,
   and give keyboard focus a visible state.

---

## How to work

**Look at what you build.** Render every page you change with `google-chrome`,
in light *and* dark, and with scripting disabled, and iterate on what you see. A
page you have not viewed is not done. This has caught, in this project already:
overlapping titles, a legend sitting on its own inset, type silently scaled to
5.4 pt, and a formula rendering as raw TeX.

**Study the field first, and say what you took.** Before designing, look at how
strong library sites solve your aspect — PyTorch, Hugging Face, POT, scikit-learn,
Polars, Tailwind, Astro, Three.js, Observable, Stripe docs are a reasonable
starting set. Report what you borrowed and what you rejected. An aspect designed
without looking is the one that ends up generic.

**Prefer removing.** The site is dull partly because it is uniform, and
uniformity is usually too much of something. Cutting a weak element is a valid
deliverable.

---

## Ownership

Ten aspects, ten owners. **Edit only what your row names.** Everything else on
the site is read-only to you — including the other nine agents' stylesheets.

| # | aspect | zone | owns |
|---|---|---|---|
| 1 | Visual identity and typography | all | `docs/stylesheets/tokens.css`, `docs/stylesheets/base.css` |
| 2 | The landing page | dev-tool | `docs/index.md`, `docs/stylesheets/home.css` |
| 3 | The operator plate | **visual** | `docs/includes/operator.html`, `docs/javascripts/operator.js`, `docs/stylesheets/operator.css` |
| 4 | The taxonomy tree | **visual** | `docs/gen_taxonomy.py`, `docs/javascripts/taxonomy.js`, `docs/stylesheets/taxonomy.css` |
| 5 | Browse and search | instrument | `docs/javascripts/bank.js`, `docs/stylesheets/browse.css`, and `browser()` + its helpers in `docs/gen_methods.py` |
| 6 | Method pages | instrument | `docs/stylesheets/method.css`, and `page()` + its block builders in `docs/gen_methods.py` |
| 7 | The comparison view | instrument | `docs/stylesheets/compare.css`, and `compare_payload()` + `GROUPS` in `bank.js` |
| 8 | The examples gallery | dev-tool | `docs/examples/index.md`, `docs/gen_gallery.py`, `docs/stylesheets/gallery.css` |
| 9 | The long-form pages | instrument | `docs/quickstart.md`, `docs/invariant.md`, `docs/conformance.md`, `docs/backends.md`, `docs/stylesheets/prose.css` |
| 10 | Chrome, motion and states | all | `overrides/`, `docs/stylesheets/chrome.css`, `docs/stylesheets/motion.css`, `docs/javascripts/motion.js` |

Three files are shared, and the rule that makes that safe is simple:

- **`docs/gen_methods.py`** is split between agents 5 and 6, and **`bank.js`**
  between 5 and 7. In a shared file, use `Edit` — never `Write`. Concurrent
  edits to different functions merge; a whole-file write destroys the other
  agent's work silently.
- **`mkdocs.yml` is read-only for everyone.** Every stylesheet and script you
  could need is already registered and loads in cascade order. If you need a new
  page in the nav, a new theme feature or a new asset, **ask the coordinator** —
  ten agents editing one YAML file is the one collision nothing recovers from.

`docs/javascripts/clip.js`, `docs/javascripts/mathjax.js`, `docs/tex_source.py`,
`docs/griffe_roles.py` and everything outside `docs/` and `overrides/` are
read-only to all ten.

### What was done before you started

`attnbank.css` and `bank.css` were two files that all ten of you would have
edited at once. They are now twelve, one per aspect, split along their existing
section comments — the rule multiset is unchanged, verified. `attnbank.js` was
likewise split into `operator.js` and `taxonomy.js`; `operator.js` publishes
`window.AttnOperator = {SYMS, figure, symsOf}` and `taxonomy.js` consumes it,
because the tree drives the same plate. `motion.css`, `motion.js`,
`gallery.css` and `prose.css` are new and deliberately empty.

So your stylesheet already holds the rules for your aspect, and the site renders
exactly as it did before. You are redesigning from a working baseline, not a
blank file.

### Reporting

Finish with: what you studied and what you took from it, what you changed, what
you deliberately did not change, anything you want from another aspect's owner,
and anything you believe is wrong with this contract.

---

## Amendment: why the first pass came out grey

The ten-aspect redesign landed and the author's verdict was still "dull". That
is a fair reading of the result, and two clauses above are why.

**"Colour is never decoration" was too strong.** It was written to stop the
generic gradient-hero look, and it did. But combined with "hue only as rules,
ticks and underlines", it left seven category colours appearing as 2px marks
and nothing else, on 4,700 pixels of grey text on white. The rule stays in
substance — hue still *means* category, never ornament, and still never sits
behind text it would make illegible — but a category may now own a visible
area, not only a line: a filled chip, a tinted well, a band. If a reader cannot
tell the seven apart at a glance, the colour system is not doing its job.

**The site has no pictures, and that is the bigger miss.** This library exists
to measure things, and the site shows not one measurement. Error splits,
convergence, cost against capacity — all of it is computed by code in this
repository and none of it is drawn anywhere on the site.

Figures are now a first-class surface, under two rules:

1. **The site draws its own.** `figures/` in the monograph repository is the
   author's unpublished work and does not belong on a public site before the
   paper. The site's figures are computed from the library at build time, by a
   generator in `docs/`, so they are reproducible by anyone who clones the
   public repository and are covered by "nothing is invented" like every other
   number here.
2. **They use the house palette.** `attnbank.plotting.style.CATEGORICAL`, the
   same seven hues as the site and the monograph, so a category is one colour
   everywhere.

**What is still forbidden** is unchanged and still load-bearing: zero
third-party requests, works with JavaScript off, generated pages stay
generated, nothing invented, `--strict` clean, `prefers-reduced-motion`
respected, and the generic ML-landing-page furniture — gradient hero, floating
feature cards, icon rows, logo clouds.
