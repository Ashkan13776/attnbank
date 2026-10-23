# Changelog

All notable changes to `attnbank` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The one thing that will never change silently is **Invariant W**
(`y == weights @ (v_hat or value)`). If a release ever relaxes it, that release is a
major version and this file says so in the first line.

## [Unreleased]

Documentation site only. No method's mathematics, no number and no result file
changed, and the site still makes **zero third-party requests**.

### Added

- **Side-by-side comparison** on the **Browse the bank** page. Tick two to
  four rows and the methods appear in columns: category, objective, paper section, the
  five declaration booleans, the oracle flag, backends, correctness anchor,
  configuration defaults, the three cost quantities at the shape every method page
  quotes, and the BibTeX keys. Every value is read from `ab.describe()` and the
  method's own `cost()` — `docs/gen_methods.py` writes them into one JSON block on the
  page. Rows the selected methods agree on fold away, a cell is marked when it is the
  odd one out, and a property not declared by *all* of the selected methods is not
  shown at all rather than guessed at.
- **URL state.** Search text, chip filters, sort column and direction, and the
  comparison selection are all in the query string, so a filtered view or a comparison
  is a link you can paste. It survives `navigation.instant` and is restored on load.
- **Copy controls.** A copy-BibTeX button beside each citation key on a method page, a
  *copy all* for the page's bibliography, and a copy-LaTeX button on every display
  formula and on the operator plate — the TeX source, not the rendered form.
- **Related methods** on every method page: the methods that change the same object
  and the methods that pursue the same objective, both derived from the registry.
- `/` focuses the search box of the method browser.

### Changed

- The *Others in `<category>`* section of a method page is now *Related methods*, and
  names the same-objective methods as well as the same-category ones.
- `docs/tex_source.py`, a new mkdocs hook, stamps each display formula's LaTeX source
  onto its element at build time; MathJax replaces the source with SVG, so scraping it
  in the browser is a race.

## [0.2.0] — 2026-10-22

The release that turns the research code into a library. No method's mathematics, no
number and no result file changed; everything below is interface, documentation and
packaging.

### Added

- **`ab.list_methods()`** — the bank as a readable table grouped by category, with the
  declaration columns that drive the test suite. Returns a `MethodTable`, which is a
  plain `list` of row dicts that happens to print well, and carries `.names` and
  `.filter(**constraints)`.
- **`ab.describe(name)`** — one method in full: what it changes, what it declares, its
  constructor signature with defaults, its correctness anchor, its paper section and
  its citations. Also reachable as `AttentionMethod.describe()`.
- **`ab.categories()`** — the taxonomy as `{category: what it changes}`.
- **`ab.check_invariant(output, value)`** and **`ab.check_conformance(name, **config)`** —
  the design contract as a runtime API, not only as a test file. `check_conformance`
  returns a printable report over shapes, Invariant W, row stochasticity, causal
  non-leakage, finiteness at large logits, input immutability, reproducibility, the
  cost model and the exact limit.
- **Errors that teach.** A new `attnbank.errors` module. Unknown method names suggest
  near matches (`difflib`); shape errors name the argument, what was expected and what
  arrived; `causal=True` on a `supports_causal = False` method explains that this is a
  statement about the operator and names the methods that do support it; an
  unrecognised config key is rejected with the accepted keys and a near match, instead
  of being silently swallowed. Every class subclasses both `AttnBankError` and the
  built-in exception a caller would catch, so existing `except KeyError` / `except
  ValueError` handlers keep working.
- **`__repr__` everywhere.** `AttentionOutput` shows the shape of each field, the keys
  in `extras`, the dtype and the device. `AttentionMethod` shows its name, category,
  objective and configuration.
- **Type hints and a `py.typed` marker**, so downstream type checkers see the API.
- **Runnable examples in every public docstring**, executed in CI by
  `tests/test_docs.py`; a public function without an example now fails the build.
- **An examples gallery** (`examples/`) of short, self-contained scripts, each executed
  in CI by `tests/test_examples.py` and rendered as a page of the documentation site.
- **A documentation site** (MkDocs + Material + mkdocstrings): quickstart, the taxonomy,
  the invariant, the conformance suite, the API reference and the gallery.
- **A packaged bibliography** (`attnbank/references.bib`) containing every key cited by
  a method, so conformance check 11 — "every citation resolves" — survives the split
  from the monograph's repository.
- **An execution-backend layer.** A method is now an *operator* (one readable file) and
  an *execution path*. `ab.backends()` reports what this machine can run and, for
  anything it cannot, why and the exact `pip install` that would fix it;
  `ab.backend_matrix()` shows which method declares which; `ab.build(name,
  backend="sdpa")` selects one, `ab.default_backend(...)` scopes a default.
  - Backends: `reference` (plain PyTorch, always available, **the default and the
    correctness anchor**), `sdpa` (`torch.nn.functional.scaled_dot_product_attention`,
    with `sdpa_kernel=` to pin the FlashAttention-2 / memory-efficient / math
    sub-backend), `flash_attn` (the `flash-attn` package, CUDA + fp16/bf16).
    `xformers` and `triton` are detected and **reported as deliberately unused**, with
    the reason, rather than left to guesswork.
  - Four methods declare a non-reference backend: `softmax`, `flash`, `gqa`, `mla`.
    The other twenty-eight declare `reference` only, which is the correct answer for an
    operator that is not a dense softmax over a mask — and `ab.describe` prints the
    reason rather than a blank.
  - **Equivalence is tested, not asserted.**
    `tests/test_conformance.py::test_backend_equivalence` compares every declared
    backend against `reference` for both causal settings, float32 and float64, on
    `m < n`, `m == n`, `n == 1`, a single decode step and logits scaled by 10, and
    again on CUDA where available. Tolerances `1e-5` (fp32) and `1e-10` (fp64), scaled
    by ‖y‖∞.
  - Selection is never silent: an undeclared backend raises
    `BackendNotSupportedError` naming what the method does declare; a declared but
    uninstalled backend warns with `BackendFallbackWarning` and falls back; and
    `return_weights=True` — which no fused kernel can serve — falls back for that call
    and says so.
- **`attnbank/components/`** — the behaviour that recurs across methods, in one
  documented place, so a method file shows what the method *adds*: the online-softmax
  monoid (`flash`, `splitk`), the quantisers (`lowprec`, `quantkv`), the renormalised
  restriction (every selector in `restrict/`, both evictions in `compress/`), the
  recurrent feature maps and readout, and the shared SDPA call. The old private module
  paths (`methods/restrict/_restriction.py`, `methods/recurrent/_state.py`) remain as
  re-export shims, so no import broke.
- `tests/test_backends.py` — detection, selection, announcement, the matrix, and the
  study pin.
- `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, this file, and a GitHub Actions workflow
  running pytest and ruff on Python 3.10–3.13, plus jobs that build the docs strictly,
  build and `twine check` the distribution, and verify that a bare `pip install
  attnbank` still works with neither matplotlib nor transformers present.

### Changed

- **The positional parameters of every forward pass are now `query, key, value`** (the
  transformers convention) rather than `q, k, v`. This frees `k=` to mean what a reader
  expects: `ab.attend("topk", q, k, v, k=64)` sets top-k's budget. The tensors remain
  positional, so every existing call is unaffected; passing a `Tensor` as `q=`, `k=` or
  `v=` is still understood, and the old escape spelling `k_=64` still works.
- `ab.available()` takes an `objective=` filter alongside `category=`.
- `pyproject.toml` is PyPI-ready: metadata, classifiers, keywords, URLs, and optional
  extras so the core install is `torch` and `numpy` only. Plotting moved to `[viz]`,
  real-activation capture to `[studies]`, the optional CUDA kernels to
  `[flash]` / `[triton]` / `[xformers]` / `[fast]`, the docs toolchain to `[docs]`, and
  everything a contributor needs to `[dev]`.
- **`attnbank/studies/` pins `backend="reference"` on import.** Every number in the
  monograph came out of those modules; a published figure must not be able to move
  because a faster execution path appeared. `tests/test_backends.py` asserts the pin in
  a subprocess and separately greps the eight study modules for an override. No result
  file was touched and no sweep was re-run.

### Deprecated

- Nothing. The trailing-underscore spelling `k_=` is redundant now that `k=` works, but
  it is supported and tested, and no removal is scheduled.

## [0.1.0]

Initial internal release: the taxonomy, thirty-two methods, `attnbank.metrics`, the
eleven-point conformance suite, and the eight studies behind the monograph.

[Unreleased]: https://github.com/Ashkan13776/attnbank/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Ashkan13776/attnbank/releases/tag/v0.2.0
[0.1.0]: https://github.com/Ashkan13776/attnbank/releases/tag/v0.1.0
