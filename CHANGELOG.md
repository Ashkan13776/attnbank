# Changelog

All notable changes to `attnbank` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The one thing that will never change silently is **Invariant W**
(`y == weights @ (v_hat or value)`). If a release ever relaxes it, that release is a
major version and this file says so in the first line.

## [Unreleased]

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
- `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, this file, and a GitHub Actions workflow
  running pytest and ruff on Python 3.10–3.13.

### Changed

- **The positional parameters of every forward pass are now `query, key, value`** (the
  transformers convention) rather than `q, k, v`. This frees `k=` to mean what a reader
  expects: `ab.attend("topk", q, k, v, k=64)` sets top-k's budget. The tensors remain
  positional, so every existing call is unaffected; passing a `Tensor` as `q=`, `k=` or
  `v=` is still understood, and the old escape spelling `k_=64` still works.
- `ab.available()` takes an `objective=` filter alongside `category=`.
- `pyproject.toml` is PyPI-ready: metadata, classifiers, keywords, URLs, and optional
  extras so the core install is `torch` and `numpy` only. Plotting moved to `[viz]`,
  real-activation capture to `[studies]`, the docs toolchain to `[docs]`, and
  everything a contributor needs to `[dev]`.

### Deprecated

- Nothing. The trailing-underscore spelling `k_=` is redundant now that `k=` works, but
  it is supported and tested, and no removal is scheduled.

## [0.1.0]

Initial internal release: the taxonomy, thirty-two methods, `attnbank.metrics`, the
eleven-point conformance suite, and the eight studies behind the monograph.

[Unreleased]: https://github.com/attnbank/attnbank/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/attnbank/attnbank/releases/tag/v0.2.0
[0.1.0]: https://github.com/attnbank/attnbank/releases/tag/v0.1.0
