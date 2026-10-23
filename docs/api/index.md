# API reference

The surface is deliberately small. One import, a handful of functions.

```python
import attnbank as ab
```

## The flat surface

| | |
|---|---|
| [`ab.attend`][attnbank.attend] | run a method once, get `y` |
| [`ab.build`][attnbank.build] | the method as an object |
| [`ab.compare`][attnbank.compare] | several methods, scored against a reference |
| [`ab.available`][attnbank.available] | the names |
| [`ab.list_methods`][attnbank.list_methods] | the readable table |
| [`ab.describe`][attnbank.describe] | one method, in full |
| [`ab.categories`][attnbank.categories] | the taxonomy |
| [`ab.method_table`][attnbank.method_table] | the machine-readable table |
| [`ab.register`][attnbank.register] | add your own |

## The contract

| | |
|---|---|
| [`ab.check_invariant`][attnbank.check_invariant] | `y == weights @ (v_hat or value)` |
| [`ab.check_conformance`][attnbank.check_conformance] | a printable pass/fail report |

## Backends

| | |
|---|---|
| [`ab.backends`][attnbank.backends] | what runs here, and why anything does not |
| [`ab.backend_matrix`][attnbank.backend_matrix] | method → declared backends |
| [`ab.set_default_backend`][attnbank.set_default_backend] | pin the process default |
| [`ab.default_backend`][attnbank.default_backend] | the same, scoped |

## Types, errors, metrics

- [Core types](core.md) — `AttentionMethod`, `AttentionOutput`, `Category`, `Objective`
- [Metrics](metrics.md) — the error decomposition, the divergences, the bounds
- [Conformance](conformance.md) — the runtime contract API
- [Errors](errors.md) — what each one means and what to do about it

Every function below carries a runnable example, and
[`tests/test_docs.py`](https://github.com/attnbank/attnbank/blob/main/tests/test_docs.py)
executes all of them in CI. A public function without an example fails the build.
