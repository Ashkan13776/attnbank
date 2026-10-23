# Examples gallery

Short, self-contained scripts. Every one of them runs as written, and
`tests/test_examples.py` executes all of them in CI, so a broken example
is a failing build rather than a stale page.

```bash
git clone https://github.com/attnbank/attnbank
python attnbank/examples/01_quickstart.py
```

| | example | what it shows |
|---|---|---|
| 1 | [Quickstart: one import, three calls](01_quickstart.md) | The whole library behind one name. `attend` for the one-liner, `build` for the object you keep, `compare` for the numbers you report. |
| 2 | [Invariant W: every method is a measure times a value field](02_the_invariant.md) | `y == weights @ (v_hat if v_hat is not None else value)` holds for every registered method -- that is the claim the library is built on, and here it is, checked over the whole bank in float64. |
| 3 | [Splitting output error into its two causes](03_error_decomposition.md) | An approximation can be wrong in two ways: it attends to the wrong places, or it aggregates the wrong values. `metrics.error_decomposition` separates them exactly: ```text y - y_hat = (pi - pi_hat) v  +  pi_hat (v - v_hat)             \_____________/    \________________/             distributional          value ``` |
| 4 | [Top-k is not output-optimal](04_topk_is_not_output_optimal.md) | Selecting the `k` largest attention weights minimises the *omitted mass* `eps_q = pi_q(S^c)`. It does not minimise `||y - y_hat||`, because the bound is `eps_q * osc(v)` and a light token carrying an extreme value can matter more than a heavy token sitting near the mean. |
| 5 | [Random features: the error that shrinks, and the error that does not](05_random_features_converge.md) | FAVOR+ (`performer`) estimates `exp(<q, k>)` with `r` positive random features. More features, less error -- and the conformance suite checks that this is monotone, because for a Monte-Carlo method "reduces to softmax at some finite setting" is not available as a correctness anchor. |
| 6 | [Attention-matrix error is not output error](06_matrix_error_is_not_output_error.md) | Most of the literature reports `||P - P_hat||_F`. The quantity a model actually feels is `||(P - P_hat) V||`. They are not the same ranking: the Frobenius norm charges a method for mass it moved between two tokens with nearly identical values, which costs the output nothing. |
| 7 | [Finding your way around the bank](07_browse_the_bank.md) | Thirty-odd methods is too many to remember and too few to need a search engine. Four calls cover it: `categories` for the taxonomy, `list_methods` for the table, `available` for the names, `describe` for one method in full. |
| 8 | [What a method actually saves](08_cost_and_capacity.md) | Every method declares an analytic cost model: `flops`, `state_bytes` (the resident KV cache) and `bytes_moved`. Decoding is bandwidth-bound, so `bytes_moved` -- not `flops` -- is the number that predicts latency, and `state_bytes` is the number that decides whether the context fits at all. |
| 9 | [Adding a method: declare, implement, conform](09_add_your_own_method.md) | Three steps. Subclass `attnbank.AttentionMethod`, declare what the method *is*, implement `_forward`. The declaration is not documentation -- the conformance suite reads it and tests the method against it, so a wrong declaration is a failing build. |
| 10 | [Backends: the operator stays, the execution path changes](10_backends.md) | A method is two things: an *operator* (the mathematics, one readable file) and an *execution path*. The operator never changes with the backend -- and that is a tested claim, not a slogan. |

## Running them all

```bash
pytest tests/test_examples.py -q
```

Or regenerate these pages, output included:

```bash
python docs/gen_gallery.py
```
