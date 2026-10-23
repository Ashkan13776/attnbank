# The conformance suite

`tests/test_conformance.py` is parametrized over **every** registered method and is the
reason this library can be trusted. Registering a method is all it takes to be included
— there is no second list to update, and no list of exemptions anywhere in the file.

```bash
pytest tests/test_conformance.py -q
```

A method that cannot pass a check must **say so in a declaration attribute**, not in a
`skip`. The suite then tests it against what it claims. That is the whole design: the
attributes in a method's class body are commitments, and this file is where they are
collected.

## The eleven checks

| | check | what it rules out |
|---|---|---|
| 1 | **shapes** | `y` is `[B,H,m,dv]`, `weights` is `[B,H,m,n]`, `v_hat` is `[B,H,n,dv]` |
| 2 | **[Invariant W](invariant.md)** | reported weights that do not reproduce the output |
| 3 | **row stochasticity** | a method claiming a probability measure and not delivering one |
| 4 | **causal leakage** | perturb `k[…, t:, :]`; earlier outputs must be *bit-for-bit* unchanged |
| 5 | **batch and head independence** | accidental coupling across the batch or head axis |
| 6 | **exact limit** | an implementation that is plausible rather than correct |
| 7 | **convergence** | a Monte-Carlo estimator that does not actually converge |
| 8 | **dtype and device** | float32 / float64 / CPU / CUDA disagreement |
| 8 (cont.) | **backend equivalence** | a kernel that is not the operator it claims to be |
| 9 | **cost model** | three keys, finite, positive; capacity savers really save capacity |
| 10 | **no NaN / no Inf** | empty support, single-element support, logits scaled by 10 |
| 11 | **references resolve** | an implementation with no source |

## Why 6 and 7 are the interesting ones

Checks 1–5 catch bugs. Check 6 catches something else: an implementation that is
*plausible* rather than *correct*.

Every method whose `objective` is `accelerate` or `approximate` declares an anchor —
a configuration under which it **provably** reduces to dense softmax:

```python
TopKAttention.limit_config()      # {'k': None}   — keep every visible key
QuantKVAttention.limit_config()   # {'bits_k': None, 'bits_v': None}
NystromAttention.limit_config()   # {'r': None}   — landmarks = all tokens
```

The suite builds the method under that configuration and requires it to match
`softmax` to `1e-5` in the output and `1e-6` in the weights. There is nowhere for a
transposed index or a dropped scale factor to hide.

Monte-Carlo estimators have no exact limit at finite resource. `performer` and
`trigrff` declare `convergence_configs()` instead — a sequence of configurations along
which the mean error, over seeds, must **decrease monotonically**:

```python
errs = [mean_error(cfg) for cfg in cls.convergence_configs()]
assert all(a > b for a, b in zip(errs, errs[1:]))
```

Methods whose `objective` is `modify` are exempt, and the suite **asserts the
exemption** rather than silently skipping it:

```python
def test_has_an_anchor(name):
    cls = ab.get_class(name)
    if cls.objective is Objective.MODIFY:
        return          # a different operator: neither anchor is meaningful
    assert cls.limit_config() is not None or cls.convergence_configs() is not None
```

`linear_elu`, `cosformer`, `latent_stochastic`, `lowrank_ot` and the three recurrences
neither reduce to nor converge to softmax. That is what it means to define a different
operator. Demanding an anchor of them would be demanding that they stop being what they
are.

## Declarations the suite reads

| attribute | what the suite does with it |
|---|---|
| `supports_causal` | runs check 4, or skips it with the declaration as the reason |
| `expresses_weights` | requires `weights` to be present **and** absent, correspondingly |
| `row_stochastic` | runs check 3 |
| `head_independent` | `gqa` and `mla` share K/V across query heads, so check 5 verifies the *coupling boundary* instead: head `h` moves its whole group and nothing outside it. Sharing is the mechanism, not a defect |
| `objective` | decides whether an anchor is required (check 6/7) |
| `backends` | decides which equivalence cases run (check 8) |
| `references` | check 11 resolves every key in `attnbank/references.bib` |

## Backend equivalence (part of check 8)

Declaring a backend is a claim that it computes the same operator, so the suite checks
it: for every method, every declared backend, both causal settings, float32 and
float64, and the awkward shapes — `m < n`, `n == 1`, a single decode step, and logits
scaled by 10.

Tolerances are `1e-5` (fp32) and `1e-10` (fp64), scaled by $\lVert y\rVert_\infty$ so
the number is a statement about precision rather than about the test inputs. A backend
that cannot meet them is not declared. See [Backends](backends.md).

## At runtime

The same contract is available without pytest:

```python
ab.check_invariant(out, value)
ab.check_conformance("topk")           # a printable report
ab.check_conformance("topk", strict=True)   # raises on failure
```

`check_conformance` runs the fast half — shapes, Invariant W, row stochasticity, causal
non-leakage, finiteness at large logits, input immutability, reproducibility, the cost
model and the exact limit. The float64/CUDA matrix, head independence, Monte-Carlo
convergence and the bibliography check stay in the test file, where they belong.

## Rules of thumb the suite exists to enforce

- Compute softmax in log space; never materialise `exp` of an unshifted score.
- A fully masked row must produce a documented, finite result — never `nan`. Window
  methods guarantee at least one visible key by including the diagonal.
- Renormalise explicitly, with a floored denominator, and assert in tests that the
  floor was never reached.
- Never mutate `q`, `k` or `v`.
- Seeded randomness goes through `torch.Generator`, never the global RNG, so studies
  reproduce and two builds agree bit for bit.

[`DESIGN.md`](design.md) is the normative version of this page. If an implementation
and that file disagree, the file wins.
