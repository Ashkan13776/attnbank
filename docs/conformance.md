<nav class="ab-path" aria-label="The path through these pages">
  <ol>
    <li><a href="../quickstart/"><b>01</b>Quickstart</a></li>
    <li><a href="../invariant/"><b>02</b>The invariant</a></li>
    <li aria-current="page"><b>03</b>Conformance</li>
    <li><a href="../backends/"><b>04</b>Backends</a></li>
  </ol>
</nav>

# The conformance suite

<p class="ab-lead">One file, <code>tests/test_conformance.py</code>, parametrized over
every registered method, is the reason this library can be trusted. Registering a
method is all it takes to be included — there is no second list to update, and no list
of exemptions anywhere in the file.</p>

<p class="ab-stats">
<span><b>11</b> properties</span>
<span><b>78</b> methods</span>
<span><b>2228</b> tests from this file</span>
<span><b>2838</b> passed · <b>340</b> skipped across <code>tests/</code></span>
</p>

```bash
pytest tests/test_conformance.py -q
```

A method that cannot pass a check must **say so in a declaration attribute**, not in a
`skip`. The suite then tests it against what it claims — and every skip it does emit
names the declaration or the missing package that caused it. That is the whole design:
the attributes in a method's class body are commitments, and this file is where they
are collected.

## The eleven checks

<div class="ab-checks" markdown>

| | check | what it rules out |
|---|---|---|
| 01 | **shapes** | `y` is `[B,H,m,dv]`, `weights` is `[B,H,m,n]`, `v_hat` is `[B,H,n,dv]` |
| 02 | **[Invariant W](invariant.md)** | reported weights that do not reproduce the output |
| 03 | **row stochasticity** | a method claiming a probability measure and not delivering one |
| 04 | **causal leakage** | perturb keys and values from position `t` on; earlier outputs must be *bit-for-bit* unchanged |
| 05 | **batch and head independence** | accidental coupling across the batch or head axis |
| 06 | **exact limit** | an implementation that is plausible rather than correct |
| 07 | **convergence** | a Monte-Carlo estimator that does not actually converge |
| 08 | **dtype, device, backend** | float32 / float64 / CPU / CUDA disagreement; a kernel that is not the operator it claims to be |
| 09 | **cost model** | three keys, finite, positive; capacity savers that do not really save capacity |
| 10 | **no NaN / no Inf** | supports of size one (`n == 1`, the causal first row), logits scaled by 10, mutated inputs, builds that do not reproduce bit for bit |
| 11 | **references resolve** | an implementation with no source |

</div>

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

Monte-Carlo estimators have no exact limit at finite resource. Ten methods —
`performer` and `trigrff` among them — declare `convergence_configs()` instead: a
sequence of configurations along which the mean error over eight seeds must
**decrease monotonically**. From the test, verbatim:

```python
errs = []
for cfg in cfgs:
    e = []
    for seed in range(8):
        out = ab.build(name, seed=seed, **cfg)(q, k, v)
        e.append((out.y - ref).norm(dim=-1).mean().item())
    errs.append(sum(e) / len(e))
assert all(a > b for a, b in zip(errs, errs[1:])), \
    f"{name}: mean error not monotone along convergence_configs: {errs}"
```

Methods whose `objective` is `modify` are exempt, and the suite **asserts the
exemption** rather than silently skipping it:

```python
def test_has_an_anchor(name):
    """Every method that targets softmax must declare one of the two anchors."""
    cls = ab.get_class(name)
    if cls.objective is Objective.MODIFY:
        return                      # a different operator: neither anchor is meaningful
    assert cls.limit_config() is not None or cls.convergence_configs() is not None
```

Twenty-five methods — from `linear_elu`, `cosformer`, `latent_stochastic` and
`lowrank_ot` to all thirteen recurrences — neither reduce to nor converge to softmax.
That is what it means to define a different operator. Demanding an anchor of them
would be demanding that they stop being what they are.

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

Tolerances are `1e-5` (fp32) and `1e-10` (fp64), scaled by
$\max(1, \lVert y\rVert_\infty)$ so the number is a statement about precision rather
than about the test inputs. A backend that cannot meet them is not declared. See
[Backends](backends.md).

## At runtime

The same contract is available without pytest:

```python
ab.check_invariant(out, value)
ab.check_conformance("topk")                # a printable report
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

<nav class="ab-next" aria-label="Continue reading">
  <a href="../backends/">
    <span class="ab-next-kicker">Next · 04 of 04</span>
    <span class="ab-next-title">Backends</span>
    <span class="ab-next-desc">The operator never changes with the execution path —
    and why most of the bank correctly declares <code>reference</code> only.</span>
  </a>
  <p class="ab-next-also">Also from here:
  <a href="../api/conformance/">the conformance API</a> ·
  <a href="../contributing/">contributing a method</a> — what the suite will ask of
  it.</p>
</nav>
