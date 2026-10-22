<div align="center">

# attnbank

**Every efficient attention method changes one object in one formula.**
This library implements them side by side, organised by *which* object they change.

<sub>Companion code for *What Does Efficient Attention Approximate?
A Unified View of Interactions, Memory, and Error*</sub>

</div>

---

Attention is a query-conditioned expectation:

$$y(q) \;=\; \int v \,\mathrm{d}\pi_q, \qquad \pi_q \;=\; \frac{g(q,\cdot)\,\mu}{Z(q)} .$$

Efficiency has to come from somewhere, and there are only so many places it can come from.
A method either evaluates this expression differently, or it changes one of the objects in it.
That is the organising principle of the paper, and it is the directory structure here.

| Category | What it changes | Objective | Package |
|---|---|---|---|
| `exact` | nothing — only the evaluation | accelerate | `methods/exact/` |
| `interaction` | the interaction $g$ | approximate · modify | `methods/interaction/` |
| `restrict` | the support of $\pi_q$ | approximate | `methods/restrict/` |
| `compress` | the stored context $(\mu, k, v)$ | approximate · modify | `methods/compress/` |
| `factorize` | the factorisation of $\pi_q$ through latents | modify | `methods/factorize/` |
| `coupling` | normalisation, which stops being per-query | modify | `methods/coupling/` |
| `recurrent` | the operator itself (finite state) | modify | `methods/recurrent/` |

The third axis matters as much as the first. A method that **approximates** the reference
operator is judged by fidelity to it; a method that **modifies** it is a different operator and
fidelity to softmax is not its goal. Conflating the two is the most persistent confusion in this
literature, so every method here declares which it is.

## The one invariant

Every method realises, explicitly or implicitly, a measure $\widehat\pi_q$ over context positions
and a value field $\widehat v$. So every method must satisfy:

```python
y == weights @ (v_hat if v_hat is not None else v)
```

This is checked automatically for every registered method, in float32 and float64. It is the
load-bearing property of the library: the error metrics split output error into a distributional
term and a value term, and that split means nothing unless a method's reported weights and values
actually reproduce its output. Methods that genuinely cannot express their measure on the original
$n$ positions set `expresses_weights = False` and are excluded from those studies **by
declaration, not by accident**.

Correctness is anchored to an *exact limit* rather than to inspection. Each method declares the
configuration under which it provably reduces to dense softmax — top-$k$ with $k = n$,
quantisation at 16 bits, random features as $r \to \infty$ — and the conformance suite holds it
to that.

## Quick start

```bash
pip install -e lib/
```

Tensors are `[B, H, n, d]` throughout. Causal masking aligns the last query with the last key, so
the same code path serves prefill and decode.

```python
import attnbank as ab
from attnbank import metrics

ref  = ab.build("softmax")(q, k, v, causal=True, return_weights=True)
appr = ab.build("topk", k=64)(q, k, v, causal=True, return_weights=True)

# output error, split into its distributional and value parts
metrics.error_decomposition(ref.weights, appr.weights, v, appr.v_hat)
# {'distributional': ..., 'value': ..., 'total': ...}

ab.available("restrict")   # what else restricts the support?
ab.method_table()          # the whole bank, one row per method
```

## What the conformance suite checks

Every registered method, with no exemptions outside its declared attributes:

1. Shapes, for every tensor returned
2. The invariant above
3. Row stochasticity, where the method claims it
4. No causal leakage, under perturbation of future keys
5. Batch and head independence
6. Exact reduction to softmax at the declared limit
7. Monotone convergence, for Monte-Carlo methods
8. Agreement across float32, float64, CPU and CUDA
9. A cost model that is finite, positive and consistent
10. No NaN at empty support or large logits
11. Every cited BibTeX key resolves in `references.bib`

```bash
python -m pytest lib/tests -q
```

## Studies

The library exists to test claims the paper already makes. Each study names in advance the result
that would contradict the text, so a study can come back negative — and when it does, the text
changes, not the study. See [`STUDIES.md`](STUDIES.md) for the designs.

| | Claim under test | § |
|---|---|---|
| S1 | Attention-matrix error is not output error | 4 |
| S2 | Top-$k$ selection is not output-optimal | 3.3, 7 |
| S3 | Random features: bias $O(1/r)$, s.d. $O(r^{-1/2})$ | 3.2 |
| S4 | Keys are more sensitive than values | 3.4 |
| S5 | Nonnegative rank exceeds rank, measurably | 3.5, 5.2 |
| S6 | Finite state bounds recall | 5 |
| S7 | OT attention is softmax tilted by a key potential | 6 |
| S8 | Decoding is bandwidth-bound, not compute-bound | 3.1, 5.5 |

Results are committed as JSON under `results/`, so every number in the paper traces to a file.
Figures are written to `figures/` and included directly by the LaTeX sources.

## Adding a method

1. One file per method in `attnbank/methods/<category>/`. Subclass `AttentionMethod`, decorate
   with `@register`, and declare `name`, `category`, `objective`, `supports_causal`,
   `expresses_weights`, `row_stochastic`, `trainable`, `references`, `paper_section`.
2. Implement `_forward(q, k, v, causal, return_weights) -> AttentionOutput`, `cost(...)`, and
   `limit_config()` — the config under which the method reduces to softmax, or `None` with
   `convergence_configs()` instead for Monte-Carlo methods.
3. Import it in the category's `__init__.py`. The conformance suite picks it up automatically.

[`DESIGN.md`](DESIGN.md) is the binding specification — if an implementation and that file
disagree, the file wins.

## A note on the implementations

These favour clarity over speed. They are meant to be read against the paper with the relevant
section open, so no micro-optimisation, no fused tricks, and no `einsum` string longer than the
equation it implements. Kernels for wall-clock benchmarks are kept separate and must match the
reference within tolerance.

Several methods are **oracles** — `value_topk`, `nmf_oracle`, `svd_oracle`. These are not
practical methods; they are upper bounds on what a mechanism can achieve, which is exactly what
several of the paper's arguments need.

## Figures

All figures go through `attnbank.plotting.style`: vector PDF, a validated colourblind-safe
palette, colours fixed per method so a method keeps its colour across every figure in the paper,
one y-axis per panel, legends plus direct labels.

```python
from attnbank.plotting import style
style.use()
fig, ax = style.figure("full")
style.save(fig, "fig_name")
```

## Status

Under construction, v0.1. The design contract and study designs are fixed; methods, studies and
figures are landing.
