<div align="center">

# attnbank

**Every efficient attention method changes one object in one formula.**
This library implements them side by side, organised by *which* object they change —
and holds every one of them to a single invariant.

[![tests](https://github.com/Ashkan13776/attnbank/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashkan13776/attnbank/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/attnbank)](https://pypi.org/project/attnbank/)
[![Python](https://img.shields.io/pypi/pyversions/attnbank)](https://pypi.org/project/attnbank/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![docs](https://img.shields.io/badge/docs-Ashkan13776.github.io-black)](https://Ashkan13776.github.io/attnbank/)

</div>

```bash
pip install attnbank
```

```python
import attnbank as ab

y = ab.attend("topk", query, key, value, causal=True, k=64)
```

Thirty-two methods — sparse, linear, low-rank, quantised, evicted, recurrent, optimal
transport — behind one interface, one naming convention, and one promise about what
they will tell you about themselves.

---

## Why another attention library

Not for speed. For **comparison that means something**.

Everything here can tell you, exactly, *what the approximation cost you* and *in which
of the two possible ways it was wrong*:

```python
rows = ab.compare(["softmax", "topk", "performer", "quantkv"], query, key, value,
                  configs={"topk": {"k": 16}, "performer": {"r": 64},
                           "quantkv": {"bits_k": 16, "bits_v": 2}})
```

| method | output error | distributional | value |
|---|---|---|---|
| `topk` | 1.03 | **1.03** | 0.00 |
| `performer` | 2.76 | **2.76** | 0.00 |
| `quantkv` | 0.72 | 0.00 | **0.72** |

Restriction attends to the wrong places. Quantisation aggregates the wrong values.
Both appear as "error" in a single number; only the split says which mechanism is at
fault. And the split is an **identity**, not a heuristic — which is only true because
of the one invariant below.

## The idea

Attention is a query-conditioned expectation:

$$y(q) \;=\; \int v \,\mathrm{d}\pi_q, \qquad \pi_q \;=\; \frac{g(q,\cdot)\,\mu}{Z(q)} .$$

Efficiency has to come from somewhere, and there are only so many places it can come
from. A method either evaluates this expression differently, or it changes one of the
objects in it. That is the taxonomy, and it is also the directory structure.

| Category | What it changes | Methods |
|---|---|---|
| `exact` | nothing — only the evaluation | `softmax` `flash` `splitk` `lowprec` |
| `interaction` | the interaction $g$ | `performer` `trigrff` `linear_elu` `cosformer` `nystrom` |
| `restrict` | the support of $\pi_q$ | `topk` `window` `sink_window` `blocktopk` `value_topk` `lsh_sample` |
| `compress` | the stored context $(\mu, k, v)$ | `quantkv` `lowrank_kv` `evict_h2o` `evict_snapkv` `merge_kv` `gqa` `mla` `linformer` |
| `factorize` | the factorisation of $\pi_q$ through latents | `latent_stochastic` `clustered` `nmf_oracle` `svd_oracle` |
| `coupling` | normalisation, which stops being per-query | `sinkhorn` `lowrank_ot` |
| `recurrent` | the operator itself (finite state) | `linear_rnn` `decay_rnn` `delta_rnn` |

A third axis matters as much as the first. A method that **approximates** the reference
operator is judged by fidelity to it; a method that **modifies** it is a different
operator and fidelity to softmax is not its goal. Conflating the two is the most
persistent confusion in this literature, so every method here declares which it is —
and the test suite reads the declaration.

## The one invariant

```python
y == weights @ (v_hat if v_hat is not None else value)
```

Every method realises, explicitly or implicitly, a measure $\widehat\pi_q$ over context
positions and a value field $\widehat v$ — and must be able to show you both. A
Performer computes feature maps, a StreamingLLM evicts tokens, a linear RNN carries a
state matrix; none of them *looks* like a weighted average over positions, and all of
them are one.

This is checked for every registered method, in float32 and float64, on every shape the
suite throws at it. It is load-bearing: the error metrics split output error into a
distributional term and a value term, and that split means nothing unless a method's
reported weights and values actually reproduce its output.

```python
ab.check_invariant(out, value)     # max |y - weights @ v_hat|, raises if too large
ab.check_conformance("topk")       # the whole fast contract, as a printable report
```

Methods that genuinely cannot express their measure on the original $n$ positions set
`expresses_weights = False` and are excluded from those studies **by declaration, not
by accident**.

Correctness is anchored to an *exact limit* rather than to inspection. Each method
declares the configuration under which it provably reduces to dense softmax — top-$k$
with $k = n$, quantisation at 16 bits, Nyström with landmarks = all tokens — and the
conformance suite holds it to that. Monte-Carlo estimators, which have no exact limit
at finite resource, declare a sequence along which the error must decrease
monotonically instead.

## Five minutes

Tensors are `query [B, H, m, d]`, `key [B, H, n, d]`, `value [B, H, n, dv]`. Causal
masking aligns the **last** query with the **last** key, so the same code path serves
prefill and decode.

```python
import attnbank as ab

# one-liner
y = ab.attend("topk", q, k, v, causal=True, k=64)

# the object, when you want more than y
attn = ab.build("performer", r=256)
out = attn(q, k, v, causal=True, return_weights=True)
out.y, out.weights, out.v_hat, out.extras

# find your way around
ab.list_methods()                  # the whole bank, grouped by category
ab.describe("nystrom")             # one method, in full
ab.categories()                    # what each category means
ab.available(objective="modify")   # the names only
```

Errors are expected to teach:

```pycon
>>> ab.build("topkk")
UnknownMethodError: unknown method 'topkk'. Did you mean: 'topk'? ab.available() lists
all 32 registered methods; ab.list_methods() prints them grouped by category.

>>> ab.build("sinkhorn")(q, k, v, causal=True)
CausalNotSupportedError: 'sinkhorn' does not support causal attention: it declares
supports_causal = False, which is a statement about the operator, not a missing
feature. No method in the coupling category supports it; 21 others do, e.g.
'blocktopk', 'cosformer', … ab.list_methods(causal=True) lists them all.
```

## Backends

A method is an **operator** (the mathematics, one readable file) and an **execution
path**. The operator never changes with the backend.

```python
print(ab.backends())                     # what runs here, and why anything does not
ab.build("softmax", backend="sdpa")      # PyTorch's fused kernel
ab.backend_matrix()                      # method -> declared backends
```

| backend | what it is | needs |
|---|---|---|
| `reference` | plain PyTorch; **the correctness anchor and the default** | nothing |
| `sdpa` | `torch.nn.functional.scaled_dot_product_attention`; on CUDA this *is* FlashAttention-2 | torch ≥ 2.2 |
| `flash_attn` | the `flash-attn` package | CUDA, fp16/bf16, `pip install attnbank[flash]` |
| `xformers`, `triton` | detected and reported, **declared by nothing** — and `ab.backends()` says why | — |

Twenty-eight of thirty-two methods declare `reference` only, and that is the correct
answer: `nmf_oracle` is a factorisation, `merge_kv` is a clustering step, `topk` needs
the full score matrix for the omitted mass it reports. None of them is a kernel call.

Declaring a backend is a *claim that it equals the reference*, and the conformance
suite checks it — every method, every declared backend, causal and not, float32 and
float64, including `m < n`, single-element support and logits scaled by 10. Selection
is explicit; an undeclared backend raises rather than downgrading, and an unavailable
one falls back with a warning. Never silently.

## What the conformance suite checks

Every registered method, with no exemptions outside its declared attributes:

1. Shapes, for every tensor returned
2. **The invariant above**
3. Row stochasticity, where the method claims it
4. No causal leakage, under perturbation of future keys
5. Batch and head independence (and, for `gqa` / `mla`, the exact coupling boundary)
6. Exact reduction to softmax at the declared limit
7. Monotone convergence, for Monte-Carlo methods
8. Agreement across float32, float64, CPU and CUDA — **and across backends**
9. A cost model that is finite, positive and consistent
10. No NaN at empty support or large logits
11. Every cited BibTeX key resolves in `attnbank/references.bib`

```bash
pytest -q         # 1250+ tests, about twenty seconds
```

## Documentation

**[Ashkan13776.github.io/attnbank](https://Ashkan13776.github.io/attnbank/)** — quickstart,
the taxonomy, the invariant, the conformance suite, backends, a full API reference, and
a gallery of short runnable scripts.

Every public function carries a runnable example, and `tests/test_docs.py` executes all
of them in CI; a public function without one fails the build. The same is true of the
gallery: `tests/test_examples.py` runs every script.

## Adding a method

```python
@ab.register
class MyAttention(ab.AttentionMethod):
    name = "my_method"
    category = ab.Category.RESTRICT       # which object it changes
    objective = ab.Objective.APPROXIMATE  # accelerate | approximate | modify
    supports_causal = True
    expresses_weights = True              # can it name pi_hat on the original n?
    row_stochastic = True
    references = ("author2024paper",)
    paper_section = "sec:sparse"

    def _forward(self, q, k, v, causal, return_weights): ...
    def cost(self, m, n, d, dv, bytes_per_el=2): ...
    @classmethod
    def limit_config(cls): ...            # where it reduces to softmax
```

Registration is the only wiring step: the conformance suite is parametrized over the
registry. [`CONTRIBUTING.md`](CONTRIBUTING.md) has the long version, and
[`DESIGN.md`](DESIGN.md) is the binding specification — if an implementation and that
file disagree, the file wins.

## A note on the implementations

These favour clarity over speed. They are meant to be read against the monograph with
the relevant section open: no micro-optimisation, no fused tricks, and no `einsum`
string longer than the equation it implements. That is what the `reference` backend
*is*, and why it is the anchor rather than the fast path.

Behaviour that recurs across methods — masking, renormalisation, the online-softmax
monoid, quantise/dequantise, the renormalised restriction, feature maps, the SDPA call
— lives in [`attnbank/components/`](attnbank/components/), so a method file shows what
the method *adds*.

Three methods are **oracles** — `value_topk`, `nmf_oracle`, `svd_oracle`. They are not
practical; they are upper bounds on what a mechanism can achieve, which is exactly what
several of the arguments need.

## The studies

The library also reproduces the eight studies behind the monograph. Each names in
advance the result that would contradict the text, so a study can come back negative —
and when it does, the text changes, not the study. See
[`STUDIES.md`](STUDIES.md).

| | Claim under test |
|---|---|
| S1 | Attention-matrix error is not output error |
| S2 | Top-$k$ selection is not output-optimal |
| S3 | Random features: bias $O(1/r)$, s.d. $O(r^{-1/2})$ |
| S4 | Keys are more sensitive than values |
| S5 | Nonnegative rank exceeds rank, measurably |
| S6 | Finite state bounds recall |
| S7 | OT attention is softmax tilted by a key potential |
| S8 | Decoding is bandwidth-bound, not compute-bound |

```bash
pip install "attnbank[studies]"
python -m attnbank.studies.S2_restriction
```

Results are committed as JSON under `results/`, so every published number traces to a
file. The studies pin `backend="reference"` on import, so a faster execution path can
never move one of them.

## Citing

`attnbank` is the companion code for *What Does Efficient Attention Approximate? A
Unified View of Interactions, Memory, and Error*. If the library or its taxonomy is
useful in your work, please cite the monograph:

```bibtex
@article{shahbazi2026attention,
  title  = {What Does Efficient Attention Approximate?
            A Unified View of Interactions, Memory, and Error},
  author = {Shahbazi, Ashkan},
  year   = {2026},
  note   = {Manuscript in preparation},
}
```

## Licence

[Apache License 2.0](LICENSE).
