---
hide:
  - toc
---

# Every efficient attention method changes one object in one formula

<p class="ab-eyebrow">attnbank v0.3.0 &middot; 78 methods &middot; one invariant &middot; 2838 tests</p>

<div class="ab-measure" markdown="1">
Attention is a query-conditioned expectation. Efficiency has to come from somewhere,
and there are only so many places it can come from: a method either evaluates the
expectation differently, or it changes one of the objects in it. There are seven
objects, so there are seven categories — and that is the library's directory structure,
not a diagram drawn afterwards.
</div>

<div class="op" data-op markdown="1">

<div class="op-plate" data-plate="the operator">
--8<-- "docs/includes/operator.html"
<div class="op-keys" role="group" aria-label="Choose a category" hidden></div>
<p class="op-prompt" hidden>Seven keys, one per object a method can change. Choosing one shows where that category reaches into the operator.</p>
</div>

<div class="op-readout" markdown="1">

<div class="op-panel" data-cat="exact" data-label="exact" data-syms="" data-frame markdown="1">
Changes
:   Nothing. The integrand, the measure and the normaliser are the reference operator's; only the order of the arithmetic changes.

Consequence
:   Tiling, online softmax and partitioned reductions reassociate the same sum, so the answer is identical and only the memory traffic moves. The interesting question is *where exactness stops* — `flash` and `splitk` are exact to floating point, `lowprec` is not, and that boundary is the point of the category.

Read on
:   [`exact` in the taxonomy](taxonomy.md#exact)
</div>

<div class="op-panel" data-cat="interaction" data-label="interaction g" data-syms="g" markdown="1">
Changes
:   The interaction <i>g</i>(<i>q</i>,&#8202;<i>x</i>), replaced by a cheaper surrogate <i>g&#770;</i> on the same <i>n</i> positions.

Consequence
:   Separable surrogates turn the numerator and the normaliser into finite moments. The values are untouched, so the whole error is distributional. Positive features keep &#960;&#770;<sub><i>q</i></sub> a probability measure; trigonometric and Nyström ones do not, and say so in an attribute rather than in a footnote.

Read on
:   [`interaction` in the taxonomy](taxonomy.md#interaction)
</div>

<div class="op-panel" data-cat="restrict" data-label="support" data-syms="int" markdown="1">
Changes
:   The support of &#960;<sub><i>q</i></sub>, which shrinks to a subset <i>S</i><sub><i>q</i></sub> and is renormalised on it.

Consequence
:   One operator, differing only in how <i>S</i><sub><i>q</i></sub> is chosen — and the error has an exact closed form: the omitted mass times the contrast between the omitted and the retained conditional means. Nothing is discarded, so the context stays available to later queries.

Read on
:   [`restrict` in the taxonomy](taxonomy.md#restrict)
</div>

<div class="op-panel" data-cat="compress" data-label="context μ, v" data-syms="mu v" markdown="1">
Changes
:   The stored context (&#956;,&#8202;<i>k</i>,&#8202;<i>v</i>) itself — quantised, factored, evicted, merged, or shared across heads.

Consequence
:   The only category that routinely perturbs the integrand, so <i>v&#770;</i> &#8800; <i>v</i> and the error has a value term as well as a distributional one. Key error passes through the softmax; value error is only reweighted, which is why keys cost more per bit.

Read on
:   [`compress` in the taxonomy](taxonomy.md#compress)
</div>

<div class="op-panel" data-cat="factorize" data-label="factorisation" data-syms="pi" markdown="1">
Changes
:   The shape of &#960;<sub><i>q</i></sub>: it is routed through <i>r</i> latents, &#960;<sub><i>q</i></sub> = <i>a</i>(<i>q</i>,&#8201;&#183;)&#8202;<i>b</i>(&#183;,&#8201;<i>x</i>), which caps the rank of the attention matrix at <i>r</i>.

Consequence
:   A composition of two Markov kernels is still a probability measure, so normalisation survives. The price is structural: nonnegative rank exceeds rank, so a stochastic factorisation cannot reach a signed one of the same size — and the bank carries both oracles that measure the gap.

Read on
:   [`factorize` in the taxonomy](taxonomy.md#factorize)
</div>

<div class="op-panel" data-cat="coupling" data-label="normaliser Z" data-syms="Z" markdown="1">
Changes
:   The normalisation. <i>Z</i>(<i>q</i>) stops being a per-query constant and becomes a global constraint.

Consequence
:   Imposing marginals on the joint law couples queries that softmax treats independently, and the conditionals that come back are exactly softmax tilted by one potential per key. Queries are no longer independent, so there is no causal version — and asking for one raises an error that says so.

Read on
:   [`coupling` in the taxonomy](taxonomy.md#coupling)
</div>

<div class="op-panel" data-cat="recurrent" data-label="the operator" data-syms="y int v pi g mu Z" data-replace markdown="1">
Changes
:   The operator. <i>y</i> comes from a finite-state recurrence over positions rather than from an integral against a measure.

Consequence
:   A fixed-size state must serve every future query, which bounds recall however it is trained. It is still linear in <i>v</i>, so the invariant holds and the implied weights can be written down — they are simply not a probability measure. Fidelity to softmax is not the goal here.

Read on
:   [`recurrent` in the taxonomy](taxonomy.md#recurrent)
</div>

</div>
</div>

<div class="ab-measure" markdown="1">
A second axis cuts across all seven, and it decides how a method should be judged at
all: an `approximate` method targets dense softmax and is fairly measured by output
error; a `modify` method is a *different operator*, and fidelity to softmax was never
its goal. Conflating the two is the most persistent confusion in this literature, so
every method here declares which it is — and the test suite reads the declaration.
</div>

---

## One invariant, held by all 78

<div class="ab-measure" markdown="1">
Every method realises, explicitly or implicitly, a measure $\widehat\pi_q$ over context
positions and a value field $\widehat v$ — and must be able to hand you both.
</div>

<div class="ab-invariant" markdown="1">
`y == weights @ (v_hat if v_hat is not None else value)`

Checked for every registered method, in float32 and float64.
</div>

<div class="ab-measure" markdown="1">
This is the load-bearing property of the library. The error metrics decompose output
error into a distributional term and a value term, and that decomposition is
meaningless unless each method's reported weights and values really do reproduce its
own output. A method that genuinely cannot express its measure on the original $n$
positions declares `expresses_weights = False` and is excluded from those studies **by
declaration, not by accident**.

Correctness is anchored to an *exact limit* rather than to inspection. Each method
declares the configuration under which it provably reduces to dense softmax — top-$k$
with $k = n$, quantisation at full precision, Nyström with every token a landmark — and
the suite holds it to that, to `1e-5` in the output and `1e-6` in the weights. There is
nowhere for a transposed index or a dropped scale factor to hide.
</div>

<p class="ab-eyebrow">the conformance suite, parametrised over every registered method</p>

<ol class="ab-checks">
<li>Shapes, for every tensor the method returns</li>
<li>The invariant above</li>
<li>Row stochasticity, where the method claims it</li>
<li>No causal leakage, under perturbation of future keys</li>
<li>Batch and head independence</li>
<li>Exact reduction to softmax at the declared limit</li>
<li>Monotone convergence, for the Monte-Carlo estimators</li>
<li>Agreement across float32, float64, CPU and CUDA</li>
<li>A cost model that is finite, positive and consistent</li>
<li>No NaN at empty support or at large logits</li>
<li>Every cited key resolves in the bibliography</li>
</ol>

<div class="ab-measure" markdown="1">
A method that cannot pass a check must say so in a declaration attribute, not in a
`skip`. [How the suite reads those declarations](conformance.md).
</div>

---

## What the approximation cost you, and in which of the two ways

```python
import attnbank as ab

rows = ab.compare(["softmax", "topk", "performer", "quantkv"], query, key, value,
                  configs={"topk": {"k": 16}, "performer": {"r": 64},
                           "quantkv": {"bits_k": 16, "bits_v": 2}})
```

<div class="ab-measure ab-split-table" markdown="1">
| | output error | distributional | value |
|---|---|---|---|
| `topk` | 1.03 | **1.03** | 0.00 |
| `performer` | 2.76 | **2.76** | 0.00 |
| `quantkv` | 0.72 | 0.00 | **0.72** |

Restriction moves mass to the wrong places. Quantisation aggregates the wrong values.
Both arrive as "error" in a single number; only the split says which mechanism is at
fault — and the split is algebra, not a heuristic. It closes to $8.5\times10^{-15}$
over 1.78 million query rows.
</div>

---

## Five minutes

```bash
pip install attnbank
```

```python
import attnbank as ab

y = ab.attend("topk", query, key, value, causal=True, k=64)
```

<div class="ab-measure" markdown="1">
Tensors are `[B, H, n, d]` throughout, and causal masking aligns the last query with
the last key, so the same code path serves prefill and decode.
</div>

<div class="ab-split" markdown="1">
[The taxonomy](taxonomy.md)
:   The seven categories as a tree you can open, with what each one preserves and which error term it incurs.

[Browse the bank](methods/index.md)
:   All 78 methods, searchable by name, category, objective, backend and flag — one generated page each.

[The invariant](invariant.md)
:   Why `weights @ v_hat` is the property everything else rests on, and what it costs a method to be unable to supply it.

[Backends](backends.md)
:   `reference`, `sdpa`, `flash_attn`, and why most of the bank correctly declares only the first.

[Examples gallery](examples/index.md)
:   Ten short scripts, each executed to produce the output shown beside it.

[API reference](api/index.md)
:   Every public function, with a runnable example in its docstring.
</div>

---

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

Released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
