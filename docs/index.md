---
hide:
  - toc
---

<!-- The landing page. Owned by one aspect of the site redesign; see lib/SITE_DESIGN.md.
     Every number on this page comes from the registry or the test suite:
     tools/check_sync.py guards the method counts written in prose; the version is
     attnbank.__version__; the per-category counts in the spectrum bar come from
     ab.method_table() and must be updated together with their aria-label. -->

<header class="ab-hero" markdown="1">

<p class="ab-eyebrow">attnbank 0.2.0 &middot; efficient attention, side by side</p>

# Every efficient attention method changes *one object* in *one formula*

<p class="ab-lede">Attention is a query-conditioned expectation. Efficiency has to come
from somewhere, and there are only so many places it can come from: a method either
evaluates the expectation differently, or it changes one of the objects in it. There
are seven objects, so there are seven categories — and that is the library's directory
structure, not a diagram drawn afterwards.</p>

<div class="ab-act" markdown="1">

```bash
pip install attnbank   # torch + numpy, nothing else
```

<p class="ab-act-links" markdown="1">[Five minutes to a first result](quickstart.md) <span class="ab-arrow">&rarr;</span></p>

</div>

<div class="ab-tape">
<a class="ab-stat" href="methods/"><b>78</b> <span>methods</span></a>
<a class="ab-stat" href="taxonomy/"><b>7</b> <span>categories</span></a>
<a class="ab-stat" href="invariant/"><b>1</b> <span>invariant</span></a>
<a class="ab-stat" href="conformance/"><b>3334</b> <span>tests</span></a>
</div>

<div class="ab-spectrum" role="img" aria-label="78 methods by category — exact 4, interaction 14, restrict 15, compress 18, factorize 8, coupling 6, recurrent 13">
<span data-cat="exact" style="flex-grow:4" title="exact &middot; 4 methods"></span>
<span data-cat="interaction" style="flex-grow:14" title="interaction &middot; 14 methods"></span>
<span data-cat="restrict" style="flex-grow:15" title="restrict &middot; 15 methods"></span>
<span data-cat="compress" style="flex-grow:18" title="compress &middot; 18 methods"></span>
<span data-cat="factorize" style="flex-grow:8" title="factorize &middot; 8 methods"></span>
<span data-cat="coupling" style="flex-grow:6" title="coupling &middot; 6 methods"></span>
<span data-cat="recurrent" style="flex-grow:13" title="recurrent &middot; 13 methods"></span>
</div>

</header>

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
Every method realises, explicitly or implicitly, a measure &#960;&#770;<sub><i>q</i></sub>
over context positions and a value field <i>v&#770;</i> — and must be able to hand you
both.
</div>

<div class="ab-invariant" data-label="invariant W" markdown="1">
`y == weights @ (v_hat if v_hat is not None else value)`

Checked for every registered method, in float32 and float64.
</div>

<div class="ab-measure" markdown="1">
This is the load-bearing property of the library. The error metrics decompose output
error into a distributional term and a value term, and that decomposition is
meaningless unless each method's reported weights and values really do reproduce its
own output. A method that genuinely cannot express its measure on the original
<i>n</i> positions declares `expresses_weights = False` and is excluded from those
studies **by declaration, not by accident**.

Correctness is anchored to an *exact limit* rather than to inspection. Each method
declares the configuration under which it provably reduces to dense softmax —
top-<i>k</i> with <i>k</i>&#8201;=&#8201;<i>n</i>, quantisation at full precision,
Nyström with every token a landmark — and the suite holds it to that, to `1e-5` in the
output and `1e-6` in the weights. There is nowhere for a transposed index or a dropped
scale factor to hide.
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

<div class="ab-errwrap">
<table class="ab-errsplit">
<thead>
<tr><th>method</th><th>category</th><th>output error</th><th>distributional</th><th>value</th></tr>
</thead>
<tbody>
<tr data-cat="restrict"><td><code>topk</code></td><td class="ab-cat">restrict</td><td>1.03</td><td><strong>1.03</strong></td><td>0.00</td></tr>
<tr data-cat="interaction"><td><code>performer</code></td><td class="ab-cat">interaction</td><td>2.76</td><td><strong>2.76</strong></td><td>0.00</td></tr>
<tr data-cat="compress"><td><code>quantkv</code></td><td class="ab-cat">compress</td><td>0.72</td><td>0.00</td><td><strong>0.72</strong></td></tr>
</tbody>
</table>
</div>

<div class="ab-mech">
<p data-cat="restrict"><strong>Restriction moves mass to the wrong places.</strong>
The error is distributional; the values are untouched.</p>
<p data-cat="compress"><strong>Quantisation aggregates the wrong values.</strong>
The mass is where it should be; the integrand is not.</p>
</div>

<div class="ab-measure" markdown="1">
Both arrive as "error" in a single number; only the split says which mechanism is at
fault — and the split is algebra, not a heuristic. It closes to
8.5&#8201;&times;&#8201;10<sup>&minus;15</sup> over 1.78 million query rows.
</div>

---

## The first call

```python
import attnbank as ab

y = ab.attend("topk", query, key, value, causal=True, k=64)
```

<div class="ab-measure" markdown="1">
Tensors are `[B, H, n, d]` throughout, and causal masking aligns the last query with
the last key, so the same code path serves prefill and decode. The worked version,
with tensor conventions and the first comparison, is [the quickstart](quickstart.md).
</div>

<p class="ab-eyebrow">where to go next</p>

<dl class="ab-next">
<div><dt><a href="taxonomy/">The taxonomy <span class="ab-arrow">&rarr;</span></a></dt>
<dd>The seven categories as a tree you can open, with what each one preserves and which error term it incurs.</dd></div>
<div><dt><a href="methods/">Browse the bank <span class="ab-arrow">&rarr;</span></a></dt>
<dd>All 78 methods, searchable by name, category, objective, backend and flag — one generated page each.</dd></div>
</dl>

<dl class="ab-next ab-next--more">
<div><dt><a href="invariant/">The invariant</a></dt>
<dd>Why <code>weights @ v_hat</code> is the property everything else rests on, and what it costs a method to be unable to supply it.</dd></div>
<div><dt><a href="backends/">Backends</a></dt>
<dd><code>reference</code>, <code>sdpa</code>, <code>flash_attn</code>, and why most of the bank correctly declares only the first.</dd></div>
<div><dt><a href="examples/">Examples gallery</a></dt>
<dd>Ten short scripts, each executed to produce the output shown beside it.</dd></div>
<div><dt><a href="api/">API reference</a></dt>
<dd>Every public function, with a runnable example in its docstring.</dd></div>
</dl>

---

<div class="ab-cite" markdown="1">

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

</div>
