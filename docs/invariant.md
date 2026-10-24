<nav class="ab-path" aria-label="The path through these pages">
  <ol>
    <li><a href="../quickstart/"><b>01</b>Quickstart</a></li>
    <li aria-current="page"><b>02</b>The invariant</li>
    <li><a href="../conformance/"><b>03</b>Conformance</a></li>
    <li><a href="../backends/"><b>04</b>Backends</a></li>
  </ol>
</nav>

# Invariant W

<div class="ab-property">
<code>y == weights @ (v_hat if v_hat is not None else value)</code>
<span class="ab-property-note">Checked for every registered method, causal and
non-causal, in float64 — check 2 of <a href="../conformance/">the conformance
suite</a>.</span>
</div>

<p class="ab-lead">Every registered method satisfies this. It is the single most
important property in the library, and the reason the library exists in the form
it does.</p>

## What it says

Every efficient attention method, whatever else it does, realises a measure
$\widehat\pi_q$ over the $n$ context positions and a value field $\widehat v$:

$$\widehat y(q) \;=\; \sum_{j=1}^{n} \widehat\pi_q(j)\,\widehat v_j .$$

Methods do not usually advertise this. A Performer computes feature maps, a
StreamingLLM evicts tokens, a linear RNN carries a state matrix. None of them looks
like a weighted average over positions. All of them *are* one, and the library requires
each to show you which one.

```python
out = ab.build("performer", r=64)(query, key, value, return_weights=True)
out.weights          # the measure the Performer is implicitly using
ab.check_invariant(out, value)
# 0.0
```

## Why it matters

Because of this:

$$y - \widehat y \;=\; \underbrace{(\pi - \widehat\pi)\,v}_{\text{distributional}}
\;+\; \underbrace{\widehat\pi\,(v - \widehat v)}_{\text{value}}$$

An approximation can be wrong in exactly two ways — it attends to the wrong places, or
it aggregates the wrong values — and this identity separates them **exactly**, not
approximately.

```python
from attnbank import build, metrics

ref = build("softmax")(q, k, v, causal=True, return_weights=True)
out = build("quantkv", bits_k=4, bits_v=4)(q, k, v, causal=True,
                                           return_weights=True)
metrics.error_decomposition(ref.weights, out.weights, v, out.v_hat)
# {'distributional': …, 'value': …, 'total': …}
```

<figure class="ab-fig" id="fig-error-split">
<div class="ab-fig-plate">
--8<-- "docs/figures/error_split.svg"
--8<-- "docs/figures/error_split.narrow.svg"
</div>
<figcaption><b class="ab-fig-n">Figure 1</b><strong>Where the error comes
from.</strong> That call, run on nine configurations against dense softmax.
Each bar is one configuration, drawn to the full width and divided at the
share of the error its <em>distributional</em> term carries; the median
<code>‖y − ŷ‖</code> over all queries is printed beside it. The four at the top
keep every value exactly and are wrong only about where to look.
<code>quantkv</code> appears three times deliberately: at 2-bit keys the error
is entirely distributional, at 2-bit values entirely value, and at 3 bits each
it is roughly half of both — one method, totals of the same order, three
different mechanisms. Reading one number instead of two hides which.
<span class="ab-fig-src">attnbank.data.synthetic(n=256, d=64, dv=64, heads=4,
passages=2, seed=0), float64, causal · docs/gen_figures.py ·
examples/03_error_decomposition.py prints the same split</span></figcaption>
</figure>

The decomposition is a *measurement* only if `out.weights` and `out.v_hat` really
reproduce `out.y`. If a method reports weights that are merely plausible — the
attention matrix it would have used, say, before some later correction — the
decomposition becomes a story about a computation that did not happen. Every study in
the monograph that separates distributional from value error rests on this, so the
library tests it rather than assuming it.

It also pins down something softer: it forces a method to say what it *is* in a common
language. Two methods from different categories, with nothing in common at the level of
code, become comparable the moment both can write down $\widehat\pi_q$ and $\widehat v$.

And comparable at a matched budget is where the surprises are. Both methods below are
restriction oracles: they keep $k$ keys per query and the rest of the measure is thrown
away. One picks the $k$ largest weights. The other picks the $k$ that minimise the
output error — which is not the same set.

<figure class="ab-fig" id="fig-selection-gap">
<div class="ab-fig-plate">
--8<-- "docs/figures/selection_gap.svg"
--8<-- "docs/figures/selection_gap.narrow.svg"
</div>
<figcaption><b class="ab-fig-n">Figure 2</b><strong>Top-k is not
output-optimal.</strong> Output error against the per-query budget <code>k</code>
for two selection rules on identical inputs: <code>topk</code>, which keeps the
largest attention weights, and <code>value_topk</code>, which keeps the keys that
minimise <code>‖y − ŷ‖</code>. The shaded band is the gap — between 1.28× and 1.44×
at every budget measured, and it does not close as <code>k</code> grows. Selecting
the heaviest mass minimises the omitted mass <code>ε_q</code>, not the error; the
bound is <code>ε_q · osc(v)</code>, and a light token carrying an extreme value
outweighs a heavy one sitting near the mean. Both curves are oracles that see the
true scores, so what is being measured here is the rule and not the budget — a
deployable selector can only do worse than <code>topk</code>.
<span class="ab-fig-src">attnbank.data.synthetic(n=128, d=32, dv=32, heads=2,
passages=2, seed=0), float64, causal · docs/gen_figures.py ·
examples/04_topk_is_not_output_optimal.py</span></figcaption>
</figure>

## When a method genuinely cannot

Some methods cannot express their measure on the original $n$ positions. They declare

```python
expresses_weights = False
```

and return `weights=None`. They are then excluded from the decomposition studies **by
declaration, not by accident** — and the conformance suite checks the declaration in
both directions. This is the check, verbatim from `tests/test_conformance.py`:

```python
if not meth.expresses_weights:
    assert out.weights is None, f"{name} declares expresses_weights=False but returned weights"
    return
assert out.weights is not None, f"{name} declares expresses_weights=True but returned None"
recon = out.weights @ (out.v_hat if out.v_hat is not None else v)
assert torch.allclose(out.y, recon, atol=1e-9, rtol=0), \
    f"{name}: max |y - W v| = {(out.y - recon).abs().max():.3e}"
```

Setting the flag to `False` to dodge the invariant is the one thing that will get a
contribution rejected. It silently removes a method from every study that would have
measured it.

## Checking it yourself

```python
ab.check_invariant(output, value)   # max |y - weights @ v_hat|; raises if too large
ab.check_conformance("topk")        # a printable report including this check
```

```text
topk: 9 checks passed, 0 failed
  [pass] shapes -- AttentionOutput(y=[2, 2, 6, 6], weights=[2, 2, 6, 12], …)
  [pass] invariant W -- max |y - W v| = 0.00e+00
  [pass] row stochastic -- row sums in [1.000000, 1.000000]
  [pass] no causal leakage -- queries 0..3 unchanged
  [pass] finite at large logits -- q, k scaled by 10
  [pass] inputs not mutated
  [pass] reproducible -- two builds agree bit for bit
  [pass] cost model -- {'flops': 1800, 'state_bytes': 336, 'bytes_moved': 936}
  [pass] exact limit -- limit_config={'k': None}, max |y - softmax| = 4.44e-16
```

Tolerances: the suite runs the check in float64 at `1e-9`, where there is nowhere for
a floating-point error to hide. At runtime, `ab.check_invariant` defaults to `1e-10`
in float64 and `1e-5` in float32 — the tolerances of `DESIGN.md` Sec. 1.

See the invariant held across the whole bank, with the measure each method implies
drawn out: [Invariant W, the gallery entry](examples/02_the_invariant.md).

<nav class="ab-next" aria-label="Continue reading">
  <a href="../conformance/">
    <span class="ab-next-kicker">Next · 03 of 04</span>
    <span class="ab-next-title">Conformance</span>
    <span class="ab-next-desc">Eleven checks, read from each method's own
    declarations — how the invariant, and everything around it, is enforced rather
    than hoped for.</span>
  </a>
  <p class="ab-next-also">Also from here:
  <a href="../examples/02_the_invariant/">Invariant W across the whole bank</a> ·
  <a href="../examples/03_error_decomposition/">the error decomposition, run</a>.</p>
</nav>
