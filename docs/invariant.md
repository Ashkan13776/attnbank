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

The decomposition is a *measurement* only if `out.weights` and `out.v_hat` really
reproduce `out.y`. If a method reports weights that are merely plausible — the
attention matrix it would have used, say, before some later correction — the
decomposition becomes a story about a computation that did not happen. Every study in
the monograph that separates distributional from value error rests on this, so the
library tests it rather than assuming it.

It also pins down something softer: it forces a method to say what it *is* in a common
language. Two methods from different categories, with nothing in common at the level of
code, become comparable the moment both can write down $\widehat\pi_q$ and $\widehat v$.

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
