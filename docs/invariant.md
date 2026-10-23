# Invariant W

```python
y == weights @ (v_hat if v_hat is not None else value)
```

Every registered method satisfies this, in float32 and float64, causal and
non-causal, on every shape the conformance suite throws at it. It is the single most
important property in the library, and the reason the library exists in the form it
does.

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
out = build("quantkv", bits_k=4, bits_v=4)(q, k, v, causal=True, return_weights=True)
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
both directions:

```python
if not method.expresses_weights:
    assert out.weights is None    # declared False, so it had better not return any
else:
    assert out.weights is not None
    assert torch.allclose(out.y, out.weights @ (out.v_hat or v), atol=1e-9)
```

Setting the flag to `False` to dodge the invariant is the one thing that will get a
contribution rejected. It silently removes a method from every study that would have
measured it.

## Checking it yourself

```python
ab.check_invariant(output, value)        # -> max |y - weights @ v_hat|, raises if too large
ab.check_conformance("topk")             # -> a printable report including this check
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

Tolerances: `1e-10` in float64, `1e-5` in float32. The suite runs the check in float64,
where there is nowhere for an error to hide.

See the full gallery entry: [Invariant W across the whole bank](examples/02_the_invariant.md).
