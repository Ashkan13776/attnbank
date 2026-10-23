# Dense softmax attention

<p class="attn-standfirst" markdown="span">The reference operator, y(q) = sum_j pi_q(j) v_j.</p>

<div class="attn-facts" data-cat="exact" markdown="1">

key
:   `softmax` &middot; `ab.build("softmax")`

category
:   [`exact`](../taxonomy.md#exact)

objective
:   `accelerate` — same operator, exact up to floating point; only the evaluation changes

class
:   `attnbank.SoftmaxAttention` &middot; [`attnbank/reference.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/reference.py)

monograph
:   `sec:found`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes nothing in $y(q)=\int v\,\mathrm{d}\pi_q$. The integrand, the measure and the normaliser are the reference operator's; only the order of arithmetic changes.

It pursues **acceleration**: the operator is the reference operator, exact up to floating point. The only question worth asking of it is *where* exactness stops.

</div>

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **yes** | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

This method is **exact as built**: the conformance suite instantiates it at its default config and requires both `y` and the weights to match `softmax`. `limit_config()` returns `{}` — the empty dict, not `None`: there is a limit and no setting has to be changed to reach it.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,435,456 | 268,435,456 | 1× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 4,718,592 | 4,718,592 | 1× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    return {"flops": 2 * m * n * (d + dv), "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + 2 * m * n) * bytes_per_el}
```

!!! note "A cost model is not a benchmark"
    These are counted operations and resident bytes, not measured time.
    They are what the monograph's capacity arguments are made of;
    `examples/08_cost_and_capacity.py` walks through the comparison properly.

## Backends

<div class="attn-tab" markdown="1">

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |
| `sdpa` | `torch.nn.functional.scaled_dot_product_attention` -- on CUDA, FlashAttention-2 or the memory-efficient kernel |
| `flash_attn` | the `flash-attn` package (CUDA, fp16/bf16) |

</div>

Each non-`reference` entry is a claim that the path is **equal** to the reference, and `tests/test_conformance.py::test_backend_equivalence` checks it on five shapes, both causal settings, float32 and float64. See [Backends](../backends.md).

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `sdpa_kernel` | `None` |

</div>

`sdpa_kernel`: pin the SDPA sub-backend (`flash` / `mem_efficient` /
`math`) instead of letting torch choose. Ignored by other backends.

```python
ab.describe("softmax")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("softmax")
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = accelerate: the claim is equality with the reference, so that
# is what the example checks.
y_ref = ab.attend("softmax", q, k, v, causal=True)
print("max |y - y_softmax|:", float((out.y - y_ref).abs().max()))
```

<div class="attn-out" markdown="1">

```text
SoftmaxAttention(name='softmax', category='exact', objective='accelerate', backend='reference', sdpa_kernel=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras={}, dtype=float32, device=cpu)
invariant W: 0.0
max |y - y_softmax|: 0.0
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

pi_q(j) = exp(beta <q, k_j>) / sum_i exp(beta <q, k_i>),   beta = 1/sqrt(d)

Everything in the library is compared against this.  Section `sec:found`.

</div>

## Related methods

changes the same object — [`exact`](../taxonomy.md#exact)
:   [`flash`](flash.md), [`lowprec`](lowprec.md), [`splitk`](splitk.md) &nbsp;·&nbsp; [browse the bank, exact only](index.md?category=exact)

Both lists are `ab.available(...)` with `softmax` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `softmax` selected](index.md?compare=softmax) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Attention Is All You Need.*<br><span class="attn-cite" data-bibkey="vaswani2017attention"><code>vaswani2017attention</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="vaswani2017attention">
<pre class="attn-bib__src"><code>@inproceedings{vaswani2017attention,
  title={Attention Is All You Need}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/reference.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/reference.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
