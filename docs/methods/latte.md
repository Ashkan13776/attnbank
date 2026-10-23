# Latte

<p class="attn-standfirst" markdown="span">Latent attention as a probabilistic model — the *free* latent space.</p>

<div class="attn-facts" data-cat="factorize" markdown="1">

key
:   `latte` &middot; `ab.build("latte")`

category
:   [`factorize`](../taxonomy.md#factorize)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.LatteAttention` &middot; [`attnbank/methods/factorize/latte.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/latte.py)

monograph
:   `sec:factor`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the *shape* of $\pi_q$: it is routed through $r$ latents, $\pi_q = a(q,\cdot)\,b(\cdot,x)$, which caps the rank of the attention matrix at $r$.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

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
| `trainable` | **yes** | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

</div>

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 4,194,304 | 268,435,456 | 0.0156× |
| `state_bytes` | 2,048 | 262,144 | 0.00781× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.r
    return {"flops": 2 * m * r * d + 2 * n * r * (d + dv) + 2 * m * r * dv,
            "state_bytes": r * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
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

</div>

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `a`, `b`, `latent_keys`, `latent_queries`, `latent_values`, `r`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `8` |
| `seed` | `0` |

</div>

`r`: number of latent states. There is no `r` at which this is softmax.

```python
ab.describe("latte")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("latte", r=8)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = modify: this is a different operator and fidelity to softmax
# is not its goal, so there is no error to quote. What it *is* doing shows
# up in its own diagnostics.
print("row_stochastic =", attn.row_stochastic, "| first row sums:",
      [round(x, 4) for x in out.weights.sum(-1)[0, 0, :4].tolist()])
print("extras:", sorted(out.extras))
```

<div class="attn-out" markdown="1">

```text
LatteAttention(name='latte', category='factorize', objective='modify', backend='reference', r=8, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[a, b, latent_keys, latent_queries, latent_values, r], dtype=float32, device=cpu)
invariant W: 2.384185791015625e-07
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['a', 'b', 'latent_keys', 'latent_queries', 'latent_values', 'r']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:factor-def`, `def:factor-stochastic`.  Latte (Dolga et al., 2024) introduces a
latent variable `l` with `r` states that renders query and context conditionally
independent, `x ⊥ q | l`, and reads the attention measure straight off the
factorization

```text
pi_hat_q(x) = sum_l p(x | l) p(l | q),      Latte(Q, K, V) = softmax_r(Q) softmax_n(K)^T V,
```

with the two conditionals parameterized by one latent array each:

```text
p(l | q) = softmax_l( beta <q, w^q_l> ),    normalized over the r LATENTS,
p(x | l) = softmax_x( beta <k_x, w^k_l> ),  normalized over the n POSITIONS.
```

Both factors are Markov kernels, so this is a stochastic factorization in the sense of
`def:factor-stochastic`, `weights` is their product and `prop:factor-closure`
applies unchanged.

What separates it from `latent_stochastic` / `abc` is *where the latent keys come
from*.  There the read kernel is `softmax_z(beta <q, k_bar_z>)` with `k_bar_z =
int k db(z, .)` **summarizing the context**; here `w^q_l` is free, so

```text
the read kernel a(q, .) does not look at the context at all.
```

The query's only interaction with the keys is through which of `r` latents it
selects: `pi_hat` has rank at most `r`, and perturbing every key and value leaves
`a` bit-for-bit unchanged.  That is the cleanest instance in the bank of a
factorization whose two hops are genuinely separate — and the reason it is an
**operator modification**: no setting of `r` recovers softmax, because the free
latents never see the keys.  Both anchors are `None` by design (`tab:comparative`,
"mod.").

Causal (Latte's autoregressive form): the write kernel is normalized over the prefix,
`p(x | l, t) = softmax_{x <= t}(beta <k_x, w^k_l>)`, which the paper computes with
cumulative sums; the read kernel is untouched, since it never depended on the context.
This reference implementation materializes the prefix-normalized kernel `[m, r, n]`
and softmaxes each prefix independently — clarity over speed, and stable at any logit
scale.

The latent arrays `w^q, w^k` are the learned parameters; untrained they are seeded
Gaussians, so `trainable = True`.  Scaling: the paper writes the logits unscaled;
here they carry the bank's `beta = 1/sqrt(d)` so that the two kernels are on the same
footing as every other method's.

</div>

## Related methods

changes the same object — [`factorize`](../taxonomy.md#factorize)
:   [`abc`](abc.md), [`clustered`](clustered.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`perceiver`](perceiver.md), [`svd_oracle`](svd_oracle.md) *(oracle)* &nbsp;·&nbsp; [browse the bank, factorize only](index.md?category=factorize)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`gla`](gla.md), [`gqa`](gqa.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_elu`](linear_elu.md), [`linear_rnn`](linear_rnn.md), [`lowrank_ot`](lowrank_ot.md), [`lru`](lru.md), [`mega`](mega.md), [`mla`](mla.md), [`mlstm`](mlstm.md), [`otke`](otke.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `latte` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `latte` selected](index.md?compare=latte) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Latte: Latent Attention for Linear Time Transformers.*<br><span class="attn-cite" data-bibkey="dolga2024latte"><code>dolga2024latte</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="dolga2024latte">
<pre class="attn-bib__src"><code>@misc{dolga2024latte,
  title         = {Latte: Latent Attention for Linear Time Transformers}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/latte.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/latte.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
