# Based

<p class="attn-standfirst" markdown="span">The truncated-Taylor feature map for the exponential interaction.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `based` &middot; `ab.build("based")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.BasedAttention` &middot; [`attnbank/methods/interaction/based.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/based.py)

monograph
:   `sec:kernel`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the interaction $g(q,x)$, replaced by a surrogate $\widehat g$. $\pi_q = \widehat g(q,\cdot)\mu / \widehat Z(q)$ is then a different measure on the same $n$ positions.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

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


**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("based").convergence_configs()
[{'degree': 2}, {'degree': 4}, {'degree': 6}, {'degree': 8}]
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 1,099,302,912 | 268,435,456 | 4.1× |
| `state_bytes` | 540,930 | 262,144 | 2.06× |
| `bytes_moved` | 1,065,218 | 4,718,592 | 0.226× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.feature_dim(d)
    return {"flops": 2 * n * r * dv + 2 * m * r * dv + (m + n) * r,
            "state_bytes": r * (dv + 1) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
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

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_hat`, `degree`, `g_hat`, `r`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `degree` | `2` |
| `seed` | `0` |

</div>

`degree` is the (even) Taylor order `P`; `seed` is accepted and unused.

The estimator is deterministic — there is nothing to seed. The keyword exists
because the conformance suite's convergence harness sweeps `seed` alongside
`convergence_configs` for every method that declares one.

```python
ab.describe("based")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("based", degree=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["based"], q, k, v, causal=True,
                 configs={"based": {'degree': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
BasedAttention(name='based', category='interaction', objective='approximate', backend='reference', degree=2, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_hat, degree, g_hat, r], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.3104, 'distributional_error': 0.3104, 'value_error': 0.0, 'tv': 0.08696}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:kernel-det`, `sec:kernel-rf`.  Arora et al. (2024) replace
`g(q,x) = exp(<qt, kt_x>)` — with `qt = q / d^{1/4}`, so that `<qt, kt_x>` is
the scaled logit — by its second-order Taylor polynomial

```text
sim(q, k) = 1 + q^T k + (q^T k)^2 / 2,
```

which is *exactly* separable: `phi(u) = [1, u, u (x) u / sqrt 2]` gives
`<phi(q), phi(k)> = sim(q,k)` with `r = 1 + d + d^2` features, and the usual
normalized readout `y_i = phi(q_i)^T S / phi(q_i)^T z`.  This file implements the
interaction and the readout; Based the *architecture* interleaves it with exact
softmax over a short sliding window (`w = 64`), which in this taxonomy is a separate
`restrict/` operator and not part of `g_hat`.

Unlike every other fixed feature map in this directory, this one has a resource
parameter that actually converges.  `degree` sweeps the truncation order,

```text
g_hat(q, x) = sum_{p=0}^{P} <qt, kt_x>^p / p!   ->   exp(<qt, kt_x>) = g(q, x),
```

uniformly on bounded logits, so Based is an *approximation* with a genuine anchor
rather than an operator modification — the first entry in the family for which that
is true without randomness.  `convergence_configs` sweeps `P`; there is no finite
`P` that is exact, so `limit_config` is `None`.

Only **even** `P` is accepted, and this is not a convenience: the truncated
exponential series has no real root exactly when `P` is even, which is what makes
`g_hat > 0`, `pi_hat` a probability measure and `row_stochastic` an honest
declaration.  At odd `P` the surrogate goes negative for sufficiently negative
logits and the measure is lost.  `P = 2` is the paper's choice and the default; its
minimum `1/2` at logit `-1` is the floor that `rebased` exists to remove —
Based cannot assign a near-zero weight however negative the logit.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`linear_elu`](linear_elu.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polaformer`](polaformer.md), [`polysketch`](polysketch.md), [`rebased`](rebased.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`atom`](atom.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`hyperattention`](hyperattention.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lsh_sample`](lsh_sample.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`oval`](oval.md), [`page_pca`](page_pca.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `based` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `based` selected](index.md?compare=based) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Simple linear attention language models balance the recall-throughput tradeoff.*<br><span class="attn-cite" data-bibkey="arora2024based"><code>arora2024based</code></span>
- *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.*<br><span class="attn-cite" data-bibkey="katharopoulos2020transformers"><code>katharopoulos2020transformers</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="arora2024based">
<pre class="attn-bib__src"><code>@inproceedings{arora2024based,
  title     = {Simple linear attention language models balance the recall-throughput tradeoff}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="katharopoulos2020transformers">
<pre class="attn-bib__src"><code>@inproceedings{katharopoulos2020transformers,
  title     = {Transformers are {RNN}s: Fast Autoregressive Transformers with Linear Attention}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/based.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/based.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
