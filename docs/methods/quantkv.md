# KV-cache quantization

<p class="attn-standfirst" markdown="span">Uniform affine (asymmetric) quantization of keys and values.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `quantkv` &middot; `ab.build("quantkv")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.QuantKVAttention` &middot; [`attnbank/methods/compress/quantkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/quantkv.py)

monograph
:   `sec:compress`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

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

**Exact limit.** Built with `bits_k=None`, `bits_v=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("quantkv").limit_config
{'bits_k': None, 'bits_v': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,435,456 | 268,435,456 | 1× |
| `state_bytes` | 65,536 | 262,144 | 0.25× |
| `bytes_moved` | 327,680 | 4,718,592 | 0.0694× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    bk = (self.bits_k or 8 * bytes_per_el) / 8
    bv = (self.bits_v or 8 * bytes_per_el) / 8
    state = n * (d * bk + dv * bv)
    return {"flops": 2 * m * n * (d + dv), "state_bytes": state,
            "bytes_moved": m * (d + dv) * bytes_per_el + state}
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

**`reference` only, and that is the correct answer.** It returns both the exact and the perturbed score matrices (`s`, `s_hat`) in `extras` — the perturbation *is* the object of study, and a fused kernel hands back neither.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `delta_k`, `delta_v`, `s`, `s_hat`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `bits_k` | `4` |
| `bits_v` | `4` |
| `axis` | `'token'` |
| `group` | `None` |

</div>

```python
ab.describe("quantkv")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("quantkv", bits_k=4, bits_v=4)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["quantkv"], q, k, v, causal=True,
                 configs={"quantkv": {'bits_k': 4, 'bits_v': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
QuantKVAttention(name='quantkv', category='compress', objective='approximate', backend='reference', axis='token', bits_k=4, bits_v=4, group=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[delta_k, delta_v, s, s_hat], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.15591, 'distributional_error': 0.08556, 'value_error': 0.12244, 'tv': 0.02758}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:compress-quant` and `prop:compress-sensitivity`.  Keys and values stay on the
same domain; only their representation changes, so

```text
k_hat_j = Q_{b_k}(k_j),   v_hat_j = Q_{b_v}(v_j),
Q_b(x)  = round((x - lo) / Delta) * Delta + lo,   Delta = (hi - lo) / (2^b - 1),
```

with `lo, hi` the min and max over a *group*.  The key error enters as a log-weight
perturbation `delta_l_j = beta <q, delta k_j>` — an exponential tilt of `pi_q` —
and is amplified by `e^{osc(delta_l)} - 1` and by `||q||`; the value error enters
linearly and is attention-weighted.  That asymmetry is the content of
`prop:compress-sensitivity` and of `eq:error-joint`, and is what study S4 measures.

Grouping.  `axis='token'` shares a scale among `group` channels of one token (KIVI's
value grouping); `axis='channel'` shares it among `group` tokens of one channel
(KIVI's key grouping, which confines an outlier channel's coarse step to that channel).
Only the *token-varying* part of the key error matters, since softmax cancels a common
shift — which is exactly why per-channel grouping helps for keys.

Per-channel grouping pools statistics across tokens, so it is **not causal**; the
method raises rather than silently leaking.  `axis='token'` is causal and is the
default.

Exact limit: `bits_k = bits_v = None` (the identity quantizer).  16-bit affine
quantization is *not* bit-exact in fp32, so it would be a dishonest anchor.

The quantizer itself is `attnbank.components.quantization.affine_quantize`,
shared with `lowprec`; this file is only the decision to apply it to the *cache*.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`based`](based.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`lsh_sample`](lsh_sample.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`skyformer`](skyformer.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `quantkv` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `quantkv` selected](index.md?compare=quantkv) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.*<br><span class="attn-cite" data-bibkey="liu2024kivi"><code>liu2024kivi</code></span>
- *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.*<br><span class="attn-cite" data-bibkey="hooper2024kvquant"><code>hooper2024kvquant</code></span>
- *IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact.*<br><span class="attn-cite" data-bibkey="liu2024intactkv"><code>liu2024intactkv</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="liu2024kivi">
<pre class="attn-bib__src"><code>@inproceedings{liu2024kivi,
  title     = {{KIVI}: A Tuning-Free Asymmetric 2bit Quantization for {KV} Cache}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="hooper2024kvquant">
<pre class="attn-bib__src"><code>@inproceedings{hooper2024kvquant,
  title     = {{KVQuant}: Towards 10 Million Context Length {LLM} Inference with {KV} Cache Quantization}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="liu2024intactkv">
<pre class="attn-bib__src"><code>@inproceedings{liu2024intactkv,
  title = {{IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact}}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/quantkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/quantkv.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
