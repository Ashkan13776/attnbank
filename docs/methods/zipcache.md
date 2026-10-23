# ZipCache

<p class="attn-standfirst" markdown="span">Bit width allocated per token by a **normalised** attention score.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `zipcache` &middot; `ab.build("zipcache")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.ZipCacheAttention` &middot; [`attnbank/methods/compress/zipcache.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/zipcache.py)

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
| `supports_causal` | **no** | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `bits_salient=None`, `bits_regular=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("zipcache").limit_config
{'bits_salient': None, 'bits_regular': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 402,653,184 | 268,435,456 | 1.5× |
| `state_bytes` | 57,600 | 262,144 | 0.22× |
| `bytes_moved` | 319,744 | 4,718,592 | 0.0678× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    """Two bit widths mixed by ``salient``, one scale per token, one per channel."""
    bs = (self.bits_salient or 8 * bytes_per_el) / 8
    br = (self.bits_regular or 8 * bytes_per_el) / 8
    bits = self.salient * bs + (1.0 - self.salient) * br
    side = (d + dv) * bytes_per_el + 4 * n * bytes_per_el     # a_c, plus 2 per token
    probe = m if self.probe is None else max(1, int(round(self.probe * m)))
    return {"flops": 2 * m * n * (d + dv) + 2 * probe * n * d,
            "state_bytes": n * (d + dv) * bits + side,
            "bytes_moved": m * (d + dv) * bytes_per_el + n * (d + dv) * bits + side}
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

Diagnostics returned in `extras`: `accumulated`, `delta_k`, `delta_v`, `s`, `s_hat`, `saliency`, `salient`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `bits_salient` | `4` |
| `bits_regular` | `2` |
| `salient` | `0.5` |
| `probe` | `None` |
| `target` | `'kv'` |
| `seed` | `0` |

</div>

```python
ab.describe("zipcache")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("zipcache", bits_salient=4, bits_regular=2)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["zipcache"], q, k, v, causal=False,
                 configs={"zipcache": {'bits_salient': 4, 'bits_regular': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
ZipCacheAttention(name='zipcache', category='compress', objective='approximate', backend='reference', bits_regular=2, bits_salient=4, probe=None, salient=0.5, seed=0, target='kv')
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[accumulated, delta_k, delta_v, s, s_hat, saliency, salient], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.35175, 'distributional_error': 0.23298, 'value_error': 0.26375, 'tv': 0.08671}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:compress-quant`, and the place where this family meets `sec:compress-evict`.
KIVI, KVQuant and Atom all spend the same number of bits on every token.  ZipCache does
not: it ranks tokens by how much attention they actually receive and gives the salient
ones `bits_salient` while the rest get `bits_regular`.  Eviction is the degenerate
case of this — zero bits for the unsalient — and `prop:compress-noeviction` applies
to the ranking either way.

Two ideas, and the second is the one worth implementing carefully.

**Channel-separable token-wise quantization.**  Each channel is first divided by its
own maximum absolute value over tokens, then whole tokens are quantized with a single
scale, then the channel normalizer is multiplied back:
    a_c = max_j |x_jc|,     x_hat_j = a * Q_b(x_j / a).

What this buys is *relative* precision, channel by channel.  A plain per-token scale is
set by whichever channel is largest in that token, so every channel is given the same
**absolute** step and the small channels — the ones a query may well load on — are
obliterated.  Dividing by `a_c` first makes the step proportional to each channel's
own scale, so the relative error is near-uniform.  It is not a smaller Frobenius error,
and the test file says so.  The side information is `d + 2n` numbers rather than the
`2 n d / g` of a group-wise scheme.

**The normalised attention score.**  The usual saliency, `sum_i A_ij` accumulated down
a column, is biased twice over by the causal mask: an early token appears in more rows,
and the rows it appears in are shorter, so their softmax puts more mass on each entry.
Both biases reward *position* rather than content.  ZipCache divides by the number of
queries that could see the token:
    p_j = sum_i A_ij / nnz(A_{.,j}),     nnz(A_{.,j}) = #{i : j <= i + (n - m)}.

The saliency is always measured on the **causally masked** prefill attention, whatever
`causal` is for the forward pass itself, because that is the quantity a prefill
actually has; `nnz` is then a genuine correction rather than the constant `m` a
non-causal matrix would give.

`probe` enables the paper's cheap approximation: estimate the column sums from a
sample of query rows (half the most recent, half drawn uniformly) instead of the whole
matrix, which is what makes the metric compatible with a kernel that never materialises
`A`.  `probe = None` uses every row and is the exact metric.

**Not causal.**  The channel normalizers pool over tokens, and the salient set is chosen
after seeing the whole prompt and then applied to every query — the prefill-then-decode
pattern of `evict_snapkv`, and like it, something that cannot be written as one
causally masked pass.

Quantize–dequantize is **simulated** in the input dtype; nothing is actually stored
smaller, and `cost()` states what would be.

`target` restricts the quantizer to the keys or to the values.  The paper quantizes
both with the same allocation; the split exists because the error decomposition of
`sec:output_error` — the distributional term against the value term — can only be
run on a scheme whose two halves can be switched off independently, and this family is
the one that decomposition was written for.

Exact limit: `bits_salient = bits_regular = None`.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`based`](based.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`lsh_sample`](lsh_sample.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`skyformer`](skyformer.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `zipcache` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `zipcache` selected](index.md?compare=zipcache) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification.*<br><span class="attn-cite" data-bibkey="he2024zipcache"><code>he2024zipcache</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="he2024zipcache">
<pre class="attn-bib__src"><code>@inproceedings{he2024zipcache,
  title     = {{ZipCache}: Accurate and Efficient {KV} Cache Quantization with Salient Token Identification}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/zipcache.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/zipcache.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
