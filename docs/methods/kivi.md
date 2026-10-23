# KIVI

<p class="attn-standfirst" markdown="span">Keys quantized **per channel**, values **per token**, with a full-precision residual.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `kivi` &middot; `ab.build("kivi")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.KIVIAttention` &middot; [`attnbank/methods/compress/kivi.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/kivi.py)

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
>>> ab.describe("kivi").limit_config
{'bits_k': None, 'bits_v': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,566,528 | 268,435,456 | 1× |
| `state_bytes` | 71,136 | 262,144 | 0.271× |
| `bytes_moved` | 333,280 | 4,718,592 | 0.0706× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    """Quantized groups, the full-precision residual, and the scales both need."""
    res = min(self.residual, n)
    nq = (max(0, n - res) // self.group) * self.group
    nf = n - nq
    bk = (self.bits_k or 8 * bytes_per_el) / 8
    bv = (self.bits_v or 8 * bytes_per_el) / 8
    # per-channel key scales: 2 per (group, channel); per-token value scales: 2 each
    scales = 2 * bytes_per_el * (-(-nq // self.group) * d + nq)
    state = nq * (d * bk + dv * bv) + nf * (d + dv) * bytes_per_el + scales
    return {"flops": 2 * m * n * (d + dv) + n * (d + dv),
            "state_bytes": state,
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

Diagnostics returned in `extras`: `delta_k`, `delta_v`, `n_sealed`, `s`, `s_hat`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `bits_k` | `2` |
| `bits_v` | `2` |
| `group` | `8` |
| `residual` | `8` |

</div>

```python
ab.describe("kivi")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("kivi", bits_k=2, bits_v=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["kivi"], q, k, v, causal=True,
                 configs={"kivi": {'bits_k': 2, 'bits_v': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
KIVIAttention(name='kivi', category='compress', objective='approximate', backend='reference', bits_k=2, bits_v=2, group=8, residual=8)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[delta_k, delta_v, n_sealed, s, s_hat], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.6196, 'distributional_error': 0.17481, 'value_error': 0.56148, 'tv': 0.05733}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:compress-quant` and `prop:compress-sensitivity`.  KIVI is the scheme that
reads the key/value asymmetry off the operator and groups the two caches along
different axes:

* **Keys, per channel.**  The scale and zero-point are shared by `group` consecutive
  *tokens* of one channel, so an outlier channel's coarse step stays inside that
  channel.  What matters for `pi_q` is only the token-varying part of
  `delta l_j = beta <q, delta k_j>` — a channel-constant error is absorbed by the
  softmax normalizer — and per-channel grouping is exactly the choice that makes the
  surviving part small.
* **Values, per token.**  `y = sum_j pi_j v_j` is a weighted average over tokens, so
  a per-token scale confines each token's representation error to its own summand;
  no token can poison another.

**The residual window is what makes per-channel grouping causal.**  A per-channel scale
pools statistics over tokens, which is why `quantkv(axis='channel')` refuses
`causal=True`.  KIVI avoids this by *streaming*: the last `residual` tokens are kept
in full precision, and a group of `group` tokens is quantized only once it is complete
and has fallen out of the residual window.  A query at absolute position `p` therefore
sees quantized keys only for the groups entirely contained in `[0, p + 1 - residual)`,
and full-precision keys for everything more recent:
    L(p) = group * max(0, (p + 1 - residual) // group)
    k_hat_j(p) = Q_channel(k)_j   if j < L(p),   else   k_j

The key representation is thus *query-dependent*, which is the honest rendering of a
streaming cache; the score matrix is assembled by selecting between the two score
matrices rather than by materialising `[m, n, d]` keys.  Nothing before position `t`
can depend on a token at or after `t`, which is what the conformance suite checks.

Values need no residual: per-token quantization is already causal, and Invariant W
requires a single `v_hat` per position, so the value residual of Algorithm 1 is
dropped here and recorded as such.

Quantize–dequantize is **simulated** in the input dtype (via
`attnbank.components.quantization.affine_quantize`).  The error is real; the
memory saving is not — `cost()` reports what the scheme *would* store, including the
per-group scale and zero-point, which is the term that stops `group` from shrinking
to one.

Exact limit: `bits_k = bits_v = None` (the identity quantizer).  Affine quantization
at 16 bits is not bit-exact in fp32, so it would be a dishonest anchor.

Paper defaults are `group = 32` and `residual = 128`; the defaults here are scaled
to the bank's test sizes so that the default configuration actually quantizes something.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`based`](based.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`lsh_sample`](lsh_sample.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`skyformer`](skyformer.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `kivi` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `kivi` selected](index.md?compare=kivi) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.*<br><span class="attn-cite" data-bibkey="liu2024kivi"><code>liu2024kivi</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="liu2024kivi">
<pre class="attn-bib__src"><code>@inproceedings{liu2024kivi,
  title     = {{KIVI}: A Tuning-Free Asymmetric 2bit Quantization for {KV} Cache}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/kivi.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/kivi.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
