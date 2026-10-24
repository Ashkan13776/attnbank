# KVQuant

<p class="attn-standfirst" markdown="span">Pre-RoPE per-channel keys, a sensitivity-weighted non-uniform datatype, and isolated outliers.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `kvquant` &middot; `ab.build("kvquant")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.KVQuantAttention` &middot; [`attnbank/methods/compress/kvquant.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/kvquant.py)

monograph
:   `sec:compress`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="no">no</span> | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `bits_k=None`, `bits_v=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("kvquant").limit_config
{'bits_k': None, 'bits_v': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 269,484,032 | 268,435,456 | 1× |
| `state_bytes` | 77,302 | 262,144 | 0.295× |
| `bytes_moved` | 339,446 | 4,718,592 | 0.0719× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    """Payload, per-channel/per-token scales, the NUQ codebook and the fp16 outliers."""
    bk = (self.bits_k or 8 * bytes_per_el) / 8
    bv = (self.bits_v or 8 * bytes_per_el) / 8
    book = 0.0
    if self.nuq:                                   # one codebook per quantized vector
        book = bytes_per_el * ((2 ** (self.bits_k or 4)) * d
                               + (2 ** (self.bits_v or 4)) * n)
    scales = 0.0 if self.nuq else 2 * bytes_per_el * (d + n)
    sparse = 2 * self.sparsity * (n * d + n * dv) * (bytes_per_el + 2)
    pinned = self.keep_first * (d + dv) * bytes_per_el
    state = n * (d * bk + dv * bv) + book + scales + sparse + pinned
    return {"flops": 2 * m * n * (d + dv) + self.iters * n * (d + dv),
            "state_bytes": state,
            "bytes_moved": m * (d + dv) * bytes_per_el + state}
```

## Backends

**`reference` only, and that is the correct answer.** It returns both the exact and the perturbed score matrices (`s`, `s_hat`) in `extras` — the perturbation *is* the object of study, and a fused kernel hands back neither.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `delta_k`, `delta_v`, `s`, `s_hat`, `sensitivity`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `bits_k` | `3` |
| `bits_v` | `3` |
| `pre_rope` | `True` |
| `nuq` | `True` |
| `sensitivity` | `True` |
| `sparsity` | `0.01` |
| `keep_first` | `1` |
| `theta` | `10000.0` |
| `iters` | `8` |

</div>

`ab.describe("kvquant")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("kvquant", bits_k=3, bits_v=3)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["kvquant"], q, k, v, causal=False,
                 configs={"kvquant": {'bits_k': 3, 'bits_v': 3}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
KVQuantAttention(name='kvquant', category='compress', objective='approximate', backend='reference', bits_k=3, bits_v=3, iters=8, keep_first=1, nuq=True, pre_rope=True, sensitivity=True, sparsity=0.01, theta=10000.0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[delta_k, delta_v, s, s_hat, sensitivity], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.17481, 'distributional_error': 0.12803, 'value_error': 0.11221, 'tv': 0.05101}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-quant</code> and <code>prop:compress-sensitivity</code></p>

Where `kivi` changes only
the *grouping* axis, KVQuant changes the **coordinate system**, the **datatype** and the
**support** of the quantizer, in four independent moves that this file exposes as four
switches:

1. **Per-channel keys** (`bits_k`).  The scale and zero-point are shared by all tokens
   of one channel, which is where the key outliers live.  Values stay **per token**,
   because `y` averages over tokens.
2. **Pre-RoPE** (`pre_rope`).  RoPE rotates channel pairs by a *position-dependent*
   angle, so after it the "outlier channel" structure that per-channel grouping exploits
   is smeared across each pair.  KVQuant stores the key **before** the rotation and
   re-applies RoPE on the fly after dequantization.  The bank's tensors are post-RoPE by
   convention, so this file un-rotates, quantizes, and re-rotates — the same three
   operations the cache performs, in the same order.
3. **Non-uniform datatype** (`nuq`).  The quantization signposts are the centroids of a
   **sensitivity-weighted k-means** on the entries of each vector, minimising
   `sum_i F_i (x_i - Q(x_i))^2` rather than the unweighted error a uniform grid
   minimises.  KVQuant takes `F` from the diagonal Fisher information of the language
   model on a calibration set.  No such gradient exists inside a single attention block,
   so the bank uses the *exact* local sensitivity instead, which is the first-order term
   of <span class="attn-ref">eq:error-joint</span>:
       F[j, c] = sum_i (beta pi_ij)^2 ||v_j - y_i||^2 q_ic^2

   This is the same objective with the same role for `F`; it is a different (and, for
   one layer in isolation, sharper) estimate of it.  Under `pre_rope` the weight is
   averaged within each RoPE channel pair, which makes it invariant to the rotation it
   has to survive.  For **values** the local sensitivity is `sum_i pi_ij^2`, constant
   along the channel axis, so sensitivity weighting is provably a no-op for a per-token
   value quantizer and is not applied — a small result worth stating rather than
   hiding behind a flag.
4. **Dense-and-sparse** (`sparsity`).  A per-vector fraction of numerical outliers is
   held in full precision and *excluded from the range*, so the remaining levels are
   spent on the dense part.  Thresholds are per vector: per channel for the keys, per
   token for the values, as in the paper.  `keep_first` additionally pins the first
   token exactly — the attention sink, which carries disproportionate mass and whose
   value error is therefore not averaged away.

**Not causal.**  A per-channel scale, an outlier threshold and a codebook are all
statistics over tokens.  KVQuant computes them offline on a calibration set, which makes
them constants at serving time; the bank has no calibration set and derives them from
the sequence itself, which is a genuinely non-causal operation.  Declaring
`supports_causal = False` records that rather than hiding it behind a mask.  `kivi`
is the member of this family that is causal, and the residual window is what it pays.

Quantize–dequantize is **simulated** in the input dtype; no memory is actually saved.
`cost()` reports what the scheme would store, including the per-channel scales, the
codebook and the fp16 outliers — the three overheads that decide whether 2 bits is 2 bits.

Exact limit: `bits_k = bits_v = None`.  The RoPE round trip is then skipped entirely,
so the limit is exact and not merely accurate.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`
:   30 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `kvquant` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `kvquant` selected](index.md?compare=kvquant) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Hooper et al. (2024).** *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)<br><span class="attn-cite" data-bibkey="hooper2024kvquant"><code>hooper2024kvquant</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="hooper2024kvquant">
<pre class="attn-bib__src"><code>@inproceedings{hooper2024kvquant,
  title     = {{KVQuant}: Towards 10 Million Context Length {LLM} Inference with {KV} Cache Quantization},
  author    = {Hooper, Coleman and Kim, Sehoon and Mohammadzadeh, Hiva and Mahoney, Michael W. and Shao, Yakun Sophia and Keutzer, Kurt and Gholami, Amir},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  eprint    = {2401.18079},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/kvquant.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/kvquant.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
