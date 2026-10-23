# Low-precision attention

<p class="attn-standfirst" markdown="span">Simulated FP8 / INT8 logits and values.</p>

<div class="attn-facts" data-cat="exact" markdown="1">

key
:   `lowprec` &middot; `ab.build("lowprec")`

category
:   [`exact`](../taxonomy.md#exact)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.LowPrecisionAttention` &middot; [`attnbank/methods/exact/lowprec.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/lowprec.py)

monograph
:   `sec:exact-boundary`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes nothing in $y(q)=\int v\,\mathrm{d}\pi_q$. The integrand, the measure and the normaliser are the reference operator's; only the order of arithmetic changes.

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

**Exact limit.** Built with `bits=None`, `fmt=None`, `bits_p=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("lowprec").limit_config
{'bits': None, 'fmt': None, 'bits_p': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,435,456 | 268,435,456 | 1× |
| `state_bytes` | 131,072 | 262,144 | 0.5× |
| `bytes_moved` | 262,144 | 4,718,592 | 0.0556× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    b = bytes_per_el if self.bits is None and self.fmt is None else \
        max(1, (8 if self.fmt else self.bits) // 8)
    return {"flops": 2 * m * n * (d + dv), "state_bytes": n * (d + dv) * b,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * b}
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

**`reference` only, and that is the correct answer.** The method exists to expose the *intermediate* states of the evaluation, which a single fused call does not hand back.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `delta_v`, `s_hat`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `bits` | `8` |
| `fmt` | `None` |
| `bits_p` | `None` |
| `hadamard_qk` | `False` |
| `smooth_k` | `False` |
| `seed` | `0` |

</div>

`bits`: INT width for q, k, v (`None` = no quantization, the exact limit).
`fmt`: `'e4m3'` / `'e5m2'` to use FP8 instead of INT.
`bits_p`: optional INT width for the attention weights themselves.

```python
ab.describe("lowprec")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("lowprec", bits=8, hadamard_qk=False)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["lowprec"], q, k, v, causal=True,
                 configs={"lowprec": {'bits': 8, 'hadamard_qk': False}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
LowPrecisionAttention(name='lowprec', category='exact', objective='approximate', backend='reference', bits=8, bits_p=None, fmt=None, hadamard_qk=False, seed=0, smooth_k=False)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[delta_v, s_hat], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.01213, 'distributional_error': 0.0085, 'value_error': 0.0074, 'tv': 0.00273}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:exact-boundary`.  This is the method that marks *where exactness ends*.  An
FP8/INT8 kernel keeps the evaluation order of `prop:exact-monoid` but stores the
operands at a precision `u' >> u`, so it satisfies the first clause of
`def:exact-scheme` and fails the second.  By the criterion of "which object is
changed" it belongs with context compression: quantizing `q, k` replaces `g` by
`g_hat` (a distributional error) and quantizing `v` or `P` replaces `v` by
`v_hat` (a value-representation error).  Its objective is therefore `approximate`,
not `accelerate`, which is the whole point of putting it in `exact/`.

Simulation is quantize–dequantize in the kernel, per attention head and per tensor:

```text
Q_b(x) = clamp(round(x / Delta), -2^{b-1}, 2^{b-1} - 1) * Delta,  Delta = max|x| / (2^{b-1} - 1)
```

for signed INT-b, and round-to-nearest on the FP8 E4M3 / E5M2 grids.  Optional
*incoherent processing* (`hadamard=True`) multiplies `q` and `k` by a shared
random orthogonal Hadamard matrix before quantizing, which is exact in infinite
precision (`<Rq, Rk> = <q, k>`) and spreads outlier coordinates, as in
FlashAttention-3's FP8 path.  Optional *smoothing* (`smooth_k=True`) subtracts the
token mean of `k`, which shifts every logit of a query by one constant and so
leaves `pi_q` exactly invariant (SageAttention).

The quantizers themselves live in `attnbank.components.quantization`, shared with
`quantkv`: one implementation, one place where the rounding rule is argued about.

</div>

## Related methods

changes the same object — [`exact`](../taxonomy.md#exact)
:   [`flash`](flash.md), [`softmax`](softmax.md), [`splitk`](splitk.md) &nbsp;·&nbsp; [browse the bank, exact only](index.md?category=exact)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`atom`](atom.md), [`based`](based.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`hyperattention`](hyperattention.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`longformer`](longformer.md), [`lowrank_kv`](lowrank_kv.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`lsh_sample`](lsh_sample.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`skyformer`](skyformer.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `lowprec` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `lowprec` selected](index.md?compare=lowprec) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.*<br><span class="attn-cite" data-bibkey="shah2024flashattention3"><code>shah2024flashattention3</code></span>
- *SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration.*<br><span class="attn-cite" data-bibkey="zhang2025sageattention"><code>zhang2025sageattention</code></span>
- *FP8 Formats for Deep Learning.*<br><span class="attn-cite" data-bibkey="micikevicius2022fp8"><code>micikevicius2022fp8</code></span>
- *INT-FlashAttention: Enabling Flash Attention for INT8 Quantization.*<br><span class="attn-cite" data-bibkey="chen2024intflashattention"><code>chen2024intflashattention</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="shah2024flashattention3">
<pre class="attn-bib__src"><code>@inproceedings{shah2024flashattention3,
  title={{FlashAttention-3}: Fast and Accurate Attention with Asynchrony and Low-precision}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="zhang2025sageattention">
<pre class="attn-bib__src"><code>@inproceedings{zhang2025sageattention,
  title={{SageAttention}: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="micikevicius2022fp8">
<pre class="attn-bib__src"><code>@misc{micikevicius2022fp8,
  title = {{FP8 Formats for Deep Learning}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="chen2024intflashattention">
<pre class="attn-bib__src"><code>@misc{chen2024intflashattention,
  title={{INT-FlashAttention}: Enabling Flash Attention for {INT8} Quantization}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/exact/lowprec.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/lowprec.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
