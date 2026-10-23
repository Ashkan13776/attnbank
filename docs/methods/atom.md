# Atom

<p class="attn-standfirst" markdown="span">Mixed-precision quantization with outlier **channel reordering**.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `atom` &middot; `ab.build("atom")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.AtomAttention` &middot; [`attnbank/methods/compress/atom.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/atom.py)

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

**Exact limit.** Built with `bits_k=None`, `bits_v=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("atom").limit_config
{'bits_k': None, 'bits_v': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,566,528 | 268,435,456 | 1× |
| `state_bytes` | 82,944 | 262,144 | 0.316× |
| `bytes_moved` | 345,088 | 4,718,592 | 0.0731× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    """Low-bit dense channels, INT8 outlier channels, and one scale pair per group."""
    bo = (self.bits_outlier or 8 * bytes_per_el) / 8
    bk = (self.bits_k or 8 * bytes_per_el) / 8
    bv = (self.bits_v or 8 * bytes_per_el) / 8
    ok, ov = min(self.outliers, d), min(self.outliers, dv)
    g_k, g_v = self.group or max(1, d - ok), self.group or max(1, dv - ov)
    groups = -(-(d - ok) // g_k) + -(-(dv - ov) // g_v) + 2
    state = n * ((d - ok) * bk + ok * bo + (dv - ov) * bv + ov * bo
                 + 2 * groups * bytes_per_el)
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

Diagnostics returned in `extras`: `delta_k`, `delta_v`, `outlier_channels_k`, `outlier_channels_v`, `s`, `s_hat`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `bits_k` | `4` |
| `bits_v` | `4` |
| `bits_outlier` | `8` |
| `outliers` | `1` |
| `group` | `None` |

</div>

```python
ab.describe("atom")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("atom", bits_k=4, bits_v=4)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["atom"], q, k, v, causal=False,
                 configs={"atom": {'bits_k': 4, 'bits_v': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
AtomAttention(name='atom', category='compress', objective='approximate', backend='reference', bits_k=4, bits_outlier=8, bits_v=4, group=None, outliers=1)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[delta_k, delta_v, outlier_channels_k, outlier_channels_v, s, s_hat], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.10929, 'distributional_error': 0.06819, 'value_error': 0.08639, 'tv': 0.02788}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:compress-quant`.  Atom is the third answer in this family to the same fact —
that a few channels of the key/value activations carry far more energy than the rest —
and it is the one that changes neither the axis (`kivi`) nor the datatype
(`kvquant`) but the **bit width per channel**:

* reorder the channels by energy, `sum_j x_jc^2`, so that the outliers are contiguous
  at the end of the matrix;
* keep the `outliers` highest-energy channels at `bits_outlier` bits;
* quantize the rest asymmetrically in contiguous groups of `group` channels *within
  each token*, which is what the paper calls fine-grained group quantization;
* undo the permutation.

**What the reordering is for, and what it is not for.**  The paper is explicit: Atom
"reorders the scattered outlier channels of activations to the end of the matrix with
the goal of enabling the efficient implementation of mixed-precision within a single
kernel".  The permutation is a *layout* device; the accuracy comes from the mixed
precision it makes implementable.  Numerically the permutation is exactly the identity
when `group = None` — one scale per token either way — and merely *different*, not
reliably better, when `group` is finite: sorting by channel energy can put two
outliers in the same group and widen it.  The test file asserts both of those rather
than the flattering version.  What does hold, and holds strongly, is that giving the
top channels 8 bits beats spending a *larger* uniform budget on every channel.

**What is verbatim and what is transplanted.**  Section 4.4 of the paper quantizes the
KV cache with *asymmetric low-bit quantization within each attention head* and nothing
else — that is `outliers = 0, group = None`, and it coincides exactly with
`quantkv(axis='token')`.  The reordered mixed precision of Section 4.2 ("we reorder
the activations and weights based on the square sum of each activation channel.  After
reordering, we select 128 channels with the highest square sum values as outliers and
keep them in INT8") is applied there to the linear layers' activations, not to the
cache.  This file applies it to the cache, because the cache *is* an activation matrix
with exactly the outlier-channel structure KIVI and KVQuant attack by other means, and
because the comparison is the reason to have all three in one bank.  The docstring says
so rather than letting the name imply otherwise.

Note also what Atom argues and what the monograph argues.  Section 4.4: "Due to the
normalization of Softmax, the quantization error of the K cache has less influence on
the output."  `prop:compress-sensitivity` says the opposite — the softmax
*amplifies* a key error by `e^{osc(delta l)} - 1` while averaging a value error —
and `bits_k` / `bits_v` are split here precisely so that study S4 can settle it on
the same scheme the claim is made about.

**Not causal.**  The channel permutation is a statistic over tokens.  Atom computes it
offline from calibration data, so at serving time it is a constant; the bank has no
calibration set and derives it from the sequence, which is not causal.  The grouping
*within* a token is, which is why `quantkv(axis='token')` — Atom's Section 4.4
scheme — supports `causal=True` and this does not.

Quantize–dequantize is **simulated** in the input dtype; no memory is saved here.

Exact limit: `bits_k = bits_v = None` — which switches off the outlier channels too,
so that `None` means the same thing here as everywhere else in the family.  The
permutation and its inverse are exact, so the identity quantizer leaves `k` and `v`
bit-for-bit unchanged.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`based`](based.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`lsh_sample`](lsh_sample.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`skyformer`](skyformer.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `atom` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `atom` selected](index.md?compare=atom) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="zhao2024atom"><code>zhao2024atom</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="zhao2024atom">
<pre class="attn-bib__src"><code>@inproceedings{zhao2024atom,
  title     = {{Atom}: Low-Bit Quantization for Efficient and Accurate {LLM} Serving}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/atom.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/atom.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
