# CaM

<p class="attn-standfirst" markdown="span">The token leaving the cache is merged into its neighbours instead of dropped.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `merge_cam` &middot; `ab.build("merge_cam")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.CaMAttention` &middot; [`attnbank/methods/compress/merge_cam.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/merge_cam.py)

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

**Exact limit.** Built with `budget=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("merge_cam").limit_config
{'budget': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 1,638,400 | 268,435,456 | 0.0061× |
| `state_bytes` | 1,536 | 262,144 | 0.00586× |
| `bytes_moved` | 1,835,008 | 4,718,592 | 0.389× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    b = max(1, min(self.budget or n, n))
    return {"flops": 2 * m * b * (d + dv) + n * dv,
            "state_bytes": b * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}
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

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `assign`, `budget`, `chi2`, `cluster_radius`, `eps`, `k_realized`, `merge_gate`, `merge_window`, `prominence`, `recent`, `retained_idx`, `sinks`, `sizes`, `support`, `v_bar`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `budget` | `6` |
| `sinks` | `2` |
| `merge_window` | `4` |
| `obs` | `4` |
| `mode` | `'expected'` |
| `seed` | `0` |

</div>

`budget` = sinks + recent window; `merge_window` = CaM's `merge_budget`.

```python
ab.describe("merge_cam")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("merge_cam", budget=6, sinks=2)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["merge_cam"], q, k, v, causal=False,
                 configs={"merge_cam": {'budget': 6, 'sinks': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
CaMAttention(name='merge_cam', category='compress', objective='approximate', backend='reference', budget=6, merge_window=4, mode='expected', obs=4, seed=0, sinks=2)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[assign, budget, chi2, cluster_radius, eps, k_realized, merge_gate, merge_window, prominence, recent, retained_idx, sinks, sizes, support, v_bar], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 3.35363, 'distributional_error': 2.78493, 'value_error': 1.93814, 'tv': 0.90913}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:compress-evict`, `sec:compress-merge`, `prop:compress-noeviction`.  CaM sits
on a StreamingLLM cache — `sinks` attention sinks plus a recent window of
`recent = budget - sinks` — and asks what to do with the token that falls off the
back of the window.  Eviction throws it away.  CaM writes its value into the next
`merge_window` entries first:

```text
rho_j   = pi(j) / mean_{j < i <= j + recent} pi(i)        (clamped to [0, 1])
g_j     ~ Bernoulli(rho_j)
v_hat_i += g_j v_hat_j / merge_window,   j < i <= j + merge_window
```

`rho_j` is the *prominence* of the departing token relative to the window it is
leaving, so a token nobody looked at is usually dropped and a heavy hitter is usually
carried.  Because `v_hat_j` on the right is the already-merged value, a token that is
carried and then itself evicted cascades further forward; running the loop in
increasing `j` on a mutable value field reproduces that exactly, as the released code
does by mutating `value_states` in place.

**What CaM does and does not preserve.**  Keys are never merged, so the measure is a
plain renormalised restriction: `extras['eps']` is the mass CaM deletes, and setting
`merge_window = 0` — the same method with the repair switched off — deletes it
bit for bit and leaves `v` alone.  What CaM repairs is
the *value field*: `v_hat != v`, and the evicted values survive inside the retained
ones.  This is therefore **not** the pushforward of `merge_kv`, where the mass itself
is carried and `pi_hat` picks up the `log |C_i|` gauge bias; it is a restriction
with a repaired `v_hat`, and `eq:error_decomposition` splits it that way — CaM
leaves the distributional term exactly where eviction left it and attacks the value
term alone.  Saying so is the point: the two repairs are different objects, and
`tests/test_compress_evict.py` measures both.

**Semantics: eviction, with a destructive twist.**  The merge pass is the prefill: the
whole stream is ingested, the cache is reduced to `budget`, and *every* query is then
served from that one reduced, rewritten cache — which is why the method declares
`supports_causal = False`.  The twist is that the damage is worse than hiding in a
way no member of `restrict/` can imitate: `sink_window` leaves `v` untouched, so
a later pass can still read the original values, whereas CaM has overwritten them.
The cache cannot be un-merged.

`mode` defaults to `'expected'`, which uses `rho_j` itself as the gate — the
conditional expectation of the released code's Bernoulli draw.  The bank's conformance
suite requires a method to agree between fp32 and fp64 and between CPU and CUDA, and a
coin flip on a threshold does not; `mode='sample'` reproduces the released sampler
through a seeded `torch.Generator` for anyone who wants the variance.

Exact limit: `budget = None` (nothing leaves the window, nothing is merged).

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`based`](based.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`lsh_sample`](lsh_sample.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`skyformer`](skyformer.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `merge_cam` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `merge_cam` selected](index.md?compare=merge_cam) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *CaM: Cache Merging for Memory-efficient LLMs Inference.*<br><span class="attn-cite" data-bibkey="zhang2024cam"><code>zhang2024cam</code></span>
- *Efficient Streaming Language Models with Attention Sinks.*<br><span class="attn-cite" data-bibkey="xiao2024efficient"><code>xiao2024efficient</code></span>
- *Token Merging: Your ViT But Faster.*<br><span class="attn-cite" data-bibkey="bolya2023token"><code>bolya2023token</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="zhang2024cam">
<pre class="attn-bib__src"><code>@inproceedings{zhang2024cam,
  title = {{CaM}: Cache Merging for Memory-efficient {LLMs} Inference}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="xiao2024efficient">
<pre class="attn-bib__src"><code>@inproceedings{xiao2024efficient,
  title = {{Efficient Streaming Language Models with Attention Sinks}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="bolya2023token">
<pre class="attn-bib__src"><code>@inproceedings{bolya2023token,
  title = {{Token Merging: Your ViT But Faster}}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/merge_cam.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/merge_cam.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
