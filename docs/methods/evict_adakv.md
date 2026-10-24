# Ada-KV

<p class="attn-standfirst" markdown="span">One eviction budget for the whole layer, allocated across heads.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `evict_adakv` &middot; `ab.build("evict_adakv")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.AdaKVAttention` &middot; [`attnbank/methods/compress/evict_adakv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_adakv.py)

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
| `head_independent` | <span class="attn-flag" data-v="no">no</span> | it shares $K$/$V$ across query heads, so heads are not independent. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `budget=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("evict_adakv").limit_config
{'budget': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,359,296 | 268,435,456 | 0.00879× |
| `state_bytes` | 2,048 | 262,144 | 0.00781× |
| `bytes_moved` | 2,359,296 | 4,718,592 | 0.5× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    b = max(1, min(self.budget or n, n))
    return {"flops": 2 * m * b * (d + dv) + self.obs * n * d,
            "state_bytes": b * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `adaptive_budget`, `budget`, `chi2`, `eps`, `head_budget`, `k_realized`, `obs`, `pooled_scores`, `support`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `budget` | `8` |
| `floor_alpha` | `0.2` |
| `obs` | `4` |
| `pool` | `3` |
| `pooling` | `'max'` |
| `recent` | `2` |

</div>

`budget` is the per-head *average*; `floor_alpha = 1` recovers SnapKV.

`ab.describe("evict_adakv")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("evict_adakv", budget=8, floor_alpha=0.2)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["evict_adakv"], q, k, v, causal=False,
                 configs={"evict_adakv": {'budget': 8, 'floor_alpha': 0.2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
AdaKVAttention(name='evict_adakv', category='compress', objective='approximate', backend='reference', budget=8, floor_alpha=0.2, obs=4, pool=3, pooling='max', recent=2)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[adaptive_budget, budget, chi2, eps, head_budget, k_realized, obs, pooled_scores, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.51478, 'distributional_error': 2.51478, 'value_error': 0.0, 'tv': 0.87563}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-evict</code>, <code>prop:compress-noeviction</code></p>

SnapKV, H2O and PyramidKV give
every head of a layer the same cache size.  Ada-KV observes that heads differ enormously
in how concentrated their attention is — a head whose mass sits on three tokens wastes a
budget of sixty, while a diffuse head is crippled by it — and that the top-`k`
eviction rule is itself the minimiser of an upper bound on the output perturbation.
Allocating the layer's *total* budget `H b` so as to minimise that bound jointly over
heads turns out to be a single global selection:

```text
take the `H b` largest observation scores over the concatenation of all heads,
let `f_h` be how many of them belong to head `h`, and give head `h` a cache
of size `f_h`.
```

The released code then blends that with the uniform allocation through a safeguard
`alpha`, `b_h = round((1 - alpha) f_h + alpha b)`, which stops a head from being
starved to the point where the bound's assumptions stop holding.  `alpha = 1` is
uniform allocation and recovers SnapKV head for head; `alpha = 0` is the raw
argmin.  Either way `sum_h b_h = H b` up to rounding: the budget is *moved* between
heads, never increased, and `tests/test_compress_evict.py` checks it.

**Heads are coupled, and that is the mechanism.**  The method declares
`head_independent = False`: perturbing head 1's keys moves head 2's cache size,
because they are drawing on one pool.  This is the same kind of honest declaration
`gqa` and `mla` make, for a different reason — they share the stored K/V, Ada-KV
shares the *budget*.

**Semantics: eviction, not restriction.**  As in SnapKV the support is fixed once from
the observation window and then serves every query, so a query outside that window is
answered from a cache that was pruned without knowing what it would ask.  Hence
`supports_causal = False`, and hence <span class="attn-ref">prop:compress-noeviction</span>: adaptive allocation
buys a better *average*, not a uniform guarantee.

Exact limit: `budget = None` (nothing is evicted).

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`
:   30 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `evict_adakv` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `evict_adakv` selected](index.md?compare=evict_adakv) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Feng et al. (2025).** *Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2407.11550](https://arxiv.org/abs/2407.11550)<br><span class="attn-cite" data-bibkey="feng2025adakv"><code>feng2025adakv</code></span>
- **Li et al. (2024).** *SnapKV: LLM Knows What You are Looking for Before Generation.* Advances in Neural Information Processing Systems (NeurIPS). [doi:10.52202/079017-0722](https://doi.org/10.52202/079017-0722)<br><span class="attn-cite" data-bibkey="li2024snapkv"><code>li2024snapkv</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="feng2025adakv">
<pre class="attn-bib__src"><code>@inproceedings{feng2025adakv,
  title = {{Ada-KV}: Optimizing {KV} Cache Eviction by Adaptive Budget Allocation for Efficient {LLM} Inference},
  author = {Yuan Feng and Junlin Lv and Yukun Cao and Xike Xie and S. Kevin Zhou},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2025},
  eprint = {2407.11550},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="li2024snapkv">
<pre class="attn-bib__src"><code>@inproceedings{li2024snapkv,
  title={{SnapKV}: {LLM} Knows What You are Looking for Before Generation},
  author={Li, Yuhong and Huang, Yingbing and Yang, Bowen and Venkitesh, Bharat and Locatelli, Acyr and Ye, Hanchen and Cai, Tianle and Lewis, Patrick and Chen, Deming},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  doi={10.52202/079017-0722}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/evict_adakv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_adakv.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
