# PyramidKV

<p class="attn-standfirst" markdown="span">The eviction budget is allocated across layers, not uniformly.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `evict_pyramidkv` &middot; `ab.build("evict_pyramidkv")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.PyramidKVAttention` &middot; [`attnbank/methods/compress/evict_pyramidkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_pyramidkv.py)

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

**Exact limit.** Built with `budget=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("evict_pyramidkv").limit_config
{'budget': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 3,670,016 | 268,435,456 | 0.0137× |
| `state_bytes` | 3,328 | 262,144 | 0.0127× |
| `bytes_moved` | 3,670,016 | 4,718,592 | 0.778× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    b = self._layer_budgets(n)[self.layer]
    b = max(1, min(b if b is not None else n, n))
    return {"flops": 2 * m * b * (d + dv) + self.obs * n * d,
            "state_bytes": b * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `budget`, `chi2`, `eps`, `k_realized`, `layer_budgets`, `obs`, `pooled_scores`, `support`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `budget` | `8` |
| `n_layers` | `4` |
| `layer` | `0` |
| `beta` | `4` |
| `obs` | `4` |
| `pool` | `3` |
| `pooling` | `'max'` |
| `recent` | `2` |

</div>

`budget` is the *uniform* budget the pyramid redistributes; `layer` of
`n_layers` picks this call's share of it.

`ab.describe("evict_pyramidkv")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("evict_pyramidkv", budget=8, n_layers=4)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["evict_pyramidkv"], q, k, v, causal=False,
                 configs={"evict_pyramidkv": {'budget': 8, 'n_layers': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
PyramidKVAttention(name='evict_pyramidkv', category='compress', objective='approximate', backend='reference', beta=4, budget=8, layer=0, n_layers=4, obs=4, pool=3, pooling='max', recent=2)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[budget, chi2, eps, k_realized, layer_budgets, obs, pooled_scores, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.98871, 'distributional_error': 1.98871, 'value_error': 0.0, 'tv': 0.78498}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-evict</code>, <code>prop:compress-noeviction</code></p>

SnapKV and H2O give every layer
the same cache size.  PyramidKV observes that attention concentrates as depth grows —
lower layers spread mass broadly, upper layers funnel it onto a few tokens — and so
spends the same *total* budget unevenly: the per-layer budgets form a decreasing
arithmetic sequence, wide at the bottom and narrow at the top.

```text
base   = budget - recent                      # what a uniform scheme would give
lo     = base // beta,   hi = 2 base - lo     # the ends of the sequence
b^(l)  = hi - l (hi - lo) / (L - 1) + recent
```

The mean of the sequence is `base`, so the allocation *moves* cache between layers
rather than buying more of it; `pyramid_budgets` returns the whole schedule and
`tests/test_compress_evict.py` checks the conservation.  The official implementation
uses integer floor division for the step, which loses up to one token per layer; the
exact arithmetic sequence is used here so that the conservation is a statement and not
an approximation.

A single attention call has no notion of depth, so the layer is a config key:
`ab.build("evict_pyramidkv", layer=l, n_layers=L)` is layer `l` of the stack, and a
sweep over `l` is what exhibits the pyramid.  Within a layer the selection is
SnapKV's — the last `obs` queries score the prompt tokens, a 1-D pooling of width
`pool` keeps neighbourhoods rather than isolated tokens, and the recent window is
always resident.  `pooling='max'` is the default here, which is what PyramidKV and
SnapKV's released code use; `evict_snapkv` in this package uses average pooling, so
the pair isolates the effect of the pooling operator.

**Semantics: eviction, not restriction.**  The support is chosen once, from the
observation window, and the *same* support then serves every query — including the
queries that are not in that window and whose own top-`b` is a different set.  The
context those queries would have needed is gone, which is why the method declares
`supports_causal = False`: query 0's support depends on query `m-1`, and no causal
mask can express that.  `tests/test_compress_evict.py` measures the gap against the
per-query restriction `topk` at the same budget; it is the price of deleting rather
than hiding, and <span class="attn-ref">prop:compress-noeviction</span> says no budget makes it uniformly small.

Exact limit: `budget = None` (every layer keeps everything).

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`
:   30 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `evict_pyramidkv` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `evict_pyramidkv` selected](index.md?compare=evict_pyramidkv) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Cai et al. (2024).** *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.* [arXiv:2406.02069](https://arxiv.org/abs/2406.02069)<br><span class="attn-cite" data-bibkey="cai2024pyramidkv"><code>cai2024pyramidkv</code></span>
- **Li et al. (2024).** *SnapKV: LLM Knows What You are Looking for Before Generation.* Advances in Neural Information Processing Systems (NeurIPS). [doi:10.52202/079017-0722](https://doi.org/10.52202/079017-0722)<br><span class="attn-cite" data-bibkey="li2024snapkv"><code>li2024snapkv</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="cai2024pyramidkv">
<pre class="attn-bib__src"><code>@misc{cai2024pyramidkv,
  title = {{PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling}},
  author = {Zefan Cai and Yichi Zhang and Bofei Gao and Yuliang Liu and Yucheng Li and Tianyu Liu and Keming Lu and Wayne Xiong and Yue Dong and Junjie Hu and Wen Xiao},
  year = {2024},
  howpublished = {arXiv preprint arXiv:2406.02069},
  eprint = {2406.02069},
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

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/evict_pyramidkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_pyramidkv.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
