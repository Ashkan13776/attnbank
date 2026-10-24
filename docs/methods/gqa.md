# Grouped-query attention

<p class="attn-standfirst" markdown="span">K and V shared across groups of query heads.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `gqa` &middot; `ab.build("gqa")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.GQAAttention` &middot; [`attnbank/methods/compress/gqa.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/gqa.py)

monograph
:   `sec:compress`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="yes">yes</span> | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="no">no</span> | it shares $K$/$V$ across query heads, so heads are not independent. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `group_size=1` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("gqa").limit_config
{'group_size': 1}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,435,456 | 268,435,456 | 1× |
| `state_bytes` | 131,072 | 262,144 | 0.5× |
| `bytes_moved` | 393,216 | 4,718,592 | 0.0833× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    g = max(1, self.group_size)
    return {"flops": 2 * m * n * (d + dv),
            "state_bytes": n * (d + dv) * bytes_per_el / g,
            "bytes_moved": m * (d + dv) * bytes_per_el + n * (d + dv) * bytes_per_el / g}
```

## Backends

<div class="attn-tab" markdown="1">

| backend | what it is |
|---|---|
| `reference` | plain PyTorch — the correctness anchor and the default |
| `sdpa` | `torch.nn.functional.scaled_dot_product_attention` — on CUDA, FlashAttention-2 or the memory-efficient kernel |

</div>

Each non-`reference` entry is a claim that the path is **equal** to the reference, and `tests/test_conformance.py::test_backend_equivalence` checks it on five shapes, both causal settings, float32 and float64. See [Backends](../backends.md).

Diagnostics returned in `extras`: `group_size`, `n_kv_heads`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `group_size` | `2` |
| `sdpa_kernel` | `None` |

</div>

`ab.describe("gqa")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("gqa", group_size=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = modify: this is a different operator and fidelity to softmax
# is not its goal, so there is no error to quote. What it *is* doing shows
# up in its own diagnostics.
print("row_stochastic =", attn.row_stochastic, "| first row sums:",
      [round(x, 4) for x in out.weights.sum(-1)[0, 0, :4].tolist()])
print("extras:", sorted(out.extras))
```

<div class="attn-out" markdown="1">

```text
GQAAttention(name='gqa', category='compress', objective='modify', backend='reference', group_size=2, sdpa_kernel=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[group_size, n_kv_heads], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['group_size', 'n_kv_heads']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-arch</code>, <code>sec:exact-roofline</code></p>

Query heads are partitioned into groups
of `group_size`; the heads of a group share one key–value pair, here obtained by
mean-pooling the group's `K` and `V` (the conversion recipe of Ainslie et al.,
2023).  `group_size = 1` is multi-head attention, `group_size = h` is multi-query.

This is an **operator modification** (<span class="attn-ref">tab:comparative</span>: "MQA / GQA / MLA ... mod."),
trained as such; measuring it against a multi-head reference answers a conversion
question, not an approximation question.  Its purpose is the roofline: by
<span class="attn-ref">eq:exact-decode-intensity</span> the decode arithmetic intensity is `(16/b)(h/h_kv)`, so
sharing KV is one of the very few interventions that *raises* it.  `cost()` divides
`state_bytes` and the KV traffic by `group_size` accordingly.

**Heads are deliberately coupled**, so `head_independent = False`: perturbing head
`h` changes every head in its group and nothing outside it.  `tests/test_compress.py`
checks exactly that boundary.

Exact limit: `group_size = 1`.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `modify`
:   27 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `gqa` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `gqa` selected](index.md?compare=gqa) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Ainslie et al. (2023).** *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP). [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)<br><span class="attn-cite" data-bibkey="ainslie2023gqa"><code>ainslie2023gqa</code></span>
- **Shazeer (2019).** *Fast Transformer Decoding: One Write-Head is All You Need.* [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)<br><span class="attn-cite" data-bibkey="shazeer2019fast"><code>shazeer2019fast</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="ainslie2023gqa">
<pre class="attn-bib__src"><code>@inproceedings{ainslie2023gqa,
  title = {{GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints}},
  author = {Joshua Ainslie and James Lee-Thorp and Michiel de Jong and Yury Zemlyanskiy and Federico Lebrón and Sumit Sanghai},
  year = {2023},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  eprint = {2305.13245},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="shazeer2019fast">
<pre class="attn-bib__src"><code>@misc{shazeer2019fast,
  title = {{Fast Transformer Decoding: One Write-Head is All You Need}},
  author = {Noam Shazeer},
  year = {2019},
  howpublished = {arXiv preprint arXiv:1911.02150},
  eprint = {1911.02150},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/gqa.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/gqa.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
