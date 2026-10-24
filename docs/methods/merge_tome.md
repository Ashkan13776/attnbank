# ToMe

<p class="attn-standfirst" markdown="span">Bipartite soft matching — the pushforward whose partition is *matched*.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `merge_tome` &middot; `ab.build("merge_tome")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.ToMeAttention` &middot; [`attnbank/methods/compress/merge_tome.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/merge_tome.py)

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

**Exact limit.** Built with `n_clusters=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("merge_tome").limit_config
{'n_clusters': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 35,127,296 | 268,435,456 | 0.131× |
| `state_bytes` | 1,560 | 262,144 | 0.00595× |
| `bytes_moved` | 263,680 | 4,718,592 | 0.0559× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = max(1, min(self.n_clusters or n, n))
    return {"flops": 2 * m * r * (d + dv) + n * n * d // 2,   # + the matching
            "state_bytes": r * (d + dv) * bytes_per_el + r * 4,
            "bytes_moved": (m * d + r * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** The compressed context is built by this method, and the measure it reports lives on the **original** $n$ positions — a kernel run over the compressed cache would return neither.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `assign`, `cluster_radius`, `k_bar`, `sizes`, `v_bar`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_clusters` | `6` |

</div>

`n_clusters`: how many tokens survive; `None` merges nothing.

`ab.describe("merge_tome")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("merge_tome", n_clusters=6)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["merge_tome"], q, k, v, causal=False,
                 configs={"merge_tome": {'n_clusters': 6}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
ToMeAttention(name='merge_tome', category='compress', objective='approximate', backend='reference', n_clusters=6)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[assign, cluster_radius, k_bar, sizes, v_bar], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.70902, 'distributional_error': 0.70902, 'value_error': 0.0, 'tv': 0.33552}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-merge</code>, <code>prop:compress-merge</code>, <code>prop:error-merge</code></p>

`merge_kv`
already implements the pushforward `mu_hat = T_# mu` and ToMe's proportional
attention; what it does not implement is ToMe's actual choice of `T`.  Contiguous
blocks ignore similarity and Lloyd's algorithm is iterative and order-dependent; ToMe's
point was that a *single* round of bipartite matching is almost as good and costs one
matrix product:

1. split the current tokens alternately into `A` (even slots) and `B` (odd slots);
2. draw one edge from each `a in A` to its most cosine-similar `b in B`, using the
   attention **keys** as the metric;
3. keep the `r` heaviest edges;
4. merge each matched `a` into its `b` by averaging, and concatenate back.

A round can remove at most half the tokens, so reaching a target of `n_clusters` takes
`ceil(log2(n / n_clusters))` rounds; ToMe spends them one per transformer layer, this
file spends them inside one call, which is the same map.  The matching is greedy and
deterministic — no seed, unlike `merge_kv(mode='kmeans')`.

Mass.  Exactly as in `merge_kv`: the representatives `k_hat_i, v_hat_i` are the
within-cluster means and are *lifted back* to the original `n` positions, so

```text
pi~_j ∝ e^{beta <q, k_hat_{T(j)}>},   sum_j pi~_j = 1,   pi~ @ v~ = y_hat,
```

and a cluster of size `|C_i|` contributes `|C_i|` identical atoms — ToMe's
`+ log s` proportional-attention term, appearing automatically rather than by hand.
Dropping it would replace `T_# mu` by counting measure on the compressed domain.
`extras` carries `assign`, `sizes` and `cluster_radius` so that
`metrics.merge_bound` applies, and `v_bar` so it can be called directly.

One deviation, stated rather than hidden.  ToMe merges tokens of the residual stream,
so a single partition is shared by every head.  Here each `(batch, head)` matches on
its own keys, which keeps the head independence the conformance suite checks (DESIGN.md
Sec. 4, check 5) and makes the method comparable with the rest of `compress/`;
`assign` is therefore `[B, H, n]` and `merge_bound` is applied per slice.

Not causal: a cluster spans positions on both sides of a query.

Exact limit: `n_clusters = None` (no edge is kept; every token is its own cluster).

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`
:   30 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `merge_tome` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `merge_tome` selected](index.md?compare=merge_tome) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Bolya et al. (2023).** *Token Merging: Your ViT But Faster.* International Conference on Learning Representations (ICLR). [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)<br><span class="attn-cite" data-bibkey="bolya2023token"><code>bolya2023token</code></span>
- **Wang et al. (2024).** *Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks.* [arXiv:2407.08454](https://arxiv.org/abs/2407.08454)<br><span class="attn-cite" data-bibkey="wang2024model"><code>wang2024model</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="bolya2023token">
<pre class="attn-bib__src"><code>@inproceedings{bolya2023token,
  title = {{Token Merging: Your ViT But Faster}},
  author = {Daniel Bolya and Cheng-Yang Fu and Xiaoliang Dai and Peizhao Zhang and Christoph Feichtenhofer and Judy Hoffman},
  year = {2023},
  booktitle = {International Conference on Learning Representations (ICLR)},
  eprint = {2210.09461},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="wang2024model">
<pre class="attn-bib__src"><code>@misc{wang2024model,
  title = {{Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks}},
  author = {Zheng Wang and Boxiao Jin and Zhongzhi Yu and Minjia Zhang},
  year = {2024},
  howpublished = {arXiv preprint arXiv:2407.08454},
  eprint = {2407.08454},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/merge_tome.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/merge_tome.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
