# Token merging

<p class="attn-standfirst" markdown="span">A pushforward compression of the context measure.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `merge_kv` &middot; `ab.build("merge_kv")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.MergeKVAttention` &middot; [`attnbank/methods/compress/merge_kv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/merge_kv.py)

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
>>> ab.describe("merge_kv").limit_config
{'n_clusters': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 1,835,008 | 268,435,456 | 0.00684× |
| `state_bytes` | 1,560 | 262,144 | 0.00595× |
| `bytes_moved` | 263,680 | 4,718,592 | 0.0559× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = min(self.n_clusters or n, n)
    return {"flops": 2 * m * r * (d + dv) + 2 * n * (d + dv),
            "state_bytes": r * (d + dv) * bytes_per_el + r * 4,      # + multiplicities
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
| `mode` | `'contiguous'` |
| `seed` | `0` |

</div>

`mode`: `'contiguous'` (consecutive tokens) or `'kmeans'` (on keys).

`ab.describe("merge_kv")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("merge_kv", n_clusters=6, mode='contiguous')
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["merge_kv"], q, k, v, causal=False,
                 configs={"merge_kv": {'n_clusters': 6}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
MergeKVAttention(name='merge_kv', category='compress', objective='approximate', backend='reference', mode='contiguous', n_clusters=6, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[assign, cluster_radius, k_bar, sizes, v_bar], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.72675, 'distributional_error': 0.72675, 'value_error': 0.0, 'tv': 0.34538}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-merge</code>, <code>prop:compress-merge</code>, <code>prop:error-merge</code></p>

A partition
`C_1, ..., C_r` of the positions and the map `T: x -> cluster(x)` define

```text
mu_hat = T_# mu,    mu_hat({i}) = mu(C_i),
y_hat(q) = sum_i mu(C_i) e^{beta <q, k_hat_i>} v_hat_i / sum_i mu(C_i) e^{beta <q, k_hat_i>},
```

i.e. ToMe's *proportional attention*: the merged token carries the `mu`-mass of its
preimage, which in log form is the `log |C_i|` bias.  Without it the pushforward is
replaced by counting measure on the compressed domain and large clusters are
systematically under-weighted.

Expressing it on the original positions.  Lift the representatives back,
`k~_j = k_hat_{T(j)}` and `v~_j = v_hat_{T(j)}`; then
`pi~_j ∝ e^{beta <q, k~_j>}` on the *original* `n` positions satisfies
`pi~ @ v~ = y_hat` and `sum_j pi~_j = 1`.  The multiplicity appears automatically,
because a cluster of size `|C_i|` contributes `|C_i|` identical atoms.  That is the
sense in which the pushforward is mass-preserving, and it is what
`tests/test_compress.py` checks.

Representatives.  `k_hat_i` and `v_hat_i` are the within-cluster means.  The paper
notes this is *not* the right key representative even for one query: the exact
contribution is `mu(C_i) exp(Lambda_i(q))` with `Lambda_i` the log-MGF of the
cluster's keys, and by Jensen mean-key merging under-weights every cluster, to leading
order by `exp(-beta^2 q^T Sigma_i q / 2)`.  The within-cluster covariance is returned
in `extras` so that correction can be studied.

Not causal: clusters span positions on both sides of a query.

Exact limit: `n_clusters = None` (singleton clusters).

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`
:   30 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `merge_kv` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `merge_kv` selected](index.md?compare=merge_kv) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Bolya et al. (2023).** *Token Merging: Your ViT But Faster.* International Conference on Learning Representations (ICLR). [arXiv:2210.09461](https://arxiv.org/abs/2210.09461)<br><span class="attn-cite" data-bibkey="bolya2023token"><code>bolya2023token</code></span>
- **Wang et al. (2024).** *Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks.* [arXiv:2407.08454](https://arxiv.org/abs/2407.08454)<br><span class="attn-cite" data-bibkey="wang2024model"><code>wang2024model</code></span>
- **Nawrot et al. (2024).** *Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2403.09636](https://arxiv.org/abs/2403.09636)<br><span class="attn-cite" data-bibkey="nawrot2024dynamic"><code>nawrot2024dynamic</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
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
<div class="attn-bib__entry" data-bibkey="nawrot2024dynamic">
<pre class="attn-bib__src"><code>@inproceedings{nawrot2024dynamic,
  title = {{Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference}},
  author = {Piotr Nawrot and Adrian Łańcucki and Marcin Chochowski and David Tarjan and Edoardo M. Ponti},
  year = {2024},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  eprint = {2403.09636},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/merge_kv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/merge_kv.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
