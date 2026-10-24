# Routing Transformer

<p class="attn-standfirst" markdown="span">The support is a learned cluster.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `routing` &middot; `ab.build("routing")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.RoutingAttention` &middot; [`attnbank/methods/restrict/routing.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/routing.py)

monograph
:   `sec:sparse`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="yes">yes</span> | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="yes">yes</span> | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

</div>

## Correctness anchor

**Exact limit.** Built with `n_clusters=1` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("routing").limit_config
{'n_clusters': 1}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 68,419,584 | 268,435,456 | 0.255× |
| `state_bytes` | 262,656 | 262,144 | 1× |
| `bytes_moved` | 67,764,736 | 4,718,592 | 14.4× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    c = max(1, self.n_clusters)
    kept = max(1.0, n / c) + 2 * self.w + 1
    return {"flops": 2 * (m + n) * d * c + 2 * m * kept * (d + dv),
            "state_bytes": (n * (d + dv) + c * d) * bytes_per_el,
            "bytes_moved": (m * d + n * d + c * d + m * kept * (d + dv) + m * dv)
                           * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `cluster_k`, `cluster_q`, `eps`, `k_realized`, `n_clusters`, `support`, `trained_centroids`, `w`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_clusters` | `4` |
| `w` | `0` |
| `seed` | `0` |
| `centroids` | `None` |

</div>

`w` adds a sliding window of radius `w`; `centroids` is `[c, d]`.

`ab.describe("routing")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("routing", n_clusters=4, w=0)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["routing"], q, k, v, causal=True,
                 configs={"routing": {'n_clusters': 4, 'w': 0}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
RoutingAttention(name='routing', category='restrict', objective='approximate', backend='reference', centroids=None, n_clusters=4, seed=0, w=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, cluster_k, cluster_q, eps, k_realized, n_clusters, support, trained_centroids, w], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.0325, 'distributional_error': 2.0325, 'value_error': 0.0, 'tv': 0.66957}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:sparse-content</code> (&quot;Content-based supports&quot;)</p>

Where `reformer` partitions the
sphere at random and `topk` reads the true scores, routing places `c` centroids in
the query/key space and lets a query attend to the keys routed to *its* centroid:

```text
r(x) = argmax_i  <x / ||x||, mu_i>,    S_q = { j : r(k_j) = r(q) }  (and the
                                               query's own position),
```

so the support costs `O((m + n) c d)` to compute and the partition is shared across
queries rather than recomputed per query.  Given `S_q` the operator is again the
renormalized restriction of <span class="attn-ref">prop:sparse-identity</span>.  Optionally `w > 0` adds a
sliding window, because the published model combines content routing with local
attention and attributes part of its result to the combination.

**The centroids are trained.** In the paper they are parameters updated by online
k-means: an exponential moving average of the queries and keys assigned to each
centroid, carried across minibatches, with the centroids re-normalized to the sphere.
They are never fitted to the sequence being attended to, and they could not be —
doing so inside a causal decode would let the support of an early query depend on
later keys.  This class therefore declares `trainable = True`, and its **untrained
behaviour is the paper's initialization**: `c` directions drawn from a seeded
standard Gaussian and normalized.  That is a random partition of the sphere into
Voronoi cones, which is a *lower* bound on what routing buys — it is roughly
`reformer` with one round and cones that are not axis-paired, and it should be read
as the floor the training is supposed to lift, not as the method's accuracy.  Pass
`centroids=` a `[c, d]` tensor to supply trained ones; the forward pass is the
same either way.

Load balance.  The published implementation takes, per cluster, the `n/c` keys with
the largest centroid dot product, so that every cluster has the same size and the
attention can be batched as a dense `c x (n/c)` tensor.  Nearest-centroid membership
— what is implemented here — is the routing rule that device approximates; it leaves
clusters uneven, which is visible in `extras["k_realized"]` and is part of what the
selector costs.

Exact limit: `n_clusters = 1` routes everything to one centroid, so `S_q` is the
whole visible context and `pi_hat = pi`.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`
:   31 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `routing` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `routing` selected](index.md?compare=routing) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Roy et al. (2021).** *Efficient Content-Based Sparse Attention with Routing Transformers.* Transactions of the Association for Computational Linguistics. [arXiv:2003.05997](https://arxiv.org/abs/2003.05997) [doi:10.1162/tacl_a_00353](https://doi.org/10.1162/tacl_a_00353)<br><span class="attn-cite" data-bibkey="roy2021routing"><code>roy2021routing</code></span>
- **Vyas et al. (2020).** *Fast Transformers with Clustered Attention.* Advances in Neural Information Processing Systems 33 (NeurIPS). [arXiv:2007.04825](https://arxiv.org/abs/2007.04825)<br><span class="attn-cite" data-bibkey="vyas2020fast"><code>vyas2020fast</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="roy2021routing">
<pre class="attn-bib__src"><code>@article{roy2021routing,
  title = {Efficient Content-Based Sparse Attention with Routing Transformers},
  author = {Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  year = {2021},
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {9},
  pages = {53--68},
  doi = {10.1162/tacl_a_00353},
  eprint = {2003.05997},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="vyas2020fast">
<pre class="attn-bib__src"><code>@inproceedings{vyas2020fast,
  title         = {Fast Transformers with Clustered Attention},
  author        = {Vyas, Apoorv and Katharopoulos, Angelos and Fleuret, Fran{\c{c}}ois},
  booktitle     = {Advances in Neural Information Processing Systems 33 (NeurIPS)},
  year          = {2020},
  eprint        = {2007.04825},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/routing.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/routing.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
