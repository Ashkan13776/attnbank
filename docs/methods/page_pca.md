# Page-local key PCA

<p class="attn-standfirst" markdown="span">The key-only spectral page summary.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `page_pca` &middot; `ab.build("page_pca")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.PagePCAAttention` &middot; [`attnbank/methods/restrict/page_pca.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/page_pca.py)

monograph
:   `sec:sparse`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

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

**Exact limit.** Built with `n_pages=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("page_pca").limit_config
{'n_pages': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 106,954,752 | 268,435,456 | 0.398× |
| `state_bytes` | 364,544 | 262,144 | 1.39× |
| `bytes_moved` | 2,461,696 | 4,718,592 | 0.522× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    nb = (n + self.page - 1) // self.page
    r = min(self.rank, d)
    kept = min(self.n_pages or nb, nb) * self.page
    summary_el = nb * (d + d * r + self.page * r)      # mu, B, R per page
    return {"flops": 2 * m * nb * (d + d * r + self.page * r)
                     + 2 * m * kept * (d + dv),
            "state_bytes": (n * (d + dv) + summary_el) * bytes_per_el,
            "bytes_moved": (m * d + summary_el + m * kept * (d + dv) + m * dv)
                           * bytes_per_el}
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

Diagnostics returned in `extras`: `chi2`, `eps`, `eta`, `k_realized`, `n_pages`, `page_scores`, `page_size`, `pages_selected`, `rank`, `rank_used`, `support`.

## Configuration

This method takes no configuration of its own — only the base class's `scale` and `backend`.

```python
ab.describe("page_pca")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("page_pca", page=4, n_pages=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["page_pca"], q, k, v, causal=True,
                 configs={"page_pca": {'page': 4, 'n_pages': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
PagePCAAttention(name='page_pca', category='restrict', objective='approximate', backend='reference', eps=1e-12, n_pages=2, page=4, rank=2)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, eta, k_realized, n_pages, page_scores, page_size, pages_selected, rank, rank_used, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.38114, 'distributional_error': 1.38114, 'value_error': 0.0, 'tv': 0.49796}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:sparse-query` ("Blocks and pages").  The same page selector as `oval` — keys
are paged, each page stores `(mu, B, R = X B)` with `B` orthonormal of rank `r`,
and a page is ranked by the estimated log-mass

```text
l_B(q) = beta mu^T q + lse(beta R (B^T q)),
```

the top `n_pages` complete pages being fetched and attended exactly.  The only
difference is the basis: here `B` holds the leading eigenvectors of `X^T X`, the
directions along which the page's *keys* vary most.  Under isotropic queries that is
exactly the subspace minimizing the expected squared logit residual
`E_q ||(I - P_B) X q||^2`, so it is the right answer to the question "which `r`
directions reconstruct this page's logits best" — which is the question LOCKS and
Loki ask.

It is in the bank as the thing `oval` is argued against: `oval` claims that logit
fidelity is the wrong target because a logit error only matters when it moves mass
between tokens with different values, and this file is the `eta = 0` endpoint of
that argument made separately runnable.  The two are identical in stored size, in
decode-time scoring cost, and in every part of the pipeline except the eigenproblem,
which is the comparison the claim needs;  `ab.build("oval", eta=0.0)` is this method
and `tests/test_restrict_learned.py` asserts it.

Loki applies one *global* PCA basis to the whole key cache and selects tokens; LOCKS
fits a basis per page and selects pages.  This file is the per-page form, because that
is what makes it comparable to `oval` and to `blocktopk`.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`atom`](atom.md), [`based`](based.md), [`clustered`](clustered.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`skyformer`](skyformer.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`trigrff`](trigrff.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `page_pca` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `page_pca` selected](index.md?compare=page_pca) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *LOCKS: Page-Local Compact Key Summaries for Efficient Long-Context Decoding.*<br><span class="attn-cite" data-bibkey="hwang2026locks"><code>hwang2026locks</code></span>
- *Loki: Low-rank Keys for Efficient Sparse Attention.*<br><span class="attn-cite" data-bibkey="singhania2024loki"><code>singhania2024loki</code></span>
- *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.*<br><span class="attn-cite" data-bibkey="tang2024quest"><code>tang2024quest</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="hwang2026locks">
<pre class="attn-bib__src"><code>@misc{hwang2026locks,
  title = {{LOCKS: Page-Local Compact Key Summaries for Efficient Long-Context Decoding}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="singhania2024loki">
<pre class="attn-bib__src"><code>@inproceedings{singhania2024loki,
  title = {{Loki: Low-rank Keys for Efficient Sparse Attention}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="tang2024quest">
<pre class="attn-bib__src"><code>@inproceedings{tang2024quest,
  title     = {{Quest}: Query-Aware Sparsity for Efficient Long-Context {LLM} Inference}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/page_pca.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/page_pca.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
