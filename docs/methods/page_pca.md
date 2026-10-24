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

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="yes">yes</span> | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

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

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

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

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `eta`, `k_realized`, `n_pages`, `page_scores`, `page_size`, `pages_selected`, `rank`, `rank_used`, `support`.

## Configuration

This method takes no configuration of its own — only the base class's `scale` and `backend`.

`ab.describe("page_pca")` returns the same keys alongside everything else this page states.

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

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:sparse-query</code> (&quot;Blocks and pages&quot;)</p>

The same page selector as `oval` — keys
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

judged the same way — `approximate`
:   31 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `page_pca` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `page_pca` selected](index.md?compare=page_pca) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Hwang (2026).** *LOCKS: Page-Local Compact Key Summaries for Efficient Long-Context Decoding.* [arXiv:2607.24555](https://arxiv.org/abs/2607.24555)<br><span class="attn-cite" data-bibkey="hwang2026locks"><code>hwang2026locks</code></span>
- **Singhania et al. (2024).** *Loki: Low-rank Keys for Efficient Sparse Attention.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2406.02542](https://arxiv.org/abs/2406.02542)<br><span class="attn-cite" data-bibkey="singhania2024loki"><code>singhania2024loki</code></span>
- **Tang et al. (2024).** *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)<br><span class="attn-cite" data-bibkey="tang2024quest"><code>tang2024quest</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="hwang2026locks">
<pre class="attn-bib__src"><code>@misc{hwang2026locks,
  title = {{LOCKS: Page-Local Compact Key Summaries for Efficient Long-Context Decoding}},
  author = {Junsung Hwang},
  year = {2026},
  howpublished = {arXiv preprint arXiv:2607.24555},
  eprint = {2607.24555},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2607.24555}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="singhania2024loki">
<pre class="attn-bib__src"><code>@inproceedings{singhania2024loki,
  title = {{Loki: Low-rank Keys for Efficient Sparse Attention}},
  author = {Prajwal Singhania and Siddharth Singh and Shwai He and Soheil Feizi and Abhinav Bhatele},
  year = {2024},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  eprint = {2406.02542},
  archivePrefix = {arXiv},
  url = {https://openreview.net/forum?id=raABeiV71j}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="tang2024quest">
<pre class="attn-bib__src"><code>@inproceedings{tang2024quest,
  title     = {{Quest}: Query-Aware Sparsity for Efficient Long-Context {LLM} Inference},
  author    = {Tang, Jiaming and Zhao, Yilong and Zhu, Kan and Xiao, Guangxuan and Kasikci, Baris and Han, Song},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  year      = {2024},
  eprint    = {2406.10774},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/page_pca.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/page_pca.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
