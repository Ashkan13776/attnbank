# OVAL

<p class="attn-standfirst" markdown="span">An output-aware local page basis for page selection.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `oval` &middot; `ab.build("oval")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.OvalAttention` &middot; [`attnbank/methods/restrict/oval.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/oval.py)

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
>>> ab.describe("oval").limit_config
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

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `page` | `4` |
| `n_pages` | `2` |
| `rank` | `2` |
| `eta` | `0.5` |
| `g` | `None` |
| `eps` | `1e-12` |

</div>

`eta` mixes key fidelity (0) with output sensitivity (1); `g` is the PSD
value-space metric, `None` meaning the identity (see the module docstring).

`ab.describe("oval")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("oval", page=4, n_pages=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["oval"], q, k, v, causal=True,
                 configs={"oval": {'page': 4, 'n_pages': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
OvalAttention(name='oval', category='restrict', objective='approximate', backend='reference', eps=1e-12, eta=0.5, g=None, n_pages=2, page=4, rank=2)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, eta, k_realized, n_pages, page_scores, page_size, pages_selected, rank, rank_used, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.32509, 'distributional_error': 1.32509, 'value_error': 0.0, 'tv': 0.49796}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<span class="attn-ref">sec:sparse-query</span> ("Blocks and pages"), alongside `blocktopk`.  Keys are
partitioned into contiguous pages; a page is summarised once, offline, and ranked at
decode time by an *estimate of its attention mass*.  What OVAL changes is the summary.

Page score.  For a page with keys `K in R^{m x d}`, write `mu = K^T 1 / m` and
`X = K - 1 mu^T`.  The log of the page's unnormalized mass is exactly

```text
l(q) = log sum_i exp(beta k_i^T q) = beta mu^T q + lse(z(q)),   z(q) = beta X q,
```

and since the softmax denominator is shared by all pages, ranking by `l` ranks them
by total attention probability.  A rank-`r` orthonormal basis `B in R^{d x r}`
gives the storable surrogate `l_B(q) = beta mu^T q + lse(beta X P_B q)`, evaluated as
`R (B^T q)` from the stored `R = X B in R^{m x r}`.  Values never enter scoring.

Which basis.  The key-only answer — page-local key PCA, i.e. the leading eigenvectors
of `X^T X` (`page_pca`, LOCKS / Loki) — keeps the directions with the most key
energy.  OVAL's observation is that a logit error only matters when it moves mass
between tokens with *different values*.  Centering the values as `Y = V - 1 nu^T`,
the conditional within-page output is `f(z) = nu + Y^T softmax(z)`; linearized at
uniform attention its sensitivity to a logit perturbation `h` is `Y^T h / m`.
Measuring that in a PSD seminorm `G` and the logit residual in `l_2`, and taking
the expectation over isotropic queries (`E[q q^T] = I`), the two risks are

```text
R_key(B) = tr((I - P_B) X^T X (I - P_B)) / d,
R_out(B) = tr((I - P_B) M_out (I - P_B)) / (m^2 d),   M_out = X^T Y G Y^T X >= 0.
```

Both are the energy of a PSD matrix outside the retained subspace, so both are
minimized by leading invariant subspaces.  Scaling each by its total energy and mixing
with `eta in [0, 1]` gives the matrix this file diagonalizes,

```text
M_eta = (1 - eta) c_key X^T X + eta c_out M_out,
c_key = (tr(X^T X) + eps)^-1,   c_out = (tr(M_out) + eps)^-1,
```

whose leading `r` eigenvectors are `B_eta`.  `eta = 0` *is* key PCA; `eta = 1`
minimizes the linearized output surrogate alone.  Values are read once, when the page
summary is built; the stored summary `(mu, B_eta, R_eta)` and the decode-time cost
are those of a key-only spectral summary of the same rank.

What this buys and what it does not.  The basis is chosen for a *linearization at
uniform within-page attention* under an isotropic-query assumption, so nothing here is
a bound on the realised output error.  The bound that *does* hold is on retrieval:
lse is 1-Lipschitz in the sup norm, so `|l(q) - l_B(q)| <= ||beta X (I - P_B) q||_inf`
for any basis, OVAL's included — and its expected square is `R_key`, which is what
`eta = 0` minimizes.  Moving `eta` off zero deliberately gives retrieval accuracy
up; whether that is a good trade depends on the key-value coupling actually present,
and on isotropic data with none it is not (`tests/test_restrict_learned.py`).
Selection is still at page granularity, so the omitted mass `eps_q` of
<span class="attn-ref">prop:sparse-identity</span> is still at least the page-level minimum and the realised
error is still `eps_q (m_{S^c} - m_S)`.

Differences from the paper, stated rather than hidden:

* `G = W_O^T W_O` in the paper, which measures the perturbation after the head's
  output projection.  The bank has no `W_O`, so `g=None` (the identity: the
  perturbation measured in value space) is the default.  Pass `g=` a PSD
  `[dv, dv]` tensor to recover the paper's choice.
* Algorithm 1 reduces the `d x d` eigenproblem to `m x m` for short pages and
  completes a rank-deficient basis to exactly `r` columns.  This file diagonalizes
  `M_eta` directly (clarity over speed, DESIGN.md Sec. 6) and performs the same
  completion from the residual `X - X P_B`; columns beyond that are reached only
  when `span(B)` already contains the row space of `X`, so they contribute exactly
  zero to `R_eta` and are stored as zeros here instead of as arbitrary orthonormal
  directions.  The scored function of `q` is identical.
* The paper's harness builds on FreeKV and adds sink and recent regions to the
  selected pages.  Those are orthogonal to the basis and live in `sink_window`;
  this file is the page selector alone, so that what it is compared against — the
  same selector with `eta = 0` — differs in nothing but the basis.

Causality follows `blocktopk`: only *complete past* pages are rankable, the page
containing the query is always selected, and later pages are excluded.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`
:   31 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `oval` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `oval` selected](index.md?compare=oval) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Shahbazi et al. (2026).** *OVAL: Output-Aware Local Page Bases for KV Cache Retrieval.* [arXiv:2610.06686](https://arxiv.org/abs/2610.06686)<br><span class="attn-cite" data-bibkey="shahbazi2026oval"><code>shahbazi2026oval</code></span>
- **Hwang (2026).** *LOCKS: Page-Local Compact Key Summaries for Efficient Long-Context Decoding.* [arXiv:2607.24555](https://arxiv.org/abs/2607.24555)<br><span class="attn-cite" data-bibkey="hwang2026locks"><code>hwang2026locks</code></span>
- **Tang et al. (2024).** *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)<br><span class="attn-cite" data-bibkey="tang2024quest"><code>tang2024quest</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="shahbazi2026oval">
<pre class="attn-bib__src"><code>@misc{shahbazi2026oval,
  title = {{OVAL: Output-Aware Local Page Bases for KV Cache Retrieval}},
  author = {Ashkan Shahbazi and Chayne Thrash and Soheil Kolouri},
  year = {2026},
  howpublished = {arXiv preprint arXiv:2610.06686},
  eprint = {2610.06686},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2610.06686}
}</code></pre>
</div>
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

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/oval.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/oval.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
