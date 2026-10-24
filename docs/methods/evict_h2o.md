# H2O

<p class="attn-standfirst" markdown="span">Accumulated-attention heavy-hitter eviction plus a recent window.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `evict_h2o` &middot; `ab.build("evict_h2o")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.H2OAttention` &middot; [`attnbank/methods/compress/evict_h2o.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_h2o.py)

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
| `supports_causal` | <span class="attn-flag" data-v="yes">yes</span> | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `budget=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("evict_h2o").limit_config
{'budget': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,105,344 | 268,435,456 | 0.00784× |
| `state_bytes` | 2,080 | 262,144 | 0.00793× |
| `bytes_moved` | 2,359,296 | 4,718,592 | 0.5× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    b = min(self.budget or n, n)
    return {"flops": 2 * m * b * (d + dv) + m * b,
            "state_bytes": b * (d + dv) * bytes_per_el + b * 4,   # +accumulated scores
            "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `accumulated`, `budget`, `chi2`, `eps`, `k_realized`, `recent`, `support`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `budget` | `8` |
| `recent` | `2` |

</div>

`budget`: cache size; `recent`: most recent positions never evicted.

`ab.describe("evict_h2o")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("evict_h2o", budget=8, recent=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["evict_h2o"], q, k, v, causal=True,
                 configs={"evict_h2o": {'budget': 8, 'recent': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
H2OAttention(name='evict_h2o', category='compress', objective='approximate', backend='reference', budget=8, recent=2)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[accumulated, budget, chi2, eps, k_realized, recent, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.17848, 'distributional_error': 2.17848, 'value_error': 0.0, 'tv': 0.77822}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-evict</code>, <code>prop:compress-noeviction</code></p>

Eviction is a restriction whose
set is chosen *before* the queries that will use it exist, and the sets are nested:
`S_{t+1} subset S_t union {t+1}`.  A deleted token is gone for every future query, so
<span class="attn-ref">prop:compress-noeviction</span> says no query-oblivious rule has a uniform guarantee —
for any retained `S` and any evicted `j`, the query `q = tau k_j` drives
`eps_q -> 1` while `y_hat` stays in `conv{v_i : i in S}`.

H2O estimates the future-query distribution `nu` by the empirical distribution of
*past* queries: token `j` is scored by `sum_{t' <= t} pi_{q_{t'}}(j)`, and the
lowest-scoring non-recent token is dropped whenever the cache exceeds its budget.  This
file runs that process literally, one query at a time, which is why it is causal: the
state at step `t` depends on keys and queries up to `t` only.

In non-causal mode the whole context is admitted at the first step and the same
accumulate-and-evict loop runs over the queries in order; the semantics are then "a
streaming cache served by a fixed context".

Exact limit: `budget = None` (no eviction).

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`
:   30 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `evict_h2o` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `evict_h2o` selected](index.md?compare=evict_h2o) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Zhang et al. (2023).** *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)<br><span class="attn-cite" data-bibkey="zhang2023h2o"><code>zhang2023h2o</code></span>
- **Liu et al. (2023).** *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2305.17118](https://arxiv.org/abs/2305.17118)<br><span class="attn-cite" data-bibkey="liu2023scissorhands"><code>liu2023scissorhands</code></span>
- **Oren et al. (2024).** *Transformers are Multi-State RNNs.* Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP). [arXiv:2401.06104](https://arxiv.org/abs/2401.06104)<br><span class="attn-cite" data-bibkey="oren2024transformers"><code>oren2024transformers</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="zhang2023h2o">
<pre class="attn-bib__src"><code>@inproceedings{zhang2023h2o,
  title     = {{H2O}: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models},
  author    = {Zhang, Zhenyu and Sheng, Ying and Zhou, Tianyi and Chen, Tianlong and Zheng, Lianmin and Cai, Ruisi and Song, Zhao and Tian, Yuandong and R{\'e}, Christopher and Barrett, Clark and Wang, Zhangyang and Chen, Beidi},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  year      = {2023},
  eprint    = {2306.14048},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="liu2023scissorhands">
<pre class="attn-bib__src"><code>@inproceedings{liu2023scissorhands,
  title     = {Scissorhands: Exploiting the Persistence of Importance Hypothesis for {LLM} {KV} Cache Compression at Test Time},
  author    = {Liu, Zichang and Desai, Aditya and Liao, Fangshuo and Wang, Weitao and Xie, Victor and Xu, Zhaozhuo and Kyrillidis, Anastasios and Shrivastava, Anshumali},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {36},
  year      = {2023},
  eprint    = {2305.17118},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="oren2024transformers">
<pre class="attn-bib__src"><code>@inproceedings{oren2024transformers,
  title = {{Transformers are Multi-State RNNs}},
  author = {Matanel Oren and Michael Hassid and Nir Yarden and Yossi Adi and Roy Schwartz},
  year = {2024},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  eprint = {2401.06104},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/evict_h2o.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_h2o.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
