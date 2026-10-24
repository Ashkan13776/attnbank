# Hidden-dimension low rank

<p class="attn-standfirst" markdown="span">Truncated SVD of the key and value channel matrices.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `lowrank_kv` &middot; `ab.build("lowrank_kv")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.LowRankKVAttention` &middot; [`attnbank/methods/compress/lowrank_kv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/lowrank_kv.py)

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

**Exact limit.** Built with `rank_k=None`, `rank_v=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("lowrank_kv").limit_config
{'rank_k': None, 'rank_v': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 269,484,032 | 268,435,456 | 1× |
| `state_bytes` | 16,384 | 262,144 | 0.0625× |
| `bytes_moved` | 278,528 | 4,718,592 | 0.059× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    rk, rv = min(self.rank_k or d, d), min(self.rank_v or dv, dv)
    state = n * (rk + rv) * bytes_per_el
    return {"flops": 2 * m * n * (d + dv) + 2 * n * (d * rk + dv * rv),
            "state_bytes": state,
            "bytes_moved": m * (d + dv) * bytes_per_el + state}
```

## Backends

**`reference` only, and that is the correct answer.** It returns both the exact and the perturbed score matrices (`s`, `s_hat`) in `extras` — the perturbation *is* the object of study, and a fused kernel hands back neither.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `basis_k`, `basis_v`, `q_residual`, `s`, `s_hat`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `rank_k` | `4` |
| `rank_v` | `4` |

</div>

`ab.describe("lowrank_kv")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("lowrank_kv", rank_k=4, rank_v=4)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["lowrank_kv"], q, k, v, causal=False,
                 configs={"lowrank_kv": {'rank_k': 4, 'rank_v': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
LowRankKVAttention(name='lowrank_kv', category='compress', objective='approximate', backend='reference', rank_k=4, rank_v=4)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[basis_k, basis_v, q_residual, s, s_hat], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.92801, 'distributional_error': 0.67869, 'value_error': 0.6605, 'tv': 0.30813}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-lowrank</code></p>

With `Pi_r` an orthogonal projector of rank `r` on the
channel space,

```text
k_hat_j = Pi_r^K k_j,   v_hat_j = Pi_r^V v_j,
```

where `Pi_r^K` and `Pi_r^V` are the leading right singular subspaces of `K` and
`V`.  The two errors are structurally different, which is the point of the file:

* Keys.  `delta_l_j = -beta <(I - Pi_r) q, (I - Pi_r) k_j>`, so a subspace chosen from
  the keys alone fails exactly when the queries load on low-variance key directions —
  no amount of key energy captured says anything until the query is known.
* Values.  Since `pi_hat` is unaffected by a value projection, the value error is
  `sum_j pi_j (I - Pi_r) v_j = (I - Pi_r) y(q)`: the projection residual of the
  **output**, not of the approximate output and not of the raw value matrix.  So the
  optimal value subspace is the leading eigenspace of the attention-weighted second
  moment of the *outputs*.

Not causal: the SVD pools over all tokens.  The factorization is computed in float64
so that the exact limit (`rank_k = rank_v = None`) is exact to machine precision.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`
:   30 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `lowrank_kv` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `lowrank_kv` selected](index.md?compare=lowrank_kv) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Xu et al. (2025).** *ThinK: Thinner Key Cache by Query-Driven Pruning.* International Conference on Learning Representations (ICLR). [arXiv:2407.21018](https://arxiv.org/abs/2407.21018)<br><span class="attn-cite" data-bibkey="xu2025think"><code>xu2025think</code></span>
- **Chang et al. (2025).** *Palu: Compressing KV-Cache with Low-Rank Projection.* International Conference on Learning Representations (ICLR). [arXiv:2407.21118](https://arxiv.org/abs/2407.21118)<br><span class="attn-cite" data-bibkey="chang2025palu"><code>chang2025palu</code></span>
- **Wang et al. (2025).** *SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression.* International Conference on Learning Representations (ICLR). [arXiv:2403.07378](https://arxiv.org/abs/2403.07378)<br><span class="attn-cite" data-bibkey="wang2025svdllm"><code>wang2025svdllm</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="xu2025think">
<pre class="attn-bib__src"><code>@inproceedings{xu2025think,
  title = {{ThinK: Thinner Key Cache by Query-Driven Pruning}},
  author = {Yuhui Xu and Zhanming Jie and Hanze Dong and Lei Wang and Xudong Lu and Aojun Zhou and Amrita Saha and Caiming Xiong and Doyen Sahoo},
  year = {2025},
  booktitle = {International Conference on Learning Representations (ICLR)},
  eprint = {2407.21018},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="chang2025palu">
<pre class="attn-bib__src"><code>@inproceedings{chang2025palu,
  title = {{Palu: Compressing KV-Cache with Low-Rank Projection}},
  author = {Chi-Chih Chang and Wei-Cheng Lin and Chien-Yu Lin and Chong-Yan Chen and Yu-Fang Hu and Pei-Shuo Wang and Ning-Chi Huang and Luis Ceze and Mohamed S. Abdelfattah and Kai-Chiang Wu},
  year = {2025},
  booktitle = {International Conference on Learning Representations (ICLR)},
  eprint = {2407.21118},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="wang2025svdllm">
<pre class="attn-bib__src"><code>@inproceedings{wang2025svdllm,
  title = {{SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression}},
  author = {Xin Wang and Yu Zheng and Zhongwei Wan and Mi Zhang},
  year = {2025},
  booktitle = {International Conference on Learning Representations (ICLR)},
  eprint = {2403.07378},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/lowrank_kv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/lowrank_kv.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
