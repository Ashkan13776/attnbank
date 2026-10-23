# Signed rank bound

<p class="tx-regkey"><code>svd_oracle</code> &middot; <code>ab.build("svd_oracle")</code></p>

!!! danger "Oracle — an upper bound, not a method"
    `svd_oracle` is not deployable and is not meant to be. It uses information no
    inference-time method has, and it is in the bank to answer *how well could this
    ever work?* — several of the monograph's arguments need that number. Read every
    result below as a bound on a family of real methods, never as a baseline you
    could ship.

**Signed rank bound: the best rank-r *signed* factorization of P (truncated SVD).**

category
:   [`factorize`](../taxonomy.md#factorize) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.SVDOracleAttention`, [`attnbank/methods/factorize/svd_oracle.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/svd_oracle.py)

monograph
:   `sec:factor`

## What it changes

Changes the *shape* of $\pi_q$: it is routed through $r$ latents, $\pi_q = a(q,\cdot)\,b(\cdot,x)$, which caps the rank of the attention matrix at $r$.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:factor-rank`.  The truncated SVD `P_r = U_r Sigma_r V_r^T` is the optimal
rank-`r` approximation of `P` in every unitarily invariant norm (Eckart–Young), so
it is the lower bound that no stochastic factorization of the same rank can beat:
`rank(P) <= rank_+(P) = rank_s(P)`, and the inequality is usually strict.  The gap
between this file and `nmf_oracle` is the empirical nonnegative-rank penalty of
study S5.

The price of optimality is the whole contract: `P_r` is **signed**, its rows do not
sum to one, and the output can leave the convex hull of the values — exactly the
failure mode `prop:factor-closure` was stated to rule out for Markov compositions.
`row_stochastic = False` records it; the row sums and the most negative weight are
returned in `extras` so the violation can be measured instead of assumed.

This is an **oracle**: no one computes a per-head SVD of `P` at inference.  It exists
to bound what any rank-`r` method could achieve.  The decomposition is taken in
float64.

Exact limit: `r = None` keeps the full spectrum.  Not causal: the truncation does not
preserve the mask's zeros.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **no** | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **no** | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Configuration

| parameter | default |
|---|---|
| `r` | `4` |

```python
ab.describe("svd_oracle")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("svd_oracle").limit_config
{'r': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `min_weight`, `r`, `row_sum`, `singular_values`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    rmax = min(m, n)
    r = min(self.r or rmax, rmax)
    return {"flops": 2 * m * n * d + 2 * (m + n) * r * dv,
            "state_bytes": r * n * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 135,266,304 | 268,435,456 | 0.504× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

!!! note "A cost model is not a benchmark"
    These are counted operations and resident bytes, not measured time.
    They are what the monograph's capacity arguments are made of;
    `examples/08_cost_and_capacity.py` walks through the comparison properly.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("svd_oracle", r=4)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["svd_oracle"], q, k, v, causal=False,
                 configs={"svd_oracle": {'r': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
SVDOracleAttention(name='svd_oracle', category='factorize', objective='approximate', backend='reference', r=4)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[min_weight, r, row_sum, singular_values], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.64546, 'distributional_error': 0.64546, 'value_error': 0.0, 'tv': 0.3167}
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `svd_oracle` removed.

changes the same object — [`factorize`](../taxonomy.md#factorize)
:   [`clustered`](clustered.md), [`latent_stochastic`](latent_stochastic.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)* &nbsp;·&nbsp; [browse the bank, factorize only](index.md?category=factorize)

judged the same way — `approximate`, beyond the ones above
:   [`blocktopk`](blocktopk.md), [`evict_h2o`](evict_h2o.md), [`evict_snapkv`](evict_snapkv.md), [`linformer`](linformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lsh_sample`](lsh_sample.md), [`merge_kv`](merge_kv.md), [`nystrom`](nystrom.md), [`performer`](performer.md), [`quantkv`](quantkv.md), [`sink_window`](sink_window.md), [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

[Open the browser with `svd_oracle` selected](index.md?compare=svd_oracle) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **Bhojanapalli et al. (2020).** *Low-Rank Bottleneck in Multi-head Attention Models.* Proceedings of the 37th International Conference on Machine Learning (ICML). [arXiv:2002.07028](https://arxiv.org/abs/2002.07028)<br><span class="attn-cite" data-bibkey="bhojanapalli2020lowrank"><code>bhojanapalli2020lowrank</code></span>
- **Wang et al. (2020).** *Linformer: Self-Attention with Linear Complexity.* [arXiv:2006.04768](https://arxiv.org/abs/2006.04768)<br><span class="attn-cite" data-bibkey="wang2020linformer"><code>wang2020linformer</code></span>
- **Gillis (2020).** *Nonnegative Matrix Factorization.* [doi:10.1137/1.9781611976410](https://doi.org/10.1137/1.9781611976410)<br><span class="attn-cite" data-bibkey="gillis2020nonnegative"><code>gillis2020nonnegative</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="bhojanapalli2020lowrank">
<pre class="attn-bib__src"><code>@inproceedings{bhojanapalli2020lowrank,
  title         = {Low-Rank Bottleneck in Multi-head Attention Models},
  author        = {Bhojanapalli, Srinadh and Yun, Chulhee and Rawat, Ankit Singh and Reddi, Sashank and Kumar, Sanjiv},
  booktitle     = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  series        = {PMLR},
  volume        = {119},
  pages         = {864--873},
  year          = {2020},
  eprint        = {2002.07028},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="wang2020linformer">
<pre class="attn-bib__src"><code>@misc{wang2020linformer,
  title = {{Linformer: Self-Attention with Linear Complexity}},
  author = {Sinong Wang and Belinda Z. Li and Madian Khabsa and Han Fang and Hao Ma},
  year = {2020},
  howpublished = {arXiv preprint arXiv:2006.04768},
  eprint = {2006.04768},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="gillis2020nonnegative">
<pre class="attn-bib__src"><code>@book{gillis2020nonnegative,
  title     = {Nonnegative Matrix Factorization},
  author    = {Gillis, Nicolas},
  publisher = {SIAM},
  address   = {Philadelphia, PA},
  year      = {2020},
  doi       = {10.1137/1.9781611976410}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/svd_oracle.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/svd_oracle.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
