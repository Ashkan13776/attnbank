# Signed rank bound

<p class="attn-standfirst" markdown="span">The best rank-r *signed* factorization of P (truncated SVD).</p>

!!! danger "Oracle — an upper bound, not a method"
    `svd_oracle` is not deployable and is not meant to be. It uses information no
    inference-time method has, and it is in the bank to answer *how well could this
    ever work?* — several of the monograph's arguments need that number. Read every
    result below as a bound on a family of real methods, never as a baseline you
    could ship.

<div class="attn-facts" data-cat="factorize" markdown="1">

key
:   `svd_oracle` &middot; `ab.build("svd_oracle")`

category
:   [`factorize`](../taxonomy.md#factorize)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.SVDOracleAttention` &middot; [`attnbank/methods/factorize/svd_oracle.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/svd_oracle.py)

monograph
:   `sec:factor`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the *shape* of $\pi_q$: it is routed through $r$ latents, $\pi_q = a(q,\cdot)\,b(\cdot,x)$, which caps the rank of the attention matrix at $r$.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="no">no</span> | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="no">no</span> | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("svd_oracle").limit_config
{'r': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 135,266,304 | 268,435,456 | 0.504× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    rmax = min(m, n)
    r = min(self.r or rmax, rmax)
    return {"flops": 2 * m * n * d + 2 * (m + n) * r * dv,
            "state_bytes": r * n * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `min_weight`, `r`, `row_sum`, `singular_values`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `4` |

</div>

`ab.describe("svd_oracle")` returns the same keys alongside everything else this page states.

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

<div class="attn-out" markdown="1">

```text
SVDOracleAttention(name='svd_oracle', category='factorize', objective='approximate', backend='reference', r=4)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[min_weight, r, row_sum, singular_values], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.64546, 'distributional_error': 0.64546, 'value_error': 0.0, 'tv': 0.3167}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:factor-rank</code></p>

The truncated SVD `P_r = U_r Sigma_r V_r^T` is the optimal
rank-`r` approximation of `P` in every unitarily invariant norm (Eckart–Young), so
it is the lower bound that no stochastic factorization of the same rank can beat:
`rank(P) <= rank_+(P) = rank_s(P)`, and the inequality is usually strict.  The gap
between this file and `nmf_oracle` is the empirical nonnegative-rank penalty of
study S5.

The price of optimality is the whole contract: `P_r` is **signed**, its rows do not
sum to one, and the output can leave the convex hull of the values — exactly the
failure mode <span class="attn-ref">prop:factor-closure</span> was stated to rule out for Markov compositions.
`row_stochastic = False` records it; the row sums and the most negative weight are
returned in `extras` so the violation can be measured instead of assumed.

This is an **oracle**: no one computes a per-head SVD of `P` at inference.  It exists
to bound what any rank-`r` method could achieve.  The decomposition is taken in
float64.

Exact limit: `r = None` keeps the full spectrum.  Not causal: the truncation does not
preserve the mask's zeros.

</div>

## Related methods

changes the same object — [`factorize`](../taxonomy.md#factorize)
:   [`abc`](abc.md), [`clustered`](clustered.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`perceiver`](perceiver.md) &nbsp;·&nbsp; [browse the bank, factorize only](index.md?category=factorize)

judged the same way — `approximate`
:   42 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `svd_oracle` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `svd_oracle` selected](index.md?compare=svd_oracle) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Bhojanapalli et al. (2020).** *Low-Rank Bottleneck in Multi-head Attention Models.* Proceedings of the 37th International Conference on Machine Learning (ICML). [arXiv:2002.07028](https://arxiv.org/abs/2002.07028)<br><span class="attn-cite" data-bibkey="bhojanapalli2020lowrank"><code>bhojanapalli2020lowrank</code></span>
- **Wang et al. (2020).** *Linformer: Self-Attention with Linear Complexity.* [arXiv:2006.04768](https://arxiv.org/abs/2006.04768)<br><span class="attn-cite" data-bibkey="wang2020linformer"><code>wang2020linformer</code></span>
- **Gillis (2020).** *Nonnegative Matrix Factorization.* [doi:10.1137/1.9781611976410](https://doi.org/10.1137/1.9781611976410)<br><span class="attn-cite" data-bibkey="gillis2020nonnegative"><code>gillis2020nonnegative</code></span>

</div>

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

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/svd_oracle.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/svd_oracle.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
