# Latent stochastic factorization

<p class="attn-standfirst" markdown="span">Two Markov kernels through r latents.</p>

<div class="attn-facts" data-cat="factorize" markdown="1">

key
:   `latent_stochastic` &middot; `ab.build("latent_stochastic")`

category
:   [`factorize`](../taxonomy.md#factorize)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.LatentStochasticAttention` &middot; [`attnbank/methods/factorize/latent_stochastic.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/latent_stochastic.py)

monograph
:   `sec:factor`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the *shape* of $\pi_q$: it is routed through $r$ latents, $\pi_q = a(q,\cdot)\,b(\cdot,x)$, which caps the rank of the attention matrix at $r$.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

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
| `trainable` | <span class="attn-flag" data-v="yes">yes</span> | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

</div>

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 4,194,304 | 268,435,456 | 0.0156× |
| `state_bytes` | 2,048 | 262,144 | 0.00781× |
| `bytes_moved` | 264,192 | 4,718,592 | 0.056× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.r
    return {"flops": 2 * n * r * (d + dv) + 2 * m * r * (d + dv),
            "state_bytes": r * (d + dv) * bytes_per_el,
            "bytes_moved": m * (d + dv) * bytes_per_el + r * (d + dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `a`, `b`, `latent_keys`, `latent_values`, `r`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `8` |
| `seed` | `0` |

</div>

`ab.describe("latent_stochastic")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("latent_stochastic", r=8)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = modify: this is a different operator and fidelity to softmax
# is not its goal, so there is no error to quote. What it *is* doing shows
# up in its own diagnostics.
print("row_stochastic =", attn.row_stochastic, "| first row sums:",
      [round(x, 4) for x in out.weights.sum(-1)[0, 0, :4].tolist()])
print("extras:", sorted(out.extras))
```

<div class="attn-out" markdown="1">

```text
LatentStochasticAttention(name='latent_stochastic', category='factorize', objective='modify', backend='reference', r=8, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[a, b, latent_keys, latent_values, r], dtype=float32, device=cpu)
invariant W: 1.7881393432617188e-07
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['a', 'b', 'latent_keys', 'latent_values', 'r']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:factor-def</code>, <code>def:factor-stochastic</code>, <code>prop:factor-closure</code></p>

A rank-`r`
latent stochastic factorization is a pair of Markov kernels `a: Q -> Z` (read) and
`b: Z -> X` (write) with

```text
pi_hat_q(dx) = sum_z a(q, z) b(z, dx),    P_hat = A B,
A in R_{>=0}^{m x r} with A 1 = 1,   B in R_{>=0}^{r x n} with B 1 = 1.
```

Both kernels are attentions (<span class="attn-ref">eq:factor-write</span>, <span class="attn-ref">eq:factor-read</span>):

```text
b(z, .) = softmax_x( beta <w_z, k_x> ),     w_z a latent query,
a(q, z) = softmax_z( beta <q, kbar_z> ),    kbar_z = int k_x b(z, dx),
```

the ABC form in which the latent keys summarize the context rather than being free
(Peng et al., 2022).  The latent queries `w_z` are the learned parameters; here they
are a seeded Gaussian, so `trainable = True` records that the method is only
meaningful after training.

<span class="attn-ref">prop:factor-closure</span> is why this category exists separately from low-rank
approximation: a composition of Markov kernels is automatically a probability measure,
so normalization, constant preservation and convex-hull confinement all survive — none
of which holds for a generic signed rank-`r` factorization.  The output has the
two-level form `y_hat(q) = sum_z a(q,z) u(z)` with `u(z) = int v db(z,.)`, so the
context is seen only through `r` latent values.

This is an **operator modification** (<span class="attn-ref">tab:comparative</span>): there is no `r` at which
it becomes softmax, and nothing it converges to, so both correctness anchors are
`None` by design.  It loses `fs` (full support in the exact sense) and `ql`.

Not causal: the write kernel pools over the whole context.

</div>

## Related methods

changes the same object — [`factorize`](../taxonomy.md#factorize)
:   [`abc`](abc.md), [`clustered`](clustered.md), [`isab`](isab.md), [`latte`](latte.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`perceiver`](perceiver.md), [`svd_oracle`](svd_oracle.md) *(oracle)* &nbsp;·&nbsp; [browse the bank, factorize only](index.md?category=factorize)

judged the same way — `modify`
:   25 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `latent_stochastic` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `latent_stochastic` selected](index.md?compare=latent_stochastic) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Ma et al. (2021).** *Luna: Linear Unified Nested Attention.* Advances in Neural Information Processing Systems 34 (NeurIPS). [arXiv:2106.01540](https://arxiv.org/abs/2106.01540)<br><span class="attn-cite" data-bibkey="ma2021luna"><code>ma2021luna</code></span>
- **Dolga et al. (2024).** *Latte: Latent Attention for Linear Time Transformers.* [arXiv:2402.17512](https://arxiv.org/abs/2402.17512)<br><span class="attn-cite" data-bibkey="dolga2024latte"><code>dolga2024latte</code></span>
- **Peng et al. (2022).** *ABC: Attention with Bounded-memory Control.* Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL). [arXiv:2110.02488](https://arxiv.org/abs/2110.02488) [doi:10.18653/v1/2022.acl-long.515](https://doi.org/10.18653/v1/2022.acl-long.515)<br><span class="attn-cite" data-bibkey="peng2022abc"><code>peng2022abc</code></span>
- **Lee et al. (2019).** *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* Proceedings of the 36th International Conference on Machine Learning (ICML). [arXiv:1810.00825](https://arxiv.org/abs/1810.00825)<br><span class="attn-cite" data-bibkey="lee2019set"><code>lee2019set</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="ma2021luna">
<pre class="attn-bib__src"><code>@inproceedings{ma2021luna,
  title         = {Luna: Linear Unified Nested Attention},
  author        = {Ma, Xuezhe and Kong, Xiang and Wang, Sinong and Zhou, Chunting and May, Jonathan and Ma, Hao and Zettlemoyer, Luke},
  booktitle     = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
  year          = {2021},
  eprint        = {2106.01540},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="dolga2024latte">
<pre class="attn-bib__src"><code>@misc{dolga2024latte,
  title         = {Latte: Latent Attention for Linear Time Transformers},
  author        = {Dolga, Rares and Maystre, Lucas and Cobzarenco, Marius and Barber, David},
  year          = {2024},
  eprint        = {2402.17512},
  archivePrefix = {arXiv},
  note          = {Later versions (v5, 2025) retitled ``Unifying Linear-Time Attention via Latent Probabilistic Modelling''}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="peng2022abc">
<pre class="attn-bib__src"><code>@inproceedings{peng2022abc,
  title         = {{ABC}: Attention with Bounded-memory Control},
  author        = {Peng, Hao and Kasai, Jungo and Pappas, Nikolaos and Yogatama, Dani and Wu, Zhaofeng and Kong, Lingpeng and Schwartz, Roy and Smith, Noah A.},
  booktitle     = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year          = {2022},
  doi           = {10.18653/v1/2022.acl-long.515},
  eprint        = {2110.02488},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="lee2019set">
<pre class="attn-bib__src"><code>@inproceedings{lee2019set,
  title         = {Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
  author        = {Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam R. and Choi, Seungjin and Teh, Yee Whye},
  booktitle     = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  series        = {PMLR},
  volume        = {97},
  year          = {2019},
  eprint        = {1810.00825},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/latent_stochastic.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/latent_stochastic.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
