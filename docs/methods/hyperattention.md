# HyperAttention

<p class="attn-standfirst" markdown="span">A hashed heavy block plus an unbiased uniform estimate of the rest.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `hyperattention` &middot; `ab.build("hyperattention")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.HyperAttention` &middot; [`attnbank/methods/restrict/hyperattention.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/hyperattention.py)

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

**Exact limit.** Built with `n_sampled=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("hyperattention").limit_config
{'n_sampled': None}
```

**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("hyperattention").convergence_configs()
[{'n_sampled': 1}, {'n_sampled': 2}, {'n_sampled': 4}, {'n_sampled': 8}]
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 3,145,728 | 268,435,456 | 0.0117× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 2,490,368 | 4,718,592 | 0.528× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    half = max(1, self.n_buckets // 2)
    kept = self.block + (self.n_sampled if self.n_sampled is not None else n)
    return {"flops": 2 * (m + n) * d * half + 2 * m * kept * (d + dv),
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * d + m * kept * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `eps`, `heavy`, `heavy_mass`, `k_realized`, `n_sampled_realized`, `support`, `u`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `block` | `4` |
| `n_sampled` | `4` |
| `n_buckets` | `8` |
| `seed` | `0` |
| `hash_seed` | `0` |

</div>

`hash_seed` draws the sortLSH rotation and `seed` the uniform sample.

They are separate generators so that either randomization can be held fixed
while the other varies — which is what makes the residual estimator's bias
and variance measurable at a fixed mask.

`ab.describe("hyperattention")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("hyperattention", block=4, n_sampled=4)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["hyperattention"], q, k, v, causal=True,
                 configs={"hyperattention": {'block': 4, 'n_sampled': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
HyperAttention(name='hyperattention', category='restrict', objective='approximate', backend='reference', block=4, hash_seed=0, n_buckets=8, n_sampled=4, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[eps, heavy, heavy_mass, k_realized, n_sampled_realized, support, u], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.40806, 'distributional_error': 2.40806, 'value_error': 0.0, 'tv': 0.75214}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<span class="attn-ref">sec:sparse-content</span>, and the boundary <span class="attn-ref">sec:sparse-content</span> draws between
restricting and *estimating*.  HyperAttention splits the row of `pi_q` in two and
treats the halves differently:

1. **Heavy part.** `sortLSH` hashes queries and keys, sorts by bucket, and keeps a
   contiguous block of `block` keys per query — a mask `H_q` that, after the
   permutation, is block diagonal and so can be run by a dense kernel.  These entries
   are kept exactly.
2. **Light part.** The remaining visible keys are sampled uniformly at inclusion
   probability `u = min(1, n_sampled / |S_q^c|)` and reweighted by `1/u`.

The implied measure is therefore neither a restriction nor a pure importance sample:

```text
w_j = pi_j                   j in H_q,
w_j = pi_j / u               j sampled outside H_q,
pi_hat = w / sum_i w_i.
```

Numerator and denominator of the *residual* are unbiased, so the only systematic error
is the `O(1/n_sampled)` self-normalization bias of <span class="attn-ref">prop:error-ratio-est</span>, and the
hashed block is what keeps the variance of the residual small: the keys whose
`pi_j / u` is large are the ones the hash was meant to catch.  The contrast with
`topk` — which simply discards the tail and absorbs `eps_q` into the head — and
with `lsh_sample` — which samples everything — is the point of having all three.

Scope of the paper's guarantee.  Han et al. prove a spectral-norm bound on the
approximate attention under two conditions on the matrix (a bounded ratio of column
norms, `kappa`, and a bounded max column `l_1` mass), together with their specific
sampling scheme.  Nothing here re-derives or checks that bound; this file implements
the estimator, and the only error statement it makes is the one `extras` lets you
verify numerically.

Deviations from the paper, stated rather than hidden:

* The paper sorts queries and keys *independently* by bucket and takes the block
  diagonal of the permuted `m x n` matrix, which pairs a query with keys at the same
  *rank quantile*.  This file places the query at its own bucket's position in the
  sorted key order and takes the `block` keys around it.  The two agree when queries
  and keys have the same bucket distribution; pairing by bucket value is what remains
  well defined for `m != n` and under causal masking, where the sorted key order is
  the order of the *visible prefix* and therefore differs per query.  That restriction
  to the prefix is also what makes the support causal: a key after the query can
  neither enter the block nor move another key into it.
* The paper handles causal masking by a recursive decomposition into two halves plus a
  diagonal block, for which it can reuse the same subroutine.  The prefix formulation
  above is a different route to the same requirement and is not that algorithm.
* Uniform column sampling is done here as independent per-row Bernoulli draws rather
  than as one shared column subset, so each row's estimator is unbiased on its own.

Exact limit: `n_sampled = None` sets `u = 1` on the whole complement, so every
visible key carries its exact weight and `pi_hat = pi` — independent of the hash,
which is what makes it a test of the reweighting rather than of the mask.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`
:   31 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `hyperattention` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `hyperattention` selected](index.md?compare=hyperattention) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Han et al. (2024).** *HyperAttention: Long-context Attention in Near-Linear Time.* The Twelfth International Conference on Learning Representations (ICLR). [arXiv:2310.05869](https://arxiv.org/abs/2310.05869)<br><span class="attn-cite" data-bibkey="han2024hyperattention"><code>han2024hyperattention</code></span>
- **Kitaev et al. (2020).** *Reformer: The Efficient Transformer.* International Conference on Learning Representations (ICLR). [arXiv:2001.04451](https://arxiv.org/abs/2001.04451)<br><span class="attn-cite" data-bibkey="kitaev2020reformer"><code>kitaev2020reformer</code></span>
- **Owen (2013).** *Monte Carlo Theory, Methods and Examples.*<br><span class="attn-cite" data-bibkey="owen2013monte"><code>owen2013monte</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="han2024hyperattention">
<pre class="attn-bib__src"><code>@inproceedings{han2024hyperattention,
  title = {{HyperAttention: Long-context Attention in Near-Linear Time}},
  author = {Insu Han and Rajesh Jayaram and Amin Karbasi and Vahab Mirrokni and David P. Woodruff and Amir Zandieh},
  year = {2024},
  booktitle = {The Twelfth International Conference on Learning Representations (ICLR)},
  eprint = {2310.05869},
  archivePrefix = {arXiv},
  url = {https://openreview.net/forum?id=Eh0Od2BJIM}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="kitaev2020reformer">
<pre class="attn-bib__src"><code>@inproceedings{kitaev2020reformer,
  title={Reformer: The Efficient Transformer},
  author={Kitaev, Nikita and Kaiser, {\L}ukasz and Levskaya, Anselm},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
  eprint={2001.04451},
  archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="owen2013monte">
<pre class="attn-bib__src"><code>@book{owen2013monte,
  title     = {Monte Carlo Theory, Methods and Examples},
  author    = {Owen, Art B.},
  year      = {2013},
  note      = {Book manuscript, available online at \url{https://artowen.su.domains/mc/}}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/hyperattention.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/hyperattention.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
