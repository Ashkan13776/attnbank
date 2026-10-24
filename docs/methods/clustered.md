# Clustered attention

<p class="attn-standfirst" markdown="span">Route queries through r query centroids.</p>

<div class="attn-facts" data-cat="factorize" markdown="1">

key
:   `clustered` &middot; `ab.build("clustered")`

category
:   [`factorize`](../taxonomy.md#factorize)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.ClusteredAttention` &middot; [`attnbank/methods/factorize/clustered.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/clustered.py)

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
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("clustered").limit_config
{'r': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 1,310,720 | 268,435,456 | 0.00488× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = min(self.r or m, m)
    return {"flops": 2 * r * n * (d + dv) + m * r * d,
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `assign`, `b`, `centroid_dist`, `centroids`, `r`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `4` |
| `seed` | `0` |

</div>

`r`: number of query clusters; `None` is one cluster per query (exact).

`ab.describe("clustered")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("clustered", r=4)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["clustered"], q, k, v, causal=False,
                 configs={"clustered": {'r': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
ClusteredAttention(name='clustered', category='factorize', objective='approximate', backend='reference', r=4, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[assign, b, centroid_dist, centroids, r], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.71176, 'distributional_error': 0.71176, 'value_error': 0.0, 'tv': 0.33493}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:factor-methods</code></p>

The queries are clustered; one attention row is computed per
centroid and shared by every query in the cluster:

```text
pi_hat_{q_i} = softmax_x( beta <c_{g(i)}, k_x> ).
```

In the language of <span class="attn-ref">def:factor-stochastic</span> this is `P_hat = A B` with `A` the
(0/1) cluster-assignment matrix — row-stochastic — and `B` the `r` centroid rows,
also row-stochastic: a latent stochastic factorization whose latent space is a
partition of the *queries* rather than of the context.  It therefore inherits all of
<span class="attn-ref">prop:factor-closure</span>.

Unlike `latent_stochastic` it is an **approximation** with a resource parameter:
`r = None` gives every query its own cluster and reproduces softmax exactly, which is
the correctness anchor.  Its error is controlled by how far a query sits from its
centroid, via <span class="attn-ref">prop:error-score</span> with `delta = beta ||q - c|| max_j ||k_j||`.

Not causal: queries in one cluster have different visibility, so a shared row would
leak.  Clustering is Lloyd's algorithm with a seeded initialization.

</div>

## Related methods

changes the same object — [`factorize`](../taxonomy.md#factorize)
:   [`abc`](abc.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`perceiver`](perceiver.md), [`svd_oracle`](svd_oracle.md) *(oracle)* &nbsp;·&nbsp; [browse the bank, factorize only](index.md?category=factorize)

judged the same way — `approximate`
:   42 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `clustered` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `clustered` selected](index.md?compare=clustered) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Vyas et al. (2020).** *Fast Transformers with Clustered Attention.* Advances in Neural Information Processing Systems 33 (NeurIPS). [arXiv:2007.04825](https://arxiv.org/abs/2007.04825)<br><span class="attn-cite" data-bibkey="vyas2020fast"><code>vyas2020fast</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
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

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/clustered.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/clustered.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
