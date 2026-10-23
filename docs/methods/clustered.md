# `clustered`

**Clustered attention: route queries through r query centroids.**

category
:   [`factorize`](../taxonomy.md#factorize) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.ClusteredAttention`, [`attnbank/methods/factorize/clustered.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/clustered.py)

monograph
:   `sec:factor`

## What it changes

Changes the *shape* of $\pi_q$: it is routed through $r$ latents, $\pi_q = a(q,\cdot)\,b(\cdot,x)$, which caps the rank of the attention matrix at $r$.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:factor-methods`.  The queries are clustered; one attention row is computed per
centroid and shared by every query in the cluster:

```text
pi_hat_{q_i} = softmax_x( beta <c_{g(i)}, k_x> ).
```

In the language of `def:factor-stochastic` this is `P_hat = A B` with `A` the
(0/1) cluster-assignment matrix — row-stochastic — and `B` the `r` centroid rows,
also row-stochastic: a latent stochastic factorization whose latent space is a
partition of the *queries* rather than of the context.  It therefore inherits all of
`prop:factor-closure`.

Unlike `latent_stochastic` it is an **approximation** with a resource parameter:
`r = None` gives every query its own cluster and reproduces softmax exactly, which is
the correctness anchor.  Its error is controlled by how far a query sits from its
centroid, via `prop:error-score` with `delta = beta ||q - c|| max_j ||k_j||`.

Not causal: queries in one cluster have different visibility, so a shared row would
leak.  Clustering is Lloyd's algorithm with a seeded initialization.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **no** | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Configuration

| parameter | default |
|---|---|
| `r` | `4` |
| `seed` | `0` |

`r`: number of query clusters; `None` is one cluster per query (exact).

```python
ab.describe("clustered")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("clustered").limit_config
{'r': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `assign`, `b`, `centroid_dist`, `centroids`, `r`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = min(self.r or m, m)
    return {"flops": 2 * r * n * (d + dv) + m * r * d,
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 1,310,720 | 268,435,456 | 0.00488× |
| `state_bytes` | 262,144 | 262,144 | 1× |
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

```text
ClusteredAttention(name='clustered', category='factorize', objective='approximate', backend='reference', r=4, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[assign, b, centroid_dist, centroids, r], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.71176, 'distributional_error': 0.71176, 'value_error': 0.0, 'tv': 0.33493}
```

## Others in `factorize`

[`latent_stochastic`](latent_stochastic.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`svd_oracle`](svd_oracle.md) *(oracle)* — see [the taxonomy](../taxonomy.md#factorize).

## References

- **Vyas et al. (2020).** *Fast Transformers with Clustered Attention.* Advances in Neural Information Processing Systems 33 (NeurIPS). [arXiv:2007.04825](https://arxiv.org/abs/2007.04825)<br>`vyas2020fast`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/clustered.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/clustered.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
