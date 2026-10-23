# `nmf_oracle`

!!! danger "Oracle — an upper bound, not a method"
    `nmf_oracle` is not deployable and is not meant to be. It uses information no
    inference-time method has, and it is in the bank to answer *how well could this
    ever work?* — several of the monograph's arguments need that number. Read every
    result below as a bound on a family of real methods, never as a baseline you
    could ship.

**Best rank-r *nonnegative* factorization of the attention matrix.**

category
:   [`factorize`](../taxonomy.md#factorize) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.NMFOracleAttention`, [`attnbank/methods/factorize/nmf_oracle.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/nmf_oracle.py)

monograph
:   `sec:factor`

## What it changes

Changes the *shape* of $\pi_q$: it is routed through $r$ latents, $\pi_q = a(q,\cdot)\,b(\cdot,x)$, which caps the rank of the attention matrix at $r$.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:factor-rank`, `prop:factor-rank`.  For a row-stochastic `P` the hierarchy is

```text
rank(P) <= rank_+(P) = rank_s(P),
```

so the smallest rank of a *stochastic* factorization equals the nonnegative rank, and
stochastic factorization is NMF with a free renormalization.  This file is the empirical
side of that statement: it fits `P ~ W H` with `W, H >= 0` of inner dimension `r`
by Lee–Seung multiplicative updates on the Frobenius loss,

```text
W <- W * (P H^T) / (W H H^T),     H <- H * (W^T P) / (W^T W H),
```

from `restarts` seeded initializations, keeping the best, and then renormalizes the
rows of `W H` so the result is a genuine row-stochastic `P_hat` — legitimate by
the rescaling argument in the proof of `prop:factor-rank`.  The *unnormalized*
product and its Frobenius residual are returned in `extras` so that the fit quality
can be separated from the renormalization.

This is an **oracle**, not a method anyone deploys: it gives the best a nonnegative
rank-`r` factorization *could* do, so that the gap against `svd_oracle` — which
has no nonnegativity constraint and is optimal by Eckart–Young — measures the
nonnegative-rank penalty that every stochastic latent attention pays.  Multiplicative
updates converge to a stationary point, not a global optimum, so the gap it reports is
an upper bound on the true penalty.

Exact limit: `r = None` returns `W = P`, `H = I` — the exact nonnegative
factorization of rank `n`.

Not causal: a nonnegative factorization of a masked `P` only approximates the zeros.

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
| `iters` | `200` |
| `restarts` | `3` |
| `seed` | `0` |

```python
ab.describe("nmf_oracle")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("nmf_oracle").limit_config
{'r': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `H`, `W`, `fro_residual`, `r`, `raw`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = min(self.r or n, n)
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

attn = ab.build("nmf_oracle", r=4, iters=200)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["nmf_oracle"], q, k, v, causal=False,
                 configs={"nmf_oracle": {'r': 4, 'iters': 200}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
NMFOracleAttention(name='nmf_oracle', category='factorize', objective='approximate', backend='reference', iters=200, r=4, restarts=3, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[H, W, fro_residual, r, raw], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.64372, 'distributional_error': 0.64372, 'value_error': 0.0, 'tv': 0.3159}
```

## Others in `factorize`

[`clustered`](clustered.md), [`latent_stochastic`](latent_stochastic.md), [`svd_oracle`](svd_oracle.md) *(oracle)* — see [the taxonomy](../taxonomy.md#factorize).

## References

- **Lee and Seung (1999).** *Learning the Parts of Objects by Non-negative Matrix Factorization.* Nature. [doi:10.1038/44565](https://doi.org/10.1038/44565)<br>`lee1999learning`
- **Cohen and Rothblum (1993).** *Nonnegative Ranks, Decompositions, and Factorizations of Nonnegative Matrices.* Linear Algebra and its Applications. [doi:10.1016/0024-3795(93)90224-C](https://doi.org/10.1016/0024-3795(93)90224-C)<br>`cohen1993nonnegative`
- **Gillis (2020).** *Nonnegative Matrix Factorization.* [doi:10.1137/1.9781611976410](https://doi.org/10.1137/1.9781611976410)<br>`gillis2020nonnegative`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/nmf_oracle.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/nmf_oracle.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
