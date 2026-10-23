# Linformer

<p class="tx-regkey"><code>linformer</code> &middot; <code>ab.build("linformer")</code></p>

**Linformer: a fixed linear projection of K and V along the *sequence* axis.**

category
:   [`compress`](../taxonomy.md#compress) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.LinformerAttention`, [`attnbank/methods/compress/linformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/linformer.py)

monograph
:   `sec:compress`

## What it changes

Changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:compress-lowrank`, `sec:factor-methods`.  With `E, F in R^{r x n}`,

```text
P_hat = softmax(beta Q (E K)^T) in R^{m x r},    y_hat = P_hat (F V),
```

so `E K` and `F V` are `r` learned pseudo-tokens: the method compresses keys and
values separately, changes both `mu` and `v`, fixes `n`, and is not causal.

It *can* be written on the original positions, and that is how the library stores it:
`y_hat = (P_hat F) V`, so `weights = P_hat F in R^{m x n}` with the **original**
values.  Those weights are generally signed and their rows do not sum to one unless
`F` is row-stochastic, which is the paper's point — `tab:comparative` lists
Linformer as losing `nm` and `hl`.  `row_stochastic = False` records it, and the
row sums are returned in `extras` so the deviation can be measured.

Note also that the latent keys `(EK)_z` enter the softmax *linearly*, and
`exp(beta <q, sum_x E_zx k_x>) != sum_x E_zx exp(beta <q, k_x>)`, so the entries of
`P` are not approximated at all; the Johnson–Lindenstrauss argument is an existence
statement about the reference operator with random `E, F`, not an error bound for the
trained operator.

Exact limit: `r = None` sets `E = F = I_n`, which gives dense softmax.

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
| `r` | `8` |
| `seed` | `0` |

```python
ab.describe("linformer")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("linformer").limit_config
{'r': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** The compressed context is built by this method, and the measure it reports lives on the **original** $n$ positions — a kernel run over the compressed cache would return neither.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `latent_weights`, `r`, `row_sum`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = min(self.r or n, n)
    return {"flops": 2 * n * r * (d + dv) + 2 * m * r * (d + dv),
            "state_bytes": r * (d + dv) * bytes_per_el,
            "bytes_moved": m * (d + dv) * bytes_per_el + r * (d + dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 4,194,304 | 268,435,456 | 0.0156× |
| `state_bytes` | 2,048 | 262,144 | 0.00781× |
| `bytes_moved` | 264,192 | 4,718,592 | 0.056× |

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

attn = ab.build("linformer", r=8)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["linformer"], q, k, v, causal=False,
                 configs={"linformer": {'r': 8}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
LinformerAttention(name='linformer', category='compress', objective='approximate', backend='reference', r=8, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[latent_weights, r, row_sum], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 10.14872, 'distributional_error': 10.14872, 'value_error': 0.0, 'tv': 5.86149}
```

## Others in `compress`

[`evict_h2o`](evict_h2o.md), [`evict_snapkv`](evict_snapkv.md), [`gqa`](gqa.md), [`lowrank_kv`](lowrank_kv.md), [`merge_kv`](merge_kv.md), [`mla`](mla.md), [`quantkv`](quantkv.md) — see [the taxonomy](../taxonomy.md#compress).

## References

- **Wang et al. (2020).** *Linformer: Self-Attention with Linear Complexity.* [arXiv:2006.04768](https://arxiv.org/abs/2006.04768)<br>`wang2020linformer`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/linformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/linformer.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
