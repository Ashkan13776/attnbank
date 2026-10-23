# `nystrom`

**Nystromformer: landmark Nystrom approximation of the row-stochastic matrix.**

category
:   [`interaction`](../taxonomy.md#interaction) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.NystromAttention`, [`attnbank/methods/interaction/nystrom.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/nystrom.py)

monograph
:   `sec:kernel`

## What it changes

Changes the interaction $g(q,x)$, replaced by a surrogate $\widehat g$. $\pi_q = \widehat g(q,\cdot)\mu / \widehat Z(q)$ is then a different measure on the same $n$ positions.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:kernel-matrix` and `sec:factor-methods`.  With `r` landmark queries and keys
obtained by *segment means* of `Q` and `K`,

```text
P_hat = softmax(beta Q Kt^T) * G^+ * softmax(beta Qt K^T),
G     = softmax(beta Qt Kt^T),
```

where `Qt, Kt` are the landmarks and `G^+` the pseudo-inverse (the original uses an
iterative approximation of it; this reference implementation uses an exact pinv in
float64, so that the only error measured is the method's, not the solver's).

Three caveats the paper makes and this file honours.

1. `P` is neither symmetric nor PSD when `Q != K`, so Nystrom theory does not
    apply; there is no spectral guarantee here.

2. The pseudo-inverse makes `P_hat` **signed**: `pi_hat` is a signed measure of
    unit mass, so the contract items `nm` and `hl` are lost — `row_stochastic` is
    `False`, and outputs can leave the convex hull of the values.  (Rows do sum to one
    when `G` is invertible, since `G^{-1} 1 = 1` for a stochastic `G`; the library
    does not rely on that.)

3. Segment-mean landmarks summarize the context, so the method also modifies `mu`.

Exact limit.  With `r = None` the landmarks are the data themselves (`Qt = Q`,
`Kt = K`, which requires `m = n`), giving `P_hat = P P^+ P = P` exactly by the
Moore–Penrose identity.  That is the correctness anchor.

Not causal: the segment means pool over the whole sequence.

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
| `r` | `16` |

`r`: number of landmarks; `None` uses the data as landmarks (exact limit).

```python
ab.describe("nystrom")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("nystrom").limit_config
{'r': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `min_weight`, `n_landmarks`, `row_sum`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.r or min(m, n)
    return {"flops": 2 * (m + n) * r * d + 2 * r ** 3 + 2 * (m + n) * r * dv,
            "state_bytes": r * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (d + dv)) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 8,396,800 | 268,435,456 | 0.0313× |
| `state_bytes` | 4,096 | 262,144 | 0.0156× |
| `bytes_moved` | 528,384 | 4,718,592 | 0.112× |

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

attn = ab.build("nystrom", r=16)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["nystrom"], q, k, v, causal=False,
                 configs={"nystrom": {'r': 16}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
NystromAttention(name='nystrom', category='interaction', objective='approximate', backend='reference', r=16)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[min_weight, n_landmarks, row_sum], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 13.3981, 'distributional_error': 13.3981, 'value_error': 0.0, 'tv': 7.27394}
```

## Others in `interaction`

[`cosformer`](cosformer.md), [`linear_elu`](linear_elu.md), [`performer`](performer.md), [`trigrff`](trigrff.md) — see [the taxonomy](../taxonomy.md#interaction).

## References

- **Xiong et al. (2021).** *Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention.* Proceedings of the AAAI Conference on Artificial Intelligence. [arXiv:2102.03902](https://arxiv.org/abs/2102.03902)<br>`xiong2021nystromformer`
- **Williams and Seeger (2001).** *Using the Nystrom Method to Speed Up Kernel Machines.* Advances in Neural Information Processing Systems 13 (NIPS 2000).<br>`williams2001using`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/nystrom.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/nystrom.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
