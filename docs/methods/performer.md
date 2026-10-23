# `performer`

**Performer / FAVOR+: positive random features for the exponential kernel.**

category
:   [`interaction`](../taxonomy.md#interaction) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.PerformerAttention`, [`attnbank/methods/interaction/performer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/performer.py)

monograph
:   `sec:kernel`

## What it changes

Changes the interaction $g(q,x)$, replaced by a surrogate $\widehat g$. $\pi_q = \widehat g(q,\cdot)\mu / \widehat Z(q)$ is then a different measure on the same $n$ positions.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:kernel-rf`.  With `qt = q / d^{1/4}` and `kt = k / d^{1/4}` the softmax
interaction is `g(q,x) = exp(<qt, kt_x>)` and FAVOR+ uses the positive, unbiased
estimator

```text
phi_omega(u) = exp(omega^T u - ||u||^2 / 2),
g_hat(q,x)   = (1/r) sum_{l=1}^r phi_{omega_l}(qt) phi_{omega_l}(kt_x),
omega_l ~ N(0, I_d)   i.i.d. or exactly orthogonal in blocks of d.
```

Because `phi >= 0` the normalizer `Z_hat > 0` and `pi_hat` is a genuine
probability measure (`prop:error-additive`: positivity is essential).  The relative
variance at a pair is `(exp(||qt + kt||^2) - 1) / r` (`eq:kernel-pos-var`), so the
cost of a fixed accuracy is exponential in the squared logit scale.

The output is a *ratio* of two unbiased estimators, so it is itself biased:
`prop:error-ratio-est` gives bias `O(1/r)` and standard deviation `O(r^{-1/2})`.
`convergence_configs` exists for exactly this reason — there is no finite `r` at
which the method is exact.

Implementation note.  The sum over features is evaluated as a log-sum-exp rather than
as the `O(nr)` moment product, because the library needs `pi_hat` on the original
positions and because `phi` is bounded above by `exp(||omega||^2/2)` but can
underflow to zero in the tail.  The value computed is the same estimator.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **yes** | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Configuration

| parameter | default |
|---|---|
| `r` | `64` |
| `orthogonal` | `True` |
| `seed` | `0` |

```python
ab.describe("performer")     # the same keys, with everything else it declares
```

## Correctness anchor


**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("performer").convergence_configs()
[{'r': 8}, {'r': 32}, {'r': 128}, {'r': 512}]
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_rel`, `log_g_hat`, `r`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.r
    return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv,
            "state_bytes": r * (dv + 1) * bytes_per_el,       # the moments (H, z)
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 33,554,432 | 268,435,456 | 0.125× |
| `state_bytes` | 8,320 | 262,144 | 0.0317× |
| `bytes_moved` | 532,608 | 4,718,592 | 0.113× |

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

attn = ab.build("performer", r=64, orthogonal=True)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["performer"], q, k, v, causal=True,
                 configs={"performer": {'r': 64, 'orthogonal': True}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
PerformerAttention(name='performer', category='interaction', objective='approximate', backend='reference', orthogonal=True, r=64, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_rel, log_g_hat, r], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.55857, 'distributional_error': 1.55857, 'value_error': 0.0, 'tv': 0.4355}
```

## Others in `interaction`

[`cosformer`](cosformer.md), [`linear_elu`](linear_elu.md), [`nystrom`](nystrom.md), [`trigrff`](trigrff.md) — see [the taxonomy](../taxonomy.md#interaction).

## References

- **Choromanski et al. (2021).** *Rethinking Attention with Performers.* International Conference on Learning Representations (ICLR). [arXiv:2009.14794](https://arxiv.org/abs/2009.14794)<br>`choromanski2021rethinking`
- **Yu et al. (2016).** *Orthogonal Random Features.* Advances in Neural Information Processing Systems 29 (NIPS). [arXiv:1610.09072](https://arxiv.org/abs/1610.09072)<br>`yu2016orthogonal`
- **Likhosherstov et al. (2022).** *Chefs' Random Tables: Non-Trigonometric Random Features.* Advances in Neural Information Processing Systems 35 (NeurIPS). [arXiv:2205.15317](https://arxiv.org/abs/2205.15317)<br>`likhosherstov2022chefs`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/performer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/performer.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
