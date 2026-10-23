# `trigrff`

**Random Fourier features for attention: the trigonometric (signed) estimator.**

category
:   [`interaction`](../taxonomy.md#interaction) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.TrigRFFAttention`, [`attnbank/methods/interaction/trigrff.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/trigrff.py)

monograph
:   `sec:kernel`

## What it changes

Changes the interaction $g(q,x)$, replaced by a surrogate $\widehat g$. $\pi_q = \widehat g(q,\cdot)\mu / \widehat Z(q)$ is then a different measure on the same $n$ positions.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:kernel-rf`.  Bochner's theorem for the Gaussian kernel plus
`eq:kernel-softmax-gauss` gives the unbiased estimator

```text
xi_omega(q,x) = exp((||qt||^2 + ||kt_x||^2)/2) * cos(omega^T (qt - kt_x)),
omega ~ N(0, I_d),   qt = q/d^{1/4},  kt = k/d^{1/4},
```

separable with `2r` features through `cos(a-b) = cos a cos b + sin a sin b`.  This
is the basis of RFA (Peng et al., 2021) and of the trigonometric variant of
Choromanski et al. (2021).

Two things go wrong, and this file exists so that they can be measured.
(i) The relative variance is `2 sinh^2(||qt - kt_x||^2 / 2)` (`eq:kernel-trig-var`),
largest for keys *far* from the query — the many small entries whose accumulated noise
makes `Z_hat` unreliable.
(ii) `xi` takes both signs, so `Z_hat` can be zero or negative; then `pi_hat` is
not a probability measure and no bound of the form `prop:error-additive` holds.  The
fraction of queries with `Z_hat <= 0` is reported in `extras["Z_rel"]`.

Numerics.  The factor `exp(||kt_j||^2/2)` does not cancel in `pi_hat` and overflows
fp32 for large inputs, so the estimator is carried as a (log-magnitude, sign) pair and
normalized by `attnbank.weights_from_log_magnitude`, which subtracts the maximum
over the *visible* keys.  When `|Z_hat|` is below the floor the result is finite but
large; this is a reported failure, not a silent repair.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **yes** | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **no** | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
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
ab.describe("trigrff")     # the same keys, with everything else it declares
```

## Correctness anchor


**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("trigrff").convergence_configs()
[{'r': 8}, {'r': 32}, {'r': 128}, {'r': 512}]
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_rel`, `log_g_hat`, `negative_Z`, `r`, `sign`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = 2 * self.r                 # cos and sin halves
    return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv,
            "state_bytes": r * (dv + 1) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 67,108,864 | 268,435,456 | 0.25× |
| `state_bytes` | 16,640 | 262,144 | 0.0635× |
| `bytes_moved` | 540,928 | 4,718,592 | 0.115× |

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

attn = ab.build("trigrff", r=64, orthogonal=True)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["trigrff"], q, k, v, causal=True,
                 configs={"trigrff": {'r': 64, 'orthogonal': True}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
TrigRFFAttention(name='trigrff', category='interaction', objective='approximate', backend='reference', orthogonal=True, r=64, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_rel, log_g_hat, negative_Z, r, sign], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 7.51573, 'distributional_error': 7.51573, 'value_error': 0.0, 'tv': 2.26019}
```

## Others in `interaction`

[`cosformer`](cosformer.md), [`linear_elu`](linear_elu.md), [`nystrom`](nystrom.md), [`performer`](performer.md) — see [the taxonomy](../taxonomy.md#interaction).

## References

- **Rahimi and Recht (2007).** *Random Features for Large-Scale Kernel Machines.* Advances in Neural Information Processing Systems (NIPS).<br>`rahimi2007random`
- **Peng et al. (2021).** *Random Feature Attention.* International Conference on Learning Representations (ICLR). [arXiv:2103.02143](https://arxiv.org/abs/2103.02143)<br>`peng2021random`
- **Choromanski et al. (2021).** *Rethinking Attention with Performers.* International Conference on Learning Representations (ICLR). [arXiv:2009.14794](https://arxiv.org/abs/2009.14794)<br>`choromanski2021rethinking`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/trigrff.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/trigrff.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
