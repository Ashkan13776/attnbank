# Linear transformer

<p class="tx-regkey"><code>linear_elu</code> &middot; <code>ab.build("linear_elu")</code></p>

**Linear transformer: the fixed feature map `phi(u) = elu(u) + 1`.**

category
:   [`interaction`](../taxonomy.md#interaction) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.LinearELUAttention`, [`attnbank/methods/interaction/linear_elu.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/linear_elu.py)

monograph
:   `sec:kernel`

## What it changes

Changes the interaction $g(q,x)$, replaced by a surrogate $\widehat g$. $\pi_q = \widehat g(q,\cdot)\mu / \widehat Z(q)$ is then a different measure on the same $n$ positions.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

## In the paper's notation

`sec:kernel-det`.  Katharopoulos et al. (2020) replace the interaction by

```text
g_hat(q, x) = <phi(q), phi(k_x)>,   phi(u) = elu(u) + 1 > 0,   r = d,
```

with no resource parameter, so `<phi(q), phi(k)> -> exp(beta <q,k>)` holds *in no
sense*: this is an **operator modification**, not an approximation
(`def:kernel-regimes`, `tab:kernel-methods` row "Linear transformer ... no").  The
library records that by `objective = modify` and by returning `None` from both
correctness anchors: there is no configuration under which it becomes softmax, and
nothing it converges to.

Because `phi > 0` the induced measure is a genuine probability measure, so this is
one of the normalized members of the family; `lem:kernel-mixture` then says
`pi_hat_q` is a mixture of `r = d` *query-independent* context distributions, and
`prop:kernel-retrieval` bounds what it can retrieve.  `rem:kernel-homogeneous`
applies approximately: for large positive coordinates `phi` is linear, so the query
norm cannot sharpen `pi_hat_q`.

The normalized form of `eq:kernel-moments` is used (divide by `phi(q)^T z`).  Many
deployed descendants drop the denominator and normalize the output instead
(`sec:memory-state`); that variant lives in `recurrent/linear_rnn.py`, where the
loss of normalization invariance is the point.

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

This method takes no configuration of its own — only the base class's `scale` and `backend`.

```python
ab.describe("linear_elu")     # the same keys, with everything else it declares
```

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_hat`, `g_hat`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = d
    return {"flops": 2 * n * r * dv + 2 * m * r * dv + (m + n) * r,
            "state_bytes": r * (dv + 1) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 16,908,288 | 268,435,456 | 0.063× |
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

attn = ab.build("linear_elu")
out = attn(q, k, v, causal=True,
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

```text
LinearELUAttention(name='linear_elu', category='interaction', objective='modify', backend='reference')
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_hat, g_hat], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['Z_hat', 'g_hat']
```

## Others in `interaction`

[`cosformer`](cosformer.md), [`nystrom`](nystrom.md), [`performer`](performer.md), [`trigrff`](trigrff.md) — see [the taxonomy](../taxonomy.md#interaction).

## References

- **Katharopoulos et al. (2020).** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* Proceedings of the 37th International Conference on Machine Learning (ICML). [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)<br>`katharopoulos2020transformers`
- **Schlag et al. (2021).** *Linear Transformers Are Secretly Fast Weight Programmers.* Proceedings of the 38th International Conference on Machine Learning (ICML). [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)<br>`schlag2021linear`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/linear_elu.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/linear_elu.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
