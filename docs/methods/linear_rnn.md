# Linear attention as a recurrence

<p class="tx-regkey"><code>linear_rnn</code> &middot; <code>ab.build("linear_rnn")</code></p>

**Linear attention as a recurrence: H_t = H_{t-1} + psi(k_t) v_t^T.**

category
:   [`recurrent`](../taxonomy.md#recurrent) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.LinearRNNAttention`, [`attnbank/methods/recurrent/linear_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/linear_rnn.py)

monograph
:   `sec:memory-state`

## What it changes

Changes the operator. $y$ is produced by a finite-state recurrence over positions rather than by an integral against a measure — though it is still linear in $v$, which is why the weights can be recovered at all.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

## In the paper's notation

`sec:memory-state`, `prop:memory-suff`, `prop:memory-rank`.  A separable
interaction of rank `r` admits a sufficient statistic of size `r x d_v`: the prefix
moments `H_t = sum_{i<=t} psi(k_i) v_i^T` (and `z_t = sum psi(k_i)`) are all a
future query needs.  `prop:memory-rank` is the converse: an *additive* finite-state
memory forces the interaction to have finite rank, so recall is bounded by the state.

With the normalizer kept this is `linear_elu`.  The deployed form, and the one here,
drops it (`readout = H_t^T phi(q_t)`) because the denominator can approach zero and
produce unbounded gradients.  Three properties of `eq:attention_expectation` go with
it (`sec:memory-state`): normalization invariance under `g -> c(q) g`, constant
preservation (`y_hat = c <phi(q), z>` now scales with context length), and the
convex-hull bound.  Set `normalize=True` to put the denominator back and watch all
three return.

The weights on the original positions are `w_{tj} = <phi(q_t), psi(k_j)>` for
`j <= t`: nonnegative for a positive feature map, but *not* summing to one.  Invariant
W still holds exactly, which is why this category is in the library at all.

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
| `feature` | `'elu'` |
| `normalize` | `False` |

```python
ab.describe("linear_rnn")     # the same keys, with everything else it declares
```

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `mass`, `state_rank`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = d
    return {"flops": 2 * n * r * dv + 2 * m * r * dv,
            "state_bytes": r * dv * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 16,777,216 | 268,435,456 | 0.0625× |
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

attn = ab.build("linear_rnn", feature='elu', normalize=False)
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
LinearRNNAttention(name='linear_rnn', category='recurrent', objective='modify', backend='reference', feature='elu', normalize=False)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[mass, state_rank], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [43.946, 84.5266, 119.8822, 162.848]
extras: ['mass', 'state_rank']
```

## Others in `recurrent`

[`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md) — see [the taxonomy](../taxonomy.md#recurrent).

## References

- **Katharopoulos et al. (2020).** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* Proceedings of the 37th International Conference on Machine Learning (ICML). [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)<br>`katharopoulos2020transformers`
- **Schlag et al. (2021).** *Linear Transformers Are Secretly Fast Weight Programmers.* Proceedings of the 38th International Conference on Machine Learning (ICML). [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)<br>`schlag2021linear`
- **Sun et al. (2024).** *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)<br>`sun2024learning`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/linear_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/linear_rnn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
