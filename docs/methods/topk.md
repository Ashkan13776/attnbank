# Oracle top-k restriction

<p class="tx-regkey"><code>topk</code> &middot; <code>ab.build("topk")</code></p>

**Oracle top-k restriction: the `k` largest true scores, renormalized.**

category
:   [`restrict`](../taxonomy.md#restrict) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.TopKAttention`, [`attnbank/methods/restrict/topk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/topk.py)

monograph
:   `sec:sparse`

## What it changes

Changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:sparse-operator`, `sec:sparse-query`.  `S_q` is the set of the `k` largest
exact attention weights, which by `prop:error-topk` minimizes the omitted mass
`eps_q` — and hence the bound `eps_q osc(v)` — over all supports of size `k`.
It is the **oracle** against which every proxy selector (Quest, SparQ, Loki, H2O) is
measured: computing all scores saves only the value aggregation.

The proposition's second half matters more for this library: minimizing `eps_q` is
*not* minimizing `||y - y_hat||`.  With `pi = (0.45, 0.35, 0.20)` and scalar values
`(0, 0, 1)`, top-2 retains `{1,2}` and errs by `0.2` while `{1,3}` omits more
mass and errs by `0.11`.  `value_topk` is the output-aware alternative; the gap
between them is the quantity study S2 reports.

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
| `k` | `8` |

`k`: budget; `None` keeps every visible key (the exact limit).

```python
ab.describe("topk")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `k=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("topk").limit_config
{'k': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `k_realized`, `scores`, `support`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    kk = self.k or n
    return {"flops": 2 * m * n * d + 2 * m * kk * dv + m * n,
            "state_bytes": n * (d + dv) * bytes_per_el,     # full context stays resident
            "bytes_moved": (m * d + n * d + m * kk * dv + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 136,314,880 | 268,435,456 | 0.508× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 1,441,792 | 4,718,592 | 0.306× |

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

attn = ab.build("topk", k=8)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["topk"], q, k, v, causal=True,
                 configs={"topk": {'k': 8}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
TopKAttention(name='topk', category='restrict', objective='approximate', backend='reference', k=8)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, k_realized, scores, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.98508, 'distributional_error': 0.98508, 'value_error': 0.0, 'tv': 0.39513}
```

## Others in `restrict`

[`blocktopk`](blocktopk.md), [`lsh_sample`](lsh_sample.md), [`sink_window`](sink_window.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) — see [the taxonomy](../taxonomy.md#restrict).

## References

- **Gupta et al. (2021).** *Memory-efficient Transformers via Top-k Attention.* Proceedings of the Second Workshop on Simple and Efficient Natural Language Processing (SustaiNLP). [arXiv:2106.06899](https://arxiv.org/abs/2106.06899)<br>`gupta2021memory`
- **Tang et al. (2024).** *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)<br>`tang2024quest`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/topk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/topk.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
