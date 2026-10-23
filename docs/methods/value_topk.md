# `value_topk`

!!! danger "Oracle — an upper bound, not a method"
    `value_topk` is not deployable and is not meant to be. It uses information no
    inference-time method has, and it is in the bank to answer *how well could this
    ever work?* — several of the monograph's arguments need that number. Read every
    result below as a bound on a family of real methods, never as a baseline you
    could ship.

**Output-aware selection: the support that minimizes output error, chosen greedily.**

category
:   [`restrict`](../taxonomy.md#restrict) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.ValueTopKAttention`, [`attnbank/methods/restrict/value_topk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/value_topk.py)

monograph
:   `sec:sparse`

## What it changes

Changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:sparse-operator` ("Score-optimal is not output-optimal"), `prop:error-topk`,
and the open problem of `sec:open-retrieval`.  Top-k minimizes the omitted mass
`eps_q` and therefore the *bound* `eps_q osc(v)`; it does not minimize the error.
The right target, named in the text, is

```text
min_{|S| <= k}  eps_S ||m_{S^c} - m_S||.
```

That objective simplifies.  For a renormalized restriction `y_hat = m_S`, so with
`W_S = pi(S)` and `U_S = sum_{j in S} pi_j v_j`,

```text
eps_S ||m_{S^c} - m_S|| = ||y - m_S|| = ||y - U_S / W_S||,
```

i.e. *the objective is just the realised output error*, and no conditional mean of the
complement needs to be formed.

Selection.  Forward greedy alone is badly myopic here: its first pick minimizes
`||y - v_j||` and so can seed the support with a light outlier that later additions
cannot undo.  The selector therefore evaluates four cheap candidate supports and keeps,
*per query*, whichever realises the smallest error:

  1. score top-k (so the result is **never worse than** `topk`, by construction);
  2. forward greedy from the empty set;
  3. forward greedy seeded with the single heaviest position;
  4. backward greedy from the full set, deleting the position whose removal costs least
     — the elementary step of every eviction policy, and exactly `prop:error-single`.

Complexity.  Each greedy step scores all `n` candidates, and a candidate is scored in
`O(d_v)` from the running `(U_S, W_S)` rather than by recomputing the conditional
mean, so a forward pass is `O(n k d_v)` per query and the backward pass
`O(n (n - k) d_v)` — the naive form of either is `O(n^2 k)`.  All queries are
processed in parallel, at a peak memory of `B H m n d_v`; pass `chunk` to process
the query axis in slices if that is too large.

None of this is optimal: `exhaustive=True` enumerates every support of size `k`
(guarded by `max_subsets`) and gives the true minimum, which is what
`tests/test_restrict.py` uses to bound the greedy gap.

This is an **oracle**: it reads `v` and the exact `pi_q`, which no deployed
selector can.  Its purpose is to upper-bound what output-aware selection could buy, so
correctness matters and speed does not.

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
| `exhaustive` | `False` |
| `max_subsets` | `200000` |
| `chunk` | `None` |

```python
ab.describe("value_topk")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `k=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("value_topk").limit_config
{'k': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `k_realized`, `selector`, `support`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    kk = self.k or n
    return {"flops": 2 * m * n * d + 2 * m * n * kk * dv + 2 * m * kk * dv,
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 1,209,008,128 | 268,435,456 | 4.5× |
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

attn = ab.build("value_topk", k=8, exhaustive=False)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["value_topk"], q, k, v, causal=True,
                 configs={"value_topk": {'k': 8, 'exhaustive': False}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
ValueTopKAttention(name='value_topk', category='restrict', objective='approximate', backend='reference', chunk=None, exhaustive=False, k=8, max_subsets=200000)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, k_realized, selector, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.82577, 'distributional_error': 0.82577, 'value_error': 0.0, 'tv': 0.42706}
```

## Others in `restrict`

[`blocktopk`](blocktopk.md), [`lsh_sample`](lsh_sample.md), [`sink_window`](sink_window.md), [`topk`](topk.md), [`window`](window.md) — see [the taxonomy](../taxonomy.md#restrict).

## References

- **Ribar et al. (2024).** *SparQ Attention: Bandwidth-Efficient LLM Inference.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2312.04985](https://arxiv.org/abs/2312.04985)<br>`ribar2024sparq`
- **Feng et al. (2025).** *CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective.* [arXiv:2502.03805](https://arxiv.org/abs/2502.03805)<br>`feng2025criticalkv`
- **Gupta et al. (2021).** *Memory-efficient Transformers via Top-k Attention.* Proceedings of the Second Workshop on Simple and Efficient Natural Language Processing (SustaiNLP). [arXiv:2106.06899](https://arxiv.org/abs/2106.06899)<br>`gupta2021memory`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/value_topk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/value_topk.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
