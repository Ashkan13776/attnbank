# Quest-style page / block selection

<p class="tx-regkey"><code>blocktopk</code> &middot; <code>ab.build("blocktopk")</code></p>

**Quest-style page / block selection: restriction at block granularity.**

category
:   [`restrict`](../taxonomy.md#restrict) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.BlockTopKAttention`, [`attnbank/methods/restrict/blocktopk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/blocktopk.py)

monograph
:   `sec:sparse`

## What it changes

Changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:sparse-query` ("Blocks and pages") and `prop:sparse-certificate`.  Keys are
partitioned into contiguous pages of `block` positions.  A page is ranked by an upper
bound on the scores it can contain,

```text
tau(B) = beta * sum_c max( q_c * kmin_{B,c},  q_c * kmax_{B,c} ),
```

from the elementwise minima and maxima of the keys in the page (Tang et al., 2024).
The top `n_blocks` pages are fetched and attended *exactly*; everything else is
omitted, so the error is again `eps_q (m_{S^c} - m_S)`.

The point of having it alongside `topk` is granularity: a page-level selector can
only ever omit or retain whole pages, so its realized `eps_q` exceeds the top-k
minimum by a selection error.  `score="max"` and `score="mean"` are provided to
separate the cost of the *bound* from the cost of the *granularity*: the paper notes
that an upper bound never misses a page containing a heavy key but may waste budget,
whereas a pooled representative can average a heavy key away.

Causality.  Page statistics are computed over whole pages, so only *complete* past
pages may be ranked.  The page containing the query's own position is always selected
(as in Quest, where the current page is resident anyway) and later pages are excluded.
This makes the support depend on nothing after the query.

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
| `block` | `4` |
| `n_blocks` | `2` |
| `score` | `'bound'` |

`score`: `'bound'` (Quest), `'max'` (exact page maximum) or `'mean'`.

```python
ab.describe("blocktopk")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `n_blocks=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("blocktopk").limit_config
{'n_blocks': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `k_realized`, `n_pages`, `page_scores`, `page_size`, `pages_selected`, `support`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    nb = (n + self.block - 1) // self.block
    kept = min(self.n_blocks or nb, nb) * self.block
    return {"flops": 2 * m * nb * d + 2 * m * kept * (d + dv),
            "state_bytes": n * (d + dv) * bytes_per_el + 2 * nb * d * bytes_per_el,
            "bytes_moved": (m * d + nb * 2 * d + m * kept * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 35,651,584 | 268,435,456 | 0.133× |
| `state_bytes` | 327,680 | 262,144 | 1.25× |
| `bytes_moved` | 2,424,832 | 4,718,592 | 0.514× |

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

attn = ab.build("blocktopk", block=4, n_blocks=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["blocktopk"], q, k, v, causal=True,
                 configs={"blocktopk": {'block': 4, 'n_blocks': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
BlockTopKAttention(name='blocktopk', category='restrict', objective='approximate', backend='reference', block=4, n_blocks=2, score='bound')
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, k_realized, n_pages, page_scores, page_size, pages_selected, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.53824, 'distributional_error': 1.53824, 'value_error': 0.0, 'tv': 0.5296}
```

## Others in `restrict`

[`lsh_sample`](lsh_sample.md), [`sink_window`](sink_window.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) — see [the taxonomy](../taxonomy.md#restrict).

## References

- **Tang et al. (2024).** *Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2406.10774](https://arxiv.org/abs/2406.10774)<br>`tang2024quest`
- **Xiao et al. (2024).** *InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory.* Advances in Neural Information Processing Systems (NeurIPS). [doi:10.52202/079017-3801](https://doi.org/10.52202/079017-3801)<br>`xiao2024infllm`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/blocktopk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/blocktopk.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
