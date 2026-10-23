# H2O

<p class="tx-regkey"><code>evict_h2o</code> &middot; <code>ab.build("evict_h2o")</code></p>

**H2O: accumulated-attention heavy-hitter eviction plus a recent window.**

category
:   [`compress`](../taxonomy.md#compress) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.H2OAttention`, [`attnbank/methods/compress/evict_h2o.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_h2o.py)

monograph
:   `sec:compress`

## What it changes

Changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:compress-evict`, `prop:compress-noeviction`.  Eviction is a restriction whose
set is chosen *before* the queries that will use it exist, and the sets are nested:
`S_{t+1} subset S_t union {t+1}`.  A deleted token is gone for every future query, so
`prop:compress-noeviction` says no query-oblivious rule has a uniform guarantee —
for any retained `S` and any evicted `j`, the query `q = tau k_j` drives
`eps_q -> 1` while `y_hat` stays in `conv{v_i : i in S}`.

H2O estimates the future-query distribution `nu` by the empirical distribution of
*past* queries: token `j` is scored by `sum_{t' <= t} pi_{q_{t'}}(j)`, and the
lowest-scoring non-recent token is dropped whenever the cache exceeds its budget.  This
file runs that process literally, one query at a time, which is why it is causal: the
state at step `t` depends on keys and queries up to `t` only.

In non-causal mode the whole context is admitted at the first step and the same
accumulate-and-evict loop runs over the queries in order; the semantics are then "a
streaming cache served by a fixed context".

Exact limit: `budget = None` (no eviction).

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
| `budget` | `8` |
| `recent` | `2` |

`budget`: cache size; `recent`: most recent positions never evicted.

```python
ab.describe("evict_h2o")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `budget=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("evict_h2o").limit_config
{'budget': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `accumulated`, `budget`, `chi2`, `eps`, `k_realized`, `recent`, `support`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    b = min(self.budget or n, n)
    return {"flops": 2 * m * b * (d + dv) + m * b,
            "state_bytes": b * (d + dv) * bytes_per_el + b * 4,   # +accumulated scores
            "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,105,344 | 268,435,456 | 0.00784× |
| `state_bytes` | 2,080 | 262,144 | 0.00793× |
| `bytes_moved` | 2,359,296 | 4,718,592 | 0.5× |

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

attn = ab.build("evict_h2o", budget=8, recent=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["evict_h2o"], q, k, v, causal=True,
                 configs={"evict_h2o": {'budget': 8, 'recent': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
H2OAttention(name='evict_h2o', category='compress', objective='approximate', backend='reference', budget=8, recent=2)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[accumulated, budget, chi2, eps, k_realized, recent, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.17848, 'distributional_error': 2.17848, 'value_error': 0.0, 'tv': 0.77822}
```

## Others in `compress`

[`evict_snapkv`](evict_snapkv.md), [`gqa`](gqa.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_kv`](merge_kv.md), [`mla`](mla.md), [`quantkv`](quantkv.md) — see [the taxonomy](../taxonomy.md#compress).

## References

- **Zhang et al. (2023).** *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)<br>`zhang2023h2o`
- **Liu et al. (2023).** *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2305.17118](https://arxiv.org/abs/2305.17118)<br>`liu2023scissorhands`
- **Oren et al. (2024).** *Transformers are Multi-State RNNs.* Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP). [arXiv:2401.06104](https://arxiv.org/abs/2401.06104)<br>`oren2024transformers`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/evict_h2o.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_h2o.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
