# `evict_snapkv`

**SnapKV: prompt-token scores pooled over an observation window.**

category
:   [`compress`](../taxonomy.md#compress) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.SnapKVAttention`, [`attnbank/methods/compress/evict_snapkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_snapkv.py)

monograph
:   `sec:compress`

## What it changes

Changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:compress-evict` ("Estimating importance from past queries").  SnapKV scores
prompt tokens with the attention of the *last* `obs` prompt queries, applies a 1-D
pooling of width `pool` along the key axis so that whole local neighbourhoods are
kept rather than isolated tokens, keeps the top `budget - recent` of them, and adds
the recent window.  The compressed cache then serves all subsequent decoding queries.

This works when the question ends the prompt — the observation window is then a good
sample of `nu` in `eq:compress-evict-objective` — and degrades when it arrives
earlier.

**Not causal, deliberately.**  The support is chosen after seeing the whole prompt and
is then applied to every query, including earlier ones.  That is what SnapKV does in a
prefill-then-decode pipeline, and it cannot be expressed as a causally masked single
forward pass: query 0's support would depend on query `m-1`.  Declaring
`supports_causal = False` records the fact rather than hiding it behind a mask.

Exact limit: `budget = None`.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **no** | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Configuration

| parameter | default |
|---|---|
| `budget` | `8` |
| `obs` | `4` |
| `pool` | `3` |
| `recent` | `2` |

```python
ab.describe("evict_snapkv")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `budget=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("evict_snapkv").limit_config
{'budget': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `budget`, `chi2`, `eps`, `k_realized`, `obs`, `pooled_scores`, `support`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    b = min(self.budget or n, n)
    return {"flops": 2 * m * b * (d + dv) + self.obs * n * d,
            "state_bytes": b * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,359,296 | 268,435,456 | 0.00879× |
| `state_bytes` | 2,048 | 262,144 | 0.00781× |
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

attn = ab.build("evict_snapkv", budget=8, obs=4)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["evict_snapkv"], q, k, v, causal=False,
                 configs={"evict_snapkv": {'budget': 8, 'obs': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
SnapKVAttention(name='evict_snapkv', category='compress', objective='approximate', backend='reference', budget=8, obs=4, pool=3, recent=2)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[budget, chi2, eps, k_realized, obs, pooled_scores, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.30297, 'distributional_error': 2.30297, 'value_error': 0.0, 'tv': 0.87141}
```

## Others in `compress`

[`evict_h2o`](evict_h2o.md), [`gqa`](gqa.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_kv`](merge_kv.md), [`mla`](mla.md), [`quantkv`](quantkv.md) — see [the taxonomy](../taxonomy.md#compress).

## References

- **Li et al. (2024).** *SnapKV: LLM Knows What You are Looking for Before Generation.* Advances in Neural Information Processing Systems (NeurIPS). [doi:10.52202/079017-0722](https://doi.org/10.52202/079017-0722)<br>`li2024snapkv`
- **Cai et al. (2024).** *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling.* [arXiv:2406.02069](https://arxiv.org/abs/2406.02069)<br>`cai2024pyramidkv`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/evict_snapkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_snapkv.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
