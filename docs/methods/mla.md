# `mla`

**Multi-head latent attention: one joint low-rank KV latent, per-head up-projection.**

category
:   [`compress`](../taxonomy.md#compress) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.MLAAttention`, [`attnbank/methods/compress/mla.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/mla.py)

monograph
:   `sec:compress`

## What it changes

Changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

## In the paper's notation

`sec:compress-arch`.  Instead of caching `h` key–value pairs per position, MLA
caches a single latent `c_j = W_down [k_j^{(1)}; ...; k_j^{(h)}; v_j^{(1)}; ...]` of
dimension `r`, and reconstructs each head's key and value by a per-head
up-projection.  The cached state per position drops from `h (d + d_v)` to `r`.

Like GQA this is an **operator modification** trained as such; it is in the library for
the roofline (it is the other intervention that raises decode arithmetic intensity) and
because it is the cleanest example of a compression that couples heads: the latent is
shared, so `head_independent = False`.

The projections are *trained* in DeepSeek-V2.  Here `W_down` is a fixed seeded
Gaussian and the up-projection is its pseudo-inverse, so the reconstruction is the
orthogonal projection onto `row(W_down)`.  That choice is data-independent, hence
causal, and makes the exact limit provable: at `r = h (d + d_v)` the matrix is square
and invertible, the reconstruction is exact, and the operator is dense softmax.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **yes** | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **no** | it shares $K$/$V$ across query heads, so heads are not independent. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Configuration

| parameter | default |
|---|---|
| `r` | `16` |
| `seed` | `0` |
| `sdpa_kernel` | `None` |

`r`: latent width; `None` uses the full width (the exact limit).

```python
ab.describe("mla")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("mla").limit_config
{'r': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |
| `sdpa` | `torch.nn.functional.scaled_dot_product_attention` -- on CUDA, FlashAttention-2 or the memory-efficient kernel |

Each non-`reference` entry is a claim that the path is **equal** to the reference, and `tests/test_conformance.py::test_backend_equivalence` checks it on five shapes, both causal settings, float32 and float64. See [Backends](../backends.md).

Diagnostics returned in `extras`: `latent`, `latent_width`, `r`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2, heads: int = 1):
    wide = heads * (d + dv)
    r = min(self.r or wide, wide)
    per_head = n * r * bytes_per_el / max(1, heads)
    return {"flops": 2 * m * n * (d + dv) + 2 * n * r * (d + dv),
            "state_bytes": per_head,
            "bytes_moved": m * (d + dv) * bytes_per_el + per_head}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 272,629,760 | 268,435,456 | 1.02× |
| `state_bytes` | 32,768 | 262,144 | 0.125× |
| `bytes_moved` | 294,912 | 4,718,592 | 0.0625× |

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

attn = ab.build("mla", r=16)
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
MLAAttention(name='mla', category='compress', objective='modify', backend='reference', r=16, sdpa_kernel=None, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[latent, latent_width, r], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['latent', 'latent_width', 'r']
```

## Others in `compress`

[`evict_h2o`](evict_h2o.md), [`evict_snapkv`](evict_snapkv.md), [`gqa`](gqa.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_kv`](merge_kv.md), [`quantkv`](quantkv.md) — see [the taxonomy](../taxonomy.md#compress).

## References

- **DeepSeek-AI (2024).** *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)<br>`deepseekai2024deepseekv2`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/mla.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/mla.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
