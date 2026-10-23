# `flash`

**FlashAttention: tiled online softmax (`sec:exact`, `sec:exact-tiling`).**

category
:   [`exact`](../taxonomy.md#exact) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `accelerate` — same operator, exact up to floating point; only the evaluation changes

class
:   `attnbank.FlashAttention`, [`attnbank/methods/exact/flash.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/flash.py)

monograph
:   `sec:exact`

## What it changes

Changes nothing in $y(q)=\int v\,\mathrm{d}\pi_q$. The integrand, the measure and the normaliser are the reference operator's; only the order of arithmetic changes.

It pursues **acceleration**: the operator is the reference operator, exact up to floating point. The only question worth asking of it is *where* exactness stops.

## In the paper's notation

The operator is unchanged.  For a block `J` of key indices the partial state is

```text
sigma_J = (a_J, l_J, u_J),
a_J = max_{j in J} s_j,  l_J = sum_{j in J} e^{s_j - a_J},  u_J = sum_{j in J} e^{s_j - a_J} v_j,
```

with merge (`eq:exact-merge`)

```text
(a1,l1,u1) + (a2,l2,u2) = (a, e^{a1-a} l1 + e^{a2-a} l2, e^{a1-a} u1 + e^{a2-a} u2),
a = max(a1, a2),
```

readout `rd(a,l,u) = u/l` and identity `(-inf, 0, 0)`.  `prop:exact-monoid`
states that `(Sigma, +, 0)` is a commutative monoid and that the readout of the
fold over *any* partition is `y(q)`.  This file folds left-to-right over key
blocks in one pass, which is what a FlashAttention kernel does.

The monoid itself lives in `attnbank.components.online_softmax`, shared with
`splitk`; this file is only the decision to fold it left-to-right in one pass.

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
| `block_n` | `16` |
| `sdpa_kernel` | `None` |

```python
ab.describe("flash")     # the same keys, with everything else it declares
```

## Correctness anchor

This method is **exact as built**: the conformance suite instantiates it at its default config and requires both `y` and the weights to match `softmax`. `limit_config()` returns `{}` — the empty dict, not `None`: there is a limit and no setting has to be changed to reach it.

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |
| `sdpa` | `torch.nn.functional.scaled_dot_product_attention` -- on CUDA, FlashAttention-2 or the memory-efficient kernel |

Each non-`reference` entry is a claim that the path is **equal** to the reference, and `tests/test_conformance.py::test_backend_equivalence` checks it on five shapes, both causal settings, float32 and float64. See [Backends](../backends.md).

Diagnostics returned in `extras`: `lse`, `n_blocks`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    # Same arithmetic as dense; intermediate storage is one state per query.
    return {"flops": 2 * m * n * (d + dv),
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,435,456 | 268,435,456 | 1× |
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

attn = ab.build("flash", block_n=16)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = accelerate: the claim is equality with the reference, so that
# is what the example checks.
y_ref = ab.attend("softmax", q, k, v, causal=True)
print("max |y - y_softmax|:", float((out.y - y_ref).abs().max()))
```

```text
FlashAttention(name='flash', category='exact', objective='accelerate', backend='reference', block_n=16, sdpa_kernel=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[lse, n_blocks], dtype=float32, device=cpu)
invariant W: 4.172325134277344e-07
max |y - y_softmax|: 4.172325134277344e-07
```

## Others in `exact`

[`lowprec`](lowprec.md), [`softmax`](softmax.md), [`splitk`](splitk.md) — see [the taxonomy](../taxonomy.md#exact).

## References

- **Dao et al. (2022).** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)<br>`dao2022flashattention`
- **Dao (2024).** *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* International Conference on Learning Representations (ICLR). [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)<br>`dao2024flashattention2`
- **Milakov and Gimelshein (2018).** *Online normalizer calculation for softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)<br>`milakov2018online`
- **Rabe and Staats (2021).** *Self-attention Does Not Need $O(n^2)$ Memory.* [arXiv:2112.05682](https://arxiv.org/abs/2112.05682)<br>`rabe2021selfattention`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/exact/flash.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/flash.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
