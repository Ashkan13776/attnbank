# Low-precision attention

<p class="tx-regkey"><code>lowprec</code> &middot; <code>ab.build("lowprec")</code></p>

**Low-precision attention: simulated FP8 / INT8 logits and values.**

category
:   [`exact`](../taxonomy.md#exact) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.LowPrecisionAttention`, [`attnbank/methods/exact/lowprec.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/lowprec.py)

monograph
:   `sec:exact-boundary`

## What it changes

Changes nothing in $y(q)=\int v\,\mathrm{d}\pi_q$. The integrand, the measure and the normaliser are the reference operator's; only the order of arithmetic changes.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:exact-boundary`.  This is the method that marks *where exactness ends*.  An
FP8/INT8 kernel keeps the evaluation order of `prop:exact-monoid` but stores the
operands at a precision `u' >> u`, so it satisfies the first clause of
`def:exact-scheme` and fails the second.  By the criterion of "which object is
changed" it belongs with context compression: quantizing `q, k` replaces `g` by
`g_hat` (a distributional error) and quantizing `v` or `P` replaces `v` by
`v_hat` (a value-representation error).  Its objective is therefore `approximate`,
not `accelerate`, which is the whole point of putting it in `exact/`.

Simulation is quantize–dequantize in the kernel, per attention head and per tensor:

```text
Q_b(x) = clamp(round(x / Delta), -2^{b-1}, 2^{b-1} - 1) * Delta,  Delta = max|x| / (2^{b-1} - 1)
```

for signed INT-b, and round-to-nearest on the FP8 E4M3 / E5M2 grids.  Optional
*incoherent processing* (`hadamard=True`) multiplies `q` and `k` by a shared
random orthogonal Hadamard matrix before quantizing, which is exact in infinite
precision (`<Rq, Rk> = <q, k>`) and spreads outlier coordinates, as in
FlashAttention-3's FP8 path.  Optional *smoothing* (`smooth_k=True`) subtracts the
token mean of `k`, which shifts every logit of a query by one constant and so
leaves `pi_q` exactly invariant (SageAttention).

The quantizers themselves live in `attnbank.components.quantization`, shared with
`quantkv`: one implementation, one place where the rounding rule is argued about.

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
| `bits` | `8` |
| `fmt` | `None` |
| `bits_p` | `None` |
| `hadamard_qk` | `False` |
| `smooth_k` | `False` |
| `seed` | `0` |

`bits`: INT width for q, k, v (`None` = no quantization, the exact limit).
`fmt`: `'e4m3'` / `'e5m2'` to use FP8 instead of INT.
`bits_p`: optional INT width for the attention weights themselves.

```python
ab.describe("lowprec")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `bits=None`, `fmt=None`, `bits_p=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("lowprec").limit_config
{'bits': None, 'fmt': None, 'bits_p': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** The method exists to expose the *intermediate* states of the evaluation, which a single fused call does not hand back.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `delta_v`, `s_hat`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    b = bytes_per_el if self.bits is None and self.fmt is None else \
        max(1, (8 if self.fmt else self.bits) // 8)
    return {"flops": 2 * m * n * (d + dv), "state_bytes": n * (d + dv) * b,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * b}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,435,456 | 268,435,456 | 1× |
| `state_bytes` | 131,072 | 262,144 | 0.5× |
| `bytes_moved` | 262,144 | 4,718,592 | 0.0556× |

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

attn = ab.build("lowprec", bits=8, hadamard_qk=False)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["lowprec"], q, k, v, causal=True,
                 configs={"lowprec": {'bits': 8, 'hadamard_qk': False}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
LowPrecisionAttention(name='lowprec', category='exact', objective='approximate', backend='reference', bits=8, bits_p=None, fmt=None, hadamard_qk=False, seed=0, smooth_k=False)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[delta_v, s_hat], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.01213, 'distributional_error': 0.0085, 'value_error': 0.0074, 'tv': 0.00273}
```

## Others in `exact`

[`flash`](flash.md), [`softmax`](softmax.md), [`splitk`](splitk.md) — see [the taxonomy](../taxonomy.md#exact).

## References

- **Shah et al. (2024).** *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)<br>`shah2024flashattention3`
- **Zhang et al. (2025).** *SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration.* International Conference on Learning Representations (ICLR). [arXiv:2410.02367](https://arxiv.org/abs/2410.02367)<br>`zhang2025sageattention`
- **Micikevicius et al. (2022).** *FP8 Formats for Deep Learning.* [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)<br>`micikevicius2022fp8`
- **Chen et al. (2024).** *INT-FlashAttention: Enabling Flash Attention for INT8 Quantization.* [arXiv:2409.16997](https://arxiv.org/abs/2409.16997)<br>`chen2024intflashattention`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/exact/lowprec.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/lowprec.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
