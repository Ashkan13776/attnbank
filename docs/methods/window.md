# `window`

**Sliding-window (banded) attention: a query-position-dependent, content-free support.**

category
:   [`restrict`](../taxonomy.md#restrict) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.WindowAttention`, [`attnbank/methods/restrict/window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/window.py)

monograph
:   `sec:sparse`

## What it changes

Changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:sparse-fixed`.  Query at absolute position `p` attends to
`{p - w + 1, ..., p}`, intersected with what the mask allows.  The diagonal is always
included, so no row is ever empty.

Applied to a dense-trained model this is an approximation whose error is exactly
`eps_q (m_{S^c} - m_S)` (`prop:sparse-identity`); it is the cleanest demonstration
of the sink effect of `ex:sparse-sink`, because once the window has scrolled past
position 0 it omits a large mass whose conditional mean is the sink value, and
renormalization inflates the remaining weights by `1/(1 - eps_q)`.  Compare with
`sink_window`, which differs only by the first `s` positions.

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
| `w` | `8` |

`w`: window width; `None` is the full context (the exact limit).

```python
ab.describe("window")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `w=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("window").limit_config
{'w': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `k_realized`, `support`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    w = self.w or n
    return {"flops": 2 * m * w * (d + dv),
            "state_bytes": w * (d + dv) * bytes_per_el,     # rolling buffer
            "bytes_moved": (m * d + m * w * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,097,152 | 268,435,456 | 0.00781× |
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

attn = ab.build("window", w=8)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["window"], q, k, v, causal=True,
                 configs={"window": {'w': 8}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
WindowAttention(name='window', category='restrict', objective='approximate', backend='reference', w=8)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, k_realized, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.23291, 'distributional_error': 2.23291, 'value_error': 0.0, 'tv': 0.74132}
```

## Others in `restrict`

[`blocktopk`](blocktopk.md), [`lsh_sample`](lsh_sample.md), [`sink_window`](sink_window.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)* — see [the taxonomy](../taxonomy.md#restrict).

## References

- **Beltagy et al. (2020).** *Longformer: The Long-Document Transformer.* [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)<br>`beltagy2020longformer`
- **Jiang et al. (2023).** *Mistral 7B.* [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)<br>`jiang2023mistral`
- **Child et al. (2019).** *Generating Long Sequences with Sparse Transformers.* [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)<br>`child2019generating`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/window.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
