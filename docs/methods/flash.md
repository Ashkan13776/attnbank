# FlashAttention

<p class="attn-standfirst" markdown="span">Tiled online softmax.</p>

<div class="attn-facts" data-cat="exact" markdown="1">

key
:   `flash` &middot; `ab.build("flash")`

category
:   [`exact`](../taxonomy.md#exact)

objective
:   `accelerate` — same operator, exact up to floating point; only the evaluation changes

class
:   `attnbank.FlashAttention` &middot; [`attnbank/methods/exact/flash.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/flash.py)

monograph
:   `sec:exact`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes nothing in $y(q)=\int v\,\mathrm{d}\pi_q$. The integrand, the measure and the normaliser are the reference operator's; only the order of arithmetic changes.

It pursues **acceleration**: the operator is the reference operator, exact up to floating point. The only question worth asking of it is *where* exactness stops.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="yes">yes</span> | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

This method is **exact as built**: the conformance suite instantiates it at its default config and requires both `y` and the weights to match `softmax`. `limit_config()` returns `{}` — the empty dict, not `None`: there is a limit and no setting has to be changed to reach it.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,435,456 | 268,435,456 | 1× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    # Same arithmetic as dense; intermediate storage is one state per query.
    return {"flops": 2 * m * n * (d + dv),
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

<div class="attn-tab" markdown="1">

| backend | what it is |
|---|---|
| `reference` | plain PyTorch — the correctness anchor and the default |
| `sdpa` | `torch.nn.functional.scaled_dot_product_attention` — on CUDA, FlashAttention-2 or the memory-efficient kernel |

</div>

Each non-`reference` entry is a claim that the path is **equal** to the reference, and `tests/test_conformance.py::test_backend_equivalence` checks it on five shapes, both causal settings, float32 and float64. See [Backends](../backends.md).

Diagnostics returned in `extras`: `lse`, `n_blocks`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `block_n` | `16` |
| `sdpa_kernel` | `None` |

</div>

`ab.describe("flash")` returns the same keys alongside everything else this page states.

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

<div class="attn-out" markdown="1">

```text
FlashAttention(name='flash', category='exact', objective='accelerate', backend='reference', block_n=16, sdpa_kernel=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[lse, n_blocks], dtype=float32, device=cpu)
invariant W: 4.172325134277344e-07
max |y - y_softmax|: 4.172325134277344e-07
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

The operator is unchanged.  For a block `J` of key indices the partial state is

```text
sigma_J = (a_J, l_J, u_J),
a_J = max_{j in J} s_j,  l_J = sum_{j in J} e^{s_j - a_J},  u_J = sum_{j in J} e^{s_j - a_J} v_j,
```

with merge (<span class="attn-ref">eq:exact-merge</span>)

```text
(a1,l1,u1) + (a2,l2,u2) = (a, e^{a1-a} l1 + e^{a2-a} l2, e^{a1-a} u1 + e^{a2-a} u2),
a = max(a1, a2),
```

readout `rd(a,l,u) = u/l` and identity `(-inf, 0, 0)`.  <span class="attn-ref">prop:exact-monoid</span>
states that `(Sigma, +, 0)` is a commutative monoid and that the readout of the
fold over *any* partition is `y(q)`.  This file folds left-to-right over key
blocks in one pass, which is what a FlashAttention kernel does.

The monoid itself lives in `attnbank.components.online_softmax`, shared with
`splitk`; this file is only the decision to fold it left-to-right in one pass.

</div>

## Related methods

changes the same object — [`exact`](../taxonomy.md#exact)
:   [`lowprec`](lowprec.md), [`softmax`](softmax.md), [`splitk`](splitk.md) &nbsp;·&nbsp; [browse the bank, exact only](index.md?category=exact)

Both come from `ab.available(...)` with `flash` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `flash` selected](index.md?compare=flash) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Dao et al. (2022).** *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)<br><span class="attn-cite" data-bibkey="dao2022flashattention"><code>dao2022flashattention</code></span>
- **Dao (2024).** *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* International Conference on Learning Representations (ICLR). [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)<br><span class="attn-cite" data-bibkey="dao2024flashattention2"><code>dao2024flashattention2</code></span>
- **Milakov and Gimelshein (2018).** *Online normalizer calculation for softmax.* [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)<br><span class="attn-cite" data-bibkey="milakov2018online"><code>milakov2018online</code></span>
- **Rabe and Staats (2021).** *Self-attention Does Not Need $O(n^2)$ Memory.* [arXiv:2112.05682](https://arxiv.org/abs/2112.05682)<br><span class="attn-cite" data-bibkey="rabe2021selfattention"><code>rabe2021selfattention</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="dao2022flashattention">
<pre class="attn-bib__src"><code>@inproceedings{dao2022flashattention,
  title     = {{FlashAttention}: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
  author    = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {35},
  year      = {2022},
  eprint    = {2205.14135},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="dao2024flashattention2">
<pre class="attn-bib__src"><code>@inproceedings{dao2024flashattention2,
  title={{FlashAttention-2}: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}, eprint={2307.08691}, archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="milakov2018online">
<pre class="attn-bib__src"><code>@misc{milakov2018online,
  title={Online normalizer calculation for softmax},
  author={Milakov, Maxim and Gimelshein, Natalia},
  year={2018}, eprint={1805.02867}, archivePrefix={arXiv}, primaryClass={cs.PF}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="rabe2021selfattention">
<pre class="attn-bib__src"><code>@misc{rabe2021selfattention,
  title={Self-attention Does Not Need $O(n^2)$ Memory},
  author={Rabe, Markus N. and Staats, Charles},
  year={2021}, eprint={2112.05682}, archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/exact/flash.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/flash.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
