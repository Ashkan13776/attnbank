# StreamingLLM

<p class="tx-regkey"><code>sink_window</code> &middot; <code>ab.build("sink_window")</code></p>

**StreamingLLM: the first `s` "attention sink" positions plus a sliding window.**

category
:   [`restrict`](../taxonomy.md#restrict) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.SinkWindowAttention`, [`attnbank/methods/restrict/sink_window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/sink_window.py)

monograph
:   `sec:sparse`

## What it changes

Changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:sparse-fixed`, `sec:compress-evict`, `ex:sparse-sink`.  The support is
`{0, ..., s-1} union {p-w+1, ..., p}`: content-independent, and therefore an eviction
policy as much as a sparsity pattern.

The reason it works is `prop:sparse-identity` read backwards.  A head that routes
mass `eps` to a sink with a near-null value computes `y = (1-eps) m_S`; a pure
window outputs `m_S`, an amplification by `1/(1-eps)`.  Retaining the sink restores
the normalizer, not information: the sink carries *normalization*.  Comparing
`sink_window(s>0)` with `window` at the same total budget isolates that effect,
which is what makes the pair worth having in the library.

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
| `sinks` | `2` |
| `w` | `8` |

```python
ab.describe("sink_window")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `sinks=0`, `w=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("sink_window").limit_config
{'sinks': 0, 'w': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `k_realized`, `sinks`, `support`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    kept = (self.w or n) + self.sinks
    return {"flops": 2 * m * kept * (d + dv),
            "state_bytes": kept * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,621,440 | 268,435,456 | 0.00977× |
| `state_bytes` | 2,560 | 262,144 | 0.00977× |
| `bytes_moved` | 2,883,584 | 4,718,592 | 0.611× |

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

attn = ab.build("sink_window", sinks=2, w=8)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["sink_window"], q, k, v, causal=True,
                 configs={"sink_window": {'sinks': 2, 'w': 8}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
SinkWindowAttention(name='sink_window', category='restrict', objective='approximate', backend='reference', sinks=2, w=8)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, k_realized, sinks, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.9121, 'distributional_error': 1.9121, 'value_error': 0.0, 'tv': 0.66426}
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `sink_window` removed.

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`blocktopk`](blocktopk.md), [`lsh_sample`](lsh_sample.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`, beyond the ones above
:   [`clustered`](clustered.md), [`evict_h2o`](evict_h2o.md), [`evict_snapkv`](evict_snapkv.md), [`linformer`](linformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`merge_kv`](merge_kv.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`quantkv`](quantkv.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

[Open the browser with `sink_window` selected](index.md?compare=sink_window) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **Xiao et al. (2024).** *Efficient Streaming Language Models with Attention Sinks.* International Conference on Learning Representations (ICLR). [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)<br><span class="attn-cite" data-bibkey="xiao2024efficient"><code>xiao2024efficient</code></span>
- **Han et al. (2024).** *LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models.* Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL). [arXiv:2308.16137](https://arxiv.org/abs/2308.16137)<br><span class="attn-cite" data-bibkey="han2024lminfinite"><code>han2024lminfinite</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="xiao2024efficient">
<pre class="attn-bib__src"><code>@inproceedings{xiao2024efficient,
  title = {{Efficient Streaming Language Models with Attention Sinks}},
  author = {Guangxuan Xiao and Yuandong Tian and Beidi Chen and Song Han and Mike Lewis},
  year = {2024},
  booktitle = {International Conference on Learning Representations (ICLR)},
  eprint = {2309.17453},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="han2024lminfinite">
<pre class="attn-bib__src"><code>@inproceedings{han2024lminfinite,
  title = {{LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models}},
  author = {Han, Chi and Wang, Qifan and Peng, Hao and Xiong, Wenhan and Chen, Yu and Ji, Heng and Wang, Sinong},
  year = {2024},
  booktitle = {Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)},
  eprint = {2308.16137},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/sink_window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/sink_window.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
