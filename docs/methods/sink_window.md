# StreamingLLM

<p class="attn-standfirst" markdown="span">The first `s` "attention sink" positions plus a sliding window.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `sink_window` &middot; `ab.build("sink_window")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.SinkWindowAttention` &middot; [`attnbank/methods/restrict/sink_window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/sink_window.py)

monograph
:   `sec:sparse`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

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

**Exact limit.** Built with `sinks=0`, `w=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("sink_window").limit_config
{'sinks': 0, 'w': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,621,440 | 268,435,456 | 0.00977× |
| `state_bytes` | 2,560 | 262,144 | 0.00977× |
| `bytes_moved` | 2,883,584 | 4,718,592 | 0.611× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    kept = (self.w or n) + self.sinks
    return {"flops": 2 * m * kept * (d + dv),
            "state_bytes": kept * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `k_realized`, `sinks`, `support`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `sinks` | `2` |
| `w` | `8` |

</div>

`ab.describe("sink_window")` returns the same keys alongside everything else this page states.

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

<div class="attn-out" markdown="1">

```text
SinkWindowAttention(name='sink_window', category='restrict', objective='approximate', backend='reference', sinks=2, w=8)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, k_realized, sinks, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.9121, 'distributional_error': 1.9121, 'value_error': 0.0, 'tv': 0.66426}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:sparse-fixed</code>, <code>sec:compress-evict</code>, <code>ex:sparse-sink</code></p>

The support is
`{0, ..., s-1} union {p-w+1, ..., p}`: content-independent, and therefore an eviction
policy as much as a sparsity pattern.

The reason it works is <span class="attn-ref">prop:sparse-identity</span> read backwards.  A head that routes
mass `eps` to a sink with a near-null value computes `y = (1-eps) m_S`; a pure
window outputs `m_S`, an amplification by `1/(1-eps)`.  Retaining the sink restores
the normalizer, not information: the sink carries *normalization*.  Comparing
`sink_window(s>0)` with `window` at the same total budget isolates that effect,
which is what makes the pair worth having in the library.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`strided`](strided.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`
:   31 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `sink_window` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `sink_window` selected](index.md?compare=sink_window) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Xiao et al. (2024).** *Efficient Streaming Language Models with Attention Sinks.* International Conference on Learning Representations (ICLR). [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)<br><span class="attn-cite" data-bibkey="xiao2024efficient"><code>xiao2024efficient</code></span>
- **Han et al. (2024).** *LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models.* Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL). [arXiv:2308.16137](https://arxiv.org/abs/2308.16137)<br><span class="attn-cite" data-bibkey="han2024lminfinite"><code>han2024lminfinite</code></span>

</div>

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

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/sink_window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/sink_window.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
