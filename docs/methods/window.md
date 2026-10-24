# Sliding-window (banded) attention

<p class="attn-standfirst" markdown="span">A query-position-dependent, content-free support.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `window` &middot; `ab.build("window")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.WindowAttention` &middot; [`attnbank/methods/restrict/window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/window.py)

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

**Exact limit.** Built with `w=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("window").limit_config
{'w': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,097,152 | 268,435,456 | 0.00781× |
| `state_bytes` | 2,048 | 262,144 | 0.00781× |
| `bytes_moved` | 2,359,296 | 4,718,592 | 0.5× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    w = self.w or n
    return {"flops": 2 * m * w * (d + dv),
            "state_bytes": w * (d + dv) * bytes_per_el,     # rolling buffer
            "bytes_moved": (m * d + m * w * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `k_realized`, `support`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `w` | `8` |

</div>

`w`: window width; `None` is the full context (the exact limit).

`ab.describe("window")` returns the same keys alongside everything else this page states.

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

<div class="attn-out" markdown="1">

```text
WindowAttention(name='window', category='restrict', objective='approximate', backend='reference', w=8)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, k_realized, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.23291, 'distributional_error': 2.23291, 'value_error': 0.0, 'tv': 0.74132}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:sparse-fixed</code></p>

Query at absolute position `p` attends to
`{p - w + 1, ..., p}`, intersected with what the mask allows.  The diagonal is always
included, so no row is ever empty.

Applied to a dense-trained model this is an approximation whose error is exactly
`eps_q (m_{S^c} - m_S)` (<span class="attn-ref">prop:sparse-identity</span>); it is the cleanest demonstration
of the sink effect of <span class="attn-ref">ex:sparse-sink</span>, because once the window has scrolled past
position 0 it omits a large mass whose conditional mean is the sink value, and
renormalization inflates the remaining weights by `1/(1 - eps_q)`.  Compare with
`sink_window`, which differs only by the first `s` positions.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md) *(oracle)*, [`value_topk`](value_topk.md) *(oracle)* &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`
:   31 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `window` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `window` selected](index.md?compare=window) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Beltagy et al. (2020).** *Longformer: The Long-Document Transformer.* [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)<br><span class="attn-cite" data-bibkey="beltagy2020longformer"><code>beltagy2020longformer</code></span>
- **Jiang et al. (2023).** *Mistral 7B.* [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)<br><span class="attn-cite" data-bibkey="jiang2023mistral"><code>jiang2023mistral</code></span>
- **Child et al. (2019).** *Generating Long Sequences with Sparse Transformers.* [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)<br><span class="attn-cite" data-bibkey="child2019generating"><code>child2019generating</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="beltagy2020longformer">
<pre class="attn-bib__src"><code>@misc{beltagy2020longformer,
  title = {{Longformer: The Long-Document Transformer}},
  author = {Beltagy, Iz and Peters, Matthew E. and Cohan, Arman},
  year = {2020},
  howpublished = {arXiv preprint arXiv:2004.05150},
  eprint = {2004.05150},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="jiang2023mistral">
<pre class="attn-bib__src"><code>@misc{jiang2023mistral,
  title = {{Mistral 7B}},
  author = {Jiang, Albert Q. and Sablayrolles, Alexandre and Mensch, Arthur and Bamford, Chris and Chaplot, Devendra Singh and Casas, Diego de las and Bressand, Florian and Lengyel, Gianna and Lample, Guillaume and Saulnier, Lucile and others},
  year = {2023},
  howpublished = {arXiv preprint arXiv:2310.06825},
  eprint = {2310.06825},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="child2019generating">
<pre class="attn-bib__src"><code>@misc{child2019generating,
  title = {{Generating Long Sequences with Sparse Transformers}},
  author = {Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
  year = {2019},
  howpublished = {arXiv preprint arXiv:1904.10509},
  eprint = {1904.10509},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/window.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
