# Longformer

<p class="attn-standfirst" markdown="span">A *dilated* sliding window plus task-global tokens.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `longformer` &middot; `ab.build("longformer")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.LongformerAttention` &middot; [`attnbank/methods/restrict/longformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/longformer.py)

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

**Exact limit.** Built with `w=None`, `dilation=1` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("longformer").limit_config
{'w': None, 'dilation': 1}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,359,296 | 268,435,456 | 0.00879× |
| `state_bytes` | 2,304 | 262,144 | 0.00879× |
| `bytes_moved` | 2,621,440 | 4,718,592 | 0.556× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    kept = min(n, (self.w + 1 if self.w is not None else n) + self.n_global)
    return {"flops": 2 * m * kept * (d + dv),
            "state_bytes": kept * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `dilation`, `eps`, `head_dilations`, `k_realized`, `n_global`, `support`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `w` | `8` |
| `dilation` | `1` |
| `n_global` | `0` |
| `head_dilation` | `False` |

</div>

`w`: Longformer's window size — `w//2` per side (`None` = unbounded).

`dilation`: gap between attended positions.  `head_dilation`: give head
`h` the dilation `1 + (h % dilation)`, so head 0 stays local while later
heads reach further — Longformer's per-head configuration.  `n_global`: the
number of leading positions that are globally attended *and* attend globally.

`ab.describe("longformer")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("longformer", w=8, dilation=1)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["longformer"], q, k, v, causal=True,
                 configs={"longformer": {'w': 8, 'dilation': 1}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
LongformerAttention(name='longformer', category='restrict', objective='approximate', backend='reference', dilation=1, head_dilation=False, n_global=0, w=8)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, dilation, eps, head_dilations, k_realized, n_global, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.91424, 'distributional_error': 2.91424, 'value_error': 0.0, 'tv': 0.86072}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:sparse-fixed</code></p>

Longformer's support is the union of three query-independent
pieces (Beltagy et al., 2020, Sec. 3):

* **sliding window** — "given a fixed window size `w`, each token attends to
  `w/2` tokens on each side";
* **dilated sliding window** — the window is given "gaps of size dilation `d`", so
  `l` stacked layers reach `l * d * w`.  The paper uses *different dilations on
  different heads*, "allowing some heads without dilation to focus on local context,
  while others with dilation focus on longer context"; `head_dilation=True`
  reproduces that;
* **global attention** — "a token with a global attention attends to all tokens
  across the sequence, and all tokens in the sequence attend to it".  The pattern is
  therefore *not* symmetric in the two directions of the restriction: a global
  position widens every row, and its own row is dense.

Written in the paper's notation, query `p` keeps

```text
S_p = { p + i*d : -w/2 <= i <= w/2 }  u  { 0, ..., n_global - 1 },
```

and `S_p` is the whole visible row when `p < n_global`.  Offset `i = 0` is always
present, so no row is ever empty; the support is then intersected with the causal
mask, which both removes the right half of the window (Longformer's own
autoregressive configuration) and makes the global prefix behave like the sinks of
`sink_window` — except that the global rows stay dense, which is the whole
difference between a *sink* and a *global token*.

The error is purely distributional: `y - y_hat = eps_p (m_{S^c} - m_S)`
(<span class="attn-ref">prop:sparse-identity</span>), with `eps_p` returned in `extras`.

Dilation alone (`w=None, dilation=d`) is the pure "strided column" pattern that
LongNet later scales up with exponentially growing dilation rates.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md) *(oracle)*, [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`
:   31 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `longformer` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `longformer` selected](index.md?compare=longformer) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Beltagy et al. (2020).** *Longformer: The Long-Document Transformer.* [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)<br><span class="attn-cite" data-bibkey="beltagy2020longformer"><code>beltagy2020longformer</code></span>
- **Child et al. (2019).** *Generating Long Sequences with Sparse Transformers.* [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)<br><span class="attn-cite" data-bibkey="child2019generating"><code>child2019generating</code></span>
- **Ding et al. (2023).** *LongNet: Scaling Transformers to 1,000,000,000 Tokens.* [arXiv:2307.02486](https://arxiv.org/abs/2307.02486)<br><span class="attn-cite" data-bibkey="ding2023longnet"><code>ding2023longnet</code></span>

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
<div class="attn-bib__entry" data-bibkey="ding2023longnet">
<pre class="attn-bib__src"><code>@misc{ding2023longnet,
  title = {{LongNet: Scaling Transformers to 1,000,000,000 Tokens}},
  author = {Jiayu Ding and Shuming Ma and Li Dong and Xingxing Zhang and Shaohan Huang and
            Wenhui Wang and Nanning Zheng and Furu Wei},
  year = {2023},
  howpublished = {arXiv preprint arXiv:2307.02486},
  eprint = {2307.02486},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/longformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/longformer.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
