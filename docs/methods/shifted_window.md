# Swin-style shifted local windows

<p class="attn-standfirst" markdown="span">A *block-diagonal* support, optionally displaced.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `shifted_window` &middot; `ab.build("shifted_window")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.ShiftedWindowAttention` &middot; [`attnbank/methods/restrict/shifted_window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/shifted_window.py)

monograph
:   `sec:sparse`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

</div>

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

## Correctness anchor

**Exact limit.** Built with `w=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("shifted_window").limit_config
{'w': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 1,048,576 | 268,435,456 | 0.00391× |
| `state_bytes` | 1,024 | 262,144 | 0.00391× |
| `bytes_moved` | 1,310,720 | 4,718,592 | 0.278× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    kept = min(n, self.w or n)
    return {"flops": 2 * m * kept * (d + dv),
            "state_bytes": kept * (d + dv) * bytes_per_el,   # one resident window
            "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}
```

!!! note "A cost model is not a benchmark"
    These are counted operations and resident bytes, not measured time.
    They are what the monograph's capacity arguments are made of;
    `examples/08_cost_and_capacity.py` walks through the comparison properly.

## Backends

<div class="attn-tab" markdown="1">

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

</div>

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `k_realized`, `n_windows`, `shift`, `support`, `window_offset`, `window_size`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `w` | `4` |
| `shift` | `None` |

</div>

`w`: window size (`None` = one window, the exact limit).

`shift`: displacement of the partition; `None` means Swin's
`floor(w/2)`, `0` is the regular (block-diagonal) partition.

```python
ab.describe("shifted_window")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("shifted_window", w=4)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["shifted_window"], q, k, v, causal=True,
                 configs={"shifted_window": {'w': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
ShiftedWindowAttention(name='shifted_window', category='restrict', objective='approximate', backend='reference', shift=None, w=4)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, k_realized, n_windows, shift, support, window_offset, window_size], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 4.1183, 'distributional_error': 4.1183, 'value_error': 0.0, 'tv': 0.93419}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:sparse-fixed`.  Unlike the banded support of `window`, Swin partitions the
sequence into non-overlapping windows of `w` positions and lets each query see only
its own window (Liu et al., 2021).  Consecutive layers alternate between the regular
partition and one "displaced from the regularly partitioned windows by
`(floor(w/2), floor(w/2))` pixels", which is what connects windows that would
otherwise never communicate.  In one dimension this is

```text
S_p = { j : floor((j + shift)/w) = floor((p + shift)/w) },
```

a block-diagonal matrix whose blocks are offset by `shift`.  `shift = 0` is the
regular partition — plain chunked / block-local attention, the pattern of BlockBERT
(Qiu et al., 2020) — and `shift = floor(w/2)` is Swin's displaced partition; the
default follows Swin.  Under `shift > 0` the first window is a short one of
`w - shift` positions, exactly as in the paper's "efficient batch computation".

The difference from `window` is the point of having both: at an equal budget `w`
the banded support is *query-centred* and always keeps the `w` nearest keys, while
the block support keeps between `1` and `w` of them depending on where in its
block the query sits.  The shift is what repairs the queries that land at a block
boundary, and `extras['window_offset']` reports the realised distance to the start
of the block so that the effect can be measured rather than argued.

A query is always in its own window, so the diagonal is kept and no row is empty.
Intersecting with the causal mask turns this into *block-causal* attention: query `p`
sees the earlier positions of its own block only. Values are untouched, so the error
is `eps_q (m_{S^c} - m_S)` (`prop:sparse-identity`); `w = None` is a single
window over the whole sequence and recovers dense softmax exactly.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`atom`](atom.md), [`based`](based.md), [`clustered`](clustered.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`skyformer`](skyformer.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`trigrff`](trigrff.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `shifted_window` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `shifted_window` selected](index.md?compare=shifted_window) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.*<br><span class="attn-cite" data-bibkey="liu2021swin"><code>liu2021swin</code></span>
- *Blockwise Self-Attention for Long Document Understanding.*<br><span class="attn-cite" data-bibkey="qiu2020blockwise"><code>qiu2020blockwise</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="liu2021swin">
<pre class="attn-bib__src"><code>@inproceedings{liu2021swin,
  title = {{Swin Transformer: Hierarchical Vision Transformer using Shifted Windows}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="qiu2020blockwise">
<pre class="attn-bib__src"><code>@inproceedings{qiu2020blockwise,
  title = {{Blockwise Self-Attention for Long Document Understanding}}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/shifted_window.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/shifted_window.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
