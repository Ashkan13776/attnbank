# BigBird

<p class="tx-regkey"><code>bigbird</code> &middot; <code>ab.build("bigbird")</code></p>

**BigBird: window blocks + global blocks + *random* blocks, at block granularity.**

category
:   [`restrict`](../taxonomy.md#restrict) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.BigBirdAttention`, [`attnbank/methods/restrict/bigbird.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/bigbird.py)

monograph
:   `sec:sparse`

## What it changes

Changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:sparse-fixed`.  Zaheer et al. (2020) build the support as a graph on blocks of
`block` consecutive positions and take the union of three families:

* **window** — the `window_blocks` blocks centred on the query's own block
  (`w/2` neighbouring blocks on each side);
* **global** — the first `global_blocks` blocks, in the *internal* (ITC)
  construction: a global token has `A(i,:) = 1` and `A(:,i) = 1`, so it is
  attended by every query **and** its own row is dense;
* **random** — `random_blocks` blocks drawn uniformly, without replacement, from
  the blocks that are neither window nor global.  This is the piece that makes the
  pattern a (sparse) expander and gives the paper its reachability argument; it is
  also the only randomness in the family, so it is drawn through a seeded
  `torch.Generator` and is reproducible across builds, devices and dtypes.

The draw depends on the query *block index*, the sequence length and `causal` —
nothing else.  It is shared across batch and head, which keeps the pattern a property
of the geometry rather than of the data; per-head draws would change nothing that the
library measures.

**Causality.**  Under a causal mask the candidate set for query block `pb` is
restricted to blocks strictly before the window, so no random block can ever sit in
the future; the support is intersected with the causal mask afterwards in any case.
The query's own block is always in the window (`window_blocks` down to 1 still keeps
it), so the diagonal is retained and no row is empty.

BigBird is an encoder pattern; as everywhere in `restrict/` the values are untouched,
so the error is exactly `eps_q (m_{S^c} - m_S)` of `prop:sparse-identity`.  Put
beside `sink_window` it isolates what the *random* blocks buy: the same budget spent
on uniformly scattered mass instead of on the prefix.

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
| `block` | `4` |
| `window_blocks` | `3` |
| `random_blocks` | `2` |
| `global_blocks` | `1` |
| `seed` | `0` |

`window_blocks`: blocks centred on the query's own (`None` = all, the limit).

`random_blocks` blocks are drawn per query block; `global_blocks` leading
blocks are global in the ITC sense.  `block` is the block size (64 in the
paper; small here because the bank's test shapes are small).

```python
ab.describe("bigbird")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `window_blocks=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("bigbird").limit_config
{'window_blocks': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `block_size`, `blocks_selected`, `chi2`, `eps`, `k_realized`, `n_blocks`, `support`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    nb = (n + self.block - 1) // self.block
    blocks = nb if self.window_blocks is None else min(
        nb, self.window_blocks + self.random_blocks + self.global_blocks)
    kept = min(n, blocks * self.block)
    return {"flops": 2 * m * kept * (d + dv),
            "state_bytes": n * (d + dv) * bytes_per_el,   # random blocks reach anywhere
            "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 6,291,456 | 268,435,456 | 0.0234× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 6,553,600 | 4,718,592 | 1.39× |

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

attn = ab.build("bigbird", block=4, window_blocks=3)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["bigbird"], q, k, v, causal=True,
                 configs={"bigbird": {'block': 4, 'window_blocks': 3}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
BigBirdAttention(name='bigbird', category='restrict', objective='approximate', backend='reference', block=4, global_blocks=1, random_blocks=2, seed=0, window_blocks=3)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[block_size, blocks_selected, chi2, eps, k_realized, n_blocks, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.11778, 'distributional_error': 1.11778, 'value_error': 0.0, 'tv': 0.40706}
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `bigbird` removed.

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`atom`](atom.md), [`based`](based.md), [`clustered`](clustered.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`skyformer`](skyformer.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`trigrff`](trigrff.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

[Open the browser with `bigbird` selected](index.md?compare=bigbird) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **zaheer2020bigbird (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="zaheer2020bigbird"><code>zaheer2020bigbird</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="zaheer2020bigbird">
<pre class="attn-bib__src"><code>@inproceedings{zaheer2020bigbird,
  title = {{Big Bird: Transformers for Longer Sequences}}

@inproceedings{zandieh2025qjl,
  title     = {{QJL}: 1-Bit Quantized {JL} Transform for {KV} Cache Quantization with Zero Overhead}

@inproceedings{zhang2023h2o,
  title     = {{H2O}: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models}

@inproceedings{zhang2024cam,
  title = {{CaM}: Cache Merging for Memory-efficient {LLMs} Inference}

@inproceedings{zhang2024hedgehog,
  title     = {The Hedgehog \&amp; the Porcupine: Expressive Linear Attentions with Softmax Mimicry}

@inproceedings{zhang2025sageattention,
  title={{SageAttention}: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}

@inproceedings{zhao2024atom,
  title     = {{Atom}: Low-Bit Quantization for Efficient and Accurate {LLM} Serving}
</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/bigbird.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/bigbird.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
