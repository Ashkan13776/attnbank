# Multi-head latent attention

<p class="attn-standfirst" markdown="span">One joint low-rank KV latent, per-head up-projection.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `mla` &middot; `ab.build("mla")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.MLAAttention` &middot; [`attnbank/methods/compress/mla.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/mla.py)

monograph
:   `sec:compress`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="yes">yes</span> | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="no">no</span> | it shares $K$/$V$ across query heads, so heads are not independent. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("mla").limit_config
{'r': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 272,629,760 | 268,435,456 | 1.02× |
| `state_bytes` | 32,768 | 262,144 | 0.125× |
| `bytes_moved` | 294,912 | 4,718,592 | 0.0625× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2, heads: int = 1):
    wide = heads * (d + dv)
    r = min(self.r or wide, wide)
    per_head = n * r * bytes_per_el / max(1, heads)
    return {"flops": 2 * m * n * (d + dv) + 2 * n * r * (d + dv),
            "state_bytes": per_head,
            "bytes_moved": m * (d + dv) * bytes_per_el + per_head}
```

## Backends

<div class="attn-tab" markdown="1">

| backend | what it is |
|---|---|
| `reference` | plain PyTorch — the correctness anchor and the default |
| `sdpa` | `torch.nn.functional.scaled_dot_product_attention` — on CUDA, FlashAttention-2 or the memory-efficient kernel |

</div>

Each non-`reference` entry is a claim that the path is **equal** to the reference, and `tests/test_conformance.py::test_backend_equivalence` checks it on five shapes, both causal settings, float32 and float64. See [Backends](../backends.md).

Diagnostics returned in `extras`: `latent`, `latent_width`, `r`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `16` |
| `seed` | `0` |
| `sdpa_kernel` | `None` |

</div>

`r`: latent width; `None` uses the full width (the exact limit).

`ab.describe("mla")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("mla", r=16)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = modify: this is a different operator and fidelity to softmax
# is not its goal, so there is no error to quote. What it *is* doing shows
# up in its own diagnostics.
print("row_stochastic =", attn.row_stochastic, "| first row sums:",
      [round(x, 4) for x in out.weights.sum(-1)[0, 0, :4].tolist()])
print("extras:", sorted(out.extras))
```

<div class="attn-out" markdown="1">

```text
MLAAttention(name='mla', category='compress', objective='modify', backend='reference', r=16, sdpa_kernel=None, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[latent, latent_width, r], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['latent', 'latent_width', 'r']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-arch</code></p>

Instead of caching `h` key–value pairs per position, MLA
caches a single latent `c_j = W_down [k_j^{(1)}; ...; k_j^{(h)}; v_j^{(1)}; ...]` of
dimension `r`, and reconstructs each head's key and value by a per-head
up-projection.  The cached state per position drops from `h (d + d_v)` to `r`.

Like GQA this is an **operator modification** trained as such; it is in the library for
the roofline (it is the other intervention that raises decode arithmetic intensity) and
because it is the cleanest example of a compression that couples heads: the latent is
shared, so `head_independent = False`.

The projections are *trained* in DeepSeek-V2.  Here `W_down` is a fixed seeded
Gaussian and the up-projection is its pseudo-inverse, so the reconstruction is the
orthogonal projection onto `row(W_down)`.  That choice is data-independent, hence
causal, and makes the exact limit provable: at `r = h (d + d_v)` the matrix is square
and invertible, the reconstruction is exact, and the operator is dense softmax.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `modify`
:   27 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `mla` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `mla` selected](index.md?compare=mla) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **DeepSeek-AI (2024).** *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model.* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)<br><span class="attn-cite" data-bibkey="deepseekai2024deepseekv2"><code>deepseekai2024deepseekv2</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="deepseekai2024deepseekv2">
<pre class="attn-bib__src"><code>@misc{deepseekai2024deepseekv2,
  title = {{DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model}},
  author = {{DeepSeek-AI}},
  year = {2024},
  howpublished = {arXiv preprint arXiv:2405.04434},
  eprint = {2405.04434},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/mla.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/mla.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
