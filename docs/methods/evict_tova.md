# TOVA

<p class="attn-standfirst" markdown="span">Drop the token the *current* query cares least about, one per step.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `evict_tova` &middot; `ab.build("evict_tova")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.TOVAAttention` &middot; [`attnbank/methods/compress/evict_tova.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_tova.py)

monograph
:   `sec:compress`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

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

**Exact limit.** Built with `budget=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("evict_tova").limit_config
{'budget': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,105,344 | 268,435,456 | 0.00784× |
| `state_bytes` | 2,048 | 262,144 | 0.00781× |
| `bytes_moved` | 2,359,296 | 4,718,592 | 0.5× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    b = max(1, min(self.budget or n, n))
    return {"flops": 2 * m * b * (d + dv) + m * b,
            "state_bytes": b * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}
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

Diagnostics returned in `extras`: `budget`, `chi2`, `eps`, `evicted_at`, `k_realized`, `support`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `budget` | `8` |

</div>

`budget`: the multi-state size, i.e. how many (k, v) pairs survive.

```python
ab.describe("evict_tova")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("evict_tova", budget=8)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["evict_tova"], q, k, v, causal=True,
                 configs={"evict_tova": {'budget': 8}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
TOVAAttention(name='evict_tova', category='compress', objective='approximate', backend='reference', budget=8)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[budget, chi2, eps, evicted_at, k_realized, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.13116, 'distributional_error': 2.13116, 'value_error': 0.0, 'tv': 0.74278}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:compress-evict`, `prop:compress-noeviction`.  Reading a decoder as a
multi-state RNN, the KV cache is the state and a cache policy is a rule for keeping it
bounded.  TOVA is the simplest non-trivial such rule: at every step the new token is
admitted, and if the cache is over budget the state with the **smallest weight under
the query that just arrived** is deleted.  No accumulation over history (that is
`evict_h2o`), no protected recent window (that is `sink_window`), no observation
window (that is `evict_snapkv`): one query decides, once, and the decision is
permanent.

```text
S_0 = {0},   S_t = topkeep_{b}( S_{t-1} u {t}; pi_{q_t} ),   S_t \ {t} subset S_{t-1}
```

**Semantics: eviction, and the file is built so that this is visible.**  The supports
are produced by a loop over queries in order, so the set serving query `t` was fixed
by queries `<= t` and nothing later can restore what they threw away.  Two
consequences are tested in `tests/test_compress_evict.py` rather than asserted here:

* the supports are nested — `S_{t+1} subset S_t u {t+1}`.  `window` and
  `sink_window` are nested too, which is exactly why they are listed in both
  `restrict/` and `sec:compress-evict`; the *content-dependent* selectors are not,
  because a restriction recomputes its set per query and is free to come back to a
  position it ignored a step ago.  `topk` at the same budget is the comparison;
* at equal budget the error against dense softmax is larger than `topk`'s, and the
  gap *grows along the query axis*: early queries are served by a cache that still
  holds almost everything, late ones by a cache pruned long before their question was
  asked.  That gap is the whole content of `prop:compress-noeviction` — restriction
  hides context, eviction destroys it.

The current position is the one protection the policy keeps, so that no row is ever
left with an empty support (DESIGN.md Sec. 4, rule 10); TOVA keeps it in practice
anyway, since a token's own key usually dominates its row.  `extras['evicted_at']`
records the step at which each position was deleted, or `-1` if it survived, which is
the permanence made inspectable.

In non-causal mode the whole context is admitted at the first step and the same
evict-per-query loop runs over the queries in order: "a streaming cache served by a
fixed context", the convention `evict_h2o` also uses.

Exact limit: `budget = None` (nothing is ever evicted).

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`based`](based.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`lsh_sample`](lsh_sample.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`skyformer`](skyformer.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `evict_tova` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `evict_tova` selected](index.md?compare=evict_tova) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Transformers are Multi-State RNNs.*<br><span class="attn-cite" data-bibkey="oren2024transformers"><code>oren2024transformers</code></span>
- *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.*<br><span class="attn-cite" data-bibkey="zhang2023h2o"><code>zhang2023h2o</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="oren2024transformers">
<pre class="attn-bib__src"><code>@inproceedings{oren2024transformers,
  title = {{Transformers are Multi-State RNNs}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="zhang2023h2o">
<pre class="attn-bib__src"><code>@inproceedings{zhang2023h2o,
  title     = {{H2O}: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/evict_tova.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/evict_tova.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
