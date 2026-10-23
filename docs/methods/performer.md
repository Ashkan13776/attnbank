# Performer / FAVOR+

<p class="attn-standfirst" markdown="span">Positive random features for the exponential kernel.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `performer` &middot; `ab.build("performer")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.PerformerAttention` &middot; [`attnbank/methods/interaction/performer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/performer.py)

monograph
:   `sec:kernel`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the interaction $g(q,x)$, replaced by a surrogate $\widehat g$. $\pi_q = \widehat g(q,\cdot)\mu / \widehat Z(q)$ is then a different measure on the same $n$ positions.

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


**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("performer").convergence_configs()
[{'r': 8}, {'r': 32}, {'r': 128}, {'r': 512}]
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 33,554,432 | 268,435,456 | 0.125× |
| `state_bytes` | 8,320 | 262,144 | 0.0317× |
| `bytes_moved` | 532,608 | 4,718,592 | 0.113× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.r
    return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv,
            "state_bytes": r * (dv + 1) * bytes_per_el,       # the moments (H, z)
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
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

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_rel`, `log_g_hat`, `r`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `64` |
| `orthogonal` | `True` |
| `seed` | `0` |

</div>

```python
ab.describe("performer")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("performer", r=64, orthogonal=True)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["performer"], q, k, v, causal=True,
                 configs={"performer": {'r': 64, 'orthogonal': True}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
PerformerAttention(name='performer', category='interaction', objective='approximate', backend='reference', orthogonal=True, r=64, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_rel, log_g_hat, r], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.55857, 'distributional_error': 1.55857, 'value_error': 0.0, 'tv': 0.4355}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:kernel-rf`.  With `qt = q / d^{1/4}` and `kt = k / d^{1/4}` the softmax
interaction is `g(q,x) = exp(<qt, kt_x>)` and FAVOR+ uses the positive, unbiased
estimator

```text
phi_omega(u) = exp(omega^T u - ||u||^2 / 2),
g_hat(q,x)   = (1/r) sum_{l=1}^r phi_{omega_l}(qt) phi_{omega_l}(kt_x),
omega_l ~ N(0, I_d)   i.i.d. or exactly orthogonal in blocks of d.
```

Because `phi >= 0` the normalizer `Z_hat > 0` and `pi_hat` is a genuine
probability measure (`prop:error-additive`: positivity is essential).  The relative
variance at a pair is `(exp(||qt + kt||^2) - 1) / r` (`eq:kernel-pos-var`), so the
cost of a fixed accuracy is exponential in the squared logit scale.

The output is a *ratio* of two unbiased estimators, so it is itself biased:
`prop:error-ratio-est` gives bias `O(1/r)` and standard deviation `O(r^{-1/2})`.
`convergence_configs` exists for exactly this reason — there is no finite `r` at
which the method is exact.

Implementation note.  The sum over features is evaluated as a log-sum-exp rather than
as the `O(nr)` moment product, because the library needs `pi_hat` on the original
positions and because `phi` is bounded above by `exp(||omega||^2/2)` but can
underflow to zero in the tail.  The value computed is the same estimator.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`linear_elu`](linear_elu.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`nystrom`](nystrom.md), [`polaformer`](polaformer.md), [`polysketch`](polysketch.md), [`rebased`](rebased.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`atom`](atom.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`hyperattention`](hyperattention.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lsh_sample`](lsh_sample.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`oval`](oval.md), [`page_pca`](page_pca.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `performer` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `performer` selected](index.md?compare=performer) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Rethinking Attention with Performers.*<br><span class="attn-cite" data-bibkey="choromanski2021rethinking"><code>choromanski2021rethinking</code></span>
- *Orthogonal Random Features.*<br><span class="attn-cite" data-bibkey="yu2016orthogonal"><code>yu2016orthogonal</code></span>
- *Chefs' Random Tables: Non-Trigonometric Random Features.*<br><span class="attn-cite" data-bibkey="likhosherstov2022chefs"><code>likhosherstov2022chefs</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="choromanski2021rethinking">
<pre class="attn-bib__src"><code>@inproceedings{choromanski2021rethinking,
  title     = {Rethinking Attention with Performers}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="yu2016orthogonal">
<pre class="attn-bib__src"><code>@inproceedings{yu2016orthogonal,
  title={Orthogonal Random Features}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="likhosherstov2022chefs">
<pre class="attn-bib__src"><code>@inproceedings{likhosherstov2022chefs,
  title={Chefs' Random Tables: Non-Trigonometric Random Features}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/performer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/performer.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
