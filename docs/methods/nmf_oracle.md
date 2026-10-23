# Nonnegative rank bound

<p class="attn-standfirst" markdown="span">The best rank-r *nonnegative* factorization of P.</p>

!!! danger "Oracle — an upper bound, not a method"
    `nmf_oracle` is not deployable and is not meant to be. It uses information no
    inference-time method has, and it is in the bank to answer *how well could this
    ever work?* — several of the monograph's arguments need that number. Read every
    result below as a bound on a family of real methods, never as a baseline you
    could ship.

<div class="attn-facts" data-cat="factorize" markdown="1">

key
:   `nmf_oracle` &middot; `ab.build("nmf_oracle")`

category
:   [`factorize`](../taxonomy.md#factorize)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.NMFOracleAttention` &middot; [`attnbank/methods/factorize/nmf_oracle.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/nmf_oracle.py)

monograph
:   `sec:factor`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the *shape* of $\pi_q$: it is routed through $r$ latents, $\pi_q = a(q,\cdot)\,b(\cdot,x)$, which caps the rank of the attention matrix at $r$.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

</div>

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **no** | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("nmf_oracle").limit_config
{'r': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 135,266,304 | 268,435,456 | 0.504× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = min(self.r or n, n)
    return {"flops": 2 * m * n * d + 2 * (m + n) * r * dv,
            "state_bytes": r * n * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
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

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `H`, `W`, `fro_residual`, `r`, `raw`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `4` |
| `iters` | `200` |
| `restarts` | `3` |
| `seed` | `0` |

</div>

```python
ab.describe("nmf_oracle")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("nmf_oracle", r=4, iters=200)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["nmf_oracle"], q, k, v, causal=False,
                 configs={"nmf_oracle": {'r': 4, 'iters': 200}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
NMFOracleAttention(name='nmf_oracle', category='factorize', objective='approximate', backend='reference', iters=200, r=4, restarts=3, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[H, W, fro_residual, r, raw], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.64372, 'distributional_error': 0.64372, 'value_error': 0.0, 'tv': 0.3159}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:factor-rank`, `prop:factor-rank`.  For a row-stochastic `P` the hierarchy is

```text
rank(P) <= rank_+(P) = rank_s(P),
```

so the smallest rank of a *stochastic* factorization equals the nonnegative rank, and
stochastic factorization is NMF with a free renormalization.  This file is the empirical
side of that statement: it fits `P ~ W H` with `W, H >= 0` of inner dimension `r`
by Lee–Seung multiplicative updates on the Frobenius loss,

```text
W <- W * (P H^T) / (W H H^T),     H <- H * (W^T P) / (W^T W H),
```

from `restarts` seeded initializations, keeping the best, and then renormalizes the
rows of `W H` so the result is a genuine row-stochastic `P_hat` — legitimate by
the rescaling argument in the proof of `prop:factor-rank`.  The *unnormalized*
product and its Frobenius residual are returned in `extras` so that the fit quality
can be separated from the renormalization.

This is an **oracle**, not a method anyone deploys: it gives the best a nonnegative
rank-`r` factorization *could* do, so that the gap against `svd_oracle` — which
has no nonnegativity constraint and is optimal by Eckart–Young — measures the
nonnegative-rank penalty that every stochastic latent attention pays.  Multiplicative
updates converge to a stationary point, not a global optimum, so the gap it reports is
an upper bound on the true penalty.

Exact limit: `r = None` returns `W = P`, `H = I` — the exact nonnegative
factorization of rank `n`.

Not causal: a nonnegative factorization of a masked `P` only approximates the zeros.

</div>

## Related methods

changes the same object — [`factorize`](../taxonomy.md#factorize)
:   [`abc`](abc.md), [`clustered`](clustered.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`perceiver`](perceiver.md), [`svd_oracle`](svd_oracle.md) *(oracle)* &nbsp;·&nbsp; [browse the bank, factorize only](index.md?category=factorize)

judged the same way — `approximate`, beyond the ones above
:   [`atom`](atom.md), [`based`](based.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`hyperattention`](hyperattention.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`lsh_sample`](lsh_sample.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nystrom`](nystrom.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`skyformer`](skyformer.md), [`strided`](strided.md), [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `nmf_oracle` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `nmf_oracle` selected](index.md?compare=nmf_oracle) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Learning the Parts of Objects by Non-negative Matrix Factorization.*<br><span class="attn-cite" data-bibkey="lee1999learning"><code>lee1999learning</code></span>
- *Nonnegative Ranks, Decompositions, and Factorizations of Nonnegative Matrices.*<br><span class="attn-cite" data-bibkey="cohen1993nonnegative"><code>cohen1993nonnegative</code></span>
- *Nonnegative Matrix Factorization.*<br><span class="attn-cite" data-bibkey="gillis2020nonnegative"><code>gillis2020nonnegative</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="lee1999learning">
<pre class="attn-bib__src"><code>@article{lee1999learning,
  title   = {Learning the Parts of Objects by Non-negative Matrix Factorization}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="cohen1993nonnegative">
<pre class="attn-bib__src"><code>@article{cohen1993nonnegative,
  title   = {Nonnegative Ranks, Decompositions, and Factorizations of Nonnegative Matrices}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="gillis2020nonnegative">
<pre class="attn-bib__src"><code>@book{gillis2020nonnegative,
  title     = {Nonnegative Matrix Factorization}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/nmf_oracle.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/nmf_oracle.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
