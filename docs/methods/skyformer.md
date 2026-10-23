# Skyformer

<p class="attn-standfirst" markdown="span">Kernelize first, then Nystrom on a matrix that is actually PSD.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `skyformer` &middot; `ab.build("skyformer")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.SkyformerAttention` &middot; [`attnbank/methods/interaction/skyformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/skyformer.py)

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
| `supports_causal` | **no** | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **no** | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("skyformer").limit_config
{'r': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 8,396,800 | 268,435,456 | 0.0313× |
| `state_bytes` | 4,096 | 262,144 | 0.0156× |
| `bytes_moved` | 528,384 | 4,718,592 | 0.112× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = min(self.r or (m + n), m + n)
    return {"flops": 2 * (m + n) * r * d + 2 * r ** 3 + 2 * (m + n) * r * dv,
            "state_bytes": r * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (d + dv)) * bytes_per_el}
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

Diagnostics returned in `extras`: `Z_rel`, `min_weight`, `n_landmarks`, `neg_mass`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `16` |
| `seed` | `0` |

</div>

`r`: sub-sample size out of `m + n`; `None` keeps all of them (the limit).

```python
ab.describe("skyformer")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("skyformer", r=16)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["skyformer"], q, k, v, causal=False,
                 configs={"skyformer": {'r': 16}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
SkyformerAttention(name='skyformer', category='interaction', objective='approximate', backend='reference', r=16, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_rel, min_weight, n_landmarks, neg_mass], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.24419, 'distributional_error': 1.24419, 'value_error': 0.0, 'tv': 0.46344}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:kernel-matrix`.  Nystromformer applies the Nystrom method to `P`, which is
neither symmetric nor PSD, so none of the Nystrom theory applies to it.  Skyformer
fixes the premise rather than the conclusion, in two steps.

*Kernelize.*  With `beta = 1/sqrt(d)` the softmax kernel factors through a Gaussian
one,

```text
SM(q, x)   = exp(beta <q, k_x>)
           = exp(beta ||q||^2 / 2) * C(q, x) * exp(beta ||k_x||^2 / 2),
C(q, x)    = exp(-beta ||q - k_x||^2 / 2),
```

i.e. `A = D_Q^{1/2} C D_K^{1/2}` with `(D_Q)_ii = exp(beta ||q_i||^2)` (Eq. 1 of
the paper).  `C` is the *kernelized attention score matrix*: the paper's operator is
`C V`, which is already "automatically normalized" and is therefore a **different**
operator — the key factor `D_K^{1/2}` does not cancel.  The bank targets softmax, so
this file keeps `C` as the object that is approximated and then undoes the diagonal:
the query factor cancels under row normalization and the key factor is multiplied back
in.  That is the use the paper states for the same machinery in Sec. 4.2 — "a low-rank
approximation to A also implies an approximation to D".

*Lift.*  `C` is a rectangular block, but it is a block of the Gram matrix of the
*union* of the two point sets, and that matrix is PSD:

```text
Cbar = kappa([Q; K] / d^{1/4}, [Q; K] / d^{1/4}) in R^{(m+n) x (m+n)},
Cbar_tilde = Cbar S (S^T Cbar S)^+ S^T Cbar,
C_tilde    = (I, 0) Cbar_tilde (0, I)^T,
```

with `S` a uniform sub-sampling matrix choosing `r` of the `m + n` positions.
Since `||C - C_tilde|| <= ||Cbar - Cbar_tilde||`, approximating the block reduces to
approximating a PSD matrix, which is what Nystrom has guarantees for (Theorem 2 of the
paper, after Musco and Musco).  This is the whole difference from `nystrom`, and it
is a difference in the premise, not in the formula.

What survives and what does not.  The core pseudo-inverse makes `C_tilde` **signed**,
so `pi_hat` is a signed measure: `row_stochastic = False` and the output can leave
the convex hull of the values.  Rows do sum to one here because the library normalizes
explicitly, which the paper's own operator does not.

Exact limit.  `r = None` samples every position, `S = I`, and
`Cbar Cbar^+ Cbar = Cbar`, so `C_tilde = C` and `pi_hat` is softmax — for any
`m` and `n`, since the lift is square by construction.  That is the anchor.

Numerics.  The whole factorization is float64, and the Gram matrix is scaled by a
single positive constant chosen to centre its log-range (Nystrom is homogeneous of
degree one, and the scalar cancels in the row normalization).  Without it the
off-diagonal block underflows to zero in the large-logit regime while the diagonal
stays at one — the Gaussian Gram matrix tends to the identity when the points are far
apart, which is the regime where this kernelization has nothing left to approximate.

Not causal: the sub-sampling draws landmarks from the whole union, future included.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`linear_elu`](linear_elu.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polaformer`](polaformer.md), [`polysketch`](polysketch.md), [`rebased`](rebased.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`atom`](atom.md), [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`hyperattention`](hyperattention.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`longformer`](longformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lsh_sample`](lsh_sample.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`oval`](oval.md), [`page_pca`](page_pca.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `skyformer` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `skyformer` selected](index.md?compare=skyformer) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Skyformer: Remodel Self-Attention with Gaussian Kernel and Nystrom Method.*<br><span class="attn-cite" data-bibkey="chen2021skyformer"><code>chen2021skyformer</code></span>
- *Using the Nystrom Method to Speed Up Kernel Machines.*<br><span class="attn-cite" data-bibkey="williams2001using"><code>williams2001using</code></span>
- *On the Nystrom Method for Approximating a Gram Matrix for Improved Kernel-Based Learning.*<br><span class="attn-cite" data-bibkey="drineas2005nystrom"><code>drineas2005nystrom</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="chen2021skyformer">
<pre class="attn-bib__src"><code>@inproceedings{chen2021skyformer,
  title     = {Skyformer: Remodel Self-Attention with {G}aussian Kernel and {N}ystr{\&quot;o}m Method}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="williams2001using">
<pre class="attn-bib__src"><code>@inproceedings{williams2001using,
  title     = {Using the {N}ystr{\&quot;o}m Method to Speed Up Kernel Machines}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="drineas2005nystrom">
<pre class="attn-bib__src"><code>@article{drineas2005nystrom,
  title={On the {N}ystr{\&quot;o}m Method for Approximating a {G}ram Matrix for Improved Kernel-Based Learning}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/skyformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/skyformer.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
