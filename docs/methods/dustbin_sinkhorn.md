# Partial-assignment attention

<p class="attn-standfirst" markdown="span">Sinkhorn on a score matrix with a dustbin.</p>

<div class="attn-facts" data-cat="coupling" markdown="1">

key
:   `dustbin_sinkhorn` &middot; `ab.build("dustbin_sinkhorn")`

category
:   [`coupling`](../taxonomy.md#coupling)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.DustbinSinkhornAttention` &middot; [`attnbank/methods/coupling/dustbin_sinkhorn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/dustbin_sinkhorn.py)

monograph
:   `sec:ot`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the **normalisation**. $Z(q)$ stops being a per-query constant and becomes a global constraint, so the queries are no longer independent of one another.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

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

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 310,460,456 | 268,435,456 | 1.16× |
| `state_bytes` | 266,244 | 262,144 | 1.02× |
| `bytes_moved` | 84,574,288 | 4,718,592 | 17.9× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    mb, nb = m + 1, n + 1
    return {"flops": 2 * m * n * (d + dv) + 2 * self.n_iter * mb * nb,
            "state_bytes": n * (d + dv) * bytes_per_el + (n + 1) * 4,
            "bytes_moved": (m * d + n * (d + dv) + m * dv
                            + 2 * self.n_iter * mb * nb) * bytes_per_el}
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

**`reference` only, and that is the correct answer.** A fixed-point iteration over the whole matrix is not a kernel call, and the normalisation it computes is not per-query.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `bin_to_keys`, `col_marginal`, `dustbin`, `dustbin_mass`, `f`, `f_bin`, `g`, `g_bin`, `gamma`, `gamma_bar`, `marginal_error`, `n_iter`, `potential`, `queries_to_bin`, `row_marginal`, `target_col`, `target_row`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_iter` | `20` |
| `eps` | `1.0` |
| `dustbin` | `0.0` |

</div>

`dustbin`: the scalar `z` filling the extra row and column (learned in
SuperGlue).  Very negative recovers `sinkhorn`; larger values let queries
abstain.  `n_iter`, `eps`: as for `sinkhorn`.

```python
ab.describe("dustbin_sinkhorn")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("dustbin_sinkhorn", n_iter=20, eps=1.0)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
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
DustbinSinkhornAttention(name='dustbin_sinkhorn', category='coupling', objective='modify', backend='reference', dustbin=0.0, eps=1.0, n_iter=20)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[bin_to_keys, col_marginal, dustbin, dustbin_mass, f, f_bin, g, g_bin, gamma, gamma_bar, marginal_error, n_iter, potential, queries_to_bin, row_marginal, target_col, target_row], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['bin_to_keys', 'col_marginal', 'dustbin', 'dustbin_mass', 'f', 'f_bin', 'g', 'g_bin', 'gamma', 'gamma_bar', 'marginal_error', 'n_iter', 'potential', 'queries_to_bin', 'row_marginal', 'target_col', 'target_row']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:ot`, "Unbalanced relaxation".  A doubly stochastic plan forces every query to
spend all of its mass on real keys and every key to receive exactly its share.  The
discrete way to buy slack is the one SuperGlue uses for unmatchable keypoints
(Sarlin et al. 2020, eq. 8-9): augment the scores with one extra row and one extra
column — a *dustbin* — filled with a single scalar `z`,

```text
Sbar[i, j]   = beta <q_i, k_j> / eps      i <= m, j <= n
Sbar[i, n+1] = Sbar[m+1, j] = Sbar[m+1, n+1] = z / eps,
```

and prescribe marginals that give the bin exactly the surplus capacity,

```text
a = [1_m, n] / (m + n),      b = [1_n, m] / (m + n),
```

so that both sides carry unit total mass.  Sinkhorn on `exp(Sbar)` then returns a
*partial* assignment: the `m x n` block is a coupling of what was matched, and the
dustbin entries hold what was not.

As an attention operator the conditional is read off the augmented row and
renormalized over the real keys.  Because the plan is still of Gibbs form, this is
exact rather than a projection:

```text
pi_hat_i(j) = softmax_j ( s_ij + g_j ),        j = 1 .. n,
```

with `g` the column potential restricted to the real keys — `prop:ot-conditional`
again, with the tilt now computed in the presence of the bin.  The mass the query sent
to the bin, `extras["dustbin_mass"]`, is the quantity renormalization removes; it is
the *abstention* the operator permits, and it is reported rather than hidden.

Two limits bracket the method.  As `z -> -infinity` the bin becomes unattractive, the
surplus is absorbed by the bin-to-bin cell, and the real block approaches the doubly
stochastic plan of `sinkhorn`.  As `z` grows, queries divert mass to the bin and
`dustbin_mass -> 1`; the renormalized conditional stays finite and tends to the
conditional induced by the bin row's own tilt.  The default `dustbin = 0.0` sits
between them; in SuperGlue `z` is a learned parameter, which is the only sense in
which this operator has one — it is a scalar, and the operator is meaningful at any
value of it, so `trainable` stays `False`.

One cost of the deep-bin limit is worth recording, because it is not visible in the
construction.  As `z` falls, the bin's entries vanish relative to the real block and
the augmented matrix loses *total support* in the sense of Sinkhorn (1967); the scaling
iterations then give up their geometric rate.  Measured at `z = -40` the distance to
the balanced plan decays like `1/n_iter`, so recovering `sinkhorn` this way costs
thousands of iterations where `sinkhorn` itself needs tens.  At the useful values of
`z` — those where abstention is actually happening — convergence is geometric and
the default `n_iter = 20` is ample.

This is the discrete counterpart of `unbalanced_sinkhorn`: a hard constraint with an
escape hatch, rather than a soft penalty.  `sec:ot` lists both, alongside explicit
register tokens, as ways to keep abstention under a column constraint.

Not causal: `prop:ot-causal`.  The dustbin does relax the marginals — the
compatibility inequality `eq:ot-hall` is no longer tight — but the slack it provides
is a single lump, not the position-decreasing column marginal that the proposition says
would be needed, and the plan of a prefix still changes when a token arrives.

</div>

## Related methods

changes the same object — [`coupling`](../taxonomy.md#coupling)
:   [`espformer`](espformer.md), [`lowrank_ot`](lowrank_ot.md), [`otke`](otke.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, coupling only](index.md?category=coupling)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`gqa`](gqa.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mla`](mla.md), [`mlstm`](mlstm.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `dustbin_sinkhorn` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `dustbin_sinkhorn` selected](index.md?compare=dustbin_sinkhorn) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *SuperGlue: Learning Feature Matching with Graph Neural Networks.*<br><span class="attn-cite" data-bibkey="sarlin2020superglue"><code>sarlin2020superglue</code></span>
- *Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances.*<br><span class="attn-cite" data-bibkey="cuturi2013sinkhorn"><code>cuturi2013sinkhorn</code></span>
- *A relationship between arbitrary positive matrices and doubly stochastic matrices.*<br><span class="attn-cite" data-bibkey="sinkhorn1964relationship"><code>sinkhorn1964relationship</code></span>
- *Computational Optimal Transport.*<br><span class="attn-cite" data-bibkey="peyre2019computational"><code>peyre2019computational</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="sarlin2020superglue">
<pre class="attn-bib__src"><code>@inproceedings{sarlin2020superglue,
  title={{SuperGlue}: Learning Feature Matching with Graph Neural Networks}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="cuturi2013sinkhorn">
<pre class="attn-bib__src"><code>@inproceedings{cuturi2013sinkhorn,
  title={Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sinkhorn1964relationship">
<pre class="attn-bib__src"><code>@article{sinkhorn1964relationship,
  title={A relationship between arbitrary positive matrices and doubly stochastic matrices}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="peyre2019computational">
<pre class="attn-bib__src"><code>@article{peyre2019computational,
  title={Computational Optimal Transport}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/coupling/dustbin_sinkhorn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/dustbin_sinkhorn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
