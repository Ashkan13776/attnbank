# Unbalanced entropic OT attention

<p class="attn-standfirst" markdown="span">The column constraint as a knob, not a law.</p>

<div class="attn-facts" data-cat="coupling" markdown="1">

key
:   `unbalanced_sinkhorn` &middot; `ab.build("unbalanced_sinkhorn")`

category
:   [`coupling`](../taxonomy.md#coupling)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.UnbalancedSinkhornAttention` &middot; [`attnbank/methods/coupling/unbalanced_sinkhorn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/unbalanced_sinkhorn.py)

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

**Exact limit.** Built with `lam_col=0.0`, `eps=1.0` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("unbalanced_sinkhorn").limit_config
{'lam_col': 0.0, 'eps': 1.0}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 289,406,976 | 268,435,456 | 1.08× |
| `state_bytes` | 266,240 | 262,144 | 1.02× |
| `bytes_moved` | 42,467,328 | 4,718,592 | 9× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    return {"flops": 2 * m * n * (d + dv) + 2 * self.n_iter * m * n,
            "state_bytes": n * (d + dv) * bytes_per_el + n * 4,   # + the key potential
            "bytes_moved": (m * d + n * (d + dv) + m * dv
                            + 2 * self.n_iter * m * n) * bytes_per_el}
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

Diagnostics returned in `extras`: `col_kl`, `col_marginal`, `f`, `g`, `gamma`, `lam_col`, `marginal_error`, `n_iter`, `potential`, `rho_col`, `rho_row`, `row_marginal`, `target_col`, `target_row`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_iter` | `10` |
| `eps` | `1.0` |
| `lam_col` | `1.0` |
| `lam_row` | `None` |

</div>

`lam_col`: KL penalty on the column marginal. `0` is softmax exactly,
`inf` (or a large value) is the doubly stochastic plan of `sinkhorn`.
`lam_row`: penalty on the row marginal; `None` means `infinity`, the hard
constraint that keeps every query's conditional a probability measure.
`eps`: entropic regularization, a temperature on the scaled logits.
`n_iter`: scaling iterations.

```python
ab.describe("unbalanced_sinkhorn")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("unbalanced_sinkhorn", n_iter=10, eps=1.0)
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
UnbalancedSinkhornAttention(name='unbalanced_sinkhorn', category='coupling', objective='modify', backend='reference', eps=1.0, lam_col=1.0, lam_row=None, n_iter=10)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[col_kl, col_marginal, f, g, gamma, lam_col, marginal_error, n_iter, potential, rho_col, rho_row, row_marginal, target_col, target_row], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['col_kl', 'col_marginal', 'f', 'g', 'gamma', 'lam_col', 'marginal_error', 'n_iter', 'potential', 'rho_col', 'rho_row', 'row_marginal', 'target_col', 'target_row']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:ot`, `eq:ot-unbalanced`.  `sinkhorn` imposes the column marginal exactly;
softmax does not impose it at all.  Replacing the hard constraints by Kullback–Leibler
penalties puts the two at the ends of one continuum (Frogner et al. 2015; Chizat et al.
2016; Liero et al. 2018; Sejourne et al. 2022):

```text
min_{gamma >= 0}  int c dgamma + eps KL(gamma || nu x mubar)
                  + lam_row KL(gamma_Q || nu) + lam_col KL(gamma_X || mubar).
```

Chizat et al. show the scaling iterations keep the form of `eq:ot-sinkhorn` with each
potential update *damped* by `rho_i = lam_i / (lam_i + eps)`:

```text
f <- rho_row ( log a - LSE_j ( s + g ) ),
g <- rho_col ( log b - LSE_i ( s + f ) ),     s = -c/eps.
```

`rho = 1` is the hard constraint of `sinkhorn`; `rho = 0` drops the constraint and
leaves the potential at zero.  The default here keeps the row constraint hard
(`lam_row = None`, meaning `lam_row = infinity`), which is what makes the operator a
probability measure per query, and treats `lam_col` as the single dial the paper
describes: **at `lam_col = 0` this is exactly softmax** — the semi-relaxed problem of
`prop:ot-gibbs` with `psi = 0` — and as `lam_col -> infinity` it is the doubly
stochastic coupling of `prop:ot-conditional`.

That is why this file, alone among the `coupling` methods, declares a `limit_config`
even though its `objective` is `MODIFY`: the family *contains* softmax at an interior
value of its own parameter.  The operator it defines away from that value is still a
different operator, not an approximation of softmax, so no convergence sequence is
claimed and fidelity is not the goal.

The practical reading is the one `sec:ot` gives: `lam_col` interpolates between
"sinks allowed" and "sinks forbidden".  A finite `lam_col` lets a key still absorb more
than its share — attention sinks survive, in proportion — while bounding how far it
can go, and it does so without the abstention mass that a dustbin introduces
(`dustbin_sinkhorn` is the discrete alternative).

`extras` matches `sinkhorn` key for key — `potential`, `f`, `g`, `gamma`,
`row_marginal`, `col_marginal`, `marginal_error` — because the plan is still of
Gibbs form `exp(s + f + g)` and `prop:ot-conditional` still applies verbatim: the
conditional is softmax tilted by one number per key.  Only the number is different, and
`col_kl`, the divergence the penalty charges, is added.

Not causal, and the relaxation does not rescue it.  `prop:ot-causal` is the limiting
obstruction — at `lam_col = infinity` a triangular doubly stochastic plan is the
identity — but at *any* `lam_col > 0` there is a second, more basic one: the column
potential `g_j` is a reduction over all queries that can see key `j`, including
queries later than the one being answered, so the conditional of query `i` moves when
a key after `i` changes.  That is leakage, not degeneracy, and the conformance suite
measures it directly.  Only at `lam_col = 0`, where the operator is softmax, does it
vanish; `supports_causal` is one declaration for the whole class, so `False` is the
one that is true of it.

</div>

## Related methods

changes the same object — [`coupling`](../taxonomy.md#coupling)
:   [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`lowrank_ot`](lowrank_ot.md), [`otke`](otke.md), [`sinkhorn`](sinkhorn.md) &nbsp;·&nbsp; [browse the bank, coupling only](index.md?category=coupling)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`gqa`](gqa.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mla`](mla.md), [`mlstm`](mlstm.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `unbalanced_sinkhorn` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `unbalanced_sinkhorn` selected](index.md?compare=unbalanced_sinkhorn) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Scaling Algorithms for Unbalanced Transport Problems.*<br><span class="attn-cite" data-bibkey="chizat2016scaling"><code>chizat2016scaling</code></span>
- *Learning with a Wasserstein Loss.*<br><span class="attn-cite" data-bibkey="frogner2015learning"><code>frogner2015learning</code></span>
- *Optimal Entropy-Transport problems and a new Hellinger–Kantorovich distance between positive measures.*<br><span class="attn-cite" data-bibkey="liero2018optimal"><code>liero2018optimal</code></span>
- *Unbalanced Optimal Transport, from Theory to Numerics.*<br><span class="attn-cite" data-bibkey="sejourne2022unbalanced"><code>sejourne2022unbalanced</code></span>
- *Sinkformers: Transformers with Doubly Stochastic Attention.*<br><span class="attn-cite" data-bibkey="sander2022sinkformers"><code>sander2022sinkformers</code></span>
- *Computational Optimal Transport.*<br><span class="attn-cite" data-bibkey="peyre2019computational"><code>peyre2019computational</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 6 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="chizat2016scaling">
<pre class="attn-bib__src"><code>@misc{chizat2016scaling,
  title={Scaling Algorithms for Unbalanced Transport Problems}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="frogner2015learning">
<pre class="attn-bib__src"><code>@inproceedings{frogner2015learning,
  title={Learning with a {W}asserstein Loss}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="liero2018optimal">
<pre class="attn-bib__src"><code>@article{liero2018optimal,
  title={Optimal Entropy-Transport problems and a new {H}ellinger--{K}antorovich distance between positive measures}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sejourne2022unbalanced">
<pre class="attn-bib__src"><code>@misc{sejourne2022unbalanced,
  title={Unbalanced Optimal Transport, from Theory to Numerics}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sander2022sinkformers">
<pre class="attn-bib__src"><code>@inproceedings{sander2022sinkformers,
  title         = {Sinkformers: Transformers with Doubly Stochastic Attention}</code></pre>
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

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/coupling/unbalanced_sinkhorn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/unbalanced_sinkhorn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
