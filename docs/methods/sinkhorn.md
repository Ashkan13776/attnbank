# Sinkhorn / entropic-OT attention

<p class="attn-standfirst" markdown="span">Prescribe the column marginal too.</p>

<div class="attn-facts" data-cat="coupling" markdown="1">

key
:   `sinkhorn` &middot; `ab.build("sinkhorn")`

category
:   [`coupling`](../taxonomy.md#coupling)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.SinkhornAttention` &middot; [`attnbank/methods/coupling/sinkhorn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/sinkhorn.py)

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

**Exact limit.** Built with `n_iter=0`, `eps=1.0` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("sinkhorn").limit_config
{'n_iter': 0, 'eps': 1.0}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 278,921,216 | 268,435,456 | 1.04× |
| `state_bytes` | 266,240 | 262,144 | 1.02× |
| `bytes_moved` | 21,495,808 | 4,718,592 | 4.56× |

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

Diagnostics returned in `extras`: `col_marginal`, `f`, `g`, `gamma`, `marginal_error`, `n_iter`, `potential`, `row_marginal`, `target_col`, `target_row`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_iter` | `5` |
| `eps` | `1.0` |

</div>

`n_iter`: Sinkhorn iterations (0 = softmax exactly).
`eps`: entropic regularization, applied as a temperature on the scaled logits.

```python
ab.describe("sinkhorn")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("sinkhorn", n_iter=5, eps=1.0)
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
SinkhornAttention(name='sinkhorn', category='coupling', objective='modify', backend='reference', eps=1.0, n_iter=5)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[col_marginal, f, g, gamma, marginal_error, n_iter, potential, row_marginal, target_col, target_row], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['col_marginal', 'f', 'g', 'gamma', 'marginal_error', 'n_iter', 'potential', 'row_marginal', 'target_col', 'target_row']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:ot-eot`.  Softmax attention is *semi-relaxed* entropic OT: it minimizes
`int c dgamma + eps KL(gamma || nu x mubar)` subject to the **row** marginal only
(`prop:ot-gibbs`).  Adding the column constraint gives entropic OT, and the central
structural fact (`prop:ot-conditional`) is that its conditionals are still softmax —
tilted by a *query-independent key potential*:

```text
pi^OT_q(dx) ∝ exp( beta ( <q, k_x> + psi(x) ) ) mubar(dx).
```

So a doubly stochastic attention operator is characterised, completely, by one vector
`psi` per context: an economic price that is negative on keys over-subscribed under
softmax and positive on under-subscribed ones.  `extras["potential"]` returns the
quantity `psi` in the form in which the identity can be checked directly,
`pi_hat = softmax_j(s_ij / eps + potential_j)`, which is what study S7 does.

Sinkhorn's algorithm alternates the two identities of `prop:ot-conditional` (b) in
log domain (`eq:ot-sinkhorn`): a row log-sum-exp (a softmax normalizer) followed by a
column log-sum-exp.  **One step from `psi = 0`, stopped after the row update, is
exactly softmax attention** — so `n_iter = 0` is the exact limit, and the whole
operator family is a continuation of softmax in the number of iterations.

Not causal, and not by omission: `prop:ot-causal` shows that a lower-triangular
matrix with equal row and column sums *is diagonal*.  Causal attention with uniform
marginals is the identity, every token attending only to itself.  A non-uniform column
marginal satisfying `sum_{j>=t} b_j <= sum_{i>=t} a_i` is the way out, and is not
implemented here.

</div>

## Related methods

changes the same object — [`coupling`](../taxonomy.md#coupling)
:   [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`lowrank_ot`](lowrank_ot.md), [`otke`](otke.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, coupling only](index.md?category=coupling)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`gqa`](gqa.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mla`](mla.md), [`mlstm`](mlstm.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `sinkhorn` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `sinkhorn` selected](index.md?compare=sinkhorn) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances.*<br><span class="attn-cite" data-bibkey="cuturi2013sinkhorn"><code>cuturi2013sinkhorn</code></span>
- *A relationship between arbitrary positive matrices and doubly stochastic matrices.*<br><span class="attn-cite" data-bibkey="sinkhorn1964relationship"><code>sinkhorn1964relationship</code></span>
- *Sinkformers: Transformers with Doubly Stochastic Attention.*<br><span class="attn-cite" data-bibkey="sander2022sinkformers"><code>sander2022sinkformers</code></span>
- *Computational Optimal Transport.*<br><span class="attn-cite" data-bibkey="peyre2019computational"><code>peyre2019computational</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="cuturi2013sinkhorn">
<pre class="attn-bib__src"><code>@inproceedings{cuturi2013sinkhorn,
  title={Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sinkhorn1964relationship">
<pre class="attn-bib__src"><code>@article{sinkhorn1964relationship,
  title={A relationship between arbitrary positive matrices and doubly stochastic matrices}</code></pre>
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

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/coupling/sinkhorn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/sinkhorn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
