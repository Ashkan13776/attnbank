# OTKE

<p class="attn-standfirst" markdown="span">Transport the context onto a trainable reference, then read it out.</p>

<div class="attn-facts" data-cat="coupling" markdown="1">

key
:   `otke` &middot; `ab.build("otke")`

category
:   [`coupling`](../taxonomy.md#coupling)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.OTKEAttention` &middot; [`attnbank/methods/coupling/otke.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/otke.py)

monograph
:   `sec:ot`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the **normalisation**. $Z(q)$ stops being a per-query constant and becomes a global constraint, so the queries are no longer independent of one another.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="no">no</span> | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="yes">yes</span> | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

</div>

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,179,072 | 268,435,456 | 0.00812× |
| `state_bytes` | 262,656 | 262,144 | 1× |
| `bytes_moved` | 524,800 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    p = self.p
    return {"flops": 2 * (m + n) * p * d + 2 * (n + m) * p * dv
                     + 2 * self.n_iter * n * p,
            # the plan needs every key, so the reference buys rank, not capacity
            "state_bytes": n * (d + dv) * bytes_per_el + p * d * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + p * d) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** A fixed-point iteration over the whole matrix is not a kernel call, and the normalisation it computes is not per-query.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `a`, `col_marginal`, `f`, `g`, `gamma`, `marginal_error`, `n_iter`, `p`, `potential`, `reference`, `row_marginal`, `slot_col_marginal`, `target_col`, `target_row`, `w`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `p` | `4` |
| `n_iter` | `10` |
| `eps` | `1.0` |
| `seed` | `0` |

</div>

`p`: size of the trainable reference. `n_iter`: Sinkhorn iterations
(the paper reports ten is enough). `eps`: entropic regularization.
`seed`: the generator the untrained reference is drawn from.

`ab.describe("otke")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("otke", p=4, n_iter=10)
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
OTKEAttention(name='otke', category='coupling', objective='modify', backend='reference', eps=1.0, n_iter=10, p=4, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[a, col_marginal, f, g, gamma, marginal_error, n_iter, p, potential, reference, row_marginal, slot_col_marginal, target_col, target_row, w], dtype=float32, device=cpu)
invariant W: 2.086162567138672e-07
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['a', 'col_marginal', 'f', 'g', 'gamma', 'marginal_error', 'n_iter', 'p', 'potential', 'reference', 'row_marginal', 'slot_col_marginal', 'target_col', 'target_row', 'w']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:ot</code>, &quot;The same constraint elsewhere&quot;, and <code>sec:ot-implications</code></p>

The optimal
transport kernel embedding of Mialon et al. (2021) is not an attention layer — it is a
*pooling* layer, with no queries at all.  It embeds a set by transporting it onto a
trainable reference `Z` of `p` points and aggregating along the plan
(their Definition 3.1, eq. 5):

```text
P = argmin_{P in Pi(1_n/n, 1_p/p)}  -sum_ij <z_s, x_j> P_js - eps H(P),
Phi_Z(x)_s = sqrt(p) * sum_j P_js phi(x_j).
```

The plan is computed by Sinkhorn, so each *reference slot* ends up with a normalized
weighting of the whole set, `W_sj = p P_js`, and `Phi` is the stack of the `p`
weighted means.  The paper's own reading of the construction is the one that places it
here: elements are pooled together when they align with *the same part of the
reference*, and the column constraint is what stops one slot from swallowing the set.

**What this file does, and what it adds.**  To fit the bank's `q, k, v` interface the
layer needs a query, and OTKE does not have one — downstream it flattens `Phi` into a
classifier.  The transport half is OTKE verbatim: keys to a trainable reference, uniform
marginals, Sinkhorn, aggregation along the plan.  The readout half, `a_is =
softmax_s(beta <q_i, z_s>)`, is *supplied here* and is not part of the paper.  The
operator is therefore

```text
pi_hat = A W ,     A in R^{m x p} row-stochastic,  W in R^{p x n} row-stochastic,
```

a pivot factorization (<span class="attn-ref">def:factor-stochastic</span>) of rank at most `p`, exactly as
<span class="attn-ref">sec:ot</span> classifies OTKE.  `lowrank_ot` is the neighbouring object and the contrast
is the point: it Sinkhorns *both* halves, so the full coupling is doubly stochastic;
OTKE Sinkhorns only the key half, so the constraint binds between keys and slots and
each key is guaranteed its share of the *reference's* attention, not of the queries'.
`extras["slot_col_marginal"]` is that guarantee, equal to `1/n` on every key.

`phi` is a kernel embedding, Nystrom-approximated in the paper when the kernel is
infinite dimensional; a linear kernel needs no approximation and is the case
implemented, so `phi(x_j) = v_j` and the aggregation is a plain weighted mean of
values.  The `sqrt(p)` factor of Definition 3.1 is a scaling of the embedding, not of
the conditional, and drops out of a row-stochastic operator.

`extras` exposes the transport diagnostics in the `sinkhorn` spelling — `f`,
`g`, `potential`, `gamma`, `row_marginal`, `col_marginal`,
`marginal_error` — for the key-to-reference coupling, which is the coupling the
method computes.  <span class="attn-ref">prop:ot-conditional</span> applies to it unchanged: a slot's weighting of
the keys is softmax of `beta <z_s, k_j>` tilted by one potential per key.

`trainable = True`, and not as a formality: `Z` carries the method.  The paper
initializes it by k-means on the training features and learns it end to end through the
differentiable Sinkhorn; here it is drawn from a seeded `torch.Generator` so that the
library reproduces, which makes the untrained operator arbitrary in the same way
`lowrank_ot`'s pivots are.

Not causal: the plan is computed over the whole key set at once, and the column
constraint is the reason <span class="attn-ref">prop:ot-causal</span> applies to this family.

</div>

## Related methods

changes the same object — [`coupling`](../taxonomy.md#coupling)
:   [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`lowrank_ot`](lowrank_ot.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, coupling only](index.md?category=coupling)

judged the same way — `modify`
:   23 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `otke` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `otke` selected](index.md?compare=otke) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Mialon et al. (2021).** *A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention.* International Conference on Learning Representations (ICLR). [arXiv:2006.12065](https://arxiv.org/abs/2006.12065)<br><span class="attn-cite" data-bibkey="mialon2021trainable"><code>mialon2021trainable</code></span>
- **Cuturi (2013).** *Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances.* Advances in Neural Information Processing Systems 26 (NIPS). [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)<br><span class="attn-cite" data-bibkey="cuturi2013sinkhorn"><code>cuturi2013sinkhorn</code></span>
- **Peyre and Cuturi (2019).** *Computational Optimal Transport.* Foundations and Trends in Machine Learning. [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)<br><span class="attn-cite" data-bibkey="peyre2019computational"><code>peyre2019computational</code></span>
- **Naderializadeh et al. (2021).** *Pooling by Sliced-Wasserstein Embedding.* Advances in Neural Information Processing Systems (NeurIPS).<br><span class="attn-cite" data-bibkey="naderializadeh2021pooling"><code>naderializadeh2021pooling</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="mialon2021trainable">
<pre class="attn-bib__src"><code>@inproceedings{mialon2021trainable,
  title={A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention},
  author={Mialon, Gr{\'e}goire and Chen, Dexiong and d'Aspremont, Alexandre and Mairal, Julien},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2021}, eprint={2006.12065}, archivePrefix={arXiv}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="cuturi2013sinkhorn">
<pre class="attn-bib__src"><code>@inproceedings{cuturi2013sinkhorn,
  title={Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances},
  author={Cuturi, Marco}, booktitle={Advances in Neural Information Processing Systems 26 (NIPS)}, pages={2292--2300}, year={2013},
  eprint={1306.0895}, archivePrefix={arXiv}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="peyre2019computational">
<pre class="attn-bib__src"><code>@article{peyre2019computational,
  title={Computational Optimal Transport}, author={Peyr{\'e}, Gabriel and Cuturi, Marco},
  journal={Foundations and Trends in Machine Learning}, volume={11}, number={5-6}, pages={355--607}, year={2019},
  eprint={1803.00567}, archivePrefix={arXiv}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="naderializadeh2021pooling">
<pre class="attn-bib__src"><code>@inproceedings{naderializadeh2021pooling,
  title={Pooling by Sliced-{W}asserstein Embedding},
  author={Naderializadeh, Navid and Comer, Joseph F. and Andrews, Reed and Hoffmann, Heiko and Kolouri, Soheil},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2021}}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/coupling/otke.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/otke.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
