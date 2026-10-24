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

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="no">no</span> | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

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

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    return {"flops": 2 * m * n * (d + dv) + 2 * self.n_iter * m * n,
            "state_bytes": n * (d + dv) * bytes_per_el + n * 4,   # + the key potential
            "bytes_moved": (m * d + n * (d + dv) + m * dv
                            + 2 * self.n_iter * m * n) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** A fixed-point iteration over the whole matrix is not a kernel call, and the normalisation it computes is not per-query.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

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

`ab.describe("unbalanced_sinkhorn")` returns the same keys alongside everything else this page states.

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

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:ot</code>, <code>eq:ot-unbalanced</code></p>

`sinkhorn` imposes the column marginal exactly;
softmax does not impose it at all.  Replacing the hard constraints by Kullback–Leibler
penalties puts the two at the ends of one continuum (Frogner et al. 2015; Chizat et al.
2016; Liero et al. 2018; Sejourne et al. 2022):

```text
min_{gamma >= 0}  int c dgamma + eps KL(gamma || nu x mubar)
                  + lam_row KL(gamma_Q || nu) + lam_col KL(gamma_X || mubar).
```

Chizat et al. show the scaling iterations keep the form of <span class="attn-ref">eq:ot-sinkhorn</span> with each
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
<span class="attn-ref">prop:ot-gibbs</span> with `psi = 0` — and as `lam_col -> infinity` it is the doubly
stochastic coupling of <span class="attn-ref">prop:ot-conditional</span>.

That is why this file, alone among the `coupling` methods, declares a `limit_config`
even though its `objective` is `MODIFY`: the family *contains* softmax at an interior
value of its own parameter.  The operator it defines away from that value is still a
different operator, not an approximation of softmax, so no convergence sequence is
claimed and fidelity is not the goal.

The practical reading is the one <span class="attn-ref">sec:ot</span> gives: `lam_col` interpolates between
"sinks allowed" and "sinks forbidden".  A finite `lam_col` lets a key still absorb more
than its share — attention sinks survive, in proportion — while bounding how far it
can go, and it does so without the abstention mass that a dustbin introduces
(`dustbin_sinkhorn` is the discrete alternative).

`extras` matches `sinkhorn` key for key — `potential`, `f`, `g`, `gamma`,
`row_marginal`, `col_marginal`, `marginal_error` — because the plan is still of
Gibbs form `exp(s + f + g)` and <span class="attn-ref">prop:ot-conditional</span> still applies verbatim: the
conditional is softmax tilted by one number per key.  Only the number is different, and
`col_kl`, the divergence the penalty charges, is added.

Not causal, and the relaxation does not rescue it.  <span class="attn-ref">prop:ot-causal</span> is the limiting
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

judged the same way — `modify`
:   23 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `unbalanced_sinkhorn` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `unbalanced_sinkhorn` selected](index.md?compare=unbalanced_sinkhorn) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Chizat et al. (2016).** *Scaling Algorithms for Unbalanced Transport Problems.* [arXiv:1607.05816](https://arxiv.org/abs/1607.05816)<br><span class="attn-cite" data-bibkey="chizat2016scaling"><code>chizat2016scaling</code></span>
- **Frogner et al. (2015).** *Learning with a Wasserstein Loss.* Advances in Neural Information Processing Systems (NIPS). [arXiv:1506.05439](https://arxiv.org/abs/1506.05439)<br><span class="attn-cite" data-bibkey="frogner2015learning"><code>frogner2015learning</code></span>
- **Liero et al. (2018).** *Optimal Entropy-Transport problems and a new Hellinger–Kantorovich distance between positive measures.* Inventiones mathematicae. [arXiv:1508.07941](https://arxiv.org/abs/1508.07941)<br><span class="attn-cite" data-bibkey="liero2018optimal"><code>liero2018optimal</code></span>
- **Sejourne et al. (2022).** *Unbalanced Optimal Transport, from Theory to Numerics.* [arXiv:2211.08775](https://arxiv.org/abs/2211.08775)<br><span class="attn-cite" data-bibkey="sejourne2022unbalanced"><code>sejourne2022unbalanced</code></span>
- **Sander et al. (2022).** *Sinkformers: Transformers with Doubly Stochastic Attention.* Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS). [arXiv:2110.11773](https://arxiv.org/abs/2110.11773)<br><span class="attn-cite" data-bibkey="sander2022sinkformers"><code>sander2022sinkformers</code></span>
- **Peyre and Cuturi (2019).** *Computational Optimal Transport.* Foundations and Trends in Machine Learning. [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)<br><span class="attn-cite" data-bibkey="peyre2019computational"><code>peyre2019computational</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 6 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="chizat2016scaling">
<pre class="attn-bib__src"><code>@misc{chizat2016scaling,
  title={Scaling Algorithms for Unbalanced Transport Problems},
  author={Chizat, L{\'e}na{\&quot;\i}c and Peyr{\'e}, Gabriel and Schmitzer, Bernhard and Vialard, Fran{\c{c}}ois-Xavier},
  year={2016}, eprint={1607.05816}, archivePrefix={arXiv}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="frogner2015learning">
<pre class="attn-bib__src"><code>@inproceedings{frogner2015learning,
  title={Learning with a {W}asserstein Loss},
  author={Frogner, Charlie and Zhang, Chiyuan and Mobahi, Hossein and Araya-Polo, Mauricio and Poggio, Tomaso},
  booktitle={Advances in Neural Information Processing Systems (NIPS)}, year={2015}, eprint={1506.05439}, archivePrefix={arXiv}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="liero2018optimal">
<pre class="attn-bib__src"><code>@article{liero2018optimal,
  title={Optimal Entropy-Transport problems and a new {H}ellinger--{K}antorovich distance between positive measures},
  author={Liero, Matthias and Mielke, Alexander and Savar{\'e}, Giuseppe}, journal={Inventiones mathematicae},
  volume={211}, pages={969--1117}, year={2018}, eprint={1508.07941}, archivePrefix={arXiv}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sejourne2022unbalanced">
<pre class="attn-bib__src"><code>@misc{sejourne2022unbalanced,
  title={Unbalanced Optimal Transport, from Theory to Numerics},
  author={S{\'e}journ{\'e}, Thibault and Peyr{\'e}, Gabriel and Vialard, Fran{\c{c}}ois-Xavier}, year={2022},
  eprint={2211.08775}, archivePrefix={arXiv}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sander2022sinkformers">
<pre class="attn-bib__src"><code>@inproceedings{sander2022sinkformers,
  title         = {Sinkformers: Transformers with Doubly Stochastic Attention},
  author        = {Sander, Michael E. and Ablin, Pierre and Blondel, Mathieu and Peyr{\'e}, Gabriel},
  booktitle     = {Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series        = {PMLR},
  volume        = {151},
  year          = {2022},
  eprint        = {2110.11773},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="peyre2019computational">
<pre class="attn-bib__src"><code>@article{peyre2019computational,
  title={Computational Optimal Transport}, author={Peyr{\'e}, Gabriel and Cuturi, Marco},
  journal={Foundations and Trends in Machine Learning}, volume={11}, number={5-6}, pages={355--607}, year={2019},
  eprint={1803.00567}, archivePrefix={arXiv}}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/coupling/unbalanced_sinkhorn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/unbalanced_sinkhorn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
