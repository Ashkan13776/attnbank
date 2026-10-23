# Sinkhorn / entropic-OT attention

<p class="tx-regkey"><code>sinkhorn</code> &middot; <code>ab.build("sinkhorn")</code></p>

**Sinkhorn / entropic-OT attention: prescribe the column marginal too.**

category
:   [`coupling`](../taxonomy.md#coupling) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.SinkhornAttention`, [`attnbank/methods/coupling/sinkhorn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/sinkhorn.py)

monograph
:   `sec:ot`

## What it changes

Changes the **normalisation**. $Z(q)$ stops being a per-query constant and becomes a global constraint, so the queries are no longer independent of one another.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

## In the paper's notation

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

## Configuration

| parameter | default |
|---|---|
| `n_iter` | `5` |
| `eps` | `1.0` |

`n_iter`: Sinkhorn iterations (0 = softmax exactly).
`eps`: entropic regularization, applied as a temperature on the scaled logits.

```python
ab.describe("sinkhorn")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `n_iter=0`, `eps=1.0` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("sinkhorn").limit_config
{'n_iter': 0, 'eps': 1.0}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** A fixed-point iteration over the whole matrix is not a kernel call, and the normalisation it computes is not per-query.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `col_marginal`, `f`, `g`, `gamma`, `marginal_error`, `n_iter`, `potential`, `row_marginal`, `target_col`, `target_row`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    return {"flops": 2 * m * n * (d + dv) + 2 * self.n_iter * m * n,
            "state_bytes": n * (d + dv) * bytes_per_el + n * 4,   # + the key potential
            "bytes_moved": (m * d + n * (d + dv) + m * dv
                            + 2 * self.n_iter * m * n) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 278,921,216 | 268,435,456 | 1.04× |
| `state_bytes` | 266,240 | 262,144 | 1.02× |
| `bytes_moved` | 21,495,808 | 4,718,592 | 4.56× |

!!! note "A cost model is not a benchmark"
    These are counted operations and resident bytes, not measured time.
    They are what the monograph's capacity arguments are made of;
    `examples/08_cost_and_capacity.py` walks through the comparison properly.

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

```text
SinkhornAttention(name='sinkhorn', category='coupling', objective='modify', backend='reference', eps=1.0, n_iter=5)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[col_marginal, f, g, gamma, marginal_error, n_iter, potential, row_marginal, target_col, target_row], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['col_marginal', 'f', 'g', 'gamma', 'marginal_error', 'n_iter', 'potential', 'row_marginal', 'target_col', 'target_row']
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `sinkhorn` removed.

changes the same object — [`coupling`](../taxonomy.md#coupling)
:   [`lowrank_ot`](lowrank_ot.md) &nbsp;·&nbsp; [browse the bank, coupling only](index.md?category=coupling)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gqa`](gqa.md), [`latent_stochastic`](latent_stochastic.md), [`linear_elu`](linear_elu.md), [`linear_rnn`](linear_rnn.md), [`mla`](mla.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

[Open the browser with `sinkhorn` selected](index.md?compare=sinkhorn) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **Cuturi (2013).** *Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances.* Advances in Neural Information Processing Systems 26 (NIPS). [arXiv:1306.0895](https://arxiv.org/abs/1306.0895)<br><span class="attn-cite" data-bibkey="cuturi2013sinkhorn"><code>cuturi2013sinkhorn</code></span>
- **Sinkhorn (1964).** *A relationship between arbitrary positive matrices and doubly stochastic matrices.* The Annals of Mathematical Statistics.<br><span class="attn-cite" data-bibkey="sinkhorn1964relationship"><code>sinkhorn1964relationship</code></span>
- **Sander et al. (2022).** *Sinkformers: Transformers with Doubly Stochastic Attention.* Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS). [arXiv:2110.11773](https://arxiv.org/abs/2110.11773)<br><span class="attn-cite" data-bibkey="sander2022sinkformers"><code>sander2022sinkformers</code></span>
- **Peyre and Cuturi (2019).** *Computational Optimal Transport.* Foundations and Trends in Machine Learning. [arXiv:1803.00567](https://arxiv.org/abs/1803.00567)<br><span class="attn-cite" data-bibkey="peyre2019computational"><code>peyre2019computational</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="cuturi2013sinkhorn">
<pre class="attn-bib__src"><code>@inproceedings{cuturi2013sinkhorn,
  title={Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances},
  author={Cuturi, Marco}, booktitle={Advances in Neural Information Processing Systems 26 (NIPS)}, pages={2292--2300}, year={2013},
  eprint={1306.0895}, archivePrefix={arXiv}}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sinkhorn1964relationship">
<pre class="attn-bib__src"><code>@article{sinkhorn1964relationship,
  title={A relationship between arbitrary positive matrices and doubly stochastic matrices},
  author={Sinkhorn, Richard}, journal={The Annals of Mathematical Statistics}, volume={35}, pages={876--879}, year={1964}}</code></pre>
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

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/coupling/sinkhorn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/sinkhorn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
