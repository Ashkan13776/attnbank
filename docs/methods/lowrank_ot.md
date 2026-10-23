# LOTFormer-style factored coupling

<p class="tx-regkey"><code>lowrank_ot</code> &middot; <code>ab.build("lowrank_ot")</code></p>

**LOTFormer-style factored coupling: transport through r pivots.**

category
:   [`coupling`](../taxonomy.md#coupling) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.LowRankOTAttention`, [`attnbank/methods/coupling/lowrank_ot.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/lowrank_ot.py)

monograph
:   `sec:ot`

## What it changes

Changes the **normalisation**. $Z(q)$ stops being a per-query constant and becomes a global constraint, so the queries are no longer independent of one another.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

## In the paper's notation

`sec:ot` ("Factorization through a pivot measure"), `eq:ot-glued`, and
`eq:factor-pivot`.  Let `rho` be a probability measure on `r` pivots.  Given
`gamma_1 in Pi(nu, rho)` and `gamma_2 in Pi(rho, mubar)`, the glued coupling

```text
gamma(dq, dx) = sum_z rho_z gamma_1(dq | z) gamma_2(dx | z)
              = Gamma_1 diag(rho)^{-1} Gamma_2
```

lies in `Pi(nu, mubar)`, has rank at most `r`, and has query conditional
`pi^piv_q = sum_z gamma_1(z | q) gamma_2(. | z)`.  Both halves are solved by
Sinkhorn on dot-product costs (queries-to-pivots, pivots-to-keys), so the whole thing
costs `O((m + n) r)` and never forms the `m x n` matrix.

Two readings of the same object.  As a *coupling* it is doubly stochastic and so
balances participation of context positions — no key absorbs the mass of all queries.
As a *factorization* it is exactly `def:factor-stochastic`: `A_{iz} = gamma_1(z|q_i)`
and `B_{zj} = gamma_2(j|z)` are both row-stochastic, so `prop:factor-closure`
applies and `pi_hat` is a probability measure by construction.  It is filed under
`coupling` because the column constraint, absent from the row-stochastic
factorizations of `sec:factor`, is the thing it adds.

It is a **new operator**: a doubly stochastic coupling does not approximate softmax,
whose column sums are unconstrained, and it is not claimed to approximate the full
plan of `prop:ot-conditional` either.  Hence no exact limit and no convergence
sequence — the pivots are learned end to end, so `trainable = True`.

Not causal, for the reason in `prop:ot-causal`.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **no** | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **yes** | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

</div>

## Configuration

| parameter | default |
|---|---|
| `r` | `4` |
| `n_iter` | `10` |
| `eps` | `1.0` |
| `seed` | `0` |

```python
ab.describe("lowrank_ot")     # the same keys, with everything else it declares
```

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** A fixed-point iteration over the whole matrix is not a kernel call, and the normalisation it computes is not per-query.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `a`, `b`, `col_marginal`, `pivots`, `r`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.r
    return {"flops": 2 * (m + n) * r * d + 2 * (m + n) * r * dv
                    + 2 * self.n_iter * (m + n) * r,
            "state_bytes": r * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,260,992 | 268,435,456 | 0.00842× |
| `state_bytes` | 1,024 | 262,144 | 0.00391× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

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

attn = ab.build("lowrank_ot", r=4, n_iter=10)
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
LowRankOTAttention(name='lowrank_ot', category='coupling', objective='modify', backend='reference', eps=1.0, n_iter=10, r=4, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[a, b, col_marginal, pivots, r], dtype=float32, device=cpu)
invariant W: 2.086162567138672e-07
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['a', 'b', 'col_marginal', 'pivots', 'r']
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `lowrank_ot` removed.

changes the same object — [`coupling`](../taxonomy.md#coupling)
:   [`sinkhorn`](sinkhorn.md) &nbsp;·&nbsp; [browse the bank, coupling only](index.md?category=coupling)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gqa`](gqa.md), [`latent_stochastic`](latent_stochastic.md), [`linear_elu`](linear_elu.md), [`linear_rnn`](linear_rnn.md), [`mla`](mla.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

[Open the browser with `lowrank_ot` selected](index.md?compare=lowrank_ot) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **Shahbazi et al. (2025).** *LOTFormer: Doubly-Stochastic Linear Attention via Low-Rank Optimal Transport.* [arXiv:2509.23436](https://arxiv.org/abs/2509.23436)<br><span class="attn-cite" data-bibkey="shahbazi2025lotformer"><code>shahbazi2025lotformer</code></span>
- **Scetbon et al. (2021).** *Low-Rank Sinkhorn Factorization.* Proceedings of the 38th International Conference on Machine Learning (ICML). [arXiv:2103.04737](https://arxiv.org/abs/2103.04737)<br><span class="attn-cite" data-bibkey="scetbon2021lowrank"><code>scetbon2021lowrank</code></span>
- **Forrow et al. (2019).** *Statistical Optimal Transport via Factored Couplings.* Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS). [arXiv:1806.07348](https://arxiv.org/abs/1806.07348)<br><span class="attn-cite" data-bibkey="forrow2019statistical"><code>forrow2019statistical</code></span>
- **Mialon et al. (2021).** *A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention.* International Conference on Learning Representations (ICLR). [arXiv:2006.12065](https://arxiv.org/abs/2006.12065)<br><span class="attn-cite" data-bibkey="mialon2021trainable"><code>mialon2021trainable</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="shahbazi2025lotformer">
<pre class="attn-bib__src"><code>@misc{shahbazi2025lotformer,
  title         = {{LOTFormer}: Doubly-Stochastic Linear Attention via Low-Rank Optimal Transport},
  author        = {Shahbazi, Ashkan and Thrash, Chayne and Bai, Yikun and Hamm, Keaton and NaderiAlizadeh, Navid and Kolouri, Soheil},
  year          = {2025},
  eprint        = {2509.23436},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {arXiv preprint, v2 February 2026}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="scetbon2021lowrank">
<pre class="attn-bib__src"><code>@inproceedings{scetbon2021lowrank,
  title         = {Low-Rank {S}inkhorn Factorization},
  author        = {Scetbon, Meyer and Cuturi, Marco and Peyr{\'e}, Gabriel},
  booktitle     = {Proceedings of the 38th International Conference on Machine Learning (ICML)},
  series        = {PMLR},
  volume        = {139},
  pages         = {9344--9354},
  year          = {2021},
  eprint        = {2103.04737},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="forrow2019statistical">
<pre class="attn-bib__src"><code>@inproceedings{forrow2019statistical,
  title         = {Statistical Optimal Transport via Factored Couplings},
  author        = {Forrow, Aden and H{\&quot;u}tter, Jan-Christian and Nitzan, Mor and Rigollet, Philippe and Schiebinger, Geoffrey and Weed, Jonathan},
  booktitle     = {Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS)},
  series        = {PMLR},
  volume        = {89},
  pages         = {2454--2465},
  year          = {2019},
  eprint        = {1806.07348},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="mialon2021trainable">
<pre class="attn-bib__src"><code>@inproceedings{mialon2021trainable,
  title={A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention},
  author={Mialon, Gr{\'e}goire and Chen, Dexiong and d'Aspremont, Alexandre and Mairal, Julien},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2021}, eprint={2006.12065}, archivePrefix={arXiv}}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/coupling/lowrank_ot.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/lowrank_ot.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
