# DeltaNet

<p class="attn-standfirst" markdown="span">The delta-rule update on a fast-weight memory.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `delta_rnn` &middot; `ab.build("delta_rnn")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.DeltaRNNAttention` &middot; [`attnbank/methods/recurrent/delta_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/delta_rnn.py)

monograph
:   `sec:memory-state`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the operator. $y$ is produced by a finite-state recurrence over positions rather than by an integral against a measure — though it is still linear in $v$, which is why the weights can be recovered at all.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

</div>

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **yes** | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **no** | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
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
| `flops` | 25,165,824 | 268,435,456 | 0.0938× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = d
    return {"flops": 4 * n * r * dv + 2 * m * r * dv,
            "state_bytes": r * dv * bytes_per_el,
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

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `beta`, `mass`, `state_rank`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `beta` | `1.0` |
| `feature` | `'elu'` |

</div>

`beta`: write strength (not an inverse temperature; see sec:memory-state).

```python
ab.describe("delta_rnn")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("delta_rnn", beta=1.0, feature='elu')
out = attn(q, k, v, causal=True,
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
DeltaRNNAttention(name='delta_rnn', category='recurrent', objective='modify', backend='reference', beta=1.0, feature='elu')
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[beta, mass, state_rank], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [5.3093, 8.0564, 8.2169, 7.149]
extras: ['beta', 'mass', 'state_rank']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:memory-state`, `eq:memory-delta`, `tab:memory-recurrences` (row "DeltaNet").
The transition is a rank-one correction rather than a decay:

```text
H_t = (I - beta_t psi_t psi_t^T) H_{t-1} + beta_t psi_t v_t^T,    psi_t = psi(k_t),
y_hat_t = H_t^T phi(q_t),
```

which is exactly one gradient step of size `beta_t` on the per-token regression loss
`L_t(H) = 1/2 ||H^T psi_t - v_t||^2` (Widrow–Hoff).  Plain linear attention is a
gradient step on the *linear* loss `-<H^T psi_t, v_t>`, which never saturates and so
never corrects interference between non-orthogonal keys; the delta rule does.  That is
the whole capacity argument: at a fixed state of `r x d_v`, the delta rule stores more
distinguishable associations than the additive recurrence, which is what study S6
measures.

The regression view also identifies the operator.  Softmax attention is a
Nadaraya–Watson estimator that never leaves the convex hull of the stored values; the
delta rule fits a *global parametric* regression of `v` on `psi(k)` online and
returns arbitrary linear combinations.  It is a new operator and is judged by downstream
quality, not by fidelity to `y(q)`.

Weights.  `H_t` is linear in `V`, so `H_t = M_t V` for an `r x n` coefficient
matrix obeying the same recurrence driven by the basis vectors,
`M_t = (I - beta psi_t psi_t^T) M_{t-1} + beta psi_t e_t^T`.  Running it gives the
exact weights `w_{tj} = <phi(q_t), M_t[:, j]>` at a cost of `O(n^2 r)`, which is the
price of exposing `pi_hat` on the original positions.  `psi` is l2-normalized, as in
DeltaNet, so that `I - beta psi psi^T` is a contraction for `beta in (0, 2)`.

Note on notation: DESIGN.md writes the update as `H_{t-1}(I - beta phi phi^T)`, which
is the same map with `H` transposed (`d_v x r`).  This file follows the paper's
`eq:memory-delta` with `H in R^{r x d_v}`.

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`gqa`](gqa.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`lowrank_ot`](lowrank_ot.md), [`mla`](mla.md), [`otke`](otke.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `delta_rnn` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `delta_rnn` selected](index.md?compare=delta_rnn) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Linear Transformers Are Secretly Fast Weight Programmers.*<br><span class="attn-cite" data-bibkey="schlag2021linear"><code>schlag2021linear</code></span>
- *Parallelizing Linear Transformers with the Delta Rule over Sequence Length.*<br><span class="attn-cite" data-bibkey="yang2024parallelizing"><code>yang2024parallelizing</code></span>
- *Adaptive Switching Circuits.*<br><span class="attn-cite" data-bibkey="widrow1960adaptive"><code>widrow1960adaptive</code></span>
- *Gated Delta Networks: Improving Mamba2 with Delta Rule.*<br><span class="attn-cite" data-bibkey="yang2025gated"><code>yang2025gated</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="schlag2021linear">
<pre class="attn-bib__src"><code>@inproceedings{schlag2021linear,
  title={Linear Transformers Are Secretly Fast Weight Programmers}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="yang2024parallelizing">
<pre class="attn-bib__src"><code>@inproceedings{yang2024parallelizing,
  title={Parallelizing Linear Transformers with the Delta Rule over Sequence Length}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="widrow1960adaptive">
<pre class="attn-bib__src"><code>@inproceedings{widrow1960adaptive,
  title={Adaptive Switching Circuits}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="yang2025gated">
<pre class="attn-bib__src"><code>@inproceedings{yang2025gated,
  title={Gated Delta Networks: Improving {Mamba2} with Delta Rule}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/delta_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/delta_rnn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
