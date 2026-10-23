# RWKV-5/6 time mixing

<p class="attn-standfirst" markdown="span">A matrix-valued state that the current token bypasses.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `rwkv` &middot; `ab.build("rwkv")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.RWKVAttention` &middot; [`attnbank/methods/recurrent/rwkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/rwkv.py)

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
| `flops` | 17,170,432 | 268,435,456 | 0.064× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.state_rank or d
    c = recurrent_cost(m, n, r, dv, bytes_per_el, gate_flops=4)
    c["flops"] += 2 * m * r                      # the bonus term
    return c
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

Diagnostics returned in `extras`: `bonus`, `data_dependent`, `decay`, `mass`, `state_rank`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `decay_base` | `-2.0` |
| `bonus` | `1.0` |
| `data_dependent` | `True` |
| `normalize` | `False` |
| `state_rank` | `None` |

</div>

`decay_base`: the `omega` of `w = exp(-exp(omega + k_t))`.

`bonus` is RWKV's `u`, the weight the current token gets *outside* the
state.  `data_dependent=False` drops the `k_t` term and gives Eagle
(RWKV-5); `True` gives Finch (RWKV-6).  `state_rank` truncates the
key/query dimension and so *is* the size of the state, `state_rank x d_v`.

```python
ab.describe("rwkv")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("rwkv", decay_base=-2.0, bonus=1.0)
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
RWKVAttention(name='rwkv', category='recurrent', objective='modify', backend='reference', bonus=1.0, data_dependent=True, decay_base=-2.0, normalize=False, state_rank=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[bonus, data_dependent, decay, mass, state_rank], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [-3.5872, -5.5343, -9.4189, 6.36]
extras: ['bonus', 'data_dependent', 'decay', 'mass', 'state_rank']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:memory-state`, `tab:memory-recurrences`.  Eagle (RWKV-5) and Finch (RWKV-6)
write the time-mixing step as

```text
wkv_t = S_{t-1} + Diag(u) k_t v_t^T,       S_t = Diag(w_t) S_{t-1} + k_t v_t^T,
y_hat_t = wkv_t^T q_t,                     w_t = exp(-exp(d_t)) in (0, 1)^r,
```

which is the family's recurrence with two deliberate departures.

*The state is matrix-valued.*  RWKV-4 carried a per-channel numerator and denominator
— `2d` numbers, a diagonal operator, no mixing between channels.  Eagle replaces the
element-wise product by the outer product `k_t v_t^T` and the state becomes
`r x d_v`.  That is the same move HGRN2 makes, and for the same reason: at a fixed
parameter count `prop:memory-rank` ties recall to the *number of state variables*,
and `r d_v` is `r` times more of them than `d`.

*The current token is not in the state.*  The readout uses `S_{t-1}` and adds the
token's own write back with a separate bonus gate `u`.  So the decay a key at `j`
suffers is `prod_{u = j+1}^{t-1} w_u` — lag `t - 1 - j`, not `t - j` — and the
diagonal weight is `u <q_t, k_t>` with no decay at all.  RWKV is therefore not a pure
exponential-recency operator: it has a recency *prior* plus an always-available
present, which is what the bonus term buys.  `bonus=0` removes it and the method
becomes ordinary gated linear attention one step out of phase.

```text
w_{tj} = <q_t, exp(c_{t-1} - c_j) * k_j>     j < t,     c = cumsum(log w),
w_{tt} = u <q_t, k_t>.
```

Finch versus Eagle.  Eagle's decay `w = exp(-exp(omega))` is a learned per-channel
constant; Finch makes `omega` a function of the token, `w_t = exp(-exp(d_t))`.
`data_dependent=False` selects the Eagle form and is the controlled comparison the
family exists to make: same state, same bonus, gate input-dependent or not.  With no
training there is no projection to produce `d_t`, so this reference uses
`d_t = decay_base + k_t` — stated plainly, and all that is needed of it is that the
gate be per channel, in `(0, 1)`, and a function of the token.  The double
exponential is the paper's and is what keeps `log w_t = -exp(d_t)` strictly negative
and finite however large the logits get.

No denominator survives — RWKV-4's was dropped precisely when the state became a
matrix — so the rows are signed and do not sum to one.  The LayerNorm, the SiLU gate
and the token shift belong to the RWKV *block*, not to this operator.

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`gqa`](gqa.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`lowrank_ot`](lowrank_ot.md), [`mla`](mla.md), [`otke`](otke.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `rwkv` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `rwkv` selected](index.md?compare=rwkv) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence.*<br><span class="attn-cite" data-bibkey="peng2024eagle"><code>peng2024eagle</code></span>
- *RWKV: Reinventing RNNs for the Transformer Era.*<br><span class="attn-cite" data-bibkey="peng2023rwkv"><code>peng2023rwkv</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="peng2024eagle">
<pre class="attn-bib__src"><code>@inproceedings{peng2024eagle,
  title={Eagle and Finch: {RWKV} with Matrix-Valued States and Dynamic Recurrence}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="peng2023rwkv">
<pre class="attn-bib__src"><code>@inproceedings{peng2023rwkv,
  title={{RWKV}: Reinventing {RNN}s for the Transformer Era}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/rwkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/rwkv.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
