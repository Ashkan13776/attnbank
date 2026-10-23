# Decayed linear recurrence

<p class="attn-standfirst" markdown="span">H_t = gamma H_{t-1} + psi(k_t) v_t^T.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `decay_rnn` &middot; `ab.build("decay_rnn")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.DecayRNNAttention` &middot; [`attnbank/methods/recurrent/decay_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/decay_rnn.py)

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
| `flops` | 17,825,792 | 268,435,456 | 0.0664× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = d
    return {"flops": 2 * n * r * dv + 2 * m * r * dv + m * n,
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

Diagnostics returned in `extras`: `decay`, `gate`, `mass`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `decay` | `0.9` |
| `feature` | `'elu'` |
| `normalize` | `False` |

</div>

```python
ab.describe("decay_rnn")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("decay_rnn", decay=0.9, feature='elu')
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
DecayRNNAttention(name='decay_rnn', category='recurrent', objective='modify', backend='reference', decay=0.9, feature='elu', normalize=False)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[decay, gate, mass], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [43.946, 78.9989, 108.2692, 140.6155]
extras: ['decay', 'gate', 'mass']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:memory-state`, `tab:memory-recurrences` (rows "Retention", "GLA / Mamba-2").
`A_t = gamma Id` with a fixed `gamma < 1` is retention (RetNet); a data-dependent
scalar or diagonal gate gives Mamba-2, GLA, HGRN2, RWKV-5/6 and the mLSTM.  The weights
on the original positions are

```text
w_{tj} = gamma^{(t - j)} <phi(q_t), psi(k_j)>,    j <= t,
```

an exponential recency prior multiplying a rank-`r` content term.

Decay restores *boundedness* without restoring normalization: the effective mass
`sum_i prod_{u>i} gamma_u` is bounded by `1/(1 - gamma)` rather than growing with
`n`, so the output cannot blow up with context length, but it is still not an
expectation under a probability measure (set `normalize=True` to recover one).  This
is the distinction `sec:memory-state` draws between "restoring boundedness" and
"restoring the contract".

State size is `r x d_v` regardless of `gamma`, so sweeping `gamma` at fixed state
separates *recency* from *capacity* — which is what study S6 needs.

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`gqa`](gqa.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`lowrank_ot`](lowrank_ot.md), [`mla`](mla.md), [`otke`](otke.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `decay_rnn` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `decay_rnn` selected](index.md?compare=decay_rnn) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Retentive Network: A Successor to Transformer for Large Language Models.*<br><span class="attn-cite" data-bibkey="sun2023retentive"><code>sun2023retentive</code></span>
- *Gated Linear Attention Transformers with Hardware-Efficient Training.*<br><span class="attn-cite" data-bibkey="yang2024gated"><code>yang2024gated</code></span>
- *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.*<br><span class="attn-cite" data-bibkey="dao2024transformers"><code>dao2024transformers</code></span>
- *RWKV: Reinventing RNNs for the Transformer Era.*<br><span class="attn-cite" data-bibkey="peng2023rwkv"><code>peng2023rwkv</code></span>
- *xLSTM: Extended Long Short-Term Memory.*<br><span class="attn-cite" data-bibkey="beck2024xlstm"><code>beck2024xlstm</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 5 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="sun2023retentive">
<pre class="attn-bib__src"><code>@misc{sun2023retentive,
  title={Retentive Network: A Successor to Transformer for Large Language Models}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="yang2024gated">
<pre class="attn-bib__src"><code>@inproceedings{yang2024gated,
  title={Gated Linear Attention Transformers with Hardware-Efficient Training}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="dao2024transformers">
<pre class="attn-bib__src"><code>@inproceedings{dao2024transformers,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="peng2023rwkv">
<pre class="attn-bib__src"><code>@inproceedings{peng2023rwkv,
  title={{RWKV}: Reinventing {RNN}s for the Transformer Era}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="beck2024xlstm">
<pre class="attn-bib__src"><code>@inproceedings{beck2024xlstm,
  title={{xLSTM}: Extended Long Short-Term Memory}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/decay_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/decay_rnn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
