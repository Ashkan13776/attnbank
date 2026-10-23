# HGRN2

<p class="attn-standfirst" markdown="span">A gated *vector* recurrence expanded to a matrix state by an outer product.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `hgrn2` &middot; `ab.build("hgrn2")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.HGRN2Attention` &middot; [`attnbank/methods/recurrent/hgrn2.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/hgrn2.py)

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
| `flops` | 16,973,824 | 268,435,456 | 0.0632× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.state_rank or d
    # no key projection: the gate is the key, so the per-step gate work is the
    # only extra over the additive recurrence.
    return recurrent_cost(m, n, r, dv, bytes_per_el, gate_flops=3)
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

Diagnostics returned in `extras`: `forget_gate`, `input_gate`, `lower_bound`, `mass`, `state_rank`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `lower_bound` | `0.9` |
| `gate_bias` | `0.0` |
| `normalize` | `False` |
| `state_rank` | `None` |

</div>

`lower_bound`: this layer's `gamma` in `f = gamma + (1 - gamma) sigmoid`.

It is the hierarchical schedule's one free number — 0 at the bottom of the
stack, approaching 1 at the top; the default sits mid-stack, because at 0 the
untrained gate averages one half and the state keeps barely a token.
`state_rank` truncates the gate/key dimension and so *is* the size of the
recurrent state, `state_rank x d_v`.

```python
ab.describe("hgrn2")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("hgrn2", lower_bound=0.9, gate_bias=0.0)
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
HGRN2Attention(name='hgrn2', category='recurrent', objective='modify', backend='reference', gate_bias=0.0, lower_bound=0.9, normalize=False, state_rank=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[forget_gate, input_gate, lower_bound, mass, state_rank], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [0.0886, 0.6624, 0.4207, -0.2597]
extras: ['forget_gate', 'input_gate', 'lower_bound', 'mass', 'state_rank']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:memory-state`, `tab:memory-recurrences`.  HGRN is an element-wise gated RNN,

```text
h_t = f_t * h_{t-1} + (1 - f_t) * i_t,                     h_t in R^d,
```

whose state is a *vector*: `prop:memory-rank` caps what it can recall at `d`
numbers however wide the layer is.  HGRN2 changes one symbol — the element-wise
product becomes an outer product —

```text
H_t = Diag(f_t) H_{t-1} + (1 - f_t) v_t^T,                 H_t in R^{d x d_v},
y_hat_t = H_t^T q_t,
```

and the state becomes a matrix of `d x d_v` numbers at exactly the same parameter
count.  In the vocabulary of this category that is GLA with the key *tied to the gate*,
`k_t = 1 - f_t` and `alpha_t = f_t`: HGRN2 has no separate key projection at all,
which is the point — the expansion is free.

Two consequences worth stating, because they are what distinguishes this row from GLA.

*Per channel the write is a convex combination.*  Summing the unrolled recurrence
channel by channel, `sum_{j <= t} (1 - f_j) prod_{u > j} f_u = 1 - prod_{u <= t} f_u
<= 1`.  So the state cannot grow without bound in `t`, and the weights

```text
w_{tj} = sum_r q_{t,r} (1 - f_{j,r}) prod_{u = j+1}^{t} f_{u,r}
```

have row sum `<q_t, 1 - prod_{u <= t} f_u>` — bounded, readable, and equal to one
only by accident.  Boundedness without normalization again (`sec:memory-state`); the
rows are not a probability measure and the operator declares
`row_stochastic = False`.

*The lower bound is where "hierarchical" lives.*  HGRN sets
`f_t = gamma + (1 - gamma) sigmoid(.)` and makes `gamma` increase monotonically
with depth (via `cumax` over layers), so lower layers forget fast and model local
structure while upper layers approach `f = 1` and carry long-range state.  A single
operator cannot be hierarchical on its own; `lower_bound=` is that layer's `gamma`,
and sweeping it from 0 to 1 traces the depth axis of the real model.  At
`lower_bound=1` the gate is stuck open, nothing is ever written (`1 - f = 0`) and
the output is exactly zero — the degenerate end of the schedule, reported rather than
clamped away.

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`gqa`](gqa.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`lowrank_ot`](lowrank_ot.md), [`mla`](mla.md), [`otke`](otke.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `hgrn2` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `hgrn2` selected](index.md?compare=hgrn2) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *HGRN2: Gated Linear RNNs with State Expansion.*<br><span class="attn-cite" data-bibkey="qin2024hgrn2"><code>qin2024hgrn2</code></span>
- *Hierarchically Gated Recurrent Neural Network for Sequence Modeling.*<br><span class="attn-cite" data-bibkey="qin2023hgrn"><code>qin2023hgrn</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="qin2024hgrn2">
<pre class="attn-bib__src"><code>@inproceedings{qin2024hgrn2,
  title={{HGRN2}: Gated Linear {RNN}s with State Expansion}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="qin2023hgrn">
<pre class="attn-bib__src"><code>@inproceedings{qin2023hgrn,
  title={Hierarchically Gated Recurrent Neural Network for Sequence Modeling}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/hgrn2.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/hgrn2.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
