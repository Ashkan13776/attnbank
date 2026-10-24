# Retention (RetNet)

<p class="attn-standfirst" markdown="span">A *multi-scale* fixed decay, one time constant per head.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `retnet` &middot; `ab.build("retnet")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.RetNetAttention` &middot; [`attnbank/methods/recurrent/retnet.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/retnet.py)

monograph
:   `sec:memory-state`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the operator. $y$ is produced by a finite-state recurrence over positions rather than by an integral against a measure — though it is still linear in $v$, which is why the weights can be recovered at all.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="yes">yes</span> | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="no">no</span> | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 16,842,752 | 268,435,456 | 0.0627× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.state_rank or d
    return recurrent_cost(m, n, r, dv, bytes_per_el)
```

## Backends

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `decay`, `gamma`, `half_life`, `mass`, `state_rank`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `decay` | `None` |
| `base_exponent` | `5` |
| `normalize` | `False` |
| `state_rank` | `None` |

</div>

`decay`: one rate for every head; `None` uses the paper's head schedule.

`base_exponent` is the `5` in `gamma_h = 1 - 2^{-5-h}`.  `state_rank`
truncates the key/query dimension, which *is* the size of the recurrent state
(`state_rank x d_v`); `None` keeps all `d` channels.

`ab.describe("retnet")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("retnet", base_exponent=5, normalize=False)
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
RetNetAttention(name='retnet', category='recurrent', objective='modify', backend='reference', base_exponent=5, decay=None, normalize=False, state_rank=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[decay, gamma, half_life, mass, state_rank], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [-0.6341, -1.0144, -1.6115, 1.0348]
extras: ['decay', 'gamma', 'half_life', 'mass', 'state_rank']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:memory-state</code>, <code>tab:memory-recurrences</code> (row &quot;Retention&quot;)</p>

The recurrence is

```text
S_n = gamma S_{n-1} + k_n v_n^T,      y_hat_n = S_n^T q_n,
```

and its parallel form is the one the paper writes,

```text
Retention(X) = (Q K^T * D) V,    D_{nm} = gamma^{n-m} for n >= m, else 0,
```

so `A_t = gamma Id` is a scalar multiple of the identity and does **not** depend on
the input.  That is the whole content of the row: retention is the input-independent
member of this family, and everything after it (GLA, HGRN2, RWKV-6, the mLSTM) is the
same recurrence with `A_t` made a function of `x_t`.

What `decay_rnn` does not already cover is the *multi-scale* part.  RetNet runs
`h` heads at `h` different time constants,

```text
gamma = 1 - 2^{-5 - arange(0, h)},
```

so head 0 keeps a half-life of about 22 tokens and head `h-1` is almost lossless.
A single scalar `gamma` — `decay_rnn` — has to choose one point on that curve;
the head schedule is how RetNet gets a range of recency priors out of one operator at
one state size, which is exactly the recency-versus-capacity separation study S6 is
after.  Pass `decay=` to collapse the schedule back to one rate and recover
`decay_rnn` with the identity feature map.

Two further differences from `decay_rnn`, both from the paper.  The interaction is
the raw scaled inner product `<q_t, k_j> / sqrt(d)`, with no positive feature map, so
the weights are **signed**; and the readout carries no denominator, so the rows do not
sum to one.  The effective mass is still bounded by `1 / (1 - gamma)` rather than
growing with `n` — boundedness without normalization, the distinction
<span class="attn-ref">sec:memory-state</span> draws.  `normalize=True` puts the denominator back.

xPos relative position embedding, the GroupNorm on the readout and the swish gate are
part of the RetNet *block*, not of the retention operator, and are left out: this file
is the operator of <span class="attn-ref">tab:memory-recurrences</span>.

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`
:   16 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `retnet` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `retnet` selected](index.md?compare=retnet) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Sun et al. (2023).** *Retentive Network: A Successor to Transformer for Large Language Models.* [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)<br><span class="attn-cite" data-bibkey="sun2023retentive"><code>sun2023retentive</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="sun2023retentive">
<pre class="attn-bib__src"><code>@misc{sun2023retentive,
  title={Retentive Network: A Successor to Transformer for Large Language Models},
  author={Sun, Yutao and Dong, Li and Huang, Shaohan and Ma, Shuming and Xia, Yuqing and Xue, Jilong and Wang, Jianyong and Wei, Furu},
  year={2023},
  eprint={2307.08621},
  archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/retnet.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/retnet.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
