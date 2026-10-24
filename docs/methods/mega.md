# Mega

<p class="attn-standfirst" markdown="span">A multi-dimensional damped EMA in front of chunked gated attention.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `mega` &middot; `ab.build("mega")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.MegaAttention` &middot; [`attnbank/methods/recurrent/mega.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/mega.py)

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
| `flops` | 3,145,728 | 268,435,456 | 0.0117× |
| `state_bytes` | 2,560 | 262,144 | 0.00977× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    c = min(n if self.chunk is None else self.chunk, n)
    return {"flops": 4 * (m + n) * d * max(self.ndim, 1) + 2 * m * c * (d + dv),
            # one ndim-dimensional EMA state per feature, plus one chunk of KV
            "state_bytes": (2 * max(self.ndim, 1) * d + c * (d + dv)) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chunk`, `ema`, `ndim`, `support_size`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `chunk` | `8` |
| `ndim` | `2` |
| `ema` | `True` |
| `seed` | `0` |

</div>

`chunk`: attention span in positions (`None` = one chunk, i.e. no chunking).

`ndim` is the EMA's expansion — the dimension of the state carried per
feature, Mega's `h` — and `ema=False` removes the moving average, leaving
the chunked attention alone.

`ab.describe("mega")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("mega", chunk=8, ndim=2)
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
MegaAttention(name='mega', category='recurrent', objective='modify', backend='reference', chunk=8, ema=True, ndim=2, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chunk, ema, ndim, support_size], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['chunk', 'ema', 'ndim', 'support_size']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:memory-state</code>, <code>sec:sparse</code></p>

Mega keeps softmax attention but puts a
recurrence in front of it.  A multi-dimensional damped exponential moving average
smooths the sequence before the queries and keys are read off it,

```text
h_t^{(j)} = alpha_j * (beta_j x_{t,j}) + (1 - alpha_j * delta_j) * h_{t-1}^{(j)},
EMA(x)_{t,j} = <eta_j, h_t^{(j)}>,
```

one `ndim`-dimensional state per feature `j`, and the attention that follows is
restricted to chunks of `c` positions so that the cost is linear in the sequence
length.  In the bank's signature that reads

```text
q' = EMA(q),   k' = EMA(k),   pi_t = softmax_{j in chunk(t), j <= t} (<q'_t, k'_j> / sqrt d),
y_t = sum_j pi_t(j) v_j,
```

with `v` left alone — in Mega the values come from the input directly while `Q`
and `K` are read off the smoothed sequence, which is the whole mechanism: the EMA
supplies the local, position-based inductive bias that lets a *chunk-local* attention
still see something about the rest of the sequence.

**Input-independent dynamics, with a real caveat.**  The EMA's coefficients
`alpha, delta, beta, eta` are parameters; the recurrence `h_t = A h_{t-1} + B x_t`
has `A` independent of `x_t`, so the EMA belongs to this family exactly as S4D and
the LRU do.  The attention branch does not: `pi_t` is computed from the content of
`q'` and `k'` and is as input-dependent as any softmax.  Mega is thus *not* an
instance of the family's defining restriction taken as a whole — only its recurrent
component is — and the operator is a hybrid rather than a pure state-space layer.
<span class="attn-ref">sec:memory-state</span> records the same caveat about the headline model's cost: full
gated attention plus an EMA is quadratic, and only the chunked variant implemented here
(`chunk` not `None`) is linear.

**Weights.**  Unlike the rest of this family, Mega *does* express a measure on the
original `n` positions, and a row-stochastic one: the chunk-restricted softmax.  The
EMA changes which positions get the mass, not whether the mass is a probability.  So
`expresses_weights = True`, `row_stochastic = True`, and Mega is the member of the
state-space lineage that the decomposition studies can actually score.

**Omitted.**  Mega's output gates (`sigmoid`-weighted residual mixing) and its
single-head shared-representation projections are not modelled: they act elementwise on
`y` after the aggregation, so including them would break Invariant W without changing
anything about the measure.  The EMA parameters and the `q`/`k` smoothing are what
the bank models.  The EMA is kept unidirectional (causal) even when `causal=False`,
so that the chunk boundary is the only thing the flag moves.

**No softmax anchor is declared.**  `objective = MODIFY`: Mega was never an
approximation of dense attention.  It happens to contain one — switch the EMA off and
remove the chunking and the operator *is* softmax — and `tests/test_recurrent_ssm.py`
uses that as a correctness check, but declaring it as `limit_config` would claim
fidelity to softmax as Mega's goal, which is not what the paper proposes.

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`
:   16 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `mega` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `mega` selected](index.md?compare=mega) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Ma et al. (2023).** *Mega: Moving Average Equipped Gated Attention.* International Conference on Learning Representations (ICLR). [arXiv:2209.10655](https://arxiv.org/abs/2209.10655)<br><span class="attn-cite" data-bibkey="ma2023mega"><code>ma2023mega</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="ma2023mega">
<pre class="attn-bib__src"><code>@inproceedings{ma2023mega,
  title     = {Mega: Moving Average Equipped Gated Attention},
  author    = {Ma, Xuezhe and Zhou, Chunting and Kong, Xiang and He, Junxian and
               Gui, Liangke and Neubig, Graham and May, Jonathan and
               Zettlemoyer, Luke},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  eprint    = {2209.10655},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2209.10655}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/mega.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/mega.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
