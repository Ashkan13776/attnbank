# mLSTM (xLSTM)

<p class="attn-standfirst" markdown="span">Exponential gating, a matrix memory, and a normaliser state.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `mlstm` &middot; `ab.build("mlstm")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.MLSTMAttention` &middot; [`attnbank/methods/recurrent/mlstm.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/mlstm.py)

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
| `flops` | 17,301,504 | 268,435,456 | 0.0645× |
| `state_bytes` | 8,320 | 262,144 | 0.0317× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.state_rank or d
    c = recurrent_cost(m, n, r, dv, bytes_per_el, gate_flops=4)
    c["state_bytes"] += r * bytes_per_el         # the normaliser state n_t
    c["flops"] += 2 * n * r + 2 * m * r          # and its update and readout
    return c
```

## Backends

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `clipped`, `forget_gate`, `input_gate`, `mass`, `stabilizer`, `state_rank`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `forget_bias` | `3.0` |
| `input_bias` | `0.0` |
| `state_rank` | `None` |

</div>

`forget_bias` shifts `f~` (positive = remember longer, the LSTM default).

`input_bias` shifts `i~`.  `state_rank` truncates the key/query dimension
and so *is* the size of the matrix memory, `state_rank x d_v` plus the
`state_rank` of the normaliser; `None` keeps all `d` channels.

`ab.describe("mlstm")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("mlstm", forget_bias=3.0, input_bias=0.0)
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
MLSTMAttention(name='mlstm', category='recurrent', objective='modify', backend='reference', forget_bias=3.0, input_bias=0.0, state_rank=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[clipped, forget_gate, input_gate, mass, stabilizer, state_rank], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [-1.0, -0.0524, -1.0, 1.0]
extras: ['clipped', 'forget_gate', 'input_gate', 'mass', 'stabilizer', 'state_rank']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:memory-state</code>, <code>tab:memory-recurrences</code></p>

The mLSTM is the one member of this
family that keeps a denominator, and keeping it is the whole reason it is here:

```text
C_t = f_t C_{t-1} + i_t v_t k_t^T,            C_t in R^{d_v x r}   (matrix memory)
n_t = f_t n_{t-1} + i_t k_t,                  n_t in R^r           (normaliser)
y_hat_t = C_t q_t / max(|n_t^T q_t|, 1),
```

with `i_t = exp(i~_t)` an *exponential* input gate and `f_t = sigmoid(f~_t)` a
scalar forget gate — so `A_t = f_t Id`, data-dependent but not diagonal: one rate
for the whole head, unlike GLA.  The interest is entirely in the second and third
lines.  `n_t` is the same recurrence driven by `k_t` instead of `k_t v_t^T`, so
`n_t^T q_t` is exactly the row sum of the unnormalised weights: the mLSTM *carries
the denominator of* <span class="attn-ref">eq:attention_expectation</span> *as part of its state*, which linear
attention threw away and which `linear_rnn` does not have.

It still does not get the contract back, and the `max(., 1)` is why.  Dividing by
`max(|n_t^T q_t|, 1)` makes the readout bounded — that was the goal, since an
exponential input gate can make the true denominator astronomically small — but it is
not division by the mass.  When the mass exceeds one the operator *is* normalised and
the row sums to one; when it does not, the row is left alone and sums to less.  Row
sums are therefore in `[0, 1]` and the weights are signed (`q_t . k_t` is a raw
inner product), so normalization invariance and constant preservation fail exactly in
the small-mass regime, which is reported in `extras["clipped"]`.  That partial
restoration is a sharper illustration of <span class="attn-ref">sec:memory-state</span> than dropping the
normaliser outright: the denominator is present and the invariant still breaks.

Exponential gating needs the paper's stabiliser or it overflows at the first large
logit.  With `c = cumsum(log f)`,

```text
m_t = max_{j <= t} ( log i_j + c_t - c_j ) = c_t + cummax( log i - c )_t,
```

and every stabilised coefficient `a_{tj} = exp(log i_j + c_t - c_j - m_t)` lies in
`(0, 1]` by construction.  The denominator floor becomes `exp(-m_t)` in the same
units.  This is the recurrence `m_t = max(log f_t + m_{t-1}, log i_t)` of the paper,
written as the prefix scan it is.

The gates.  A trained mLSTM reads `i~_t` and `f~_t` off `x_t` through learned
projections.  With no training this reference takes two fixed orthogonal directions of
the key — the mean `<k_t, 1>/sqrt(d)` for forgetting and the alternating-sign
direction `<k_t, (-1)^c>/sqrt(d)` for writing — plus the biases below.  They are
stand-ins, stated rather than hidden; what matters downstream is that both gates are
scalar per head and functions of the token, and both are.

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`
:   16 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `mlstm` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `mlstm` selected](index.md?compare=mlstm) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Beck et al. (2024).** *xLSTM: Extended Long Short-Term Memory.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2405.04517](https://arxiv.org/abs/2405.04517)<br><span class="attn-cite" data-bibkey="beck2024xlstm"><code>beck2024xlstm</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="beck2024xlstm">
<pre class="attn-bib__src"><code>@inproceedings{beck2024xlstm,
  title={{xLSTM}: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\&quot;o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\&quot;u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  eprint={2405.04517}, archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/mlstm.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/mlstm.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
