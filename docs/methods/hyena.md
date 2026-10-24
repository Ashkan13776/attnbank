# Hyena

<p class="attn-standfirst" markdown="span">Implicitly parametrised long convolutions interleaved with gating.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `hyena` &middot; `ab.build("hyena")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.HyenaAttention` &middot; [`attnbank/methods/recurrent/hyena.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/hyena.py)

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
| `expresses_weights` | <span class="attn-flag" data-v="no">no</span> | it cannot name $\widehat\pi_q$ on the original positions and is excluded from the error decomposition by declaration. |
| `row_stochastic` | <span class="attn-flag" data-v="no">no</span> | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
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
| `flops` | 44,826,624 | 268,435,456 | 0.167× |
| `state_bytes` | 393,216 | 262,144 | 1.5× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    """``state_bytes`` grows with ``n``: a long convolution has no finite state."""
    logn = max(1.0, math.log2(max(n, 2)))
    return {"flops": self.order * (15 * dv * n * logn + 2 * n * dv * dv)
            + 2 * n * dv * dv,
            "state_bytes": (self.order + 1) * n * dv * bytes_per_el,
            "bytes_moved": (n * dv + m * dv
                            + self.order * n * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `filter_length`, `filters`, `order`, `reads_qk`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `order` | `2` |
| `width` | `16` |
| `n_freqs` | `4` |
| `decay_min` | `0.5` |
| `decay_max` | `8.0` |
| `seed` | `0` |

</div>

`order`: how many (convolution, gate) stages; `2` is the paper's default.

`width` and `n_freqs` size the implicit filter network — the hidden width
and the number of Fourier features in the positional encoding of the lag —
and `decay_min` / `decay_max` bound the log-spaced exponential window, which
is what sets the effective filter length per channel.

`ab.describe("hyena")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("hyena", order=2, width=16)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = modify: this is a different operator and fidelity to softmax
# is not its goal, so there is no error to quote. What it *is* doing shows
# up in its own diagnostics.
print("expresses_weights =", attn.expresses_weights,
      "-- no measure over the original positions exists,",
      "so weights is", out.weights)
print("extras:", sorted(out.extras))
```

<div class="attn-out" markdown="1">

```text
HyenaAttention(name='hyena', category='recurrent', objective='modify', backend='reference', decay_max=8.0, decay_min=0.5, n_freqs=4, order=2, seed=0, width=16)
AttentionOutput(y=[1, 2, 64, 32], weights=None, v_hat=None, extras=[filter_length, filters, order, reads_qk], dtype=float32, device=cpu)
invariant W: 0.0
expresses_weights = False -- no measure over the original positions exists, so weights is None
extras: ['filter_length', 'filters', 'order', 'reads_qk']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:memory-state</code></p>

The Hyena operator of order `N` alternates a long causal
convolution with an elementwise ("data-controlled") gate:

```text
z^1 = x^1,     z^{i+1} = x^{i+1} * (h^i conv z^i),     y = z^{N+1},
```

where `x^1 ... x^{N+1}` are projections of the input and `h^1 ... h^N` are filters
as long as the sequence.  The filters are not stored as `n` free numbers per channel:
they are *implicit*, the output of a small network applied to a positional encoding of
the lag, multiplied by an exponential decay window and normalised.  That is what makes
a filter of unbounded length a bounded number of parameters, and it is the idea Hyena
inherits from continuous-kernel convolutions (Romero et al., 2021).

**The claim this file is here to test, and the caveat.**  `h^i` depends on the lag
alone — it is fixed before any token is seen — so Hyena's *convolutional* branch has
input-independent dynamics exactly like S4, S5 and the LRU.  Its *gating* does not:
`x^{i+1}` is a projection of the input, and multiplying by it makes the composite
operator depend on the content at the output position.  Hyena is therefore a genuine
member of this family in its filters and a partial exception in its gates, which is
precisely the distinction <span class="attn-ref">sec:memory-state</span> draws when it says the step that
produced the current generation of models was to make the *state dynamics* themselves
input-dependent.  Hyena moves the data dependence into a pointwise product sitting
outside the recurrence; Mamba and the gated linear RNNs move it inside.

**No finite state.**  Unlike every other method in `recurrent/`, Hyena's filter is
not realised by a finite-dimensional recurrence: an arbitrary length-`n` FIR filter
needs `O(n)` memory to run causally, which is why `cost()['state_bytes']` grows
with `n` here and does not for `s4d`, `s5` or `lru`.  <span class="attn-ref">prop:memory-rank</span> has
nothing to bound.  The *parameters* are few; the *state* is not.

**It does not read the queries or the keys.**  As in the rest of the family, the three
projections are taken from the value field — fixed seeded `d_v x d_v` matrices, the
stand-ins for the learned ones — because a long convolution has nowhere for `q` and
`k` to enter.  `extras['reads_qk']` is `False`.

**Weights.**  `expresses_weights = False`, twice over: each channel is convolved with
its own filter, and the gate multiplies channel by channel, so the output is not
`sum_j w_{tj} v_j` for any scalar `w` — it is not even linear in `v`.

**Simplifications.**  The implicit filter here is a two-hidden-layer MLP with sine
activations on the positional features `[t, sin(2 pi f t), cos(2 pi f t)]`, windowed
by `exp(-alpha_c t)` with `alpha_c` log-spaced across channels and L1-normalised,
which is the published recipe's shape but not its exact schedule; the short depthwise
convolution Hyena applies to the projections before the recurrence is omitted.  As
everywhere in `recurrent/` the convolution is evaluated directly rather than by FFT,
so that causality is bit-exact (see `s4d`).

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`
:   16 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `hyena` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `hyena` selected](index.md?compare=hyena) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Poli et al. (2023).** *Hyena Hierarchy: Towards Larger Convolutional Language Models.* International Conference on Machine Learning (ICML). [arXiv:2302.10866](https://arxiv.org/abs/2302.10866)<br><span class="attn-cite" data-bibkey="poli2023hyena"><code>poli2023hyena</code></span>
- **Romero et al. (2022).** *CKConv: Continuous Kernel Convolution For Sequential Data.* International Conference on Learning Representations (ICLR). [arXiv:2102.02611](https://arxiv.org/abs/2102.02611)<br><span class="attn-cite" data-bibkey="romero2022ckconv"><code>romero2022ckconv</code></span>
- **Gu et al. (2022).** *Efficiently Modeling Long Sequences with Structured State Spaces.* International Conference on Learning Representations (ICLR). [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)<br><span class="attn-cite" data-bibkey="gu2022efficiently"><code>gu2022efficiently</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="poli2023hyena">
<pre class="attn-bib__src"><code>@inproceedings{poli2023hyena,
  title     = {Hyena Hierarchy: Towards Larger Convolutional Language Models},
  author    = {Poli, Michael and Massaroli, Stefano and Nguyen, Eric and
               Fu, Daniel Y. and Dao, Tri and Baccus, Stephen and Bengio, Yoshua and
               Ermon, Stefano and R{\'e}, Christopher},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2023},
  eprint    = {2302.10866},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2302.10866}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="romero2022ckconv">
<pre class="attn-bib__src"><code>@inproceedings{romero2022ckconv,
  title     = {CKConv: Continuous Kernel Convolution For Sequential Data},
  author    = {Romero, David W. and Kuzina, Anna and Bekkers, Erik J. and
               Tomczak, Jakub M. and Hoogendoorn, Mark},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  eprint    = {2102.02611},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2102.02611}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="gu2022efficiently">
<pre class="attn-bib__src"><code>@inproceedings{gu2022efficiently,
  title     = {Efficiently Modeling Long Sequences with Structured State Spaces},
  author    = {Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  eprint    = {2111.00396},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2111.00396}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/hyena.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/hyena.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
