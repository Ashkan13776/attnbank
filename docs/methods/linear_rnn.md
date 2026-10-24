# Linear attention as a recurrence

<p class="attn-standfirst" markdown="span">H_t = H_{t-1} + psi(k_t) v_t^T.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `linear_rnn` &middot; `ab.build("linear_rnn")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.LinearRNNAttention` &middot; [`attnbank/methods/recurrent/linear_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/linear_rnn.py)

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
| `flops` | 16,777,216 | 268,435,456 | 0.0625× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = d
    return {"flops": 2 * n * r * dv + 2 * m * r * dv,
            "state_bytes": r * dv * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `mass`, `state_rank`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `feature` | `'elu'` |
| `normalize` | `False` |

</div>

`ab.describe("linear_rnn")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("linear_rnn", feature='elu', normalize=False)
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
LinearRNNAttention(name='linear_rnn', category='recurrent', objective='modify', backend='reference', feature='elu', normalize=False)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[mass, state_rank], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [43.946, 84.5266, 119.8822, 162.848]
extras: ['mass', 'state_rank']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:memory-state</code>, <code>prop:memory-suff</code>, <code>prop:memory-rank</code></p>

A separable
interaction of rank `r` admits a sufficient statistic of size `r x d_v`: the prefix
moments `H_t = sum_{i<=t} psi(k_i) v_i^T` (and `z_t = sum psi(k_i)`) are all a
future query needs.  <span class="attn-ref">prop:memory-rank</span> is the converse: an *additive* finite-state
memory forces the interaction to have finite rank, so recall is bounded by the state.

With the normalizer kept this is `linear_elu`.  The deployed form, and the one here,
drops it (`readout = H_t^T phi(q_t)`) because the denominator can approach zero and
produce unbounded gradients.  Three properties of <span class="attn-ref">eq:attention_expectation</span> go with
it (<span class="attn-ref">sec:memory-state</span>): normalization invariance under `g -> c(q) g`, constant
preservation (`y_hat = c <phi(q), z>` now scales with context length), and the
convex-hull bound.  Set `normalize=True` to put the denominator back and watch all
three return.

The weights on the original positions are `w_{tj} = <phi(q_t), psi(k_j)>` for
`j <= t`: nonnegative for a positive feature map, but *not* summing to one.  Invariant
W still holds exactly, which is why this category is in the library at all.

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`
:   16 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `linear_rnn` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `linear_rnn` selected](index.md?compare=linear_rnn) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Katharopoulos et al. (2020).** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* Proceedings of the 37th International Conference on Machine Learning (ICML). [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)<br><span class="attn-cite" data-bibkey="katharopoulos2020transformers"><code>katharopoulos2020transformers</code></span>
- **Schlag et al. (2021).** *Linear Transformers Are Secretly Fast Weight Programmers.* Proceedings of the 38th International Conference on Machine Learning (ICML). [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)<br><span class="attn-cite" data-bibkey="schlag2021linear"><code>schlag2021linear</code></span>
- **Sun et al. (2024).** *Learning to (Learn at Test Time): RNNs with Expressive Hidden States.* [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)<br><span class="attn-cite" data-bibkey="sun2024learning"><code>sun2024learning</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="katharopoulos2020transformers">
<pre class="attn-bib__src"><code>@inproceedings{katharopoulos2020transformers,
  title     = {Transformers are {RNN}s: Fast Autoregressive Transformers with Linear Attention},
  author    = {Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {5156--5165},
  year      = {2020},
  eprint    = {2006.16236},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="schlag2021linear">
<pre class="attn-bib__src"><code>@inproceedings{schlag2021linear,
  title={Linear Transformers Are Secretly Fast Weight Programmers},
  author={Schlag, Imanol and Irie, Kazuki and Schmidhuber, J{\&quot;u}rgen},
  booktitle={Proceedings of the 38th International Conference on Machine Learning (ICML)},
  series={PMLR},
  volume={139},
  pages={9355--9366},
  year={2021},
  eprint={2102.11174},
  archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sun2024learning">
<pre class="attn-bib__src"><code>@misc{sun2024learning,
  title={Learning to (Learn at Test Time): {RNN}s with Expressive Hidden States},
  author={Sun, Yu and Li, Xinhao and Dalal, Karan and Xu, Jiarui and Vikram, Arjun and Zhang, Genghan and others},
  year={2024},
  eprint={2407.04620}, archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/linear_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/linear_rnn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
