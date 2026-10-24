# Linear transformer

<p class="attn-standfirst" markdown="span">The fixed feature map `phi(u) = elu(u) + 1`.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `linear_elu` &middot; `ab.build("linear_elu")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.LinearELUAttention` &middot; [`attnbank/methods/interaction/linear_elu.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/linear_elu.py)

monograph
:   `sec:kernel`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the interaction $g(q,x)$, replaced by a surrogate $\widehat g$. $\pi_q = \widehat g(q,\cdot)\mu / \widehat Z(q)$ is then a different measure on the same $n$ positions.

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
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 16,908,288 | 268,435,456 | 0.063× |
| `state_bytes` | 8,320 | 262,144 | 0.0317× |
| `bytes_moved` | 532,608 | 4,718,592 | 0.113× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = d
    return {"flops": 2 * n * r * dv + 2 * m * r * dv + (m + n) * r,
            "state_bytes": r * (dv + 1) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_hat`, `g_hat`.

## Configuration

This method takes no configuration of its own — only the base class's `scale` and `backend`.

`ab.describe("linear_elu")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("linear_elu")
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
LinearELUAttention(name='linear_elu', category='interaction', objective='modify', backend='reference')
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_hat, g_hat], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['Z_hat', 'g_hat']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:kernel-det</code></p>

Katharopoulos et al. (2020) replace the interaction by

```text
g_hat(q, x) = <phi(q), phi(k_x)>,   phi(u) = elu(u) + 1 > 0,   r = d,
```

with no resource parameter, so `<phi(q), phi(k)> -> exp(beta <q,k>)` holds *in no
sense*: this is an **operator modification**, not an approximation
(<span class="attn-ref">def:kernel-regimes</span>, <span class="attn-ref">tab:kernel-methods</span> row "Linear transformer ... no").  The
library records that by `objective = modify` and by returning `None` from both
correctness anchors: there is no configuration under which it becomes softmax, and
nothing it converges to.

Because `phi > 0` the induced measure is a genuine probability measure, so this is
one of the normalized members of the family; <span class="attn-ref">lem:kernel-mixture</span> then says
`pi_hat_q` is a mixture of `r = d` *query-independent* context distributions, and
<span class="attn-ref">prop:kernel-retrieval</span> bounds what it can retrieve.  <span class="attn-ref">rem:kernel-homogeneous</span>
applies approximately: for large positive coordinates `phi` is linear, so the query
norm cannot sharpen `pi_hat_q`.

The normalized form of <span class="attn-ref">eq:kernel-moments</span> is used (divide by `phi(q)^T z`).  Many
deployed descendants drop the denominator and normalize the output instead
(<span class="attn-ref">sec:memory-state</span>); that variant lives in `recurrent/linear_rnn.py`, where the
loss of normalization invariance is the point.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polaformer`](polaformer.md), [`polysketch`](polysketch.md), [`rebased`](rebased.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `modify`
:   25 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `linear_elu` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `linear_elu` selected](index.md?compare=linear_elu) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Katharopoulos et al. (2020).** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* Proceedings of the 37th International Conference on Machine Learning (ICML). [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)<br><span class="attn-cite" data-bibkey="katharopoulos2020transformers"><code>katharopoulos2020transformers</code></span>
- **Schlag et al. (2021).** *Linear Transformers Are Secretly Fast Weight Programmers.* Proceedings of the 38th International Conference on Machine Learning (ICML). [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)<br><span class="attn-cite" data-bibkey="schlag2021linear"><code>schlag2021linear</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
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
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/linear_elu.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/linear_elu.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
