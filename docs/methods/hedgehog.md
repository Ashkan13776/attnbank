# Hedgehog

<p class="attn-standfirst" markdown="span">A feature map *fitted* to the softmax attention it is replacing.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `hedgehog` &middot; `ab.build("hedgehog")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.HedgehogAttention` &middot; [`attnbank/methods/interaction/hedgehog.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/hedgehog.py)

monograph
:   `sec:kernel`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the interaction $g(q,x)$, replaced by a surrogate $\widehat g$. $\pi_q = \widehat g(q,\cdot)\mu / \widehat Z(q)$ is then a different measure on the same $n$ positions.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="no">no</span> | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="yes">yes</span> | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

</div>

## Correctness anchor


**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("hedgehog").convergence_configs()
[{'steps': 0}, {'steps': 20}, {'steps': 80}, {'steps': 320}]
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 67,108,864 | 268,435,456 | 0.25× |
| `state_bytes` | 16,640 | 262,144 | 0.0635× |
| `bytes_moved` | 540,928 | 4,718,592 | 0.115× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = 2 * (self.r or d)
    return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv,
            "state_bytes": r * (dv + 1) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_hat`, `cross_entropy`, `g_hat`, `phi_k`, `phi_q`, `r`, `steps`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `None` |
| `steps` | `100` |
| `lr` | `0.05` |
| `seed` | `0` |

</div>

`r` feature directions (`2r` features); `None` means the head dim `d`.

`ab.describe("hedgehog")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("hedgehog", steps=100, lr=0.05)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["hedgehog"], q, k, v, causal=False,
                 configs={"hedgehog": {'steps': 100, 'lr': 0.05}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
HedgehogAttention(name='hedgehog', category='interaction', objective='approximate', backend='reference', lr=0.05, r=None, seed=0, steps=100)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_hat, cross_entropy, g_hat, phi_k, phi_q, r, steps], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.19257, 'distributional_error': 0.19257, 'value_error': 0.0, 'tv': 0.09522}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:kernel-det</code>, <code>sec:kernel-limits</code></p>

Every other entry in this directory picks
`phi` in advance — `exp(omega^T u - ||u||^2/2)`, `elu + 1`, `relu`, `x^2`,
`1 + z + z^2/2` — and then asks how close the induced `pi_hat` lands.  Zhang et
al. (2024) invert the question: fix the separable *form*, learn the map.  Their
diagnosis is that linear attentions fail on two measurable axes, spikiness and
monotonicity of the induced weights, and their feature map is built to restore both,

```text
phi_W(u) = softmax([ W u + b ; -(W u + b) ]) in R^{2r},
```

— an exponential map (spiky) run through a softmax over the feature axis (bounded,
so it never overflows), with the negated copy appended so that the map is monotone in
`<q,k>` in both directions.  Queries and keys get their own `(W, b)`.  The
parameters are then obtained by *distillation*: the paper's Eq. 7 minimises the
cross-entropy between the induced measure and the true one,

```text
L = - sum_i sum_j pi_q_i(j) log pi_hat_q_i(j),
```

which is the KL from softmax to `pi_hat` up to a constant, so the fit is an explicit
projection of softmax onto the rank-`2r` separable family.

This file performs that projection **inside the forward pass**, per `(batch, head)`,
on the tensors it is given, with Adam in float64 from a seeded initialization.  It is
therefore an *oracle* in the sense of `nmf_oracle` and `svd_oracle`: it reports
how well the best feature map of width `2r` that this objective can find does, which
upper-bounds what a deployed Hedgehog achieves after distillation on a corpus.  Two
consequences are declared rather than hidden.  `trainable = True` — the defaults are
not a method, they are an initialization plus a fitting budget.  And
`supports_causal = False`: the fitted `W` depends on the whole attention matrix,
so an early query's output would move when a later key changed.  A deployed Hedgehog
distills once and then runs causally; this one cannot, and saying so is cheaper than
a mask that does not mean what it says.

The anchor is `convergence_configs` over `steps`.  More optimization lowers the
cross-entropy, which lowers the KL, which lowers the output error — monotonically in
the mean, and towards softmax, which no finite budget reaches because the family has
rank `2r`.  `steps = 0` is the untrained random feature map and is the worst point
of the sequence by construction.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`linear_elu`](linear_elu.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polaformer`](polaformer.md), [`polysketch`](polysketch.md), [`rebased`](rebased.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `approximate`
:   36 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `hedgehog` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `hedgehog` selected](index.md?compare=hedgehog) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Zhang et al. (2024).** *The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry.* International Conference on Learning Representations (ICLR). [arXiv:2402.04347](https://arxiv.org/abs/2402.04347)<br><span class="attn-cite" data-bibkey="zhang2024hedgehog"><code>zhang2024hedgehog</code></span>
- **Katharopoulos et al. (2020).** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* Proceedings of the 37th International Conference on Machine Learning (ICML). [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)<br><span class="attn-cite" data-bibkey="katharopoulos2020transformers"><code>katharopoulos2020transformers</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="zhang2024hedgehog">
<pre class="attn-bib__src"><code>@inproceedings{zhang2024hedgehog,
  title     = {The Hedgehog \&amp; the Porcupine: Expressive Linear Attentions with Softmax Mimicry},
  author    = {Zhang, Michael and Bhatia, Kush and Kumbong, Hermann and R{\'e}, Christopher},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024},
  eprint    = {2402.04347},
  archivePrefix = {arXiv}
}</code></pre>
</div>
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
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/hedgehog.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/hedgehog.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
