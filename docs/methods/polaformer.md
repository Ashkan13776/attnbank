# PolaFormer

<p class="attn-standfirst" markdown="span">Polarity-aware linear attention, and the measure it cannot express.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `polaformer` &middot; `ab.build("polaformer")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.PolaFormerAttention` &middot; [`attnbank/methods/interaction/polaformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/polaformer.py)

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
| `flops` | 67,633,152 | 268,435,456 | 0.252× |
| `state_bytes` | 33,280 | 262,144 | 0.127× |
| `bytes_moved` | 557,568 | 4,718,592 | 0.118× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = 2 * d                       # [f(x+), f(x-)], used in two pairings
    return {"flops": 4 * n * r * dv + 4 * m * r * dv + 2 * (m + n) * r,
            "state_bytes": 2 * r * (dv + 1) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv
                            + 2 * r * (dv + 1)) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_opposite`, `Z_same`, `power`, `split`, `weights_opposite_sign`, `weights_same_sign`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `power` | `3.0` |

</div>

`ab.describe("polaformer")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("polaformer", power=3.0)
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
PolaFormerAttention(name='polaformer', category='interaction', objective='modify', backend='reference', power=3.0)
AttentionOutput(y=[1, 2, 64, 32], weights=None, v_hat=None, extras=[Z_opposite, Z_same, power, split, weights_opposite_sign, weights_same_sign], dtype=float32, device=cpu)
invariant W: 0.0
expresses_weights = False -- no measure over the original positions exists, so weights is None
extras: ['Z_opposite', 'Z_same', 'power', 'split', 'weights_opposite_sign', 'weights_same_sign']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:kernel-det</code></p>

Every nonnegative feature map in this directory throws away the
sign structure of the logit: `relu` deletes the negative coordinates outright, and
with them the opposite-signed part of `<q, k>`.  Meng et al. (2025) keep it.  Split
each vector into its positive and negative parts, `q = q+ - q-` with
`q+ = max(q,0)`, so that

```text
<q, k> = (<q+,k+> + <q-,k->) - (<q+,k-> + <q-,k+>),
```

and build two *separate* nonnegative interactions from the two halves, each sharpened
by an elementwise power `f(x) = x^p` with `p = 1 + alpha sigmoid(w)` learned per
channel (their Eq. 9; `p > 1` has positive first and second derivatives, which is
their entropy-reduction condition):

```text
g_same(q,x) = <f(q+), f(k+)> + <f(q-), f(k-)>,
g_opp (q,x) = <f(q-), f(k+)> + <f(q+), f(k-)>.
```

**Invariant W fails here, and that is the finding.**  The paper does not subtract the
two — that would make the surrogate signed and the measure disappear.  It row-
normalizes each separately and sends them to *different halves of the value
channels*: with `v = [v1, v2]`,

```text
y = [ pi_same v1 , pi_opp v2 ].
```

Each of `pi_same` and `pi_opp` is a genuine probability measure, but the output is
not an average of `v` against *one* measure on the `n` positions, so no single
`pi_hat` exists and the method declares `expresses_weights = False`.  It is the
clearest example in the bank of an operator that is nonnegative, normalized and
channel-wise convex, yet outside the framework's single-measure form — the value
field is not shared across output coordinates.  Both measures are returned in
`extras` so the pair can still be studied.

PolaFormer the vision backbone adds a learned per-channel temperature, a depthwise
convolution on the values and a gating branch; those are architecture, not `g`, and
are left out, exactly as `cosformer` leaves out TransNormer's surroundings.  The
one exception is the per-channel temperature: at initialization it is a single
constant on both `q` and `k`, which cancels in the row normalization, so omitting
it changes nothing at the defaults.

`power` defaults to `3`, the initialization `1 + alpha sigmoid(0)` at the
paper's `alpha = 4`.  It is learned, so `trainable = True`; `power = 1` is the
plain polarity split with no sharpening.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`linear_elu`](linear_elu.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`rebased`](rebased.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `modify`
:   25 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `polaformer` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `polaformer` selected](index.md?compare=polaformer) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Meng et al. (2025).** *PolaFormer: Polarity-aware Linear Attention for Vision Transformers.* International Conference on Learning Representations (ICLR). [arXiv:2501.15061](https://arxiv.org/abs/2501.15061)<br><span class="attn-cite" data-bibkey="meng2025polaformer"><code>meng2025polaformer</code></span>
- **Katharopoulos et al. (2020).** *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* Proceedings of the 37th International Conference on Machine Learning (ICML). [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)<br><span class="attn-cite" data-bibkey="katharopoulos2020transformers"><code>katharopoulos2020transformers</code></span>
- **Qin et al. (2022).** *cosFormer: Rethinking Softmax in Attention.* International Conference on Learning Representations (ICLR). [arXiv:2202.08791](https://arxiv.org/abs/2202.08791)<br><span class="attn-cite" data-bibkey="qin2022cosformer"><code>qin2022cosformer</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="meng2025polaformer">
<pre class="attn-bib__src"><code>@inproceedings{meng2025polaformer,
  title     = {{PolaFormer}: Polarity-aware Linear Attention for Vision Transformers},
  author    = {Meng, Weikang and Luo, Yadan and Li, Xin and Jiang, Dongmei and Zhang, Zheng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2501.15061},
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
<div class="attn-bib__entry" data-bibkey="qin2022cosformer">
<pre class="attn-bib__src"><code>@inproceedings{qin2022cosformer,
  title={{cosFormer}: Rethinking Softmax in Attention},
  author={Qin, Zhen and Sun, Weixuan and Deng, Hui and Li, Dongxu and Wei, Yunshen and Lv, Baohong and Yan, Junjie and Kong, Lingpeng and Zhong, Yiran},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
  eprint={2202.08791},
  archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/polaformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/polaformer.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
