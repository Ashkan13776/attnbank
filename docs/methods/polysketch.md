# Polynomial sketch

<p class="attn-standfirst" markdown="span">Truncate exp to degree P, then sketch each tensor power.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `polysketch` &middot; `ab.build("polysketch")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.PolySketchAttention` &middot; [`attnbank/methods/interaction/polysketch.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/polysketch.py)

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
| `supports_causal` | <span class="attn-flag" data-v="yes">yes</span> | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="no">no</span> | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor


**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("polysketch").convergence_configs()
[{'degree': 2, 'sketch': 8}, {'degree': 4, 'sketch': 32}, {'degree': 6, 'sketch': 128}, {'degree': 8, 'sketch': 512}]
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 312,741,888 | 268,435,456 | 1.17× |
| `state_bytes` | 66,690 | 262,144 | 0.254× |
| `bytes_moved` | 590,978 | 4,718,592 | 0.125× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    if self.sketch is None:                       # exact kernel: the dense path
        return {"flops": 2 * m * n * d + 2 * m * n * (self.degree + 1),
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
    s = self.sketch
    df = 1 + self.degree * s
    sketching = (m + n) * self.degree * (2 * d * s + 5 * s * math.log2(2 * s))
    return {"flops": sketching + 2 * (m + n) * df * (dv + 1),
            "state_bytes": df * (dv + 1) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv
                            + df * (dv + 1)) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_rel`, `degree`, `feature_dim`, `min_weight`, `neg_mass`, `sketch`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `degree` | `4` |
| `sketch` | `128` |
| `seed` | `0` |

</div>

`degree`: truncation order `P`.  `sketch`: buckets `s` per tensor
power, or `None` to evaluate the polynomial kernel exactly.

`ab.describe("polysketch")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("polysketch", degree=4, sketch=128)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["polysketch"], q, k, v, causal=True,
                 configs={"polysketch": {'degree': 4, 'sketch': 128}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
PolySketchAttention(name='polysketch', category='interaction', objective='approximate', backend='reference', degree=4, seed=0, sketch=128)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_rel, degree, feature_dim, min_weight, neg_mass, sketch], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 2.80379, 'distributional_error': 2.8038, 'value_error': 0.0, 'tv': 0.97174}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<span class="attn-ref">sec:kernel-rf</span>, the deterministic half.  The softmax interaction is an entire
function of the inner product,

```text
g(q, x) = exp(beta <q, k_x>) = sum_{p >= 0} <qt, kt_x>^p / p!,
qt = beta q,  kt = k,
```

so truncating the series at degree `P` gives a surrogate that is a *polynomial
kernel of degree P* — exactly representable by the feature map
`x -> (1, x, x^{(x)2}/sqrt(2!), ..., x^{(x)P}/sqrt(P!))` of dimension
`sum_p d^p`, which is why nobody writes it down.  The sketching literature does the
only sensible thing with that feature map: it never forms it.  TensorSketch
(Pham and Pagh, 2013) replaces the `p`-fold tensor power by `s` buckets,

```text
TS_p(x) = FFT^{-1}( prod_{i=1}^{p} FFT(CountSketch_i(x)) ) in R^s,
E <TS_p(x), TS_p(y)> = <x, y>^p,
```

and Ahle et al. (2019) assemble exactly this sum of sketched powers into an oblivious
sketch of the Gaussian / exponential kernel.  PolySketchFormer (Kacham et al., 2024)
is the attention instance of the same construction.  The shipped feature map is

```text
Psi(x) = [ 1, TS_1(x), TS_2(x)/sqrt(2!), ..., TS_P(x)/sqrt(P!) ] in R^{1 + P s},
g_hat(q, x) = <Psi(qt), Psi(kt_x)>,
```

so this is a linear-attention method with an explicit `1 + P s` dimensional state —
the point of the whole exercise, since the unsketched map needs `sum_p d^p`.
`sketch=None` evaluates the same polynomial exactly through powers of the inner
product and is the control the sketch is measured against; at `degree = 2` that
control is `based`, which is the same surrogate with the tensor-power feature map
written out in full. The difference between the two files is the sketch, and the
resource that converges here is the feature budget `1 + P s` rather than `P` alone.

Two error sources, and they are different in kind.  The truncation error is
**deterministic** and vanishes as `P -> infinity`, which is why this is an
`approximate` method and not a `modify` one.  The sketch error is Monte-Carlo and
vanishes as `s -> infinity` at fixed `P`; its variance grows like
`(||qt|| ||kt||)^{2p}`, so a high degree needs normalised queries and keys to be
usable at all — the reason PolySketchFormer normalises them.  `convergence_configs`
raises both at once, because that is the knob a user actually has: the pair `(P, s)`
is one feature budget and neither half converges alone.

What is lost.  An even-degree truncation of `exp` is strictly positive on the whole
real line, so with `sketch=None` and even `degree` the surrogate is nonnegative and
`pi_hat` is a genuine probability measure — that is what `based` relies on.  The
sketch destroys it: `<Psi(q), Psi(k)>` is an unbiased estimate of a positive number
and takes negative values, so `pi_hat` is a signed measure of unit mass.  The class
declares the weaker property, `row_stochastic = False`, and reports the negative mass
in `extras` rather than assuming it away.

A sharper caveat, worth more than the two above.  Softmax is invariant to adding a
constant to a row of logits; a truncated Taylor series is not.  `g_hat` is a
polynomial, so it cannot be shifted before evaluation the way `exp` can, and in the
large-logit regime the surrogate is dominated by its top term and `pi_hat` tends to
the softmax of `P log|s|` rather than of `s`.  Accuracy here is a statement about
the *scale* of the logits, not only about `P`.

Numerics.  Feature maps and their inner products are evaluated in float64 whatever
the input dtype, as in `nystrom`, so that the error measured is the estimator's.
Even so, a sketched degree-`P` polynomial is an ill-conditioned object once the
logits are large: the terms grow like `s^P` and cancel, and float32 inputs then
agree with float64 ones only to about `1e-3`.  That is the same large-logit regime
the caveat above is about, and in it the estimator has already stopped tracking
softmax — the precision loss is a symptom, not the disease.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`linear_elu`](linear_elu.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `approximate`
:   36 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `polysketch` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `polysketch` selected](index.md?compare=polysketch) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Pham and Pagh (2013).** *Fast and Scalable Polynomial Kernels via Explicit Feature Maps.* Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. [doi:10.1145/2487575.2487591](https://doi.org/10.1145/2487575.2487591)<br><span class="attn-cite" data-bibkey="pham2013fast"><code>pham2013fast</code></span>
- **Ahle et al. (2019).** *Oblivious Sketching of High-Degree Polynomial Kernels.* [arXiv:1909.01410](https://arxiv.org/abs/1909.01410)<br><span class="attn-cite" data-bibkey="ahle2019oblivious"><code>ahle2019oblivious</code></span>
- **Kacham et al. (2024).** *PolySketchFormer: Fast Transformers via Sketching Polynomial Kernels.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2310.01655](https://arxiv.org/abs/2310.01655)<br><span class="attn-cite" data-bibkey="kacham2024polysketchformer"><code>kacham2024polysketchformer</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="pham2013fast">
<pre class="attn-bib__src"><code>@inproceedings{pham2013fast,
  title={Fast and Scalable Polynomial Kernels via Explicit Feature Maps},
  author={Pham, Ninh and Pagh, Rasmus},
  booktitle={Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={239--247},
  year={2013},
  doi={10.1145/2487575.2487591}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="ahle2019oblivious">
<pre class="attn-bib__src"><code>@misc{ahle2019oblivious,
  title={Oblivious Sketching of High-Degree Polynomial Kernels},
  author={Ahle, Thomas D. and Kapralov, Michael and Knudsen, Jakob B. T. and Pagh, Rasmus and Velingker, Ameya and Woodruff, David and Zandieh, Amir},
  year={2019},
  eprint={1909.01410},
  archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="kacham2024polysketchformer">
<pre class="attn-bib__src"><code>@inproceedings{kacham2024polysketchformer,
  title={{PolySketchFormer}: Fast Transformers via Sketching Polynomial Kernels},
  author={Kacham, Praneeth and Mirrokni, Vahab and Zhong, Peilin},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series={PMLR},
  volume={235},
  pages={22748--22770},
  year={2024},
  eprint={2310.01655},
  archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/polysketch.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/polysketch.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
