# FAVOR#

<p class="attn-standfirst" markdown="span">Dense-exponential random features, the anisotropic optimum.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `favorsharp` &middot; `ab.build("favorsharp")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.FAVORSharpAttention` &middot; [`attnbank/methods/interaction/favorsharp.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/favorsharp.py)

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
| `row_stochastic` | <span class="attn-flag" data-v="yes">yes</span> | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor


**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("favorsharp").convergence_configs()
[{'r': 8}, {'r': 32}, {'r': 128}, {'r': 512}]
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 42,205,184 | 268,435,456 | 0.157× |
| `state_bytes` | 8,320 | 262,144 | 0.0317× |
| `bytes_moved` | 532,608 | 4,718,592 | 0.113× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.r
    return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv
                     + (m + n) * d * d + d ** 3,     # the pair covariance and its eigh
            "state_bytes": r * (dv + 1) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `A_diag`, `Z_rel`, `eigenvalues`, `log_g_hat`, `r`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `64` |
| `orthogonal` | `True` |
| `seed` | `0` |

</div>

`ab.describe("favorsharp")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("favorsharp", r=64, orthogonal=True)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["favorsharp"], q, k, v, causal=True,
                 configs={"favorsharp": {'r': 64, 'orthogonal': True}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
FAVORSharpAttention(name='favorsharp', category='interaction', objective='approximate', backend='reference', orthogonal=True, r=64, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[A_diag, Z_rel, eigenvalues, log_g_hat, r], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.07253, 'distributional_error': 1.07253, 'value_error': 0.0, 'tv': 0.3124}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:kernel-rf</code></p>

`favorpp` optimises one scalar `A` over the family
`D exp(A ||omega||^2 + B omega^T u + C ||u||^2)`.  Likhosherstov et al. (2023) let
the quadratic form be a full matrix — *dense-exponential random features* —

```text
f(omega, u) = D exp(omega^T A omega + omega^T B u + u^T C u),   omega ~ N(0, I_d),
```

and their Thm. 4.1 gives unbiasedness for `g(q,x) = exp(<qt, kt_x>)` iff

```text
8A < I,   B^T (I - 4A)^{-1} B = I,   C = -1/2 B^T (I - 4A)^{-1} B,
D = det(I - 4A)^{1/4},
```

which for the symmetric choice `B^(1) = B^(2) = B` collapses to `C = -I/2`.  The
optimum of the shifted log-variance over the pair set (their Thm. 4.3) is the
eigendecomposition of the **pair covariance**

```text
Sigma = mean_{i,j} (qt_i + kt_j)(qt_i + kt_j)^T
      = M^(1) + mu^(4) (mu^(5))^T + mu^(5) (mu^(4))^T + M^(2) = Q Lam Q^T,
```

after which, with `attnbank.methods.interaction.favorpp.optimal_a` applied to
each eigenvalue,

```text
A = diag(A*(Lam_1), ..., A*(Lam_d)),   B = (I - 4A)^{1/2} Q^T,
C = -I/2,   D = det(I - 4A)^{1/4} = prod_l (1 - 4 A_ll)^{1/4}.
```

So FAVOR# is FAVOR++ run independently in each eigendirection of `Sigma`, and
FAVOR++ is FAVOR# with every eigenvalue replaced by their mean `tr(Sigma)/d`: the
two closed forms are the *same* scalar map `A*`, which is why `optimal_a` lives in
one place and is used by both.  At `Sigma = 0` it collapses further still: `A = 0`,
`B = Q^T` is an orthogonal rotation, and `Q omega` is again `N(0, I_d)`, so the
estimator is FAVOR+ in distribution.

Because `Sigma` is a mean of outer products it is positive semidefinite, so every
`Lam_l >= 0`, every `A*(Lam_l) <= 0` and `1 - 4A > 0`: the feature map is real
and strictly positive, and `pi_hat` is a probability measure.

**Causality.**  As in `favorpp`, the statistic is conditioned on each query's
visible support — `Sigma_i = qt_i qt_i^T + qt_i ybar_i^T + ybar_i qt_i^T + M2_i`
with `ybar_i`, `M2_i` the prefix means of `kt_j` and `kt_j kt_j^T` — so the
operator is exactly causal and head-independent, and unbiasedness is untouched because
`Sigma` does not depend on `omega`.  The statistic and the eigendecomposition are
done in float64 so that the choice of basis, which a sign flip of an eigenvector would
otherwise make dtype-dependent, is identical in every precision; eigenvector signs are
additionally canonicalised (largest-magnitude entry positive), which the estimator's
distribution is invariant to.

The per-query `O(d^3)` eigendecomposition is the price of exact causality and is a
property of this reference implementation; `cost` reports the deployed form, in which
`Sigma` is a layer constant fitted once (`O(L d^2 + d^3)`) and the features are
separable.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`hedgehog`](hedgehog.md), [`linear_elu`](linear_elu.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polaformer`](polaformer.md), [`polysketch`](polysketch.md), [`rebased`](rebased.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `approximate`
:   36 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `favorsharp` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `favorsharp` selected](index.md?compare=favorsharp) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Likhosherstov et al. (2023).** *Dense-Exponential Random Features: Sharp Positive Estimators of the Gaussian Kernel.* Advances in Neural Information Processing Systems 36 (NeurIPS). [arXiv:2302.00787](https://arxiv.org/abs/2302.00787)<br><span class="attn-cite" data-bibkey="likhosherstov2023favorsharp"><code>likhosherstov2023favorsharp</code></span>
- **Likhosherstov et al. (2022).** *Chefs' Random Tables: Non-Trigonometric Random Features.* Advances in Neural Information Processing Systems 35 (NeurIPS). [arXiv:2205.15317](https://arxiv.org/abs/2205.15317)<br><span class="attn-cite" data-bibkey="likhosherstov2022chefs"><code>likhosherstov2022chefs</code></span>
- **Yu et al. (2016).** *Orthogonal Random Features.* Advances in Neural Information Processing Systems 29 (NIPS). [arXiv:1610.09072](https://arxiv.org/abs/1610.09072)<br><span class="attn-cite" data-bibkey="yu2016orthogonal"><code>yu2016orthogonal</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="likhosherstov2023favorsharp">
<pre class="attn-bib__src"><code>@inproceedings{likhosherstov2023favorsharp,
  title     = {Dense-Exponential Random Features: Sharp Positive Estimators of the Gaussian Kernel},
  author    = {Likhosherstov, Valerii and Choromanski, Krzysztof and Dubey, Kumar Avinava and Liu, Frederick and Sarlos, Tamas and Weller, Adrian},
  booktitle = {Advances in Neural Information Processing Systems 36 (NeurIPS)},
  year      = {2023},
  note      = {Introduces FAVOR\#; circulated as arXiv:2302.00787, ``FAVOR\#: Sharp Attention Kernel Approximations via New Classes of Positive Random Features''},
  eprint    = {2302.00787},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="likhosherstov2022chefs">
<pre class="attn-bib__src"><code>@inproceedings{likhosherstov2022chefs,
  title={Chefs' Random Tables: Non-Trigonometric Random Features},
  author={Likhosherstov, Valerii and Choromanski, Krzysztof and Dubey, Avinava and Liu, Frederick and Sarlos, Tamas and Weller, Adrian},
  booktitle={Advances in Neural Information Processing Systems 35 (NeurIPS)},
  year={2022},
  eprint={2205.15317},
  archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="yu2016orthogonal">
<pre class="attn-bib__src"><code>@inproceedings{yu2016orthogonal,
  title={Orthogonal Random Features},
  author={Yu, Felix X. and Suresh, Ananda Theertha and Choromanski, Krzysztof and Holtmann-Rice, Daniel and Kumar, Sanjiv},
  booktitle={Advances in Neural Information Processing Systems 29 (NIPS)},
  year={2016},
  eprint={1610.09072},
  archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/favorsharp.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/favorsharp.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
