# Low-rank logits

<p class="attn-standfirst" markdown="span">Truncate the *logits* to rank r, then softmax what is left.</p>

!!! danger "Oracle — an upper bound, not a method"
    `lowrank_logit` is not deployable and is not meant to be. It uses information no
    inference-time method has, and it is in the bank to answer *how well could this
    ever work?* — several of the monograph's arguments need that number. Read every
    result below as a bound on a family of real methods, never as a baseline you
    could ship.

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `lowrank_logit` &middot; `ab.build("lowrank_logit")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.LowRankLogitAttention` &middot; [`attnbank/methods/interaction/lowrank_logit.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/lowrank_logit.py)

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
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("lowrank_logit").limit_config
{'r': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 2,415,919,104 | 268,435,456 | 9× |
| `state_bytes` | 16,384 | 262,144 | 0.0625× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    rmax = min(m, n)
    r = min(self.r or rmax, rmax)
    return {"flops": 2 * m * n * d + 2 * m * n * rmax + 2 * m * n * dv,
            "state_bytes": r * (m + n) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `logit_error`, `r`, `singular_values`, `spectral_energy`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `4` |

</div>

`r`: rank kept in the logit matrix; `None` keeps all of it (the limit).

`ab.describe("lowrank_logit")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("lowrank_logit", r=4)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["lowrank_logit"], q, k, v, causal=False,
                 configs={"lowrank_logit": {'r': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
LowRankLogitAttention(name='lowrank_logit', category='interaction', objective='approximate', backend='reference', r=4)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[logit_error, r, singular_values, spectral_energy], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.63556, 'distributional_error': 0.63556, 'value_error': 0.0, 'tv': 0.29079}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:kernel-matrix</code></p>

Everything else in the bank that says "low rank" is low rank in
the wrong space.  `linformer` projects the sequence axis of `K` and `V`,
`svd_oracle` truncates `P` itself, `nystrom` factors `P` through landmarks —
all three produce a **signed** measure, because a rank-`r` approximation of a
nonnegative matrix is not nonnegative.  This file truncates one step earlier:

```text
S   = beta Q K^T,      S_r = U_r Sigma_r V_r^T   (truncated SVD, float64),
g_hat(q, x) = exp((S_r)_{qx}),   pi_hat = softmax(S_r).
```

The exponential is applied *after* the truncation, so `pi_hat` is a genuine
probability measure for every `r`: nonnegative, unit mass, output inside the convex
hull of the values.  The whole contract survives, and the error is pure `hl` — the
measure is wrong, but it is still a measure.  That is the comparison this file exists
to make: at equal rank, is it better to be optimal in `P` and signed, or suboptimal
in `P` and stochastic?  `svd_oracle` is the other arm.

The rank that matters.  `S = beta Q K^T` already has rank at most `d`, so the
method is *exactly* softmax for any `r >= min(m, n, d)` and the truncation only
bites below the head dimension — which is exactly Bhojanapalli et al.'s low-rank
bottleneck, stated as an operator rather than as an architectural constraint.  Note
also that `rank(S_r) = r` says nothing about `rank(exp(S_r))`: the entrywise
exponential of a rank-one matrix is generically full rank, so a low-rank *interaction*
is not a low-rank *attention matrix*, and this method is not a factorization in the
sense of <span class="attn-ref">sec:factor</span>.

This is an **oracle**: it needs the full `m x n` logit matrix and an SVD of it, so it
buys nothing at inference.  It exists to bound what a rank-`r` surrogate of the
interaction can achieve, and to separate that bound from the rank of `P`.

Exact limit: `r = None` keeps the full spectrum.  Not causal: the truncation does not
preserve the mask's `-inf` entries, and an SVD of a matrix containing `-inf` is not
defined.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`linear_elu`](linear_elu.md), [`nystrom`](nystrom.md), [`performer`](performer.md), [`polaformer`](polaformer.md), [`polysketch`](polysketch.md), [`rebased`](rebased.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `approximate`
:   36 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `lowrank_logit` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `lowrank_logit` selected](index.md?compare=lowrank_logit) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Bhojanapalli et al. (2020).** *Low-Rank Bottleneck in Multi-head Attention Models.* Proceedings of the 37th International Conference on Machine Learning (ICML). [arXiv:2002.07028](https://arxiv.org/abs/2002.07028)<br><span class="attn-cite" data-bibkey="bhojanapalli2020lowrank"><code>bhojanapalli2020lowrank</code></span>
- **Wang et al. (2020).** *Linformer: Self-Attention with Linear Complexity.* [arXiv:2006.04768](https://arxiv.org/abs/2006.04768)<br><span class="attn-cite" data-bibkey="wang2020linformer"><code>wang2020linformer</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="bhojanapalli2020lowrank">
<pre class="attn-bib__src"><code>@inproceedings{bhojanapalli2020lowrank,
  title         = {Low-Rank Bottleneck in Multi-head Attention Models},
  author        = {Bhojanapalli, Srinadh and Yun, Chulhee and Rawat, Ankit Singh and Reddi, Sashank and Kumar, Sanjiv},
  booktitle     = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  series        = {PMLR},
  volume        = {119},
  pages         = {864--873},
  year          = {2020},
  eprint        = {2002.07028},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="wang2020linformer">
<pre class="attn-bib__src"><code>@misc{wang2020linformer,
  title = {{Linformer: Self-Attention with Linear Complexity}},
  author = {Sinong Wang and Belinda Z. Li and Madian Khabsa and Han Fang and Hao Ma},
  year = {2020},
  howpublished = {arXiv preprint arXiv:2006.04768},
  eprint = {2006.04768},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/lowrank_logit.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/lowrank_logit.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
