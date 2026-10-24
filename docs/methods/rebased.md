# ReBased

<p class="attn-standfirst" markdown="span">A learnable quadratic feature map with a movable root.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `rebased` &middot; `ab.build("rebased")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.ReBasedAttention` &middot; [`attnbank/methods/interaction/rebased.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/rebased.py)

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
| `trainable` | <span class="attn-flag" data-v="yes">yes</span> | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

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
| `bytes_moved` | 532,608 | 4,718,592 | 0.113× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = d
    return {"flops": 2 * n * r * dv + 2 * m * r * dv + 4 * (m + n) * r,
            "state_bytes": r * (dv + 1) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `Z_hat`, `dynamic_range`, `g_hat`, `phi_k`, `phi_q`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `gamma_q` | `1.0` |
| `beta_q` | `0.0` |
| `gamma_k` | `1.0` |
| `beta_k` | `0.0` |

</div>

`ab.describe("rebased")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("rebased", gamma_q=1.0, beta_q=0.0)
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
ReBasedAttention(name='rebased', category='interaction', objective='modify', backend='reference', beta_k=0.0, beta_q=0.0, gamma_k=1.0, gamma_q=1.0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[Z_hat, dynamic_range, g_hat, phi_k, phi_q], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['Z_hat', 'dynamic_range', 'g_hat', 'phi_k', 'phi_q']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:kernel-det</code></p>

Aksenov et al. (2024) start from the observation that Based's
surrogate `1 + z + z^2/2` has minimum `1/2` at `z = -1`: however negative the
logit, the weight it assigns is bounded below, so the measure can never concentrate
the way softmax does.  Their fix keeps the quadratic but makes its root learnable —
normalize each head's queries and keys, then apply a per-channel affine before
squaring,

```text
phi(q) = (gamma_Q . norm(q) + beta_Q)^2,   phi(k) = (gamma_K . norm(k) + beta_K)^2,
g_hat(q, x) = <phi(q), phi(k_x)>,   y_i = phi(q_i)^T sum_j phi(k_j) v_j^T
                                         / phi(q_i)^T sum_j phi(k_j),
```

with `norm` a LayerNorm over the head dimension (no affine of its own: `gamma`
and `beta` are that affine, learned separately for queries and keys) and `.` the
elementwise product.  `phi >= 0` is a square, so `pi_hat` is a probability
measure, and `phi` now *reaches* zero — at `norm(x)_c = -beta_c / gamma_c` — which
is the whole point.

This is an **operator modification**: `r = d` features, no resource parameter, and
no configuration under which it becomes softmax, so both correctness anchors are
`None` by design (<span class="attn-ref">tab:kernel-methods</span>).  Two structural consequences follow from
the normalization and are worth separating from the learnability.  First
<span class="attn-ref">rem:kernel-homogeneous</span> applies *exactly*: LayerNorm discards `||q||`, so unlike
`based` — and unlike softmax — scaling the query cannot sharpen `pi_hat_q` at
all.  Second, `g_hat` is a quadratic form in `norm(q)`, hence bounded on the
sphere: the dynamic range of the surrogate is bounded uniformly in the input, where
`exp` is not.  Both are visible in `extras`.

`gamma` and `beta` are the learned parameters, so `trainable = True`.  The
defaults are their initialization, `gamma = 1` and `beta = 0`, at which
`phi(x) = norm(x)^2`; pass floats to move them (the same value in every channel) and
see the root move with them.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`linear_elu`](linear_elu.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polaformer`](polaformer.md), [`polysketch`](polysketch.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `modify`
:   25 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `rebased` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `rebased` selected](index.md?compare=rebased) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Aksenov et al. (2024).** *Linear Transformers with Learnable Kernel Functions are Better In-Context Models.* Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), Volume 1: Long Papers. [arXiv:2402.10644](https://arxiv.org/abs/2402.10644)<br><span class="attn-cite" data-bibkey="aksenov2024rebased"><code>aksenov2024rebased</code></span>
- **Arora et al. (2024).** *Simple linear attention language models balance the recall-throughput tradeoff.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2402.18668](https://arxiv.org/abs/2402.18668)<br><span class="attn-cite" data-bibkey="arora2024based"><code>arora2024based</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="aksenov2024rebased">
<pre class="attn-bib__src"><code>@inproceedings{aksenov2024rebased,
  title     = {Linear Transformers with Learnable Kernel Functions are Better In-Context Models},
  author    = {Aksenov, Yaroslav and Balagansky, Nikita and Lo Cicero Vaina, Sofia Maria and Shaposhnikov, Boris and Gorbatovski, Alexey and Gavrilov, Daniil},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), Volume 1: Long Papers},
  pages     = {9584--9597},
  year      = {2024},
  eprint    = {2402.10644},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="arora2024based">
<pre class="attn-bib__src"><code>@inproceedings{arora2024based,
  title     = {Simple linear attention language models balance the recall-throughput tradeoff},
  author    = {Arora, Simran and Eyuboglu, Sabri and Zhang, Michael and Timalsina, Aman and Alberti, Silas and Zou, James and Rudra, Atri and R{\'e}, Christopher},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {235},
  pages     = {1763--1840},
  year      = {2024},
  eprint    = {2402.18668},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/rebased.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/rebased.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
