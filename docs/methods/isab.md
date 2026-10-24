# ISAB

<p class="attn-standfirst" markdown="span">The Set Transformer's induced set attention block — a nonlinear bottleneck.</p>

<div class="attn-facts" data-cat="factorize" markdown="1">

key
:   `isab` &middot; `ab.build("isab")`

category
:   [`factorize`](../taxonomy.md#factorize)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.ISABAttention` &middot; [`attnbank/methods/factorize/isab.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/isab.py)

monograph
:   `sec:factor`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the *shape* of $\pi_q$: it is routed through $r$ latents, $\pi_q = a(q,\cdot)\,b(\cdot,x)$, which caps the rank of the attention matrix at $r$.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="no">no</span> | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
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
| `flops` | 46,465,024 | 268,435,456 | 0.173× |
| `state_bytes` | 1,024 | 262,144 | 0.00391× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r, hid = self.r, max(1, self.width * dv)
    return {"flops": 2 * n * r * (d + dv) + 2 * m * r * (d + dv)
                     + 4 * (r + m) * dv * hid + 2 * r * dv * d + 2 * m * d * dv,
            "state_bytes": r * dv * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `a`, `b`, `induced`, `note`, `r`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `8` |
| `width` | `2` |
| `seed` | `0` |

</div>

`r` inducing points; `width` is the rFF expansion factor.

`ab.describe("isab")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("isab", r=8, width=2)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
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
ISABAttention(name='isab', category='factorize', objective='modify', backend='reference', r=8, seed=0, width=2)
AttentionOutput(y=[1, 2, 64, 32], weights=None, v_hat=None, extras=[a, b, induced, note, r], dtype=float32, device=cpu)
invariant W: 0.0
expresses_weights = False -- no measure over the original positions exists, so weights is None
extras: ['a', 'b', 'induced', 'note', 'r']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:factor-methods</code></p>

ISAB routes attention through `r` learned *inducing points*
`I` in two multihead attention blocks (Lee et al., 2019):

```text
ISAB(X) = MAB(X, H),      H = MAB(I, X) in R^{r x d},
MAB(X, Y) = LayerNorm( O + rFF(O) ),   O = LayerNorm( X + Attn(X, Y, Y) ).
```

Written with this library's arguments, the context is `(k, v)` and the first block
summarizes it into `r` induced rows, the second lets the queries read them:

```text
b(z, x) = softmax_x( beta <i_z, k_x> ),   H = MAB-wrap( I_v + int v db(z, .) ),
a(q, z) = softmax_z( beta <q, (H P)_z> ), y_hat(q) = MAB-wrap( q P_q + a(q, .) H ).
```

The two attention maps are Markov kernels, but ISAB is **not** a stochastic
factorization and this method declares `expresses_weights = False`.  Three separate
things break the composition, and each one alone is enough:

1. `MAB` applies a residual, a layer norm and a row-wise feedforward *between* the
   hops, so `H` is a nonlinear function of the values — there is no matrix `B`
   with `H = B V`;
2. the second block's keys are a projection of `H`, so even the read kernel depends
   on the values;
3. the query-side residual `q P_q` puts mass on a term that does not come from the
   value field at all — with a constant value field the output is not that constant,
   which no probability measure on the `n` positions can do.

So `y` is not even linear in `V`: `y(2V) != 2 y(V)`, and no weight matrix exists
to report.  This is the contrast <span class="attn-ref">sec:factor</span> draws against `latent_stochastic`,
which cites the same paper for the *bare* two-hop kernel — strip `MAB` down to its
attention and ISAB becomes a stochastic factorization; keep the block that the Set
Transformer actually uses and the measure on the original positions is gone.  Both
kernels are still exposed in `extras`, as attention maps rather than factors.

`objective = MODIFY`: ISAB is a set-encoder architecture with a parameter-count and
`O(r n)` cost argument, not an estimator of dense attention, and no `r` makes it
softmax.  Both anchors are `None` and the suite asserts the exemption.

Untrained behaviour (`trainable = True`): the inducing points `I_q, I_v`, the two
feedforwards and the two projections are seeded Gaussians scaled by
`1/sqrt(fan_in)`; layer norms are parameter-free.  The projections `P` and `P_q`
exist because this library allows `d != dv` while a Set Transformer is uniformly
`d_model` wide; with `d == dv` they are the block's own `W_K` and the identity
residual, randomly initialized.

Not causal: the inducing points pool the whole context.

</div>

## Related methods

changes the same object — [`factorize`](../taxonomy.md#factorize)
:   [`abc`](abc.md), [`clustered`](clustered.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`perceiver`](perceiver.md), [`svd_oracle`](svd_oracle.md) *(oracle)* &nbsp;·&nbsp; [browse the bank, factorize only](index.md?category=factorize)

judged the same way — `modify`
:   25 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `isab` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `isab` selected](index.md?compare=isab) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Lee et al. (2019).** *Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks.* Proceedings of the 36th International Conference on Machine Learning (ICML). [arXiv:1810.00825](https://arxiv.org/abs/1810.00825)<br><span class="attn-cite" data-bibkey="lee2019set"><code>lee2019set</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="lee2019set">
<pre class="attn-bib__src"><code>@inproceedings{lee2019set,
  title         = {Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
  author        = {Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam R. and Choi, Seungjin and Teh, Yee Whye},
  booktitle     = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  series        = {PMLR},
  volume        = {97},
  year          = {2019},
  eprint        = {1810.00825},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/isab.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/isab.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
