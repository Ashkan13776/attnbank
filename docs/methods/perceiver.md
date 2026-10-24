# Perceiver / Perceiver IO

<p class="attn-standfirst" markdown="span">A *nonlinear* latent bottleneck — no induced measure.</p>

<div class="attn-facts" data-cat="factorize" markdown="1">

key
:   `perceiver` &middot; `ab.build("perceiver")`

category
:   [`factorize`](../taxonomy.md#factorize)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.PerceiverAttention` &middot; [`attnbank/methods/factorize/perceiver.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/perceiver.py)

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
| `flops` | 4,947,968 | 268,435,456 | 0.0184× |
| `state_bytes` | 1,024 | 262,144 | 0.00391× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r, hid = self.r, max(1, self.width * dv)
    block = 2 * r * r * (d + dv) + 2 * r * dv * d + 4 * r * dv * hid
    return {"flops": 2 * n * r * (d + dv) + self.depth * block
                     + 2 * m * r * (d + dv) + 2 * r * dv * d,
            "state_bytes": r * dv * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `a`, `b`, `depth`, `latents`, `note`, `r`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `8` |
| `depth` | `2` |
| `width` | `2` |
| `seed` | `0` |

</div>

`r` latents, `depth` latent blocks (>= 1), MLP expansion `width`.

`ab.describe("perceiver")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("perceiver", r=8, depth=2)
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
PerceiverAttention(name='perceiver', category='factorize', objective='modify', backend='reference', depth=2, r=8, seed=0, width=2)
AttentionOutput(y=[1, 2, 64, 32], weights=None, v_hat=None, extras=[a, b, depth, latents, note, r], dtype=float32, device=cpu)
invariant W: 0.0
expresses_weights = False -- no measure over the original positions exists, so weights is None
extras: ['a', 'b', 'depth', 'latents', 'note', 'r']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:factor-methods</code></p>

A fixed latent array of `r` slots cross-attends to the
context, a stack of latent self-attention + MLP blocks processes the latents, and the
real queries cross-attend to the result (Jaegle et al., 2021; Jaegle et al., 2022):

```text
z_0 = b V,          b(z, x) = softmax_x( beta <w_z, k_x> )          (encode)
z_{l+1} = Block(z_l)                                                (process)
y_hat(q) = a(q, .) z_L,   a(q, z) = softmax_z( beta <q, (z_L P)_z> )  (decode)
```

The two cross-attentions are Markov kernels, and it is tempting to call the product
`a b` the induced measure.  **It is not, and this method declares
`expresses_weights = False` for exactly that reason.**  `Block` — pre-norm latent
self-attention with a residual, then a GELU MLP with a residual — is a *nonlinear*
map of the latents, so `z_L` is not `(something) V` for any matrix depending only
on `(q, k)`, and the decoder's keys `z_L P` depend on `V` as well.  The operator
is not linear in the value field at all: `y(2V) != 2 y(V)`.  No matrix `W(q, k)`
with `y = W V` exists, let alone a probability measure on the `n` positions, so
returning `a @ b` would be a fabrication.  This is the sharp line in <span class="attn-ref">sec:factor</span>
between a *stochastic factorization*, which composes two kernels and stays a measure
(`latent_stochastic`, `abc`, `latte`), and a *latent bottleneck*, which processes
between the hops and leaves the measure behind.  Both kernels are still reported in
`extras` — as the two attention maps they are, not as a factorization.

Perceiver was never approximating softmax: it is an architecture for attending to very
large inputs at `O(r n)` cost, and there is no `r` at which it becomes dense
attention over the context.  `objective = MODIFY`; both anchors are `None` and the
conformance suite asserts the exemption.

Untrained behaviour, which is what a method bank can offer for an architecture whose
content is learned (`trainable = True`): the latent array `w`, the latent
projection `P`, the MLP `W_1, W_2` and the decoder projection `P_out` are seeded
Gaussians scaled by `1/sqrt(fan_in)`; layer norms are parameter-free (no affine).
Following Perceiver's own weight sharing, one set of block parameters is reused at
every depth.  The map is then a fixed, deterministic, well-defined nonlinear
bottleneck — the structure is exact, the content is random.

Not causal: the encode kernel pools the whole context into every latent.

</div>

## Related methods

changes the same object — [`factorize`](../taxonomy.md#factorize)
:   [`abc`](abc.md), [`clustered`](clustered.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`svd_oracle`](svd_oracle.md) *(oracle)* &nbsp;·&nbsp; [browse the bank, factorize only](index.md?category=factorize)

judged the same way — `modify`
:   25 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `perceiver` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `perceiver` selected](index.md?compare=perceiver) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Jaegle et al. (2021).** *Perceiver: General Perception with Iterative Attention.* Proceedings of the 38th International Conference on Machine Learning (ICML). [arXiv:2103.03206](https://arxiv.org/abs/2103.03206)<br><span class="attn-cite" data-bibkey="jaegle2021perceiver"><code>jaegle2021perceiver</code></span>
- **Jaegle et al. (2022).** *Perceiver IO: A General Architecture for Structured Inputs & Outputs.* International Conference on Learning Representations (ICLR). [arXiv:2107.14795](https://arxiv.org/abs/2107.14795)<br><span class="attn-cite" data-bibkey="jaegle2022perceiverio"><code>jaegle2022perceiverio</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="jaegle2021perceiver">
<pre class="attn-bib__src"><code>@inproceedings{jaegle2021perceiver,
  title         = {Perceiver: General Perception with Iterative Attention},
  author        = {Jaegle, Andrew and Gimeno, Felix and Brock, Andrew and Zisserman, Andrew and Vinyals, Oriol and Carreira, Joao},
  booktitle     = {Proceedings of the 38th International Conference on Machine Learning (ICML)},
  series        = {PMLR},
  volume        = {139},
  year          = {2021},
  eprint        = {2103.03206},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="jaegle2022perceiverio">
<pre class="attn-bib__src"><code>@inproceedings{jaegle2022perceiverio,
  title         = {Perceiver {IO}: A General Architecture for Structured Inputs \&amp; Outputs},
  author        = {Jaegle, Andrew and Borgeaud, Sebastian and Alayrac, Jean-Baptiste and Doersch, Carl and Ionescu, Catalin and Ding, David and Koppula, Skanda and Zoran, Daniel and Brock, Andrew and Shelhamer, Evan and H{\'e}naff, Olivier and Botvinick, Matthew M. and Zisserman, Andrew and Vinyals, Oriol and Carreira, Jo{\~a}o},
  booktitle     = {International Conference on Learning Representations (ICLR)},
  year          = {2022},
  eprint        = {2107.14795},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/perceiver.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/perceiver.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
