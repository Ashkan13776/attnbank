# ABC

<p class="attn-standfirst" markdown="span">Attention with bounded-memory control — a *causal* stochastic factorization.</p>

<div class="attn-facts" data-cat="factorize" markdown="1">

key
:   `abc` &middot; `ab.build("abc")`

category
:   [`factorize`](../taxonomy.md#factorize)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.ABCAttention` &middot; [`attnbank/methods/factorize/abc.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/abc.py)

monograph
:   `sec:factor`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the *shape* of $\pi_q$: it is routed through $r$ latents, $\pi_q = a(q,\cdot)\,b(\cdot,x)$, which caps the rank of the attention matrix at $r$.

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
| `trainable` | <span class="attn-flag" data-v="yes">yes</span> | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

</div>

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("abc").limit_config
{'r': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 4,194,304 | 268,435,456 | 0.0156× |
| `state_bytes` | 2,048 | 262,144 | 0.00781× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = min(self.r or n, n)
    return {"flops": 2 * n * r * (d + dv) + 2 * m * r * (d + dv),
            "state_bytes": r * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** The measure is a *product* $a(q,\cdot)\,b(\cdot,x)$ — an SVD, an NMF, a clustering, a pair of small softmaxes — and `sdpa` computes one softmax, not a product of two.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `a`, `b`, `control`, `memory_keys`, `memory_values`, `r`, `unbounded`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `8` |
| `seed` | `0` |

</div>

`r`: memory slots. `None` (or `r >= n`) is unbounded memory: exact.

`ab.describe("abc")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("abc", r=8)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["abc"], q, k, v, causal=True,
                 configs={"abc": {'r': 8}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
ABCAttention(name='abc', category='factorize', objective='approximate', backend='reference', r=8, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[a, b, control, memory_keys, memory_values, r, unbounded], dtype=float32, device=cpu)
invariant W: 2.384185791015625e-07
{'output_error': 1.01096, 'distributional_error': 1.01096, 'value_error': 0.0, 'tv': 0.35943}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:factor-def</code>, <code>def:factor-stochastic</code>, <code>prop:factor-closure</code></p>

ABC keeps a
memory of `r` slots built by a *control* `phi_x in R_{>=0}^r` attached to every
context position (Peng et al., 2022):

```text
K_tilde = sum_x phi_x ⊗ k_x,    V_tilde = sum_x phi_x ⊗ v_x,
y_hat(q) = V_tilde^T softmax(K_tilde q).
```

With the normalized control of the paper's `AbcMLP` form, `phi_{z,x} = alpha_{z,x} /
sum_{x'} alpha_{z,x'}`, the slot rows are probability measures, so the memory *is* a
Markov kernel `b: Z -> X` and the read is a Markov kernel `a: Q -> Z`:

```text
b(z, x) = softmax_x( beta <w_z, k_x> ),     k_bar_z = int k db(z, .),
a(q, z) = softmax_z( beta <q, k_bar_z> ),   pi_hat_q = a(q, .) b(., .).
```

This is a **genuine composition of two Markov kernels** — nothing happens to the
latents between the two hops — so `pi_hat_q` exists on the original `n` positions,
is row-stochastic, and inherits <span class="attn-ref">prop:factor-closure</span> whole.  `weights` is that
composition and Invariant W holds exactly.

Two things distinguish this file from `latent_stochastic`, which is the same two-hop
kernel in the non-causal case:

* **It is causal.**  ABC's memory is a running sum, `K_tilde_{t+1} = K_tilde_t +
  phi_{t+1} ⊗ k_{t+1}`, so the control normalizer at query `t` runs over the prefix
  `x <= t` only.  The write kernel therefore depends on the query position,
  `b_t(z, x) = softmax_{x <= t}( beta <w_z, k_x> )`, and `pi_hat_{q_t} = a_t b_t`
  is supported on the prefix.  It is the only factorization in the bank that decodes.
* **It has an exact limit.**  "If growing-size memory were allowed (n = N), an ABC with
  `phi_i = e_i` would fall back to softmax attention" (Peng et al., 2022, Sec. 3).
  `r = None` is that unbounded memory — one slot per position, identity control —
  and reproduces dense softmax exactly, causal or not.  That is the correctness anchor,
  and it is why ABC is an **approximation** while `latent_stochastic` and `latte`
  are operator modifications: bounded memory is a resource parameter with a limit.

The controls `w_z` are the learned parameters.  Untrained they are a seeded Gaussian
in `R^{r x d}` (the write kernel is then a random soft partition of the context), so
`trainable = True` records that the *quality* of the factorization, not its
structure, is what training buys.  Nothing in the declared contract depends on them.

Reference implementation, clarity over speed (DESIGN.md Sec. 6): the causal path
materializes the prefix-normalized kernel `[m, r, n]` and softmaxes each prefix
independently, which is stable at any logit scale.  The deployed form is the `O(r d)`
running memory above.

</div>

## Related methods

changes the same object — [`factorize`](../taxonomy.md#factorize)
:   [`clustered`](clustered.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`perceiver`](perceiver.md), [`svd_oracle`](svd_oracle.md) *(oracle)* &nbsp;·&nbsp; [browse the bank, factorize only](index.md?category=factorize)

judged the same way — `approximate`
:   42 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `abc` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `abc` selected](index.md?compare=abc) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Peng et al. (2022).** *ABC: Attention with Bounded-memory Control.* Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL). [arXiv:2110.02488](https://arxiv.org/abs/2110.02488) [doi:10.18653/v1/2022.acl-long.515](https://doi.org/10.18653/v1/2022.acl-long.515)<br><span class="attn-cite" data-bibkey="peng2022abc"><code>peng2022abc</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="peng2022abc">
<pre class="attn-bib__src"><code>@inproceedings{peng2022abc,
  title         = {{ABC}: Attention with Bounded-memory Control},
  author        = {Peng, Hao and Kasai, Jungo and Pappas, Nikolaos and Yogatama, Dani and Wu, Zhaofeng and Kong, Lingpeng and Schwartz, Roy and Smith, Noah A.},
  booktitle     = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year          = {2022},
  doi           = {10.18653/v1/2022.acl-long.515},
  eprint        = {2110.02488},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/factorize/abc.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/factorize/abc.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
