# QJL

<p class="attn-standfirst" markdown="span">A one-bit Johnson–Lindenstrauss sketch of the keys, with an unbiased inner product.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `qjl` &middot; `ab.build("qjl")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.QJLAttention` &middot; [`attnbank/methods/compress/qjl.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/qjl.py)

monograph
:   `sec:compress`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

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

**Exact limit.** Built with `sketch=None`, `bits_v=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("qjl").limit_config
{'sketch': None, 'bits_v': None}
```

**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("qjl").convergence_configs()
[{'sketch': 8, 'bits_v': None}, {'sketch': 32, 'bits_v': None}, {'sketch': 128, 'bits_v': None}, {'sketch': 512, 'bits_v': None}]
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 285,212,672 | 268,435,456 | 1.06× |
| `state_bytes` | 47,104 | 262,144 | 0.18× |
| `bytes_moved` | 309,248 | 4,718,592 | 0.0655× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    """Keys: one bit per sketch coordinate and one norm -- no scale, no zero point.

    The two extra constants per token are the *value* quantizer's; the key side is
    the part with no group overhead, which is what the paper's title is about.
    """
    r = self.sketch if self.sketch is not None else 8 * bytes_per_el * d
    bv = (self.bits_v or 8 * bytes_per_el) / 8
    state = n * (r / 8 + bytes_per_el + dv * bv + 2 * bytes_per_el)
    return {"flops": 2 * m * n * (d + dv) + 2 * n * r * d + 2 * m * r * d,
            "state_bytes": state,
            "bytes_moved": m * (d + dv) * bytes_per_el + state}
```

## Backends

**`reference` only, and that is the correct answer.** It returns both the exact and the perturbed score matrices (`s`, `s_hat`) in `extras` — the perturbation *is* the object of study, and a fused kernel hands back neither.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `delta_k`, `delta_v`, `key_norm`, `s`, `s_hat`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `sketch` | `64` |
| `bits_v` | `4` |
| `seed` | `0` |

</div>

`ab.describe("qjl")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("qjl", sketch=64, bits_v=4)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["qjl"], q, k, v, causal=True,
                 configs={"qjl": {'sketch': 64, 'bits_v': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
QJLAttention(name='qjl', category='compress', objective='approximate', backend='reference', bits_v=4, seed=0, sketch=64)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[delta_k, delta_v, key_norm, s, s_hat], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.93419, 'distributional_error': 0.93313, 'value_error': 0.14712, 'tv': 0.28403}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-quant</code></p>

Every other member of this family stores a *rounded copy* of
`k_j` and accepts the rounding error.  QJL stores no copy at all: it keeps the
`sketch` sign bits of a random projection,

```text
H_S(k_j) = sign(S k_j) in {-1, +1}^R,       S in R^{R x d},  S_ab ~ N(0, 1),
```

together with the scalar `nu_j = ||k_j||_2` in full precision, and estimates the
interaction *asymmetrically* — one side sketched and quantized, the other side merely
sketched:
    g_hat(q, k_j) = sqrt(pi/2) * (nu_j / R) * <S q, sign(S k_j)>.

This is unbiased, `E_S[g_hat] = <q, k_j>`, because for a Gaussian row `r`
`E[sign(<r, k>) <r, q>] = sqrt(2/pi) <q, k> / ||k||` — the Grothendieck / random
hyperplane identity.  Nothing is grouped, so there is no scale and no zero-point to
store per group; the only side information is one norm per token, and that is the
"zero overhead" of the title.

**It is still a key perturbation.**  Writing the estimator as an inner product with an
explicit vector,

```text
k_hat_j = sqrt(pi/2) * (nu_j / R) * S^T sign(S k_j),
```

gives `g_hat(q, k_j) = <q, k_hat_j>` identically, so QJL lands in exactly the same
frame as the rest of `compress/`: `delta k_j = k_j - k_hat_j` with
`E[k_hat_j] = k_j`, and `extras['delta_k']` is directly comparable with `kivi`'s.
This file computes `k_hat` and then calls the ordinary score path, which is both
clearer and the reason Invariant W holds without special pleading.

**Unbiased logits are not unbiased outputs.**  `E[s_hat] = s` does not give
`E[y_hat] = y`: the softmax is convex in the logits and the estimator's variance is
`O(||q||^2 nu_j^2 / R)`, so it is *token dependent* and tilts `pi_q` toward
large-norm keys.  That failure is the point of having QJL in the bank, and it is why
the anchor below is a limit rather than only a convergence sequence.

Values are **not** sketched.  The paper is explicit that "standard token-wise
quantization is very effective" for the value cache, so `bits_v` here is an ordinary
per-token affine quantizer — the same asymmetry `kivi` reaches by a different route.

Causal and head-independent: the sketch of token `j` depends on `k_j` alone and the
projection is a seeded constant, so nothing pools over tokens.

Anchors, both declared: `limit_config` disables the sketch and the value quantizer and
recovers dense softmax exactly, and `convergence_configs` increases `sketch`, along
which the Monte-Carlo error must fall.

Quantize–dequantize is **simulated**: `k_hat` is materialised in the input dtype, so
the error is the one the 1-bit cache would make while the memory is not saved.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`
:   30 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `qjl` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `qjl` selected](index.md?compare=qjl) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Zandieh et al. (2025).** *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead.* Proceedings of the AAAI Conference on Artificial Intelligence. [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)<br><span class="attn-cite" data-bibkey="zandieh2025qjl"><code>zandieh2025qjl</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="zandieh2025qjl">
<pre class="attn-bib__src"><code>@inproceedings{zandieh2025qjl,
  title     = {{QJL}: 1-Bit Quantized {JL} Transform for {KV} Cache Quantization with Zero Overhead},
  author    = {Zandieh, Amir and Daliri, Majid and Han, Insu},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {24},
  pages     = {25805--25813},
  year      = {2025},
  eprint    = {2406.03482},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/qjl.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/qjl.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
