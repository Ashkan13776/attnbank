# Linformer

<p class="attn-standfirst" markdown="span">A fixed linear projection of K and V along the *sequence* axis.</p>

<div class="attn-facts" data-cat="compress" markdown="1">

key
:   `linformer` &middot; `ab.build("linformer")`

category
:   [`compress`](../taxonomy.md#compress)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.LinformerAttention` &middot; [`attnbank/methods/compress/linformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/linformer.py)

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
| `supports_causal` | <span class="attn-flag" data-v="no">no</span> | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | <span class="attn-flag" data-v="yes">yes</span> | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | <span class="attn-flag" data-v="no">no</span> | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("linformer").limit_config
{'r': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 4,194,304 | 268,435,456 | 0.0156× |
| `state_bytes` | 2,048 | 262,144 | 0.00781× |
| `bytes_moved` | 264,192 | 4,718,592 | 0.056× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = min(self.r or n, n)
    return {"flops": 2 * n * r * (d + dv) + 2 * m * r * (d + dv),
            "state_bytes": r * (d + dv) * bytes_per_el,
            "bytes_moved": m * (d + dv) * bytes_per_el + r * (d + dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** The compressed context is built by this method, and the measure it reports lives on the **original** $n$ positions — a kernel run over the compressed cache would return neither.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `latent_weights`, `r`, `row_sum`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `8` |
| `seed` | `0` |

</div>

`ab.describe("linformer")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("linformer", r=8)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["linformer"], q, k, v, causal=False,
                 configs={"linformer": {'r': 8}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
LinformerAttention(name='linformer', category='compress', objective='approximate', backend='reference', r=8, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[latent_weights, r, row_sum], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 10.14872, 'distributional_error': 10.14872, 'value_error': 0.0, 'tv': 5.86149}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:compress-lowrank</code>, <code>sec:factor-methods</code></p>

With `E, F in R^{r x n}`,

```text
P_hat = softmax(beta Q (E K)^T) in R^{m x r},    y_hat = P_hat (F V),
```

so `E K` and `F V` are `r` learned pseudo-tokens: the method compresses keys and
values separately, changes both `mu` and `v`, fixes `n`, and is not causal.

It *can* be written on the original positions, and that is how the library stores it:
`y_hat = (P_hat F) V`, so `weights = P_hat F in R^{m x n}` with the **original**
values.  Those weights are generally signed and their rows do not sum to one unless
`F` is row-stochastic, which is the paper's point — <span class="attn-ref">tab:comparative</span> lists
Linformer as losing `nm` and `hl`.  `row_stochastic = False` records it, and the
row sums are returned in `extras` so the deviation can be measured.

Note also that the latent keys `(EK)_z` enter the softmax *linearly*, and
`exp(beta <q, sum_x E_zx k_x>) != sum_x E_zx exp(beta <q, k_x>)`, so the entries of
`P` are not approximated at all; the Johnson–Lindenstrauss argument is an existence
statement about the reference operator with random `E, F`, not an error bound for the
trained operator.

Exact limit: `r = None` sets `E = F = I_n`, which gives dense softmax.

</div>

## Related methods

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`gqa`](gqa.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`
:   30 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `linformer` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `linformer` selected](index.md?compare=linformer) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Wang et al. (2020).** *Linformer: Self-Attention with Linear Complexity.* [arXiv:2006.04768](https://arxiv.org/abs/2006.04768)<br><span class="attn-cite" data-bibkey="wang2020linformer"><code>wang2020linformer</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
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

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/linformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/linformer.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
