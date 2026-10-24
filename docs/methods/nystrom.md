# Nystromformer

<p class="attn-standfirst" markdown="span">Landmark Nystrom approximation of the row-stochastic matrix.</p>

<div class="attn-facts" data-cat="interaction" markdown="1">

key
:   `nystrom` &middot; `ab.build("nystrom")`

category
:   [`interaction`](../taxonomy.md#interaction)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.NystromAttention` &middot; [`attnbank/methods/interaction/nystrom.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/nystrom.py)

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
| `row_stochastic` | <span class="attn-flag" data-v="no">no</span> | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="no">no</span> | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `r=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("nystrom").limit_config
{'r': None}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 8,396,800 | 268,435,456 | 0.0313× |
| `state_bytes` | 4,096 | 262,144 | 0.0156× |
| `bytes_moved` | 528,384 | 4,718,592 | 0.112× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.r or min(m, n)
    return {"flops": 2 * (m + n) * r * d + 2 * r ** 3 + 2 * (m + n) * r * dv,
            "state_bytes": r * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (d + dv)) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** `sdpa` computes exactly $\mathrm{softmax}(\beta\langle q,k\rangle + \text{mask})\,v$; this operator replaces $g$, so it is not that expression and there is no kernel for it.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `min_weight`, `n_landmarks`, `row_sum`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `r` | `16` |

</div>

`r`: number of landmarks; `None` uses the data as landmarks (exact limit).

`ab.describe("nystrom")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("nystrom", r=16)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["nystrom"], q, k, v, causal=False,
                 configs={"nystrom": {'r': 16}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
NystromAttention(name='nystrom', category='interaction', objective='approximate', backend='reference', r=16)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[min_weight, n_landmarks, row_sum], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 13.3981, 'distributional_error': 13.3981, 'value_error': 0.0, 'tv': 7.27394}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:kernel-matrix</code> and <code>sec:factor-methods</code></p>

With `r` landmark queries and keys
obtained by *segment means* of `Q` and `K`,

```text
P_hat = softmax(beta Q Kt^T) * G^+ * softmax(beta Qt K^T),
G     = softmax(beta Qt Kt^T),
```

where `Qt, Kt` are the landmarks and `G^+` the pseudo-inverse (the original uses an
iterative approximation of it; this reference implementation uses an exact pinv in
float64, so that the only error measured is the method's, not the solver's).

Three caveats the paper makes and this file honours.

1. `P` is neither symmetric nor PSD when `Q != K`, so Nystrom theory does not
    apply; there is no spectral guarantee here.

2. The pseudo-inverse makes `P_hat` **signed**: `pi_hat` is a signed measure of
    unit mass, so the contract items `nm` and `hl` are lost — `row_stochastic` is
    `False`, and outputs can leave the convex hull of the values.  (Rows do sum to one
    when `G` is invertible, since `G^{-1} 1 = 1` for a stochastic `G`; the library
    does not rely on that.)

3. Segment-mean landmarks summarize the context, so the method also modifies `mu`.

Exact limit.  With `r = None` the landmarks are the data themselves (`Qt = Q`,
`Kt = K`, which requires `m = n`), giving `P_hat = P P^+ P = P` exactly by the
Moore–Penrose identity.  That is the correctness anchor.

Not causal: the segment means pool over the whole sequence.

</div>

## Related methods

changes the same object — [`interaction`](../taxonomy.md#interaction)
:   [`based`](based.md), [`cosformer`](cosformer.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`linear_elu`](linear_elu.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`performer`](performer.md), [`polaformer`](polaformer.md), [`polysketch`](polysketch.md), [`rebased`](rebased.md), [`skyformer`](skyformer.md), [`trigrff`](trigrff.md) &nbsp;·&nbsp; [browse the bank, interaction only](index.md?category=interaction)

judged the same way — `approximate`
:   36 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `nystrom` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `nystrom` selected](index.md?compare=nystrom) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Xiong et al. (2021).** *Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention.* Proceedings of the AAAI Conference on Artificial Intelligence. [arXiv:2102.03902](https://arxiv.org/abs/2102.03902)<br><span class="attn-cite" data-bibkey="xiong2021nystromformer"><code>xiong2021nystromformer</code></span>
- **Williams and Seeger (2001).** *Using the Nystrom Method to Speed Up Kernel Machines.* Advances in Neural Information Processing Systems 13 (NIPS 2000).<br><span class="attn-cite" data-bibkey="williams2001using"><code>williams2001using</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="xiong2021nystromformer">
<pre class="attn-bib__src"><code>@inproceedings{xiong2021nystromformer,
  title     = {Nystr{\&quot;o}mformer: A Nystr{\&quot;o}m-Based Algorithm for Approximating Self-Attention},
  author    = {Xiong, Yunyang and Zeng, Zhanpeng and Chakraborty, Rudrasis and Tan, Mingxing and Fung, Glenn and Li, Yin and Singh, Vikas},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {35},
  year      = {2021},
  eprint    = {2102.03902},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="williams2001using">
<pre class="attn-bib__src"><code>@inproceedings{williams2001using,
  title     = {Using the {N}ystr{\&quot;o}m Method to Speed Up Kernel Machines},
  author    = {Williams, Christopher K. I. and Seeger, Matthias},
  booktitle = {Advances in Neural Information Processing Systems 13 (NIPS 2000)},
  pages     = {682--688},
  year      = {2001}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/interaction/nystrom.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/interaction/nystrom.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
