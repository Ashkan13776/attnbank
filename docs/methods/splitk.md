# Split-K / Flash-Decoding

<p class="attn-standfirst" markdown="span">Partition the keys, attend independently, merge by LSE.</p>

<div class="attn-facts" data-cat="exact" markdown="1">

key
:   `splitk` &middot; `ab.build("splitk")`

category
:   [`exact`](../taxonomy.md#exact)

objective
:   `accelerate` — same operator, exact up to floating point; only the evaluation changes

class
:   `attnbank.SplitKAttention` &middot; [`attnbank/methods/exact/splitk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/splitk.py)

monograph
:   `sec:exact`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes nothing in $y(q)=\int v\,\mathrm{d}\pi_q$. The integrand, the measure and the normaliser are the reference operator's; only the order of arithmetic changes.

It pursues **acceleration**: the operator is the reference operator, exact up to floating point. The only question worth asking of it is *where* exactness stops.

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

This method is **exact as built**: the conformance suite instantiates it at its default config and requires both `y` and the weights to match `softmax`. `limit_config()` returns `{}` — the empty dict, not `None`: there is a limit and no setting has to be changed to reach it.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,959,744 | 268,435,456 | 1× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 1,605,632 | 4,718,592 | 0.34× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    p = self.splits
    return {"flops": 2 * m * n * (d + dv) + 2 * m * p * dv,
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv + 2 * m * p * (dv + 2))
                           * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** The method exists to expose the *intermediate* states of the evaluation, which a single fused call does not hand back.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `lse`, `n_splits`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `splits` | `4` |
| `partition` | `None` |
| `tree` | `False` |

</div>

`partition`: an explicit list of index tensors/lists; overrides `splits`.
`tree`: merge pairwise (a balanced reduction tree) instead of left-to-right.

`ab.describe("splitk")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("splitk", splits=4, tree=False)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = accelerate: the claim is equality with the reference, so that
# is what the example checks.
y_ref = ab.attend("softmax", q, k, v, causal=True)
print("max |y - y_softmax|:", float((out.y - y_ref).abs().max()))
```

<div class="attn-out" markdown="1">

```text
SplitKAttention(name='splitk', category='exact', objective='accelerate', backend='reference', partition=None, splits=4, tree=False)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[lse, n_splits], dtype=float32, device=cpu)
invariant W: 4.172325134277344e-07
max |y - y_softmax|: 4.172325134277344e-07
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:exact-tiling</code> (decode-specialized kernels)</p>

The context is split into `p`
chunks, each chunk produces a state `sigma_{J_t}` of <span class="attn-ref">def:exact-state</span>, and the
states are combined by the monoid of <span class="attn-ref">prop:exact-monoid</span>.  Parts (i)-(iii) of that
proposition say the result is independent of the partition *and* of the bracketing,
which is exactly what `tests/test_exact.py` checks: every partition, contiguous or
not, gives the identical answer.

</div>

## Related methods

changes the same object — [`exact`](../taxonomy.md#exact)
:   [`flash`](flash.md), [`lowprec`](lowprec.md), [`softmax`](softmax.md) &nbsp;·&nbsp; [browse the bank, exact only](index.md?category=exact)

Both come from `ab.available(...)` with `splitk` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `splitk` selected](index.md?compare=splitk) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Dao et al. (2023).** *Flash-Decoding for long-context inference.* Stanford CRFM blog, \urlhttps://crfm.stanford.edu/2023/10/12/flashdecoding.html.<br><span class="attn-cite" data-bibkey="dao2023flash"><code>dao2023flash</code></span>
- **Sanovar et al. (2024).** *Lean Attention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers.* [arXiv:2405.10480](https://arxiv.org/abs/2405.10480)<br><span class="attn-cite" data-bibkey="sanovar2024lean"><code>sanovar2024lean</code></span>
- **Shyam et al. (2024).** *Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters.* [arXiv:2408.04093](https://arxiv.org/abs/2408.04093)<br><span class="attn-cite" data-bibkey="shyam2024tree"><code>shyam2024tree</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="dao2023flash">
<pre class="attn-bib__src"><code>@misc{dao2023flash,
  title={Flash-Decoding for long-context inference},
  author={Dao, Tri and Haziza, Daniel and Massa, Francisco and Sizov, Grigory},
  year={2023}, howpublished={Stanford CRFM blog, \url{https://crfm.stanford.edu/2023/10/12/flashdecoding.html}},
  note={Accessed October 2026}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sanovar2024lean">
<pre class="attn-bib__src"><code>@misc{sanovar2024lean,
  title={Lean Attention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers},
  author={Sanovar, Rya and Bharadwaj, Srikant and St. Amant, Renee and R{\&quot;u}hle, Victor and Rajmohan, Saravan},
  year={2024}, eprint={2405.10480}, archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="shyam2024tree">
<pre class="attn-bib__src"><code>@misc{shyam2024tree,
  title={Tree Attention: Topology-aware Decoding for Long-Context Attention on {GPU} clusters},
  author={Shyam, Vasudev and Pilault, Jonathan and Shepperd, Emily and Anthony, Quentin and Millidge, Beren},
  year={2024}, eprint={2408.04093}, archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/exact/splitk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/exact/splitk.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
