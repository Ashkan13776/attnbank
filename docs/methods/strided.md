# Sparse Transformer

<p class="attn-standfirst" markdown="span">The *strided* and *fixed* factorized patterns.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `strided` &middot; `ab.build("strided")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.StridedAttention` &middot; [`attnbank/methods/restrict/strided.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/strided.py)

monograph
:   `sec:sparse`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

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

**Exact limit.** Built with `stride=1` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("strided").limit_config
{'stride': 1}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 68,419,584 | 268,435,456 | 0.255× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 68,681,728 | 4,718,592 | 14.6× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    l, blocks = self.stride, (n + self.stride - 1) // self.stride
    # strided: l + 1 local and one key per block; fixed: one block and c per block
    kept = min(n, l + 1 + blocks if self.pattern == "strided"
               else l + max(self.c, 1) * blocks)
    return {"flops": 2 * m * kept * (d + dv),
            "state_bytes": n * (d + dv) * bytes_per_el,   # the column heads reach back
            "bytes_moved": (m * d + m * kept * (d + dv) + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `c`, `chi2`, `eps`, `k_realized`, `n_local`, `n_periodic`, `pattern`, `stride`, `support`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `stride` | `4` |
| `pattern` | `'strided'` |
| `c` | `1` |

</div>

`stride`: the paper's `l`.  `pattern`: `'strided'` or `'fixed'`.

`c`: the width of the summarising column of the `'fixed'` pattern; ignored
by `'strided'`.

`ab.describe("strided")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("strided", stride=4, pattern='strided')
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["strided"], q, k, v, causal=True,
                 configs={"strided": {'stride': 4, 'pattern': 'strided'}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
StridedAttention(name='strided', category='restrict', objective='approximate', backend='reference', c=1, pattern='strided', stride=4)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[c, chi2, eps, k_realized, n_local, n_periodic, pattern, stride, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.65989, 'distributional_error': 1.65989, 'value_error': 0.0, 'tv': 0.63219}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:sparse-fixed</code></p>

Child et al. (2019, Sec. 4.3) factor the attention matrix into
two heads whose index sets are, for query `i` and stride `l`,

```text
strided:   A_i^(1) = {t, ..., i},  t = max(0, i - l)
           A_i^(2) = {j : (i - j) mod l = 0}

fixed:     A_i^(1) = {j : floor(j/l) = floor(i/l)}
           A_i^(2) = {j : j mod l in {l - c, ..., l - 1}}
```

with `l ~ sqrt(n)` and `c` a small integer (8, 16 or 32 in the paper, for
`l = 128` or `256`).  The strided pattern suits data with a natural period — it
is the one-dimensional reading of attending to the previous row and the same column
of an image — and the fixed pattern suits data without one, where "specific cells
summarize previous locations and propagate that information to all future cells".

This method takes the **union** `A_i^(1) u A_i^(2)`, which is what a single
restricted layer can see; the paper's own alternative of giving one head each is a
statement about *depth*, and the bank's unit of measurement is one operator.  Both
components are stated relative to the query position and then intersected with the
mask, so the two-sided reading used in a bidirectional model collapses to Child et
al.'s autoregressive sets under `causal=True` and nothing in the future is ever
reachable.

Offset `0` is in both patterns (`j = i` satisfies `(i - j) mod l = 0` and lies in
the query's own block), so every row keeps at least the diagonal.  `stride = 1`
makes `A^(2)` the whole row under either pattern, which is the exact limit.

Values are untouched: the error is `eps_q (m_{S^c} - m_S)` (<span class="attn-ref">prop:sparse-identity</span>).
Against `longformer` this isolates what a *periodic* support buys over a contiguous
one at the same budget — the two keep `O(l + n/l)` and `O(w)` keys respectively.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`topk`](topk.md) *(oracle)*, [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`
:   31 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `strided` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `strided` selected](index.md?compare=strided) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Child et al. (2019).** *Generating Long Sequences with Sparse Transformers.* [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)<br><span class="attn-cite" data-bibkey="child2019generating"><code>child2019generating</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 1 entry, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="child2019generating">
<pre class="attn-bib__src"><code>@misc{child2019generating,
  title = {{Generating Long Sequences with Sparse Transformers}},
  author = {Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
  year = {2019},
  howpublished = {arXiv preprint arXiv:1904.10509},
  eprint = {1904.10509},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/strided.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/strided.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
