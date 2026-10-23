# Reformer LSH attention

<p class="attn-standfirst" markdown="span">The support is a hash bucket.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `reformer` &middot; `ab.build("reformer")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.ReformerAttention` &middot; [`attnbank/methods/restrict/reformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/reformer.py)

monograph
:   `sec:sparse`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

</div>

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **yes** | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**Exact limit.** Built with `n_buckets=1` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("reformer").limit_config
{'n_buckets': 1}
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 69,206,016 | 268,435,456 | 0.258× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 67,502,080 | 4,718,592 | 14.3× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    half = max(1, self.n_buckets // 2)
    kept = max(1.0, n / max(1, 2 * half)) * self.n_rounds
    return {"flops": 2 * (m + n) * d * half * self.n_rounds + 2 * m * kept * (d + dv),
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * d + m * kept * (d + dv) + m * dv) * bytes_per_el}
```

!!! note "A cost model is not a benchmark"
    These are counted operations and resident bytes, not measured time.
    They are what the monograph's capacity arguments are made of;
    `examples/08_cost_and_capacity.py` walks through the comparison properly.

## Backends

<div class="attn-tab" markdown="1">

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

</div>

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `bucket_k`, `bucket_q`, `chi2`, `collision_rate`, `eps`, `k_realized`, `n_buckets`, `n_rounds`, `support`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_buckets` | `8` |
| `n_rounds` | `2` |
| `seed` | `0` |

</div>

```python
ab.describe("reformer")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("reformer", n_buckets=8, n_rounds=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["reformer"], q, k, v, causal=True,
                 configs={"reformer": {'n_buckets': 8, 'n_rounds': 2}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
ReformerAttention(name='reformer', category='restrict', objective='approximate', backend='reference', n_buckets=8, n_rounds=2, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[bucket_k, bucket_q, chi2, collision_rate, eps, k_realized, n_buckets, n_rounds, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.98338, 'distributional_error': 1.98338, 'value_error': 0.0, 'tv': 0.66014}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:sparse-content` ("Content-based supports").  Angular locality-sensitive hashing
gives a *query-dependent, content-based* support that costs `O(n d b)` to build
rather than the `O(n d)` of the full score row it replaces.  With `R` a random
`d x (b/2)` Gaussian matrix, Reformer's hash is

```text
h(x) = argmax( [x R ; -x R] )  in  {0, ..., b-1},
```

a random partition of the sphere into `b` cones whose collision probability
`P[h(q) = h(k)] = 1 - theta(q, k) / pi` decreases with the angle, so the bucket of a
query is biased towards its large-score keys.  `n_rounds` independent hashes are
drawn and the supports unioned, which raises recall at linear cost — the standard
amplification, and the knob Reformer actually reports.

Given the support the operator is the renormalized restriction of
`prop:sparse-identity`: the error is `eps_q (m_{S^c} - m_S)` with
`eps_q = pi_q(S^c)`, exactly as for `topk`.  What LSH buys is that `S_q` is
found without computing `pi_q`; what it costs is that `S_q` is not the top-k —
collisions are probabilistic, so a heavy key is missed with probability
`(theta/pi)^n_rounds` and light keys are retained.  Contrast `lsh_sample`, which
uses the same hashing to build a *proposal* and reweights by `1/u` instead of
restricting: that estimator is consistent, this one is biased, and the bias is the
omitted mass.

Deviations from the paper, which are about the kernel rather than the operator:

* Reformer *ties* queries and keys (`k_j = q_j / ||q_j||`) so that one hash serves
  both and a position always collides with itself.  The bank's queries and keys are
  independent, so both are hashed with the same `R` and the query's own position is
  added to its support unconditionally — which is what tying makes automatic, and
  what guarantees a nonempty row.  With shared QK the two agree.
* Reformer then *sorts* by bucket and attends within fixed-size chunks (plus the
  previous chunk), to bound memory and to handle uneven buckets.  That is a batching
  device: the set it approximates is the one implemented here, "keys sharing a bucket
  with the query in at least one round".  `hyperattention` does implement a sorted,
  block-structured version of the same idea.
* Reformer masks a position from attending to itself when it has other options, an
  artifact of shared QK that does not apply here.

Exact limit: `n_buckets = 1` puts every key in the query's bucket, so the support is
the whole visible context and `pi_hat = pi`.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`atom`](atom.md), [`based`](based.md), [`clustered`](clustered.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`skyformer`](skyformer.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`trigrff`](trigrff.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both lists are `ab.available(...)` with `reformer` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `reformer` selected](index.md?compare=reformer) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Reformer: The Efficient Transformer.*<br><span class="attn-cite" data-bibkey="kitaev2020reformer"><code>kitaev2020reformer</code></span>
- *Practical and Optimal LSH for Angular Distance.*<br><span class="attn-cite" data-bibkey="andoni2015practical"><code>andoni2015practical</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="kitaev2020reformer">
<pre class="attn-bib__src"><code>@inproceedings{kitaev2020reformer,
  title={Reformer: The Efficient Transformer}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="andoni2015practical">
<pre class="attn-bib__src"><code>@inproceedings{andoni2015practical,
  title = {Practical and Optimal {LSH} for Angular Distance}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/reformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/reformer.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
