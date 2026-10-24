# LSH importance sampling

<p class="attn-standfirst" markdown="span">MagicPIG-style key sampling with unbiased reweighting.</p>

<div class="attn-facts" data-cat="restrict" markdown="1">

key
:   `lsh_sample` &middot; `ab.build("lsh_sample")`

category
:   [`restrict`](../taxonomy.md#restrict)

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.LSHSampleAttention` &middot; [`attnbank/methods/restrict/lsh_sample.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/lsh_sample.py)

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

**Exact limit.** Built with `n_samples=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("lsh_sample").limit_config
{'n_samples': None}
```

**Convergence anchor.** There is no finite resource at which this method is exact, so the anchor is monotonicity instead: the mean error against `softmax` must decrease along these configs, and the suite fails if it does not.

```pycon
>>> ab.get_class("lsh_sample").convergence_configs()
[{'n_samples': 2}, {'n_samples': 4}, {'n_samples': 8}, {'n_samples': 14}]
```

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 135,266,304 | 268,435,456 | 0.504× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 1,441,792 | 4,718,592 | 0.306× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    ns = self.n_samples or n
    return {"flops": 2 * m * n * d + 2 * m * ns * dv,
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * d + m * ns * dv + m * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2_pi_lambda`, `empty_draws`, `eps`, `n_sampled`, `support`, `u`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_samples` | `8` |
| `alpha` | `1.0` |
| `seed` | `0` |

</div>

`ab.describe("lsh_sample")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("lsh_sample", n_samples=8, alpha=1.0)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["lsh_sample"], q, k, v, causal=True,
                 configs={"lsh_sample": {'n_samples': 8, 'alpha': 1.0}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

<div class="attn-out" markdown="1">

```text
LSHSampleAttention(name='lsh_sample', category='restrict', objective='approximate', backend='reference', alpha=1.0, n_samples=8, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2_pi_lambda, empty_draws, eps, n_sampled, support, u], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.31059, 'distributional_error': 1.31059, 'value_error': 0.0, 'tv': 0.53857}
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:sparse-content</code> (&quot;Sampling is not restriction&quot;) and <code>prop:error-ratio-est</code></p>

A key set `S` is *sampled* with inclusion probabilities `u_j` monotone in the
attention score, and the output is the self-normalized importance-sampling estimator
(<span class="attn-ref">eq:sparse-snis</span>)

```text
y_hat(q) = sum_{j in S} (g_j / u_j) v_j  /  sum_{j in S} (g_j / u_j),
pi_hat_j = 1{j in S} (pi_j / u_j) / sum_i 1{i in S} (pi_i / u_i).
```

Numerator and normalizer are unbiased for `N(q)` and `Z(q)`; the ratio is only
consistent, with `O(1/N_s)` bias (Owen, 2013).  Unlike deterministic restriction this
does **not** systematically reallocate tail mass to the head — it trades the
value-dependent bias of <span class="attn-ref">prop:sparse-identity</span> for variance, whose size is
`(1 + chi2(pi || lambda)) / N_s` times `sup_x ||v(x) - y||^2`
(<span class="attn-ref">prop:error-ratio-est</span> (b)).

Proposal.  MagicPIG derives `u_j` from SimHash collision probabilities; this
reference implementation abstracts that to a tempered proposal
`lambda_j ∝ exp(alpha * s_j)` on the visible keys, with `alpha = 0` uniform and
`alpha = 1` tracking `pi` exactly, and Poisson (independent Bernoulli) sampling at
`u_j = min(1, N_s * lambda_j)`.  Sweeping `alpha` is how `chi2(pi || lambda)`,
the quantity the bound is stated in, is varied.

Exact limit.  `n_samples = None` sets `u_j = 1` for every visible key, so `S` is
the whole visible context and `pi_hat = pi` exactly.

Degenerate draw.  If a row samples nothing, the single largest-score visible key is
inserted so the estimator is defined; the event is counted in
`extras["empty_draws"]`.

</div>

## Related methods

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`
:   31 further methods besides these declare `objective = approximate`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

Both come from `ab.available(...)` with `lsh_sample` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `lsh_sample` selected](index.md?compare=lsh_sample) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Chen et al. (2025).** *MagicPIG: LSH Sampling for Efficient LLM Generation.* International Conference on Learning Representations (ICLR). [arXiv:2410.16179](https://arxiv.org/abs/2410.16179)<br><span class="attn-cite" data-bibkey="chen2025magicpig"><code>chen2025magicpig</code></span>
- **Owen (2013).** *Monte Carlo Theory, Methods and Examples.*<br><span class="attn-cite" data-bibkey="owen2013monte"><code>owen2013monte</code></span>
- **Kitaev et al. (2020).** *Reformer: The Efficient Transformer.* International Conference on Learning Representations (ICLR). [arXiv:2001.04451](https://arxiv.org/abs/2001.04451)<br><span class="attn-cite" data-bibkey="kitaev2020reformer"><code>kitaev2020reformer</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="chen2025magicpig">
<pre class="attn-bib__src"><code>@inproceedings{chen2025magicpig,
  title     = {{MagicPIG}: {LSH} Sampling for Efficient {LLM} Generation},
  author    = {Chen, Zhuoming and Sadhukhan, Ranajoy and Ye, Zihao and Zhou, Yang and Zhang, Jianyu and Nolte, Niklas and Tian, Yuandong and Douze, Matthijs and Bottou, Leon and Jia, Zhihao and Chen, Beidi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  eprint    = {2410.16179},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="owen2013monte">
<pre class="attn-bib__src"><code>@book{owen2013monte,
  title     = {Monte Carlo Theory, Methods and Examples},
  author    = {Owen, Art B.},
  year      = {2013},
  note      = {Book manuscript, available online at \url{https://artowen.su.domains/mc/}}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="kitaev2020reformer">
<pre class="attn-bib__src"><code>@inproceedings{kitaev2020reformer,
  title={Reformer: The Efficient Transformer},
  author={Kitaev, Nikita and Kaiser, {\L}ukasz and Levskaya, Anselm},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020},
  eprint={2001.04451},
  archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/lsh_sample.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/lsh_sample.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
