# `lsh_sample`

**LSH importance sampling: MagicPIG-style key sampling with unbiased reweighting.**

category
:   [`restrict`](../taxonomy.md#restrict) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.LSHSampleAttention`, [`attnbank/methods/restrict/lsh_sample.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/lsh_sample.py)

monograph
:   `sec:sparse`

## What it changes

Changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:sparse-content` ("Sampling is not restriction") and `prop:error-ratio-est`.
A key set `S` is *sampled* with inclusion probabilities `u_j` monotone in the
attention score, and the output is the self-normalized importance-sampling estimator
(`eq:sparse-snis`)

```text
y_hat(q) = sum_{j in S} (g_j / u_j) v_j  /  sum_{j in S} (g_j / u_j),
pi_hat_j = 1{j in S} (pi_j / u_j) / sum_i 1{i in S} (pi_i / u_i).
```

Numerator and normalizer are unbiased for `N(q)` and `Z(q)`; the ratio is only
consistent, with `O(1/N_s)` bias (Owen, 2013).  Unlike deterministic restriction this
does **not** systematically reallocate tail mass to the head — it trades the
value-dependent bias of `prop:sparse-identity` for variance, whose size is
`(1 + chi2(pi || lambda)) / N_s` times `sup_x ||v(x) - y||^2`
(`prop:error-ratio-est` (b)).

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

## Configuration

| parameter | default |
|---|---|
| `n_samples` | `8` |
| `alpha` | `1.0` |
| `seed` | `0` |

```python
ab.describe("lsh_sample")     # the same keys, with everything else it declares
```

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

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2_pi_lambda`, `empty_draws`, `eps`, `n_sampled`, `support`, `u`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    ns = self.n_samples or n
    return {"flops": 2 * m * n * d + 2 * m * ns * dv,
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * d + m * ns * dv + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 135,266,304 | 268,435,456 | 0.504× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 1,441,792 | 4,718,592 | 0.306× |

!!! note "A cost model is not a benchmark"
    These are counted operations and resident bytes, not measured time.
    They are what the monograph's capacity arguments are made of;
    `examples/08_cost_and_capacity.py` walks through the comparison properly.

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

```text
LSHSampleAttention(name='lsh_sample', category='restrict', objective='approximate', backend='reference', alpha=1.0, n_samples=8, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2_pi_lambda, empty_draws, eps, n_sampled, support, u], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 1.31059, 'distributional_error': 1.31059, 'value_error': 0.0, 'tv': 0.53857}
```

## Others in `restrict`

[`blocktopk`](blocktopk.md), [`sink_window`](sink_window.md), [`topk`](topk.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) — see [the taxonomy](../taxonomy.md#restrict).

## References

- **Chen et al. (2025).** *MagicPIG: LSH Sampling for Efficient LLM Generation.* International Conference on Learning Representations (ICLR). [arXiv:2410.16179](https://arxiv.org/abs/2410.16179)<br>`chen2025magicpig`
- **Owen (2013).** *Monte Carlo Theory, Methods and Examples.*<br>`owen2013monte`
- **Kitaev et al. (2020).** *Reformer: The Efficient Transformer.* International Conference on Learning Representations (ICLR). [arXiv:2001.04451](https://arxiv.org/abs/2001.04451)<br>`kitaev2020reformer`

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/lsh_sample.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/lsh_sample.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
