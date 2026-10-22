# attnbank design contract

This file is the specification that every method, test and study in `attnbank` must satisfy.
It exists so that the library stays small, predictable and *provably* consistent with the paper.
If an implementation and this file disagree, this file wins — fix the implementation.

Companion to *What Does Efficient Attention Approximate?* The taxonomy below is the one in
`sections/03f_comparative.tex` (`tab:comparative`). Do not invent new categories.

---

## 1. The one invariant

Every method realises, exactly or implicitly, a measure $\widehat\pi_q$ over context positions and
a value field $\widehat v$:

$$\widehat y(q) \;=\; \sum_{j=1}^{n} \widehat\pi_q(j)\,\widehat v_j .$$

**Invariant W.** If a method returns `weights` and `v_hat`, then

```
y  ==  weights @ (v_hat if v_hat is not None else v)      # within 1e-5 (fp32) / 1e-10 (fp64)
```

This is checked for every registered method by the conformance suite. It is the single most
important property in the library: all of `attnbank.metrics` presumes it, and every study that
decomposes error into a distributional and a value term is meaningless without it.

A method that cannot express its measure on the original $n$ positions sets
`expresses_weights = False` and returns `weights=None`. It is then excluded from the
decomposition studies, by design, not by accident.

---

## 2. Public API (keep it this small)

```python
import attnbank as ab

ab.available()                        # -> ['blocktopk', 'flash', 'gqa', ...]
ab.available("restrict")              # filter by category
ab.method_table()                     # one row per method, for the paper's table
ab.build("topk", k=64)                # -> AttentionMethod (an nn.Module)

out = ab.build("topk", k=64)(q, k, v, causal=True, return_weights=True)
out.y, out.weights, out.v_hat, out.extras

ab.attend("topk", q, k, v, k_=64)     # one-liner: returns y only
ab.compare(["softmax", "topk", "performer"], q, k, v, causal=True)
                                      # -> list[dict] of metrics vs. the reference, one per method
```

Tensors: `q [B,H,m,d]`, `k [B,H,n,d]`, `v [B,H,n,dv]`. Causal masking aligns the **last** query
with the **last** key, so `m <= n` and query $i$ sees keys $0 \dots i+(n-m)$. This matches decoding.

`AttentionOutput` fields are exactly: `y`, `weights`, `v_hat`, `extras`. Nothing else.

### Class attributes a method must declare

| Attribute | Meaning |
|---|---|
| `name` | unique registry key, lowercase, no spaces |
| `category` | `Category.{EXACT,INTERACTION,RESTRICT,COMPRESS,FACTORIZE,COUPLING,RECURRENT}` |
| `objective` | `Objective.{ACCELERATE,APPROXIMATE,MODIFY}` — the paper's third axis |
| `supports_causal` | bool |
| `expresses_weights` | bool — can it return $\widehat\pi_q$ on the original $n$ positions? |
| `row_stochastic` | bool — are those weights nonnegative and summing to 1? (`False` for Nyström, Linformer, unnormalized recurrences) |
| `trainable` | bool — are the parameters meaningful only after training? |
| `references` | tuple of BibTeX keys **that exist in `references.bib`** |
| `paper_section` | the `\label` of the section that discusses it, e.g. `"sec:sparse"` |

### Methods a method must implement

```python
def _forward(self, q, k, v, causal, return_weights) -> AttentionOutput: ...
def cost(self, m, n, d, dv, bytes_per_el=2) -> dict:   # 'flops', 'state_bytes', 'bytes_moved'
@classmethod
def limit_config(cls) -> dict | None:
    """Config under which this method provably reduces to dense softmax, or None.
    e.g. TopK -> {'k': None} (all keys); Quantize -> {'bits': 16}.
    The conformance suite uses it as the correctness anchor."""
```

`limit_config` is **not optional decoration** — it is how we show a method is implemented
correctly rather than merely plausibly. Every method whose `objective` is `ACCELERATE` or
`APPROXIMATE` must declare an anchor: either an exact `limit_config()`, or — for Monte-Carlo
estimators such as `performer` and `trigrff`, where no finite `r` reproduces softmax to
tolerance — a `convergence_configs()` sequence whose error must decrease monotonically toward
softmax.

Methods whose `objective` is `MODIFY` are **exempt**, and the suite asserts the exemption rather
than silently skipping it. `linear_elu`, `cosformer`, `latent_stochastic`, `lowrank_ot`,
`linear_rnn`, `decay_rnn` and `delta_rnn` neither reduce to nor converge to softmax; that is what
it means to define a different operator (`tab:comparative`, "mod."). Demanding an anchor of them
would be demanding that they stop being what they are.

---

## 3. Method roster

Category assignment follows `tab:comparative`. The right-hand column is the paper claim the
method exists to let us test.

### `exact/` — evaluation changes, operator does not (`sec:exact`)
| name | What | Anchors |
|---|---|---|
| `softmax` | reference operator (already in `reference.py`) | — |
| `flash` | tiled online-softmax (running max + rescale), single pass over key blocks | `prop:exact-monoid`; must equal `softmax` to 1e-6 |
| `splitk` | partition keys into chunks, attend independently, merge by log-sum-exp | the monoid is **associative**: any partition gives the same answer |
| `lowprec` | simulated FP8/INT8 logits and values (quantize-dequantize in the kernel) | `sec:exact-boundary` — where acceleration stops being exact |

### `interaction/` — $g \to \widehat g$ (`sec:kernel`)
| name | What | Anchors |
|---|---|---|
| `performer` | FAVOR+ positive random features, optional orthogonal blocks | `prop:error-ratio-est`, bias $O(1/r)$ |
| `trigrff` | Rahimi–Recht trigonometric features (signed) | fails in the large-logit regime; variance blow-up |
| `linear_elu` | $\phi(x)=\mathrm{elu}(x)+1$, unnormalized-score linear attention | an *operator modification*, not an approximation |
| `cosformer` | ReLU features with a cosine re-weighting | fixed feature map, `mod.` |
| `nystrom` | landmark Nyström on the softmax matrix; signed weights | loses `hl` only — unit row mass survives when $G$ is invertible |

### `restrict/` — support of $\pi_q$ (`sec:sparse`)
| name | What | Anchors |
|---|---|---|
| `topk` | oracle top-$k$ on true scores, renormalized | `prop:sparse-identity` exactness |
| `window` | sliding local window | `prop:sparse-identity` with a query-independent support |
| `sink_window` | StreamingLLM: first $s$ sinks + window | same, plus the sink effect |
| `blocktopk` | page/block-level selection by block-max score (Quest-like) | selection granularity costs omitted mass |
| `value_topk` | **output-aware selection**: best per query of score top-$k$, forward greedy, seeded greedy, and backward greedy (the `prop:error-single` deletion step) | the paper's claim that top-$k$ is *not* output-optimal. Naive forward greedy alone loses to top-$k$ — keep it available for comparison |
| `lsh_sample` | importance sampling of keys with unbiased reweighting (MagicPIG-like) | variance $\propto (1+\chi^2)/N_s$ |

### `compress/` — the stored $(\mu,k,v)$ (`sec:compress`)
| name | What | Anchors |
|---|---|---|
| `quantkv` | uniform affine quantisation, `bits_k`, `bits_v`, `group` axis/size, asymmetric | `prop:compress-sensitivity` (keys hurt more than values) |
| `lowrank_kv` | truncated SVD of the key and/or value channel matrices | value error is $(I-\Pi_r)y$, **not** $(I-P)\widehat y$ |
| `evict_h2o` | accumulated-attention heavy-hitter eviction + recent window | `prop:compress-noeviction`: no uniform guarantee |
| `evict_snapkv` | pooled scores over an observation window | same |
| `merge_kv` | cluster keys, merge with proportional mass (a pushforward $T_\#\mu$) | `prop:compress-merge`: $W_1$ / oscillation bound |
| `gqa` | share K/V across groups of query heads (mean-pool the groups) | capacity/traffic, raises arithmetic intensity |
| `mla` | joint low-rank KV latent with per-head up-projection | same |
| `linformer` | fixed random projection of K and V along the sequence axis | signed, loses `nm` |

### `factorize/` — $\pi_q = a(q,\cdot)\circ b(\cdot,x)$ (`sec:factor`)
| name | What | Anchors |
|---|---|---|
| `latent_stochastic` | two Markov kernels through $r$ latents; row-stochastic by construction | `prop:factor-closure`, `prop:factor-output` |
| `clustered` | route queries through $r$ query centroids | `sec:factor-methods` |
| `nmf_oracle` | best rank-$r$ **nonnegative** factorization of $P$ (multiplicative updates) | nonnegative rank $\ge$ rank: the empirical gap |
| `svd_oracle` | best rank-$r$ signed factorization of $P$ | the lower bound the above cannot reach |

### `coupling/` — global constraints across queries (`sec:ot`)
| name | What | Anchors |
|---|---|---|
| `sinkhorn` | Sinkhorn normalisation of $\exp(s)$ to prescribed marginals | `prop:ot-conditional`: result is softmax **tilted by a key potential** |
| `lowrank_ot` | factored coupling through $r$ anchors (LOTFormer-style) | `sec:ot` |

### `recurrent/` — finite state (`sec:memory-state`)
| name | What | Anchors |
|---|---|---|
| `linear_rnn` | $H_t = H_{t-1} + \phi(k_t)v_t^\top$, chunked parallel form | `prop:memory-rank`: recall bounded by state |
| `decay_rnn` | scalar/diagonal decay (RetNet / GLA style) | state size vs. recall |
| `delta_rnn` | delta-rule update $H_t = (I-\beta_t\psi_t\psi_t^\top)H_{t-1} + \beta_t\psi_t v_t^\top$, $H\in\R^{r\times d_v}$ | capacity at fixed state |

---

## 4. Correctness: the conformance suite

`tests/test_conformance.py` is parametrized over **every** registered method and is the reason
this library can be trusted. Each method must pass, with no exemptions outside the declared
attributes:

1. **shapes** — `y` is `[B,H,m,dv]`; `weights` (if any) is `[B,H,m,n]`; `v_hat` (if any) is `[B,H,n,dv]`.
2. **Invariant W** — `y == weights @ (v_hat or v)`.
3. **row stochasticity** — if `row_stochastic`, weights are $\ge 0$ and rows sum to 1.
4. **causal leakage** — perturb `k[..., t:, :]` and `v[..., t:, :]`; outputs at query positions
   before $t$ must be bit-for-bit unchanged. Run for several $t$.
5. **batch and head independence** — perturbing batch $b$ or head $h$ changes nothing elsewhere.
   Methods that share K/V across query heads (`gqa`, `mla`) declare `head_independent = False`;
   for them the suite checks the exact coupling boundary instead — perturbing head $h$ moves its
   whole group and nothing outside it. Sharing across heads is the mechanism, not a defect.
6. **exact limit** — under `limit_config()`, `y` matches `softmax` to 1e-5 (fp32) and the
   weights match to 1e-6.
7. **convergence** — for Monte-Carlo methods, error decreases monotonically (in the mean over
   seeds) along `convergence_configs()`.
8. **dtype and device** — runs in fp32 and fp64 on CPU, and on CUDA when available; fp64 results
   agree with fp32 within the fp32 tolerance.
9. **cost model** — `cost()` returns the three keys, all finite and positive, and `state_bytes`
   is $\le$ the dense `softmax` value for any method claiming to save capacity.
10. **no NaN / no Inf** anywhere in the output, including for rows with empty or single-element
    support, and for inputs with large logits (`q,k` scaled by 10).
11. **references exist** — every key in `references` is present in `references.bib`.

Rules of thumb that prevent the bugs we care about:

- Compute softmax in log-space; never materialise `exp` of an unshifted score.
- A fully masked row must produce a documented, finite result — never `nan`. Window methods must
  guarantee at least one visible key (include the diagonal).
- Renormalise explicitly: `pi = pi / pi.sum(-1, keepdim=True).clamp_min(tiny)` and assert the
  denominator was not clamped in tests.
- Never use in-place ops on inputs. Methods must not mutate `q`, `k`, `v` (tested).
- Seeded randomness goes through `torch.Generator`, never the global RNG, so studies reproduce.

---

## 5. Repository layout

```
attnbank/
  attnbank/
    core.py registry.py reference.py metrics.py
    methods/<category>/<name>.py        # one method per file
    plotting/style.py
    data.py                             # real-activation capture (GPT-2 / Qwen2.5-0.5B)
    studies/_harness.py                 # sweep runner, caching, result tables
    studies/<Sn>_<tag>.py               # one study per file (see STUDIES.md)
  tests/test_core.py test_metrics.py test_conformance.py test_<category>.py
  results/<study>.json                  # numeric results, committed
figures/<name>.pdf                      # committed, included by the LaTeX sections
```

Results are committed as JSON so that every number in the paper can be traced to a file, and so
the polish pass can check prose against data without re-running anything.

## 6. Style

Reference implementations favour clarity over speed; they are read by people checking the maths
against the paper. Every method file opens with a docstring giving the operator in the paper's
notation, the section label, and the citation. No micro-optimisation, no fused tricks, no
`einsum` strings longer than the equation they implement.
