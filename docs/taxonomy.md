# The taxonomy

Attention is a query-conditioned expectation,

$$y(q) \;=\; \int v \,\mathrm{d}\pi_q, \qquad \pi_q \;=\; \frac{g(q,\cdot)\,\mu}{Z(q)} .$$

Efficiency has to come from somewhere, and there are only so many places it can come
from. A method either evaluates this expression differently, or it changes one of the
objects in it. **Seven categories, because there are seven objects to change.**

Cutting across them is a second question, the one that decides how a method should be
judged at all:

| `objective` | meaning |
|---|---|
| `accelerate` | the same operator, exact up to floating point |
| `approximate` | targets dense softmax; judged by output error |
| `modify` | a *different operator*; fidelity to softmax is not the goal |

Conflating `approximate` and `modify` is the most persistent confusion in this
literature. Asking how closely `linear_elu` approximates softmax is asking the wrong
question — it is not trying to. Every method here declares which it is, and the
conformance suite uses the declaration: an `approximate` method must supply a
correctness anchor, a `modify` method is exempt and the suite *asserts the exemption*
rather than skipping it.

!!! info "This page is generated"
    The tables below come from the live registry, so they cannot drift from the code.
    Regenerate with `python docs/gen_taxonomy.py`; `ab.list_methods()` prints the same
    thing in a terminal.

Every method name links to its own page. To search and filter the whole bank at once —
by category, objective, causal support, backend — use
[**Browse the bank**](methods/index.md).


## `exact`

**Changes:** nothing changes but the evaluation: tiling, chunking, precision.

Nothing in the formula changes — only how it is evaluated. Tiling, chunking, reduced precision. The interesting question in this category is *where exactness stops*: `flash` and `splitk` are exact up to floating point, `lowprec` is not, and the boundary between them is the point.

| method | objective | causal | weights | stochastic | backends | what it is |
|---|---|---|---|---|---|---|
| [`flash`](methods/flash.md) | accelerate | yes | yes | yes | reference, sdpa | Single-pass left fold of the online-softmax monoid over key blocks. |
| [`lowprec`](methods/lowprec.md) | approximate | yes | yes | yes | reference | Quantize--dequantize `q, k, v` (and optionally `P`) inside the kernel. |
| [`softmax`](methods/softmax.md) | accelerate | yes | yes | yes | reference, sdpa, flash_attn | Dense softmax attention (Vaswani et al., 2017); the reference operator. |
| [`splitk`](methods/splitk.md) | accelerate | yes | yes | yes | reference | Two-level reduction: per-chunk states, then one merge. |


## `interaction`

**Changes:** the interaction g(q, x) is replaced by a cheaper surrogate.

The interaction $g(q, x)$ is replaced by a cheaper surrogate $\widehat g$. Random features estimate $\exp\langle q,k\rangle$; fixed feature maps replace it outright. The first is an approximation whose error shrinks with the feature count; the second is a different operator that happens to be cheap. The `objective` column is where that difference is recorded.

| method | objective | causal | weights | stochastic | backends | what it is |
|---|---|---|---|---|---|---|
| [`cosformer`](methods/cosformer.md) | modify | yes | yes | yes | reference | `relu` features times `cos(pi (i - j) / 2M)`. |
| [`linear_elu`](methods/linear_elu.md) | modify | yes | yes | yes | reference | `phi(u) = elu(u) + 1`, normalized separable attention. |
| [`nystrom`](methods/nystrom.md) | approximate | — | yes | — | reference | Landmark Nystrom factorization of the softmax matrix; signed weights. |
| [`performer`](methods/performer.md) | approximate | yes | yes | yes | reference | FAVOR+ positive random features, optionally orthogonal. |
| [`trigrff`](methods/trigrff.md) | approximate | yes | yes | — | reference | Rahimi--Recht trigonometric random features; signed, unnormalized measure. |


## `restrict`

**Changes:** the support of pi_q is restricted to a subset S_q of positions.

The support of $\pi_q$ shrinks to a subset $S_q$, and the retained mass is renormalised. The whole category is one operator — `prop:sparse-identity` — differing only in how $S_q$ is chosen. Error is purely *distributional*: the values are untouched.

| method | objective | causal | weights | stochastic | backends | what it is |
|---|---|---|---|---|---|---|
| [`blocktopk`](methods/blocktopk.md) | approximate | yes | yes | yes | reference | Page-level selection by an upper bound on the page's scores. |
| [`lsh_sample`](methods/lsh_sample.md) | approximate | yes | yes | yes | reference | Self-normalized importance sampling over a score-monotone proposal. |
| [`sink_window`](methods/sink_window.md) | approximate | yes | yes | yes | reference | `s` initial sinks plus a left window of width `w`. |
| [`topk`](methods/topk.md) | approximate | yes | yes | yes | reference | Exact top-k selection on the true scores (an oracle for proxy selectors). |
| [`value_topk`](methods/value_topk.md) **(oracle)** | approximate | yes | yes | yes | reference | Greedy (or exhaustive) output-aware support selection. Oracle. |
| [`window`](methods/window.md) | approximate | yes | yes | yes | reference | Left sliding window of width `w`, always including the diagonal. |


## `compress`

**Changes:** the stored context (mu, k, v) is replaced by a smaller one.

The stored context $(\mu, k, v)$ is replaced by something smaller: quantised, low-rank, evicted, merged, or shared across heads. This is the only category that routinely produces *value* error, because it is the only one that changes what is aggregated.

| method | objective | causal | weights | stochastic | backends | what it is |
|---|---|---|---|---|---|---|
| [`evict_h2o`](methods/evict_h2o.md) | approximate | yes | yes | yes | reference | Online heavy-hitter eviction with a protected recent window. |
| [`evict_snapkv`](methods/evict_snapkv.md) | approximate | — | yes | yes | reference | Observation-window scoring with pooled, budgeted selection. |
| [`gqa`](methods/gqa.md) | modify | yes | yes | yes | reference, sdpa | Mean-pool K and V within groups of query heads. |
| [`linformer`](methods/linformer.md) | approximate | — | yes | — | reference | Random (or identity) sequence-axis projections of K and V. |
| [`lowrank_kv`](methods/lowrank_kv.md) | approximate | — | yes | yes | reference | Truncated SVD of the key and/or value channel matrices. |
| [`merge_kv`](methods/merge_kv.md) | approximate | — | yes | yes | reference | Cluster keys and values, merge with proportional (pushforward) mass. |
| [`mla`](methods/mla.md) | modify | yes | yes | yes | reference, sdpa | Joint low-rank KV latent shared by all heads. |
| [`quantkv`](methods/quantkv.md) | approximate | yes | yes | yes | reference | Uniform affine quantization of the key and value caches. |


## `factorize`

**Changes:** pi_q is factored through r latents: pi_q = a(q, .) b(., x).

$\pi_q$ is routed through $r$ latents, $\pi_q = a(q,\cdot) b(\cdot,x)$. Whether the factorisation can stay row-stochastic is the question: `latent_stochastic` is stochastic by construction, `svd_oracle` is the signed lower bound it cannot reach, and `nmf_oracle` measures the gap.

| method | objective | causal | weights | stochastic | backends | what it is |
|---|---|---|---|---|---|---|
| [`clustered`](methods/clustered.md) | approximate | — | yes | yes | reference | One attention row per query centroid, shared within the cluster. |
| [`latent_stochastic`](methods/latent_stochastic.md) | modify | — | yes | yes | reference | pi_hat = A B with both factors row-stochastic (two-hop attention). |
| [`nmf_oracle`](methods/nmf_oracle.md) **(oracle)** | approximate | — | yes | yes | reference | Best nonnegative rank-r factorization of P, renormalized. Oracle. |
| [`svd_oracle`](methods/svd_oracle.md) **(oracle)** | approximate | — | yes | — | reference | Truncated SVD of P: the best signed rank-r factorization. Oracle. |


## `coupling`

**Changes:** normalisation stops being per-query; queries are coupled.

Normalisation stops being per-query. Sinkhorn balances the whole matrix to prescribed marginals, which makes the result softmax *tilted by a key potential* rather than softmax. Queries are no longer independent, so causal masking is not available.

| method | objective | causal | weights | stochastic | backends | what it is |
|---|---|---|---|---|---|---|
| [`lowrank_ot`](methods/lowrank_ot.md) | modify | — | yes | yes | reference | Doubly stochastic, rank-<= r attention through a pivot measure. |
| [`sinkhorn`](methods/sinkhorn.md) | modify | — | yes | yes | reference | Entropic-OT attention: Sinkhorn normalization of exp(s) to prescribed marginals. |


## `recurrent`

**Changes:** the operator itself becomes a finite-state recurrence.

The operator becomes a finite-state recurrence: a fixed-size state $H_t$ absorbs the context. Linear in the values, so Invariant W still holds and the implied weights can still be written down — they are simply not a probability measure.

| method | objective | causal | weights | stochastic | backends | what it is |
|---|---|---|---|---|---|---|
| [`decay_rnn`](methods/decay_rnn.md) | modify | yes | yes | — | reference | Fixed exponential decay on the recurrent state. |
| [`delta_rnn`](methods/delta_rnn.md) | modify | yes | yes | — | reference | Delta-rule (Widrow--Hoff) fast-weight memory. |
| [`linear_rnn`](methods/linear_rnn.md) | modify | yes | yes | — | reference | Additive linear recurrence with an unnormalized readout. |


## Reading the columns

**`weights`** — can the method name its measure $\widehat\pi_q$ on the *original* $n$
positions? A `—` means it genuinely cannot, and the method is then excluded from the
error-decomposition studies **by declaration, not by accident**. See
[the invariant](invariant.md).

**`stochastic`** — are those weights nonnegative and summing to one? `nystrom`,
`linformer` and the unnormalised recurrences say no, and say it in an attribute rather
than in a footnote.

**`causal`** — a `—` is a statement about the operator. Sinkhorn normalises across
queries; there is no causal version of that, and asking for one raises an error that
says so.

**`backends`** — the execution paths this method claims are equal to the reference.
Most of the bank is `reference` only, which is the correct answer for an operator that
is not a dense softmax over a mask. See [Backends](backends.md).

## Oracles

Three methods — [`value_topk`](methods/value_topk.md), [`nmf_oracle`](methods/nmf_oracle.md),
[`svd_oracle`](methods/svd_oracle.md) — are not practical. They are
**upper bounds on what a mechanism can achieve**, which is exactly what several of the
monograph's arguments need: you cannot show that top-$k$ selection is suboptimal
without computing what the optimal selection would have been.
