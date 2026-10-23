---
hide:
  - toc
---

# The taxonomy

<div class="ab-measure" markdown="1">
Attention is a query-conditioned expectation. Efficiency has to come from somewhere,
and there are only so many places it can come from: a method either evaluates the
expectation differently, or it changes one of the objects in it. **Seven categories,
because there are seven objects to change.**

Open a branch to see what that category changes, what it leaves alone, which term of
the error it incurs, and which methods are in it — and which symbol of the operator it
reaches for.
</div>


<div class="op" data-op markdown="1">

<div class="op-plate tx-sticky" data-plate="the operator">
--8<-- "docs/includes/operator.html"
</div>

<div class="tx-tree" markdown="1">

<section class="tx-node" data-cat="exact" data-syms="" data-frame markdown="1">

## `exact` <span class="tx-changes">changes nothing but the order of the arithmetic</span><span class="tx-n">4 methods</span> {: #exact data-toc-label="exact" }

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   Nothing changes but the evaluation: tiling, chunking, precision.

Preserves
:   Everything. The measure, the values and the normaliser are the reference operator's, bit for bit where the arithmetic allows it.

Error term
:   **None**, up to floating point — which is a claim the conformance suite checks rather than a promise. `lowprec` is the exception, and locating the exception is the purpose of the category.

Methods
:   <ul class="op-methods"><li><a href="methods/flash/">flash</a></li> <li><a href="methods/lowprec/">lowprec</a></li> <li><a href="methods/softmax/">softmax</a></li> <li><a href="methods/splitk/">splitk</a></li></ul>

</div>

Nothing in the formula changes — only how it is evaluated. Tiling, chunking, reduced precision. The interesting question in this category is *where exactness stops*: `flash` and `splitk` are exact up to floating point, `lowprec` is not, and the boundary between them is the point.

</div>

</section>

<section class="tx-node" data-cat="interaction" data-syms="g" markdown="1">

## `interaction` <span class="tx-changes">changes the interaction <i>g</i></span><span class="tx-n">5 methods</span> {: #interaction data-toc-label="interaction" }

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   The interaction <i>g</i>(<i>q</i>,&#8201;<i>x</i>) is replaced by a cheaper surrogate.

Preserves
:   The support, the values, and per-query normalisation — &#960;&#770;<sub><i>q</i></sub> still lives on all <i>n</i> positions.

Error term
:   **Distributional only.** <i>v</i>&#770; = <i>v</i>, so every bit of the error is in the measure. For random features it shrinks with the feature count; for a fixed feature map it does not shrink at all, because there is nothing it is converging to.

Methods
:   <ul class="op-methods"><li><a href="methods/cosformer/">cosformer</a></li> <li><a href="methods/linear_elu/">linear_elu</a></li> <li><a href="methods/nystrom/">nystrom</a></li> <li><a href="methods/performer/">performer</a></li> <li><a href="methods/trigrff/">trigrff</a></li></ul>

</div>

The interaction <i>g</i>(<i>q</i>,&#8201;<i>x</i>) is replaced by a cheaper surrogate <i>g&#770;</i>. Random features estimate exp&#8202;&#10216;<i>q</i>,&#8201;<i>k</i>&#10217;; fixed feature maps replace it outright. The first is an approximation whose error shrinks with the feature count; the second is a different operator that happens to be cheap. The `objective` column is where that difference is recorded.

</div>

</section>

<section class="tx-node" data-cat="restrict" data-syms="int" markdown="1">

## `restrict` <span class="tx-changes">changes the support of &#960;<sub><i>q</i></sub></span><span class="tx-n">6 methods</span> {: #restrict data-toc-label="restrict" }

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   The support of &#960;<sub><i>q</i></sub> is restricted to a subset <i>S</i><sub><i>q</i></sub> of positions.

Preserves
:   The values and the scores of everything retained. Nothing is discarded from the context, so a later query may select what this one dropped.

Error term
:   **Distributional only**, and in closed form: the omitted mass &#949;<sub><i>q</i></sub> times the contrast between the omitted and the retained conditional means. Minimising &#949;<sub><i>q</i></sub> is not the same as minimising the error, which is what `value_topk` exists to show.

Methods
:   <ul class="op-methods"><li><a href="methods/blocktopk/">blocktopk</a></li> <li><a href="methods/lsh_sample/">lsh_sample</a></li> <li><a href="methods/sink_window/">sink_window</a></li> <li><a href="methods/topk/">topk</a></li> <li><a href="methods/value_topk/" data-oracle="" title="an oracle: an upper bound, not a practical method">value_topk</a></li> <li><a href="methods/window/">window</a></li></ul>

</div>

The support of &#960;<sub><i>q</i></sub> shrinks to a subset <i>S</i><sub><i>q</i></sub>, and the retained mass is renormalised. The whole category is one operator — `prop:sparse-identity` — differing only in how <i>S</i><sub><i>q</i></sub> is chosen. Error is purely *distributional*: the values are untouched.

</div>

</section>

<section class="tx-node" data-cat="compress" data-syms="mu v" markdown="1">

## `compress` <span class="tx-changes">changes the stored context (&#956;,&#8201;<i>k</i>,&#8201;<i>v</i>)</span><span class="tx-n">8 methods</span> {: #compress data-toc-label="compress" }

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   The stored context (&#956;,&#8201;<i>k</i>,&#8201;<i>v</i>) is replaced by a smaller one.

Preserves
:   The operator and the per-query normalisation. What is no longer preserved is the context itself, which is the point.

Error term
:   **Both.** A distributional term through the keys and a value term through <i>v</i>&#770; &#8800; <i>v</i> — the only category that routinely produces the second. Key error passes through the softmax; value error is only reweighted, so keys cost more per bit.

Methods
:   <ul class="op-methods"><li><a href="methods/evict_h2o/">evict_h2o</a></li> <li><a href="methods/evict_snapkv/">evict_snapkv</a></li> <li><a href="methods/gqa/">gqa</a></li> <li><a href="methods/linformer/">linformer</a></li> <li><a href="methods/lowrank_kv/">lowrank_kv</a></li> <li><a href="methods/merge_kv/">merge_kv</a></li> <li><a href="methods/mla/">mla</a></li> <li><a href="methods/quantkv/">quantkv</a></li></ul>

</div>

The stored context (&#956;,&#8201;<i>k</i>,&#8201;<i>v</i>) is replaced by something smaller: quantised, low-rank, evicted, merged, or shared across heads. This is the only category that routinely produces *value* error, because it is the only one that changes what is aggregated.

</div>

</section>

<section class="tx-node" data-cat="factorize" data-syms="pi" markdown="1">

## `factorize` <span class="tx-changes">changes the shape of &#960;<sub><i>q</i></sub></span><span class="tx-n">4 methods</span> {: #factorize data-toc-label="factorize" }

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   &#960;<sub><i>q</i></sub> is factored through <i>r</i> latents: &#960;<sub><i>q</i></sub> = <i>a</i>(<i>q</i>,&#8201;&#183;)&#8202;<i>b</i>(&#183;,&#8201;<i>x</i>).

Preserves
:   Row-stochasticity, when both factors are Markov kernels — a composition of two of them is still a probability measure.

Error term
:   **Distributional, with a floor set by rank.** No rank-<i>r</i> factorisation beats the truncated SVD, and a nonnegative one cannot reach it. `svd_oracle` is that lower bound and `nmf_oracle` measures the gap.

Methods
:   <ul class="op-methods"><li><a href="methods/clustered/">clustered</a></li> <li><a href="methods/latent_stochastic/">latent_stochastic</a></li> <li><a href="methods/nmf_oracle/" data-oracle="" title="an oracle: an upper bound, not a practical method">nmf_oracle</a></li> <li><a href="methods/svd_oracle/" data-oracle="" title="an oracle: an upper bound, not a practical method">svd_oracle</a></li></ul>

</div>

&#960;<sub><i>q</i></sub> is routed through <i>r</i> latents, &#960;<sub><i>q</i></sub> = <i>a</i>(<i>q</i>,&#8201;&#183;)&#8202;<i>b</i>(&#183;,&#8201;<i>x</i>). Whether the factorisation can stay row-stochastic is the question: `latent_stochastic` is stochastic by construction, `svd_oracle` is the signed lower bound it cannot reach, and `nmf_oracle` measures the gap.

</div>

</section>

<section class="tx-node" data-cat="coupling" data-syms="Z" markdown="1">

## `coupling` <span class="tx-changes">changes the normaliser <i>Z</i></span><span class="tx-n">2 methods</span> {: #coupling data-toc-label="coupling" }

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   Normalisation stops being per-query; queries are coupled.

Preserves
:   The values, and &#960;&#770;<sub><i>q</i></sub> as a probability measure. What goes is the independence of the queries, and with it causal masking.

Error term
:   **Not an approximation error at all.** The result is exactly softmax tilted by one potential per key — a deterministic change of operator, measured against softmax only to say how far it has moved.

Methods
:   <ul class="op-methods"><li><a href="methods/lowrank_ot/">lowrank_ot</a></li> <li><a href="methods/sinkhorn/">sinkhorn</a></li></ul>

</div>

Normalisation stops being per-query. Sinkhorn balances the whole matrix to prescribed marginals, which makes the result softmax *tilted by a key potential* rather than softmax. Queries are no longer independent, so causal masking is not available.

</div>

</section>

<section class="tx-node" data-cat="recurrent" data-syms="y int v pi g mu Z" data-replace markdown="1">

## `recurrent` <span class="tx-changes">changes the operator itself</span><span class="tx-n">3 methods</span> {: #recurrent data-toc-label="recurrent" }

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   The operator itself becomes a finite-state recurrence.

Preserves
:   Linearity in <i>v</i>. That is the whole reason the implied weights can be recovered at all, and why Invariant W still holds here.

Error term
:   **Not error but capacity.** A fixed-size state must answer every future query, which bounds recall however the model is trained. Fidelity to softmax is not the goal, so quoting an output error against it answers a question nobody asked.

Methods
:   <ul class="op-methods"><li><a href="methods/decay_rnn/">decay_rnn</a></li> <li><a href="methods/delta_rnn/">delta_rnn</a></li> <li><a href="methods/linear_rnn/">linear_rnn</a></li></ul>

</div>

The operator becomes a finite-state recurrence: a fixed-size state <i>H</i><sub><i>t</i></sub> absorbs the context. Linear in the values, so Invariant W still holds and the implied weights can still be written down — they are simply not a probability measure.

</div>

</section>

</div>

</div>


## The second axis

Cutting across all seven is the question that decides how a method should be judged at
all:

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


## Every method, in one table

All 32 of them, grouped by the object they change. This is the same content as the branches above, flattened — regenerated from `ab.list_methods()`, and the view that works whatever your browser is doing.

<p class="tx-legend" markdown="1">**`declares`** &mdash; <abbr class="tx-dec" title="causal">c</abbr>&#8201;causal, <abbr class="tx-dec" title="names its weights on the original positions">w</abbr>&#8201;names its weights on the original positions, <abbr class="tx-dec" title="row-stochastic">s</abbr>&#8201;row-stochastic. A letter <span class="tx-struck">struck through</span> is the method declaring it does <em>not</em>.</p>

<div class="tx-all-wrap" markdown="1">

| method | what it is | category | objective | declares | backends |
|---|---|---|---|---|---|
| [`flash`](methods/flash.md) | Single-pass left fold of the online-softmax monoid over key blocks. | <span data-cat="exact">`exact`</span> | accelerate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference, sdpa |
| [`lowprec`](methods/lowprec.md) | Quantize--dequantize `q, k, v` (and optionally `P`) inside the kernel. | <span data-cat="exact">`exact`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`softmax`](methods/softmax.md) | Dense softmax attention (Vaswani et al., 2017); the reference operator. | <span data-cat="exact">`exact`</span> | accelerate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference, sdpa, flash_attn |
| [`splitk`](methods/splitk.md) | Two-level reduction: per-chunk states, then one merge. | <span data-cat="exact">`exact`</span> | accelerate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`cosformer`](methods/cosformer.md) | `relu` features times `cos(pi (i - j) / 2M)`. | <span data-cat="interaction">`interaction`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`linear_elu`](methods/linear_elu.md) | `phi(u) = elu(u) + 1`, normalized separable attention. | <span data-cat="interaction">`interaction`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`nystrom`](methods/nystrom.md) | Landmark Nystrom factorization of the softmax matrix; signed weights. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`performer`](methods/performer.md) | FAVOR+ positive random features, optionally orthogonal. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`trigrff`](methods/trigrff.md) | Rahimi--Recht trigonometric random features; signed, unnormalized measure. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`blocktopk`](methods/blocktopk.md) | Page-level selection by an upper bound on the page's scores. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`lsh_sample`](methods/lsh_sample.md) | Self-normalized importance sampling over a score-monotone proposal. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`sink_window`](methods/sink_window.md) | `s` initial sinks plus a left window of width `w`. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`topk`](methods/topk.md) | Exact top-k selection on the true scores (an oracle for proxy selectors). | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`value_topk`](methods/value_topk.md) **(oracle)** | Greedy (or exhaustive) output-aware support selection. Oracle. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`window`](methods/window.md) | Left sliding window of width `w`, always including the diagonal. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`evict_h2o`](methods/evict_h2o.md) | Online heavy-hitter eviction with a protected recent window. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`evict_snapkv`](methods/evict_snapkv.md) | Observation-window scoring with pooled, budgeted selection. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`gqa`](methods/gqa.md) | Mean-pool K and V within groups of query heads. | <span data-cat="compress">`compress`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference, sdpa |
| [`linformer`](methods/linformer.md) | Random (or identity) sequence-axis projections of K and V. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`lowrank_kv`](methods/lowrank_kv.md) | Truncated SVD of the key and/or value channel matrices. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`merge_kv`](methods/merge_kv.md) | Cluster keys and values, merge with proportional (pushforward) mass. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`mla`](methods/mla.md) | Joint low-rank KV latent shared by all heads. | <span data-cat="compress">`compress`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference, sdpa |
| [`quantkv`](methods/quantkv.md) | Uniform affine quantization of the key and value caches. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`clustered`](methods/clustered.md) | One attention row per query centroid, shared within the cluster. | <span data-cat="factorize">`factorize`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`latent_stochastic`](methods/latent_stochastic.md) | pi_hat = A B with both factors row-stochastic (two-hop attention). | <span data-cat="factorize">`factorize`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`nmf_oracle`](methods/nmf_oracle.md) **(oracle)** | Best nonnegative rank-r factorization of P, renormalized. Oracle. | <span data-cat="factorize">`factorize`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`svd_oracle`](methods/svd_oracle.md) **(oracle)** | Truncated SVD of P: the best signed rank-r factorization. Oracle. | <span data-cat="factorize">`factorize`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`lowrank_ot`](methods/lowrank_ot.md) | Doubly stochastic, rank-<= r attention through a pivot measure. | <span data-cat="coupling">`coupling`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`sinkhorn`](methods/sinkhorn.md) | Entropic-OT attention: Sinkhorn normalization of exp(s) to prescribed marginals. | <span data-cat="coupling">`coupling`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`decay_rnn`](methods/decay_rnn.md) | Fixed exponential decay on the recurrent state. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`delta_rnn`](methods/delta_rnn.md) | Delta-rule (Widrow--Hoff) fast-weight memory. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`linear_rnn`](methods/linear_rnn.md) | Additive linear recurrence with an unnormalized readout. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |

</div>


## Reading the columns

**`declares`** is three fixed slots, in this order, and a letter
<span class="tx-struck">struck through</span> is the method declaring *no* — which on
this page is never an omission and always a claim.

**<span class="tx-dec">w</span>, weights** — can the method name its measure
$\widehat\pi_q$ on the *original* $n$ positions? A struck-through `w` means it
genuinely cannot, and the method is then excluded from the error-decomposition studies
**by declaration, not by accident**. See [the invariant](invariant.md).

**<span class="tx-dec">s</span>, row-stochastic** — are those weights nonnegative and
summing to one? `nystrom`, `linformer` and the unnormalised recurrences say no, and
say it in an attribute rather than in a footnote.

**<span class="tx-dec">c</span>, causal** — a struck-through `c` is a statement about
the operator. Sinkhorn normalises across queries; there is no causal version of that,
and asking for one raises an error that says so.

**`backends`** — the execution paths this method claims are equal to the reference.
Most of the bank is `reference` only, which is the correct answer for an operator that
is not a dense softmax over a mask. See [Backends](backends.md).

## Oracles

Three methods — [`value_topk`](methods/value_topk.md), [`nmf_oracle`](methods/nmf_oracle.md),
[`svd_oracle`](methods/svd_oracle.md) — are not practical. They are
**upper bounds on what a mechanism can achieve**, which is exactly what several of the
monograph's arguments need: you cannot show that top-$k$ selection is suboptimal
without computing what the optimal selection would have been.

!!! info "This page is generated"
    Everything but the category prose comes from the live registry, so it cannot drift
    from the code. Regenerate with `python docs/gen_taxonomy.py`; `ab.list_methods()`
    prints the same rows in a terminal. To search and filter the whole bank at once —
    by category, objective, causal support, backend — use
    [**Browse the bank**](methods/index.md).
