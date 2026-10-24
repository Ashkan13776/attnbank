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

Every method in the bank hangs off one of the seven branches below. Open a branch to
see what that category changes, what it leaves alone, and which term of the error it
incurs — and which symbol of the operator it reaches for.
</div>


<div class="op" data-op markdown="1">

<div class="op-plate tx-sticky" data-plate="the operator">
--8<-- "docs/includes/operator.html"
</div>

<div class="tx-tree" markdown="1">

<p class="tx-key">Each leaf is a method. Its glyph is the <code>objective</code> it declares: <span class="tx-g">=</span>&#8201;accelerate, the same operator &#183; <span class="tx-g">&#8776;</span>&#8201;approximate, judged by output error &#183; <span class="tx-g">&#8800;</span>&#8201;modify, a different operator. A <span class="tx-g-oracle">dashed leaf</span> is an oracle bound, not a practical method.</p>

<section class="tx-node" data-cat="exact" data-syms="" data-frame markdown="1">

## `exact` <span class="tx-changes">changes nothing but the order of the arithmetic</span><span class="tx-n">4 methods</span> {: #exact data-toc-label="exact" }

<ul class="tx-leaves"><li><a href="methods/flash/" data-obj="accelerate" title="Single-pass left fold of the online-softmax monoid over key blocks.">flash</a></li><li><a href="methods/lowprec/" data-obj="approximate" title="Quantize--dequantize `q, k, v` (and optionally `P`) inside the kernel.">lowprec</a></li><li><a href="methods/softmax/" data-obj="accelerate" title="Dense softmax attention (Vaswani et al., 2017); the reference operator.">softmax</a></li><li><a href="methods/splitk/" data-obj="accelerate" title="Two-level reduction: per-chunk states, then one merge.">splitk</a></li></ul>

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   Nothing changes but the evaluation: tiling, chunking, precision.

Preserves
:   Everything. The measure, the values and the normaliser are the reference operator's, bit for bit where the arithmetic allows it.

Error term
:   **None**, up to floating point — which is a claim the conformance suite checks rather than a promise. `lowprec` is the exception, and locating the exception is the purpose of the category.

</div>

Nothing in the formula changes — only how it is evaluated. Tiling, chunking, reduced precision. The interesting question in this category is *where exactness stops*: `flash` and `splitk` are exact up to floating point, `lowprec` is not, and the boundary between them is the point.

</div>

</section>

<section class="tx-node" data-cat="interaction" data-syms="g" markdown="1">

## `interaction` <span class="tx-changes">changes the interaction <i>g</i></span><span class="tx-n">14 methods</span> {: #interaction data-toc-label="interaction" }

<ul class="tx-leaves"><li><a href="methods/based/" data-obj="approximate" title="Second-order (by default) Taylor polynomial of `exp`, normalized.">based</a></li><li><a href="methods/cosformer/" data-obj="modify" title="`relu` features times `cos(pi (i - j) / 2M)`.">cosformer</a></li><li><a href="methods/favorpp/" data-obj="approximate" title="Optimal positive random features (OPRF); FAVOR+ is the case `a = 0`.">favorpp</a></li><li><a href="methods/favorsharp/" data-obj="approximate" title="Dense-exponential random features: FAVOR++ per eigendirection of the pair covariance.">favorsharp</a></li><li><a href="methods/hedgehog/" data-obj="approximate" title="Separable feature map distilled against the softmax weights it replaces.">hedgehog</a></li><li><a href="methods/linear_elu/" data-obj="modify" title="`phi(u) = elu(u) + 1`, normalized separable attention.">linear_elu</a></li><li><a href="methods/lowrank_logit/" data-obj="approximate" data-oracle="" title="Truncated SVD of the logit matrix; the softmax of a rank-r interaction. &mdash; an oracle: an upper bound, not a practical method">lowrank_logit</a></li><li><a href="methods/nystrom/" data-obj="approximate" title="Landmark Nystrom factorization of the softmax matrix; signed weights.">nystrom</a></li><li><a href="methods/performer/" data-obj="approximate" title="FAVOR+ positive random features, optionally orthogonal.">performer</a></li><li><a href="methods/polaformer/" data-obj="modify" title="Same-signed and opposite-signed interactions, each normalized, on split values.">polaformer</a></li><li><a href="methods/polysketch/" data-obj="approximate" title="TensorSketch of the degree-P Taylor feature map; `sketch=None` is the control.">polysketch</a></li><li><a href="methods/rebased/" data-obj="modify" title="`phi(x) = (gamma . LayerNorm(x) + beta)^2`, normalized separable attention.">rebased</a></li><li><a href="methods/skyformer/" data-obj="approximate" title="Gaussian-kernel attention, lifted to a PSD Gram matrix, then Nystrom.">skyformer</a></li><li><a href="methods/trigrff/" data-obj="approximate" title="Rahimi--Recht trigonometric random features; signed, unnormalized measure.">trigrff</a></li></ul>

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   The interaction <i>g</i>(<i>q</i>,&#8201;<i>x</i>) is replaced by a cheaper surrogate.

Preserves
:   The support, the values, and per-query normalisation — &#960;&#770;<sub><i>q</i></sub> still lives on all <i>n</i> positions.

Error term
:   **Distributional only.** <i>v</i>&#770; = <i>v</i>, so every bit of the error is in the measure. For random features it shrinks with the feature count; for a fixed feature map it does not shrink at all, because there is nothing it is converging to.

</div>

The interaction <i>g</i>(<i>q</i>,&#8201;<i>x</i>) is replaced by a cheaper surrogate <i>g&#770;</i>. Random features estimate exp&#8202;&#10216;<i>q</i>,&#8201;<i>k</i>&#10217;; fixed feature maps replace it outright. The first is an approximation whose error shrinks with the feature count; the second is a different operator that happens to be cheap. The `objective` column is where that difference is recorded.

</div>

</section>

<section class="tx-node" data-cat="restrict" data-syms="int" markdown="1">

## `restrict` <span class="tx-changes">changes the support of &#960;<sub><i>q</i></sub></span><span class="tx-n">15 methods</span> {: #restrict data-toc-label="restrict" }

<ul class="tx-leaves"><li><a href="methods/bigbird/" data-obj="approximate" title="Block-sparse window + global + random support, with a seeded random draw.">bigbird</a></li><li><a href="methods/blocktopk/" data-obj="approximate" title="Page-level selection by an upper bound on the page&#x27;s scores.">blocktopk</a></li><li><a href="methods/hyperattention/" data-obj="approximate" title="Exact on a hashed block, unbiased-with-reweighting on a uniform sample of the rest.">hyperattention</a></li><li><a href="methods/longformer/" data-obj="approximate" title="Dilated sliding window of size w (w//2 positions per side), plus global tokens.">longformer</a></li><li><a href="methods/lsh_sample/" data-obj="approximate" title="Self-normalized importance sampling over a score-monotone proposal.">lsh_sample</a></li><li><a href="methods/oval/" data-obj="approximate" title="Page selection by a rank-r basis built from the joint key-value structure.">oval</a></li><li><a href="methods/page_pca/" data-obj="approximate" title="Page selection by a rank-r PCA basis of the page&#x27;s centered keys.">page_pca</a></li><li><a href="methods/reformer/" data-obj="approximate" title="Support = the keys that share an LSH bucket with the query in some round.">reformer</a></li><li><a href="methods/routing/" data-obj="approximate" title="Support = the keys routed to the query&#x27;s centroid. Centroids are trained.">routing</a></li><li><a href="methods/shifted_window/" data-obj="approximate" title="Non-overlapping local windows of w positions, displaced by shift (Swin).">shifted_window</a></li><li><a href="methods/sink_window/" data-obj="approximate" title="`s` initial sinks plus a left window of width `w`.">sink_window</a></li><li><a href="methods/strided/" data-obj="approximate" title="The union of the two factorized index sets of a Sparse Transformer head.">strided</a></li><li><a href="methods/topk/" data-obj="approximate" data-oracle="" title="Exact top-k selection on the true scores (an oracle for proxy selectors). &mdash; an oracle: an upper bound, not a practical method">topk</a></li><li><a href="methods/value_topk/" data-obj="approximate" data-oracle="" title="Greedy (or exhaustive) output-aware support selection. Oracle. &mdash; an oracle: an upper bound, not a practical method">value_topk</a></li><li><a href="methods/window/" data-obj="approximate" title="Left sliding window of width `w`, always including the diagonal.">window</a></li></ul>

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   The support of &#960;<sub><i>q</i></sub> is restricted to a subset <i>S</i><sub><i>q</i></sub> of positions.

Preserves
:   The values and the scores of everything retained. Nothing is discarded from the context, so a later query may select what this one dropped.

Error term
:   **Distributional only**, and in closed form: the omitted mass &#949;<sub><i>q</i></sub> times the contrast between the omitted and the retained conditional means. Minimising &#949;<sub><i>q</i></sub> is not the same as minimising the error, which is what `value_topk` exists to show.

</div>

The support of &#960;<sub><i>q</i></sub> shrinks to a subset <i>S</i><sub><i>q</i></sub>, and the retained mass is renormalised. The whole category is one operator — `prop:sparse-identity` — differing only in how <i>S</i><sub><i>q</i></sub> is chosen. Error is purely *distributional*: the values are untouched.

</div>

</section>

<section class="tx-node" data-cat="compress" data-syms="mu v" markdown="1">

## `compress` <span class="tx-changes">changes the stored context (&#956;,&#8201;<i>k</i>,&#8201;<i>v</i>)</span><span class="tx-n">18 methods</span> {: #compress data-toc-label="compress" }

<ul class="tx-leaves"><li><a href="methods/atom/" data-obj="approximate" title="Energy-reordered mixed-precision quantization of the key and value caches.">atom</a></li><li><a href="methods/evict_adakv/" data-obj="approximate" title="Head-wise adaptive budget allocation over SnapKV-style observation scores.">evict_adakv</a></li><li><a href="methods/evict_h2o/" data-obj="approximate" title="Online heavy-hitter eviction with a protected recent window.">evict_h2o</a></li><li><a href="methods/evict_pyramidkv/" data-obj="approximate" title="Layer-wise budget allocation on top of SnapKV&#x27;s pooled observation-window scores.">evict_pyramidkv</a></li><li><a href="methods/evict_snapkv/" data-obj="approximate" title="Observation-window scoring with pooled, budgeted selection.">evict_snapkv</a></li><li><a href="methods/evict_tova/" data-obj="approximate" title="Greedy one-token-per-step eviction scored by the current query alone.">evict_tova</a></li><li><a href="methods/gqa/" data-obj="modify" title="Mean-pool K and V within groups of query heads.">gqa</a></li><li><a href="methods/kivi/" data-obj="approximate" title="Per-channel key / per-token value quantization with a full-precision residual.">kivi</a></li><li><a href="methods/kvquant/" data-obj="approximate" title="Pre-RoPE per-channel keys, non-uniform sensitivity-weighted levels, sparse outliers.">kvquant</a></li><li><a href="methods/linformer/" data-obj="approximate" title="Random (or identity) sequence-axis projections of K and V.">linformer</a></li><li><a href="methods/lowrank_kv/" data-obj="approximate" title="Truncated SVD of the key and/or value channel matrices.">lowrank_kv</a></li><li><a href="methods/merge_cam/" data-obj="approximate" title="Streaming eviction whose departing values are merged forward, not discarded.">merge_cam</a></li><li><a href="methods/merge_kv/" data-obj="approximate" title="Cluster keys and values, merge with proportional (pushforward) mass.">merge_kv</a></li><li><a href="methods/merge_tome/" data-obj="approximate" title="Token merging by bipartite soft matching on the keys, with proportional mass.">merge_tome</a></li><li><a href="methods/mla/" data-obj="modify" title="Joint low-rank KV latent shared by all heads.">mla</a></li><li><a href="methods/qjl/" data-obj="approximate" title="One-bit JL sketch of the keys; per-token affine quantization of the values.">qjl</a></li><li><a href="methods/quantkv/" data-obj="approximate" title="Uniform affine quantization of the key and value caches.">quantkv</a></li><li><a href="methods/zipcache/" data-obj="approximate" title="Mixed-precision KV quantization with salient tokens chosen by normalised score.">zipcache</a></li></ul>

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   The stored context (&#956;,&#8201;<i>k</i>,&#8201;<i>v</i>) is replaced by a smaller one.

Preserves
:   The operator and the per-query normalisation. What is no longer preserved is the context itself, which is the point.

Error term
:   **Both.** A distributional term through the keys and a value term through <i>v</i>&#770; &#8800; <i>v</i> — the only category that routinely produces the second. Key error passes through the softmax; value error is only reweighted, so keys cost more per bit.

</div>

The stored context (&#956;,&#8201;<i>k</i>,&#8201;<i>v</i>) is replaced by something smaller: quantised, low-rank, evicted, merged, or shared across heads. This is the only category that routinely produces *value* error, because it is the only one that changes what is aggregated.

</div>

</section>

<section class="tx-node" data-cat="factorize" data-syms="pi" markdown="1">

## `factorize` <span class="tx-changes">changes the shape of &#960;<sub><i>q</i></sub></span><span class="tx-n">8 methods</span> {: #factorize data-toc-label="factorize" }

<ul class="tx-leaves"><li><a href="methods/abc/" data-obj="approximate" title="Bounded-memory control: pi_hat = a b with b prefix-normalized, so it decodes.">abc</a></li><li><a href="methods/clustered/" data-obj="approximate" title="One attention row per query centroid, shared within the cluster.">clustered</a></li><li><a href="methods/isab/" data-obj="modify" title="Two MABs through r inducing points; the block between them is nonlinear.">isab</a></li><li><a href="methods/latent_stochastic/" data-obj="modify" title="pi_hat = A B with both factors row-stochastic (two-hop attention).">latent_stochastic</a></li><li><a href="methods/latte/" data-obj="modify" title="pi_hat = A B with A = softmax over latents, B = softmax over positions.">latte</a></li><li><a href="methods/nmf_oracle/" data-obj="approximate" data-oracle="" title="Best nonnegative rank-r factorization of P, renormalized. Oracle. &mdash; an oracle: an upper bound, not a practical method">nmf_oracle</a></li><li><a href="methods/perceiver/" data-obj="modify" title="Cross-attend into r latents, process them nonlinearly, cross-attend out.">perceiver</a></li><li><a href="methods/svd_oracle/" data-obj="approximate" data-oracle="" title="Truncated SVD of P: the best signed rank-r factorization. Oracle. &mdash; an oracle: an upper bound, not a practical method">svd_oracle</a></li></ul>

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   &#960;<sub><i>q</i></sub> is factored through <i>r</i> latents: &#960;<sub><i>q</i></sub> = <i>a</i>(<i>q</i>,&#8201;&#183;)&#8202;<i>b</i>(&#183;,&#8201;<i>x</i>).

Preserves
:   Row-stochasticity, when both factors are Markov kernels — a composition of two of them is still a probability measure.

Error term
:   **Distributional, with a floor set by rank.** No rank-<i>r</i> factorisation beats the truncated SVD, and a nonnegative one cannot reach it. `svd_oracle` is that lower bound and `nmf_oracle` measures the gap.

</div>

&#960;<sub><i>q</i></sub> is routed through <i>r</i> latents, &#960;<sub><i>q</i></sub> = <i>a</i>(<i>q</i>,&#8201;&#183;)&#8202;<i>b</i>(&#183;,&#8201;<i>x</i>). Whether the factorisation can stay row-stochastic is the question: `latent_stochastic` is stochastic by construction, `svd_oracle` is the signed lower bound it cannot reach, and `nmf_oracle` measures the gap.

</div>

</section>

<section class="tx-node" data-cat="coupling" data-syms="Z" markdown="1">

## `coupling` <span class="tx-changes">changes the normaliser <i>Z</i></span><span class="tx-n">6 methods</span> {: #coupling data-toc-label="coupling" }

<ul class="tx-leaves"><li><a href="methods/dustbin_sinkhorn/" data-obj="modify" title="Sinkhorn with a dustbin row and column: a coupling that allows abstention.">dustbin_sinkhorn</a></li><li><a href="methods/espformer/" data-obj="modify" title="Doubly stochastic attention in one pass: the expectation of sliced OT plans.">espformer</a></li><li><a href="methods/lowrank_ot/" data-obj="modify" title="Doubly stochastic, rank-&lt;= r attention through a pivot measure.">lowrank_ot</a></li><li><a href="methods/otke/" data-obj="modify" title="Sinkhorn transport of the keys onto a trainable reference of p points.">otke</a></li><li><a href="methods/sinkhorn/" data-obj="modify" title="Entropic-OT attention: Sinkhorn normalization of exp(s) to prescribed marginals.">sinkhorn</a></li><li><a href="methods/unbalanced_sinkhorn/" data-obj="modify" title="Entropic OT attention with a KL-relaxed column marginal: softmax at `lam_col=0`.">unbalanced_sinkhorn</a></li></ul>

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   Normalisation stops being per-query; queries are coupled.

Preserves
:   The values, and &#960;&#770;<sub><i>q</i></sub> as a probability measure. What goes is the independence of the queries, and with it causal masking.

Error term
:   **Not an approximation error at all.** The result is exactly softmax tilted by one potential per key — a deterministic change of operator, measured against softmax only to say how far it has moved.

</div>

Normalisation stops being per-query. Sinkhorn balances the whole matrix to prescribed marginals, which makes the result softmax *tilted by a key potential* rather than softmax. Queries are no longer independent, so causal masking is not available.

</div>

</section>

<section class="tx-node" data-cat="recurrent" data-syms="y int v pi g mu Z" data-replace markdown="1">

## `recurrent` <span class="tx-changes">changes the operator itself</span><span class="tx-n">13 methods</span> {: #recurrent data-toc-label="recurrent" }

<ul class="tx-leaves"><li><a href="methods/decay_rnn/" data-obj="modify" title="Fixed exponential decay on the recurrent state.">decay_rnn</a></li><li><a href="methods/delta_rnn/" data-obj="modify" title="Delta-rule (Widrow--Hoff) fast-weight memory.">delta_rnn</a></li><li><a href="methods/gla/" data-obj="modify" title="GLA: `S_t = Diag(alpha_t) S_{t-1} + k_t v_t^T` with a data-dependent gate.">gla</a></li><li><a href="methods/hgrn2/" data-obj="modify" title="HGRN2: outer-product state expansion with the input gate tied to `1 - f_t`.">hgrn2</a></li><li><a href="methods/hyena/" data-obj="modify" title="Hyena: order-N implicit long convolutions interleaved with elementwise gating.">hyena</a></li><li><a href="methods/linear_rnn/" data-obj="modify" title="Additive linear recurrence with an unnormalized readout.">linear_rnn</a></li><li><a href="methods/lru/" data-obj="modify" title="Linear Recurrent Unit: a stably parametrised diagonal linear RNN.">lru</a></li><li><a href="methods/mega/" data-obj="modify" title="Mega: multi-dimensional damped EMA on q and k, then chunk-local softmax attention.">mega</a></li><li><a href="methods/mlstm/" data-obj="modify" title="mLSTM: exponential input gate, scalar forget gate, matrix memory + normaliser.">mlstm</a></li><li><a href="methods/retnet/" data-obj="modify" title="Multi-scale retention: a fixed per-head exponential decay on an r x d_v state.">retnet</a></li><li><a href="methods/rwkv/" data-obj="modify" title="RWKV-5/6 time mixing: diagonal decay, matrix state, bonus on the current token.">rwkv</a></li><li><a href="methods/s4d/" data-obj="modify" title="Diagonal state-space layer (S4D): one SISO SSM per value channel.">s4d</a></li><li><a href="methods/s5/" data-obj="modify" title="Simplified state-space layer (S5): one diagonal MIMO SSM, swept by a parallel scan.">s5</a></li></ul>

<div class="tx-body" markdown="1">

<div class="tx-facts" markdown="1">
Changes
:   The operator itself becomes a finite-state recurrence.

Preserves
:   Linearity in <i>v</i>. That is the whole reason the implied weights can be recovered at all, and why Invariant W still holds here.

Error term
:   **Not error but capacity.** A fixed-size state must answer every future query, which bounds recall however the model is trained. Fidelity to softmax is not the goal, so quoting an output error against it answers a question nobody asked.

</div>

The operator becomes a finite-state recurrence: a fixed-size state <i>H</i><sub><i>t</i></sub> absorbs the context. Linear in the values, so Invariant W still holds and the implied weights can still be written down — they are simply not a probability measure.

</div>

</section>

</div>

</div>


## The second axis

Cutting across all seven is the question that decides how a method should be judged at
all:

| | `objective` | meaning |
|---|---|---|
| <span class="tx-g">=</span> | `accelerate` | the same operator, exact up to floating point |
| <span class="tx-g">&#8776;</span> | `approximate` | targets dense softmax; judged by output error |
| <span class="tx-g">&#8800;</span> | `modify` | a *different operator*; fidelity to softmax is not the goal |

The glyph on every leaf of the tree above is this declaration.

Conflating `approximate` and `modify` is the most persistent confusion in this
literature. Asking how closely `linear_elu` approximates softmax is asking the wrong
question — it is not trying to. Every method here declares which it is, and the
conformance suite uses the declaration: an `approximate` method must supply a
correctness anchor, a `modify` method is exempt and the suite *asserts the exemption*
rather than skipping it.


## Every method, in one table

All 78 of them, grouped by the object they change. This is the same content as the branches above, flattened — regenerated from `ab.list_methods()`, and the view that works whatever your browser is doing.

<p class="tx-legend" markdown="1">**`declares`** &mdash; <abbr class="tx-dec" title="causal">c</abbr>&#8201;causal, <abbr class="tx-dec" title="names its weights on the original positions">w</abbr>&#8201;names its weights on the original positions, <abbr class="tx-dec" title="row-stochastic">s</abbr>&#8201;row-stochastic. A letter <span class="tx-struck">struck through</span> is the method declaring it does <em>not</em>.</p>

<!-- Not indexed: the same 78 methods are indexed on their own
     pages, and a second copy here matched nearly every query. -->

<div class="tx-all-wrap" data-search-exclude markdown="1">

| method | what it is | category | objective | declares | backends |
|---|---|---|---|---|---|
| [`flash`](methods/flash.md) | Single-pass left fold of the online-softmax monoid over key blocks. | <span data-cat="exact">`exact`</span> | accelerate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference, sdpa |
| [`lowprec`](methods/lowprec.md) | Quantize--dequantize `q, k, v` (and optionally `P`) inside the kernel. | <span data-cat="exact">`exact`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`softmax`](methods/softmax.md) | Dense softmax attention (Vaswani et al., 2017); the reference operator. | <span data-cat="exact">`exact`</span> | accelerate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference, sdpa, flash_attn |
| [`splitk`](methods/splitk.md) | Two-level reduction: per-chunk states, then one merge. | <span data-cat="exact">`exact`</span> | accelerate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`based`](methods/based.md) | Second-order (by default) Taylor polynomial of `exp`, normalized. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`cosformer`](methods/cosformer.md) | `relu` features times `cos(pi (i - j) / 2M)`. | <span data-cat="interaction">`interaction`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`favorpp`](methods/favorpp.md) | Optimal positive random features (OPRF); FAVOR+ is the case `a = 0`. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`favorsharp`](methods/favorsharp.md) | Dense-exponential random features: FAVOR++ per eigendirection of the pair covariance. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`hedgehog`](methods/hedgehog.md) | Separable feature map distilled against the softmax weights it replaces. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`linear_elu`](methods/linear_elu.md) | `phi(u) = elu(u) + 1`, normalized separable attention. | <span data-cat="interaction">`interaction`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`lowrank_logit`](methods/lowrank_logit.md) **(oracle)** | Truncated SVD of the logit matrix; the softmax of a rank-r interaction. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`nystrom`](methods/nystrom.md) | Landmark Nystrom factorization of the softmax matrix; signed weights. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`performer`](methods/performer.md) | FAVOR+ positive random features, optionally orthogonal. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`polaformer`](methods/polaformer.md) | Same-signed and opposite-signed interactions, each normalized, on split values. | <span data-cat="interaction">`interaction`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" data-no title="cannot name its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`polysketch`](methods/polysketch.md) | TensorSketch of the degree-P Taylor feature map; `sketch=None` is the control. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`rebased`](methods/rebased.md) | `phi(x) = (gamma . LayerNorm(x) + beta)^2`, normalized separable attention. | <span data-cat="interaction">`interaction`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`skyformer`](methods/skyformer.md) | Gaussian-kernel attention, lifted to a PSD Gram matrix, then Nystrom. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`trigrff`](methods/trigrff.md) | Rahimi--Recht trigonometric random features; signed, unnormalized measure. | <span data-cat="interaction">`interaction`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`bigbird`](methods/bigbird.md) | Block-sparse window + global + random support, with a seeded random draw. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`blocktopk`](methods/blocktopk.md) | Page-level selection by an upper bound on the page's scores. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`hyperattention`](methods/hyperattention.md) | Exact on a hashed block, unbiased-with-reweighting on a uniform sample of the rest. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`longformer`](methods/longformer.md) | Dilated sliding window of size w (w//2 positions per side), plus global tokens. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`lsh_sample`](methods/lsh_sample.md) | Self-normalized importance sampling over a score-monotone proposal. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`oval`](methods/oval.md) | Page selection by a rank-r basis built from the joint key-value structure. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`page_pca`](methods/page_pca.md) | Page selection by a rank-r PCA basis of the page's centered keys. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`reformer`](methods/reformer.md) | Support = the keys that share an LSH bucket with the query in some round. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`routing`](methods/routing.md) | Support = the keys routed to the query's centroid. Centroids are trained. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`shifted_window`](methods/shifted_window.md) | Non-overlapping local windows of w positions, displaced by shift (Swin). | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`sink_window`](methods/sink_window.md) | `s` initial sinks plus a left window of width `w`. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`strided`](methods/strided.md) | The union of the two factorized index sets of a Sparse Transformer head. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`topk`](methods/topk.md) **(oracle)** | Exact top-k selection on the true scores (an oracle for proxy selectors). | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`value_topk`](methods/value_topk.md) **(oracle)** | Greedy (or exhaustive) output-aware support selection. Oracle. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`window`](methods/window.md) | Left sliding window of width `w`, always including the diagonal. | <span data-cat="restrict">`restrict`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`atom`](methods/atom.md) | Energy-reordered mixed-precision quantization of the key and value caches. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`evict_adakv`](methods/evict_adakv.md) | Head-wise adaptive budget allocation over SnapKV-style observation scores. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`evict_h2o`](methods/evict_h2o.md) | Online heavy-hitter eviction with a protected recent window. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`evict_pyramidkv`](methods/evict_pyramidkv.md) | Layer-wise budget allocation on top of SnapKV's pooled observation-window scores. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`evict_snapkv`](methods/evict_snapkv.md) | Observation-window scoring with pooled, budgeted selection. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`evict_tova`](methods/evict_tova.md) | Greedy one-token-per-step eviction scored by the current query alone. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`gqa`](methods/gqa.md) | Mean-pool K and V within groups of query heads. | <span data-cat="compress">`compress`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference, sdpa |
| [`kivi`](methods/kivi.md) | Per-channel key / per-token value quantization with a full-precision residual. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`kvquant`](methods/kvquant.md) | Pre-RoPE per-channel keys, non-uniform sensitivity-weighted levels, sparse outliers. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`linformer`](methods/linformer.md) | Random (or identity) sequence-axis projections of K and V. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`lowrank_kv`](methods/lowrank_kv.md) | Truncated SVD of the key and/or value channel matrices. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`merge_cam`](methods/merge_cam.md) | Streaming eviction whose departing values are merged forward, not discarded. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`merge_kv`](methods/merge_kv.md) | Cluster keys and values, merge with proportional (pushforward) mass. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`merge_tome`](methods/merge_tome.md) | Token merging by bipartite soft matching on the keys, with proportional mass. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`mla`](methods/mla.md) | Joint low-rank KV latent shared by all heads. | <span data-cat="compress">`compress`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference, sdpa |
| [`qjl`](methods/qjl.md) | One-bit JL sketch of the keys; per-token affine quantization of the values. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`quantkv`](methods/quantkv.md) | Uniform affine quantization of the key and value caches. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`zipcache`](methods/zipcache.md) | Mixed-precision KV quantization with salient tokens chosen by normalised score. | <span data-cat="compress">`compress`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`abc`](methods/abc.md) | Bounded-memory control: pi_hat = a b with b prefix-normalized, so it decodes. | <span data-cat="factorize">`factorize`</span> | approximate | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`clustered`](methods/clustered.md) | One attention row per query centroid, shared within the cluster. | <span data-cat="factorize">`factorize`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`isab`](methods/isab.md) | Two MABs through r inducing points; the block between them is nonlinear. | <span data-cat="factorize">`factorize`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" data-no title="cannot name its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`latent_stochastic`](methods/latent_stochastic.md) | pi_hat = A B with both factors row-stochastic (two-hop attention). | <span data-cat="factorize">`factorize`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`latte`](methods/latte.md) | pi_hat = A B with A = softmax over latents, B = softmax over positions. | <span data-cat="factorize">`factorize`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`nmf_oracle`](methods/nmf_oracle.md) **(oracle)** | Best nonnegative rank-r factorization of P, renormalized. Oracle. | <span data-cat="factorize">`factorize`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`perceiver`](methods/perceiver.md) | Cross-attend into r latents, process them nonlinearly, cross-attend out. | <span data-cat="factorize">`factorize`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" data-no title="cannot name its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`svd_oracle`](methods/svd_oracle.md) **(oracle)** | Truncated SVD of P: the best signed rank-r factorization. Oracle. | <span data-cat="factorize">`factorize`</span> | approximate | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`dustbin_sinkhorn`](methods/dustbin_sinkhorn.md) | Sinkhorn with a dustbin row and column: a coupling that allows abstention. | <span data-cat="coupling">`coupling`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`espformer`](methods/espformer.md) | Doubly stochastic attention in one pass: the expectation of sliced OT plans. | <span data-cat="coupling">`coupling`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`lowrank_ot`](methods/lowrank_ot.md) | Doubly stochastic, rank-<= r attention through a pivot measure. | <span data-cat="coupling">`coupling`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`otke`](methods/otke.md) | Sinkhorn transport of the keys onto a trainable reference of p points. | <span data-cat="coupling">`coupling`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`sinkhorn`](methods/sinkhorn.md) | Entropic-OT attention: Sinkhorn normalization of exp(s) to prescribed marginals. | <span data-cat="coupling">`coupling`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`unbalanced_sinkhorn`](methods/unbalanced_sinkhorn.md) | Entropic OT attention with a KL-relaxed column marginal: softmax at `lam_col=0`. | <span data-cat="coupling">`coupling`</span> | modify | <abbr class="tx-dec" data-no title="not causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`decay_rnn`](methods/decay_rnn.md) | Fixed exponential decay on the recurrent state. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`delta_rnn`](methods/delta_rnn.md) | Delta-rule (Widrow--Hoff) fast-weight memory. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`gla`](methods/gla.md) | GLA: `S_t = Diag(alpha_t) S_{t-1} + k_t v_t^T` with a data-dependent gate. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`hgrn2`](methods/hgrn2.md) | HGRN2: outer-product state expansion with the input gate tied to `1 - f_t`. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`hyena`](methods/hyena.md) | Hyena: order-N implicit long convolutions interleaved with elementwise gating. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" data-no title="cannot name its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`linear_rnn`](methods/linear_rnn.md) | Additive linear recurrence with an unnormalized readout. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`lru`](methods/lru.md) | Linear Recurrent Unit: a stably parametrised diagonal linear RNN. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" data-no title="cannot name its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`mega`](methods/mega.md) | Mega: multi-dimensional damped EMA on q and k, then chunk-local softmax attention. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" title="row-stochastic">s</abbr> | reference |
| [`mlstm`](methods/mlstm.md) | mLSTM: exponential input gate, scalar forget gate, matrix memory + normaliser. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`retnet`](methods/retnet.md) | Multi-scale retention: a fixed per-head exponential decay on an r x d_v state. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`rwkv`](methods/rwkv.md) | RWKV-5/6 time mixing: diagonal decay, matrix state, bonus on the current token. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" title="names its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`s4d`](methods/s4d.md) | Diagonal state-space layer (S4D): one SISO SSM per value channel. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" data-no title="cannot name its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |
| [`s5`](methods/s5.md) | Simplified state-space layer (S5): one diagonal MIMO SSM, swept by a parallel scan. | <span data-cat="recurrent">`recurrent`</span> | modify | <abbr class="tx-dec" title="causal">c</abbr><abbr class="tx-dec" data-no title="cannot name its weights on the original positions">w</abbr><abbr class="tx-dec" data-no title="not row-stochastic">s</abbr> | reference |

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

{_oracle_sentence()} are not practical. They are
**upper bounds on what a mechanism can achieve**, which is exactly what several of the
monograph's arguments need: you cannot show that top-$k$ selection is suboptimal
without computing what the optimal selection would have been.

!!! info "This page is generated"
    Everything but the category prose comes from the live registry, so it cannot drift
    from the code. Regenerate with `python docs/gen_taxonomy.py`; `ab.list_methods()`
    prints the same rows in a terminal. To search and filter the whole bank at once —
    by category, objective, causal support, backend — use
    [**Browse the bank**](methods/index.md).
