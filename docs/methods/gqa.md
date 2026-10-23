# Grouped-query attention

<p class="tx-regkey"><code>gqa</code> &middot; <code>ab.build("gqa")</code></p>

**Grouped-query attention: K and V shared across groups of query heads.**

category
:   [`compress`](../taxonomy.md#compress) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.GQAAttention`, [`attnbank/methods/compress/gqa.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/gqa.py)

monograph
:   `sec:compress`

## What it changes

Changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

## In the paper's notation

`sec:compress-arch`, `sec:exact-roofline`.  Query heads are partitioned into groups
of `group_size`; the heads of a group share one key–value pair, here obtained by
mean-pooling the group's `K` and `V` (the conversion recipe of Ainslie et al.,
2023).  `group_size = 1` is multi-head attention, `group_size = h` is multi-query.

This is an **operator modification** (`tab:comparative`: "MQA / GQA / MLA ... mod."),
trained as such; measuring it against a multi-head reference answers a conversion
question, not an approximation question.  Its purpose is the roofline: by
`eq:exact-decode-intensity` the decode arithmetic intensity is `(16/b)(h/h_kv)`, so
sharing KV is one of the very few interventions that *raises* it.  `cost()` divides
`state_bytes` and the KV traffic by `group_size` accordingly.

**Heads are deliberately coupled**, so `head_independent = False`: perturbing head
`h` changes every head in its group and nothing outside it.  `tests/test_compress.py`
checks exactly that boundary.

Exact limit: `group_size = 1`.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **yes** | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **no** | it shares $K$/$V$ across query heads, so heads are not independent. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Configuration

| parameter | default |
|---|---|
| `group_size` | `2` |
| `sdpa_kernel` | `None` |

```python
ab.describe("gqa")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `group_size=1` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("gqa").limit_config
{'group_size': 1}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |
| `sdpa` | `torch.nn.functional.scaled_dot_product_attention` -- on CUDA, FlashAttention-2 or the memory-efficient kernel |

Each non-`reference` entry is a claim that the path is **equal** to the reference, and `tests/test_conformance.py::test_backend_equivalence` checks it on five shapes, both causal settings, float32 and float64. See [Backends](../backends.md).

Diagnostics returned in `extras`: `group_size`, `n_kv_heads`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    g = max(1, self.group_size)
    return {"flops": 2 * m * n * (d + dv),
            "state_bytes": n * (d + dv) * bytes_per_el / g,
            "bytes_moved": m * (d + dv) * bytes_per_el + n * (d + dv) * bytes_per_el / g}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,435,456 | 268,435,456 | 1× |
| `state_bytes` | 131,072 | 262,144 | 0.5× |
| `bytes_moved` | 393,216 | 4,718,592 | 0.0833× |

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

attn = ab.build("gqa", group_size=2)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = modify: this is a different operator and fidelity to softmax
# is not its goal, so there is no error to quote. What it *is* doing shows
# up in its own diagnostics.
print("row_stochastic =", attn.row_stochastic, "| first row sums:",
      [round(x, 4) for x in out.weights.sum(-1)[0, 0, :4].tolist()])
print("extras:", sorted(out.extras))
```

```text
GQAAttention(name='gqa', category='compress', objective='modify', backend='reference', group_size=2, sdpa_kernel=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[group_size, n_kv_heads], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['group_size', 'n_kv_heads']
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `gqa` removed.

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`atom`](atom.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`mla`](mla.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`linear_rnn`](linear_rnn.md), [`lowrank_ot`](lowrank_ot.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`otke`](otke.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

[Open the browser with `gqa` selected](index.md?compare=gqa) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **ainslie2023gqa (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="ainslie2023gqa"><code>ainslie2023gqa</code></span>
- **shazeer2019fast (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="shazeer2019fast"><code>shazeer2019fast</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="ainslie2023gqa">
<pre class="attn-bib__src"><code>@inproceedings{ainslie2023gqa,
  title = {{GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints}}

@inproceedings{aksenov2024rebased,
  title     = {Linear Transformers with Learnable Kernel Functions are Better In-Context Models}

@inproceedings{andoni2015practical,
  title = {Practical and Optimal {LSH} for Angular Distance}

@inproceedings{arora2024based,
  title     = {Simple linear attention language models balance the recall-throughput tradeoff}

@inproceedings{beck2024xlstm,
  title={{xLSTM}: Extended Long Short-Term Memory}

@misc{beltagy2020longformer,
  title = {{Longformer: The Long-Document Transformer}}

@inproceedings{bhojanapalli2020lowrank,
  title         = {Low-Rank Bottleneck in Multi-head Attention Models}

@techreport{blelloch1990prefix,
  title       = {Prefix Sums and Their Applications}

@inproceedings{bolya2023token,
  title = {{Token Merging: Your ViT But Faster}}

@misc{cai2024pyramidkv,
  title = {{PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling}}

@inproceedings{chang2025palu,
  title = {{Palu: Compressing KV-Cache with Low-Rank Projection}}

@inproceedings{chen2021skyformer,
  title     = {Skyformer: Remodel Self-Attention with {G}aussian Kernel and {N}ystr{\&quot;o}m Method}

@misc{chen2024intflashattention,
  title={{INT-FlashAttention}: Enabling Flash Attention for {INT8} Quantization}

@inproceedings{chen2025magicpig,
  title     = {{MagicPIG}: {LSH} Sampling for Efficient {LLM} Generation}

@misc{child2019generating,
  title = {{Generating Long Sequences with Sparse Transformers}}

@misc{chizat2016scaling,
  title={Scaling Algorithms for Unbalanced Transport Problems}

@inproceedings{choromanski2021rethinking,
  title     = {Rethinking Attention with Performers}

@article{cohen1993nonnegative,
  title   = {Nonnegative Ranks, Decompositions, and Factorizations of Nonnegative Matrices}

@inproceedings{cuturi2013sinkhorn,
  title={Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances}

@inproceedings{dao2022flashattention,
  title     = {{FlashAttention}: Fast and Memory-Efficient Exact Attention with {IO}-Awareness}

@misc{dao2023flash,
  title={Flash-Decoding for long-context inference}

@inproceedings{dao2024flashattention2,
  title={{FlashAttention-2}: Faster Attention with Better Parallelism and Work Partitioning}

@inproceedings{dao2024transformers,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality}

@misc{deepseekai2024deepseekv2,
  title = {{DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model}}

@misc{ding2023longnet,
  title = {{LongNet: Scaling Transformers to 1,000,000,000 Tokens}}

@misc{dolga2024latte,
  title         = {Latte: Latent Attention for Linear Time Transformers}

@article{drineas2005nystrom,
  title={On the {N}ystr{\&quot;o}m Method for Approximating a {G}ram Matrix for Improved Kernel-Based Learning}

@inproceedings{feng2025adakv,
  title = {{Ada-KV}: Optimizing {KV} Cache Eviction by Adaptive Budget Allocation for Efficient {LLM} Inference}

@misc{feng2025criticalkv,
  title = {{CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective}}

@inproceedings{forrow2019statistical,
  title         = {Statistical Optimal Transport via Factored Couplings}

@inproceedings{frogner2015learning,
  title={Learning with a {W}asserstein Loss}

@book{gillis2020nonnegative,
  title     = {Nonnegative Matrix Factorization}

@inproceedings{gu2020hippo,
  title     = {HiPPO: Recurrent Memory with Optimal Polynomial Projections}

@inproceedings{gu2022efficiently,
  title     = {Efficiently Modeling Long Sequences with Structured State Spaces}

@inproceedings{gu2022parameterization,
  title     = {On the Parameterization and Initialization of Diagonal State Space Models}

@inproceedings{gupta2021memory,
  title     = {Memory-efficient Transformers via Top-k Attention}

@inproceedings{gupta2022diagonal,
  title     = {Diagonal State Spaces are as Effective as Structured State Spaces}

@inproceedings{han2024hyperattention,
  title = {{HyperAttention: Long-context Attention in Near-Linear Time}}

@inproceedings{han2024lminfinite,
  title = {{LM-Infinite: Zero-Shot Extreme Length Generalization for Large Language Models}}

@inproceedings{he2024zipcache,
  title     = {{ZipCache}: Accurate and Efficient {KV} Cache Quantization with Salient Token Identification}

@inproceedings{hooper2024kvquant,
  title     = {{KVQuant}: Towards 10 Million Context Length {LLM} Inference with {KV} Cache Quantization}

@misc{hwang2026locks,
  title = {{LOCKS: Page-Local Compact Key Summaries for Efficient Long-Context Decoding}}

@inproceedings{jaegle2021perceiver,
  title         = {Perceiver: General Perception with Iterative Attention}

@inproceedings{jaegle2022perceiverio,
  title         = {Perceiver {IO}: A General Architecture for Structured Inputs \&amp; Outputs}

@misc{jiang2023mistral,
  title = {{Mistral 7B}}

@inproceedings{kacham2024polysketchformer,
  title={{PolySketchFormer}: Fast Transformers via Sketching Polynomial Kernels}

@inproceedings{katharopoulos2020transformers,
  title     = {Transformers are {RNN}s: Fast Autoregressive Transformers with Linear Attention}

@inproceedings{kitaev2020reformer,
  title={Reformer: The Efficient Transformer}

@article{lee1999learning,
  title   = {Learning the Parts of Objects by Non-negative Matrix Factorization}

@inproceedings{lee2019set,
  title         = {Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks}

@inproceedings{li2024snapkv,
  title={{SnapKV}: {LLM} Knows What You are Looking for Before Generation}

@article{liero2018optimal,
  title={Optimal Entropy-Transport problems and a new {H}ellinger--{K}antorovich distance between positive measures}

@inproceedings{likhosherstov2022chefs,
  title={Chefs' Random Tables: Non-Trigonometric Random Features}

@inproceedings{likhosherstov2023favorsharp,
  title     = {Dense-Exponential Random Features: Sharp Positive Estimators of the Gaussian Kernel}

@inproceedings{liu2021swin,
  title = {{Swin Transformer: Hierarchical Vision Transformer using Shifted Windows}}

@inproceedings{liu2023scissorhands,
  title     = {Scissorhands: Exploiting the Persistence of Importance Hypothesis for {LLM} {KV} Cache Compression at Test Time}

@inproceedings{liu2024intactkv,
  title = {{IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact}}

@inproceedings{liu2024kivi,
  title     = {{KIVI}: A Tuning-Free Asymmetric 2bit Quantization for {KV} Cache}

@inproceedings{liu2025expected,
  title={Expected Sliced Transport Plans}

@inproceedings{ma2021luna,
  title         = {Luna: Linear Unified Nested Attention}

@inproceedings{ma2023mega,
  title     = {Mega: Moving Average Equipped Gated Attention}

@inproceedings{meng2025polaformer,
  title     = {{PolaFormer}: Polarity-aware Linear Attention for Vision Transformers}

@inproceedings{mialon2021trainable,
  title={A Trainable Optimal Transport Embedding for Feature Aggregation and its Relationship to Attention}

@misc{micikevicius2022fp8,
  title = {{FP8 Formats for Deep Learning}}

@misc{milakov2018online,
  title={Online normalizer calculation for softmax}

@inproceedings{naderializadeh2021pooling,
  title={Pooling by Sliced-{W}asserstein Embedding}

@inproceedings{nawrot2024dynamic,
  title = {{Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference}}

@inproceedings{oren2024transformers,
  title = {{Transformers are Multi-State RNNs}}

@inproceedings{orvieto2023resurrecting,
  title     = {Resurrecting Recurrent Neural Networks for Long Sequences}

@book{owen2013monte,
  title     = {Monte Carlo Theory, Methods and Examples}

@inproceedings{peng2021random,
  title     = {Random Feature Attention}

@inproceedings{peng2022abc,
  title         = {{ABC}: Attention with Bounded-memory Control}

@inproceedings{peng2023rwkv,
  title={{RWKV}: Reinventing {RNN}s for the Transformer Era}

@inproceedings{peng2024eagle,
  title={Eagle and Finch: {RWKV} with Matrix-Valued States and Dynamic Recurrence}

@article{peyre2019computational,
  title={Computational Optimal Transport}

@inproceedings{pham2013fast,
  title={Fast and Scalable Polynomial Kernels via Explicit Feature Maps}

@inproceedings{poli2023hyena,
  title     = {Hyena Hierarchy: Towards Larger Convolutional Language Models}

@inproceedings{prillo2020softsort,
  title={{SoftSort}: A Continuous Relaxation for the argsort Operator}

@inproceedings{qin2022cosformer,
  title={{cosFormer}: Rethinking Softmax in Attention}

@inproceedings{qin2022devil,
  title={The Devil in Linear Transformer}

@inproceedings{qin2023hgrn,
  title={Hierarchically Gated Recurrent Neural Network for Sequence Modeling}

@inproceedings{qin2024hgrn2,
  title={{HGRN2}: Gated Linear {RNN}s with State Expansion}

@inproceedings{qiu2020blockwise,
  title = {{Blockwise Self-Attention for Long Document Understanding}}

@misc{rabe2021selfattention,
  title={Self-attention Does Not Need $O(n^2)$ Memory}

@inproceedings{rahimi2007random,
  title     = {Random Features for Large-Scale Kernel Machines}

@inproceedings{ribar2024sparq,
  title = {{SparQ Attention: Bandwidth-Efficient LLM Inference}}

@inproceedings{romero2022ckconv,
  title     = {CKConv: Continuous Kernel Convolution For Sequential Data}

@article{roy2021routing,
  title = {Efficient Content-Based Sparse Attention with Routing Transformers}

@inproceedings{sander2022sinkformers,
  title         = {Sinkformers: Transformers with Doubly Stochastic Attention}

@misc{sanovar2024lean,
  title={Lean Attention: Hardware-Aware Scalable Attention Mechanism for the Decode-Phase of Transformers}

@inproceedings{sarlin2020superglue,
  title={{SuperGlue}: Learning Feature Matching with Graph Neural Networks}

@inproceedings{scetbon2021lowrank,
  title         = {Low-Rank {S}inkhorn Factorization}

@inproceedings{schlag2021linear,
  title={Linear Transformers Are Secretly Fast Weight Programmers}

@misc{sejourne2022unbalanced,
  title={Unbalanced Optimal Transport, from Theory to Numerics}

@inproceedings{shah2024flashattention3,
  title={{FlashAttention-3}: Fast and Accurate Attention with Asynchrony and Low-precision}

@inproceedings{shahbazi2025espformer,
  title={{ESPFormer}: Doubly-Stochastic Attention with Expected Sliced Transport Plans}

@misc{shahbazi2025lotformer,
  title         = {{LOTFormer}: Doubly-Stochastic Linear Attention via Low-Rank Optimal Transport}

@misc{shahbazi2026oval,
  title = {{OVAL: Output-Aware Local Page Bases for KV Cache Retrieval}}

@misc{shazeer2019fast,
  title = {{Fast Transformer Decoding: One Write-Head is All You Need}}

@misc{shyam2024tree,
  title={Tree Attention: Topology-aware Decoding for Long-Context Attention on {GPU} clusters}

@inproceedings{singhania2024loki,
  title = {{Loki: Low-rank Keys for Efficient Sparse Attention}}

@article{sinkhorn1964relationship,
  title={A relationship between arbitrary positive matrices and doubly stochastic matrices}

@inproceedings{smith2023simplified,
  title     = {Simplified State Space Layers for Sequence Modeling}

@misc{sun2023retentive,
  title={Retentive Network: A Successor to Transformer for Large Language Models}

@misc{sun2024learning,
  title={Learning to (Learn at Test Time): {RNN}s with Expressive Hidden States}

@inproceedings{tang2024quest,
  title     = {{Quest}: Query-Aware Sparsity for Efficient Long-Context {LLM} Inference}

@inproceedings{vaswani2017attention,
  title={Attention Is All You Need}

@inproceedings{vyas2020fast,
  title         = {Fast Transformers with Clustered Attention}

@misc{wang2020linformer,
  title = {{Linformer: Self-Attention with Linear Complexity}}

@misc{wang2024model,
  title = {{Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks}}

@inproceedings{wang2025svdllm,
  title = {{SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression}}

@inproceedings{widrow1960adaptive,
  title={Adaptive Switching Circuits}

@inproceedings{williams2001using,
  title     = {Using the {N}ystr{\&quot;o}m Method to Speed Up Kernel Machines}

@inproceedings{xiao2024efficient,
  title = {{Efficient Streaming Language Models with Attention Sinks}}

@inproceedings{xiao2024infllm,
  title={{InfLLM}: Training-Free Long-Context Extrapolation for {LLMs} with an Efficient Context Memory}

@inproceedings{xiong2021nystromformer,
  title     = {Nystr{\&quot;o}mformer: A Nystr{\&quot;o}m-Based Algorithm for Approximating Self-Attention}

@inproceedings{xu2025think,
  title = {{ThinK: Thinner Key Cache by Query-Driven Pruning}}

@inproceedings{yang2024gated,
  title={Gated Linear Attention Transformers with Hardware-Efficient Training}

@inproceedings{yang2024parallelizing,
  title={Parallelizing Linear Transformers with the Delta Rule over Sequence Length}

@inproceedings{yang2025gated,
  title={Gated Delta Networks: Improving {Mamba2} with Delta Rule}

@inproceedings{yu2016orthogonal,
  title={Orthogonal Random Features}

@inproceedings{zaheer2020bigbird,
  title = {{Big Bird: Transformers for Longer Sequences}}

@inproceedings{zandieh2025qjl,
  title     = {{QJL}: 1-Bit Quantized {JL} Transform for {KV} Cache Quantization with Zero Overhead}

@inproceedings{zhang2023h2o,
  title     = {{H2O}: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models}

@inproceedings{zhang2024cam,
  title = {{CaM}: Cache Merging for Memory-efficient {LLMs} Inference}

@inproceedings{zhang2024hedgehog,
  title     = {The Hedgehog \&amp; the Porcupine: Expressive Linear Attentions with Softmax Mimicry}

@inproceedings{zhang2025sageattention,
  title={{SageAttention}: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}

@inproceedings{zhao2024atom,
  title     = {{Atom}: Low-Bit Quantization for Efficient and Accurate {LLM} Serving}
</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="shazeer2019fast">
<pre class="attn-bib__src"><code>@misc{shazeer2019fast,
  title = {{Fast Transformer Decoding: One Write-Head is All You Need}}

@misc{shyam2024tree,
  title={Tree Attention: Topology-aware Decoding for Long-Context Attention on {GPU} clusters}

@inproceedings{singhania2024loki,
  title = {{Loki: Low-rank Keys for Efficient Sparse Attention}}

@article{sinkhorn1964relationship,
  title={A relationship between arbitrary positive matrices and doubly stochastic matrices}

@inproceedings{smith2023simplified,
  title     = {Simplified State Space Layers for Sequence Modeling}

@misc{sun2023retentive,
  title={Retentive Network: A Successor to Transformer for Large Language Models}

@misc{sun2024learning,
  title={Learning to (Learn at Test Time): {RNN}s with Expressive Hidden States}

@inproceedings{tang2024quest,
  title     = {{Quest}: Query-Aware Sparsity for Efficient Long-Context {LLM} Inference}

@inproceedings{vaswani2017attention,
  title={Attention Is All You Need}

@inproceedings{vyas2020fast,
  title         = {Fast Transformers with Clustered Attention}

@misc{wang2020linformer,
  title = {{Linformer: Self-Attention with Linear Complexity}}

@misc{wang2024model,
  title = {{Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks}}

@inproceedings{wang2025svdllm,
  title = {{SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression}}

@inproceedings{widrow1960adaptive,
  title={Adaptive Switching Circuits}

@inproceedings{williams2001using,
  title     = {Using the {N}ystr{\&quot;o}m Method to Speed Up Kernel Machines}

@inproceedings{xiao2024efficient,
  title = {{Efficient Streaming Language Models with Attention Sinks}}

@inproceedings{xiao2024infllm,
  title={{InfLLM}: Training-Free Long-Context Extrapolation for {LLMs} with an Efficient Context Memory}

@inproceedings{xiong2021nystromformer,
  title     = {Nystr{\&quot;o}mformer: A Nystr{\&quot;o}m-Based Algorithm for Approximating Self-Attention}

@inproceedings{xu2025think,
  title = {{ThinK: Thinner Key Cache by Query-Driven Pruning}}

@inproceedings{yang2024gated,
  title={Gated Linear Attention Transformers with Hardware-Efficient Training}

@inproceedings{yang2024parallelizing,
  title={Parallelizing Linear Transformers with the Delta Rule over Sequence Length}

@inproceedings{yang2025gated,
  title={Gated Delta Networks: Improving {Mamba2} with Delta Rule}

@inproceedings{yu2016orthogonal,
  title={Orthogonal Random Features}

@inproceedings{zaheer2020bigbird,
  title = {{Big Bird: Transformers for Longer Sequences}}

@inproceedings{zandieh2025qjl,
  title     = {{QJL}: 1-Bit Quantized {JL} Transform for {KV} Cache Quantization with Zero Overhead}

@inproceedings{zhang2023h2o,
  title     = {{H2O}: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models}

@inproceedings{zhang2024cam,
  title = {{CaM}: Cache Merging for Memory-efficient {LLMs} Inference}

@inproceedings{zhang2024hedgehog,
  title     = {The Hedgehog \&amp; the Porcupine: Expressive Linear Attentions with Softmax Mimicry}

@inproceedings{zhang2025sageattention,
  title={{SageAttention}: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}

@inproceedings{zhao2024atom,
  title     = {{Atom}: Low-Bit Quantization for Efficient and Accurate {LLM} Serving}
</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/gqa.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/gqa.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
