# Oracle top-k restriction

<p class="tx-regkey"><code>topk</code> &middot; <code>ab.build("topk")</code></p>

**Oracle top-k restriction: the `k` largest true scores, renormalized.**

category
:   [`restrict`](../taxonomy.md#restrict) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.TopKAttention`, [`attnbank/methods/restrict/topk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/topk.py)

monograph
:   `sec:sparse`

## What it changes

Changes the **support** of $\pi_q$, which shrinks to $S_q \subseteq \{1,\dots,n\}$ and is renormalised on it. The values $v$ are untouched, so the whole error is distributional.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:sparse-operator`, `sec:sparse-query`.  `S_q` is the set of the `k` largest
exact attention weights, which by `prop:error-topk` minimizes the omitted mass
`eps_q` — and hence the bound `eps_q osc(v)` — over all supports of size `k`.
It is the **oracle** against which every proxy selector (Quest, SparQ, Loki, H2O) is
measured: computing all scores saves only the value aggregation.

The proposition's second half matters more for this library: minimizing `eps_q` is
*not* minimizing `||y - y_hat||`.  With `pi = (0.45, 0.35, 0.20)` and scalar values
`(0, 0, 1)`, top-2 retains `{1,2}` and errs by `0.2` while `{1,3}` omits more
mass and errs by `0.11`.  `value_topk` is the output-aware alternative; the gap
between them is the quantity study S2 reports.

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
| `k` | `8` |

`k`: budget; `None` keeps every visible key (the exact limit).

```python
ab.describe("topk")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `k=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("topk").limit_config
{'k': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It reports the omitted mass `eps` (and the $\chi^2$ divergence of the renormalisation) in `extras`, which needs the full score matrix. A kernel that avoids materialising that matrix is a kernel that cannot produce them.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `chi2`, `eps`, `k_realized`, `scores`, `support`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    kk = self.k or n
    return {"flops": 2 * m * n * d + 2 * m * kk * dv + m * n,
            "state_bytes": n * (d + dv) * bytes_per_el,     # full context stays resident
            "bytes_moved": (m * d + n * d + m * kk * dv + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 136,314,880 | 268,435,456 | 0.508× |
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

attn = ab.build("topk", k=8)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["topk"], q, k, v, causal=True,
                 configs={"topk": {'k': 8}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
TopKAttention(name='topk', category='restrict', objective='approximate', backend='reference', k=8)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[chi2, eps, k_realized, scores, support], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.98508, 'distributional_error': 0.98508, 'value_error': 0.0, 'tv': 0.39513}
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `topk` removed.

changes the same object — [`restrict`](../taxonomy.md#restrict)
:   [`bigbird`](bigbird.md), [`blocktopk`](blocktopk.md), [`hyperattention`](hyperattention.md), [`longformer`](longformer.md), [`lsh_sample`](lsh_sample.md), [`oval`](oval.md), [`page_pca`](page_pca.md), [`reformer`](reformer.md), [`routing`](routing.md), [`shifted_window`](shifted_window.md), [`sink_window`](sink_window.md), [`strided`](strided.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, restrict only](index.md?category=restrict)

judged the same way — `approximate`, beyond the ones above
:   [`abc`](abc.md), [`atom`](atom.md), [`based`](based.md), [`clustered`](clustered.md), [`evict_adakv`](evict_adakv.md), [`evict_h2o`](evict_h2o.md), [`evict_pyramidkv`](evict_pyramidkv.md), [`evict_snapkv`](evict_snapkv.md), [`evict_tova`](evict_tova.md), [`favorpp`](favorpp.md), [`favorsharp`](favorsharp.md), [`hedgehog`](hedgehog.md), [`kivi`](kivi.md), [`kvquant`](kvquant.md), [`linformer`](linformer.md), [`lowprec`](lowprec.md), [`lowrank_kv`](lowrank_kv.md), [`lowrank_logit`](lowrank_logit.md) *(oracle)*, [`merge_cam`](merge_cam.md), [`merge_kv`](merge_kv.md), [`merge_tome`](merge_tome.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`polysketch`](polysketch.md), [`qjl`](qjl.md), [`quantkv`](quantkv.md), [`skyformer`](skyformer.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`trigrff`](trigrff.md), [`zipcache`](zipcache.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

[Open the browser with `topk` selected](index.md?compare=topk) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **gupta2021memory (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="gupta2021memory"><code>gupta2021memory</code></span>
- **tang2024quest (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="tang2024quest"><code>tang2024quest</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 2 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="gupta2021memory">
<pre class="attn-bib__src"><code>@inproceedings{gupta2021memory,
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
<div class="attn-bib__entry" data-bibkey="tang2024quest">
<pre class="attn-bib__src"><code>@inproceedings{tang2024quest,
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

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/restrict/topk.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/restrict/topk.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
