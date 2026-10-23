# LRU

<p class="tx-regkey"><code>lru</code> &middot; <code>ab.build("lru")</code></p>

**LRU: a diagonal linear RNN with a stable parametrisation, and no state-space theory.**

category
:   [`recurrent`](../taxonomy.md#recurrent) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.LRUAttention`, [`attnbank/methods/recurrent/lru.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/lru.py)

monograph
:   `sec:memory-state`

## What it changes

Changes the operator. $y$ is produced by a finite-state recurrence over positions rather than by an integral against a measure — though it is still linear in $v$, which is why the weights can be recovered at all.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

## In the paper's notation

`sec:memory-state`, `tab:memory-recurrences`.  The recurrence is exactly S5's,

```text
h_t = lambda * h_{t-1} + gamma * (B v_t),      y_t = Re(C h_t) + D * v_t,
```

run with the same parallel scan.  What is different is where `lambda` comes from.
There is no continuous-time system, no HiPPO matrix and no discretisation step.  The
eigenvalues are parametrised directly in polar form,

```text
lambda_n = exp(-exp(nu_n) + i exp(theta_n)),
```

which is stable by construction — `|lambda_n| = exp(-exp(nu_n)) < 1` for every real
`nu_n`, so no projection or clipping is ever needed — and initialised with
`|lambda_n|` uniform on a ring `[r_min, r_max]` and the phase uniform on
`[0, max_phase]`.  The input is rescaled by `gamma_n = sqrt(1 - |lambda_n|^2)`, the
normalisation that keeps the state's variance `O(1)` instead of `O(1/(1-|lambda|))`
however long the memory is.

That is the claim the method is here to make, and it is the reason the library carries
`s5` and `lru` side by side rather than one of them: Orvieto et al. show that a
linear diagonal recurrence with a careful parametrisation matches deep state-space
models on Long Range Arena, so what S4 and S5 contribute is the *structure* — linear,
diagonal, stable, normalised — and not the state-space derivation that motivated it.
`tests/test_recurrent_ssm.py` makes this concrete: one sequential loop reproduces
`s5` and `lru` from their own `extras`, because there is only one recurrence.

**Dynamics are input-independent.**  `lambda`, `gamma`, `B`, `C` and `D` are
parameters.  Nothing in the transition depends on `v_t`, and the queries and keys are
not read at all (`extras['reads_qk']` is `False`); the operator is a matrix-valued
convolution in the lag.  Letting `lambda` depend on the token is exactly the step the
gated linear RNNs took, and it is the boundary of this family.

**Weights.**  `expresses_weights = False`, for S5's reason: `C lambda^l B` is a
`d_v x d_v` matrix per lag, the operator mixes channels as well as positions, and
there is no scalar measure over the original positions to report.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **yes** | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | **no** | it cannot name $\widehat\pi_q$ on the original positions and is excluded from the error decomposition by declaration. |
| `row_stochastic` | **no** | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **yes** | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

</div>

## Configuration

| parameter | default |
|---|---|
| `n_state` | `16` |
| `r_min` | `0.0` |
| `r_max` | `1.0` |
| `max_phase` | `6.283185307179586` |
| `feedthrough` | `True` |
| `seed` | `0` |

`n_state`: the size `N` of the (channel-shared) complex state.

`r_min` / `r_max` bound `|lambda|` at initialisation — the memory
horizon, since mode `n` decays with time constant `-1/log|lambda_n|` — and
`max_phase` bounds the initial phase.  The defaults are the paper's.

```python
ab.describe("lru")     # the same keys, with everything else it declares
```

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `b_bar`, `c_eff`, `d_skip`, `lam`, `n_state`, `radius`, `reads_qk`, `state_size`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.n_state
    return {"flops": 8 * n * r * dv + 8 * n * r + 8 * m * r * dv,
            "state_bytes": 2 * r * bytes_per_el,
            "bytes_moved": (n * dv + m * dv + 4 * r * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 16,908,288 | 268,435,456 | 0.063× |
| `state_bytes` | 64 | 262,144 | 0.000244× |
| `bytes_moved` | 270,336 | 4,718,592 | 0.0573× |

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

attn = ab.build("lru", n_state=16, r_min=0.0)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = modify: this is a different operator and fidelity to softmax
# is not its goal, so there is no error to quote. What it *is* doing shows
# up in its own diagnostics.
print("expresses_weights =", attn.expresses_weights,
      "-- no measure over the original positions exists,",
      "so weights is", out.weights)
print("extras:", sorted(out.extras))
```

```text
LRUAttention(name='lru', category='recurrent', objective='modify', backend='reference', feedthrough=True, max_phase=6.283185307179586, n_state=16, r_max=1.0, r_min=0.0, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=None, v_hat=None, extras=[b_bar, c_eff, d_skip, lam, n_state, radius, reads_qk, state_size], dtype=float32, device=cpu)
invariant W: 0.0
expresses_weights = False -- no measure over the original positions exists, so weights is None
extras: ['b_bar', 'c_eff', 'd_skip', 'lam', 'n_state', 'radius', 'reads_qk', 'state_size']
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `lru` removed.

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`gqa`](gqa.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`lowrank_ot`](lowrank_ot.md), [`mla`](mla.md), [`otke`](otke.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

[Open the browser with `lru` selected](index.md?compare=lru) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **orvieto2023resurrecting (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="orvieto2023resurrecting"><code>orvieto2023resurrecting</code></span>
- **smith2023simplified (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="smith2023simplified"><code>smith2023simplified</code></span>
- **blelloch1990prefix (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="blelloch1990prefix"><code>blelloch1990prefix</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="orvieto2023resurrecting">
<pre class="attn-bib__src"><code>@inproceedings{orvieto2023resurrecting,
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
<div class="attn-bib__entry" data-bibkey="smith2023simplified">
<pre class="attn-bib__src"><code>@inproceedings{smith2023simplified,
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
<div class="attn-bib__entry" data-bibkey="blelloch1990prefix">
<pre class="attn-bib__src"><code>@techreport{blelloch1990prefix,
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
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/lru.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/lru.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
