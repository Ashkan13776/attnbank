# DeltaNet

<p class="tx-regkey"><code>delta_rnn</code> &middot; <code>ab.build("delta_rnn")</code></p>

**DeltaNet: the delta-rule update on a fast-weight memory.**

category
:   [`recurrent`](../taxonomy.md#recurrent) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.DeltaRNNAttention`, [`attnbank/methods/recurrent/delta_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/delta_rnn.py)

monograph
:   `sec:memory-state`

## What it changes

Changes the operator. $y$ is produced by a finite-state recurrence over positions rather than by an integral against a measure — though it is still linear in $v$, which is why the weights can be recovered at all.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

## In the paper's notation

`sec:memory-state`, `eq:memory-delta`, `tab:memory-recurrences` (row "DeltaNet").
The transition is a rank-one correction rather than a decay:

```text
H_t = (I - beta_t psi_t psi_t^T) H_{t-1} + beta_t psi_t v_t^T,    psi_t = psi(k_t),
y_hat_t = H_t^T phi(q_t),
```

which is exactly one gradient step of size `beta_t` on the per-token regression loss
`L_t(H) = 1/2 ||H^T psi_t - v_t||^2` (Widrow–Hoff).  Plain linear attention is a
gradient step on the *linear* loss `-<H^T psi_t, v_t>`, which never saturates and so
never corrects interference between non-orthogonal keys; the delta rule does.  That is
the whole capacity argument: at a fixed state of `r x d_v`, the delta rule stores more
distinguishable associations than the additive recurrence, which is what study S6
measures.

The regression view also identifies the operator.  Softmax attention is a
Nadaraya–Watson estimator that never leaves the convex hull of the stored values; the
delta rule fits a *global parametric* regression of `v` on `psi(k)` online and
returns arbitrary linear combinations.  It is a new operator and is judged by downstream
quality, not by fidelity to `y(q)`.

Weights.  `H_t` is linear in `V`, so `H_t = M_t V` for an `r x n` coefficient
matrix obeying the same recurrence driven by the basis vectors,
`M_t = (I - beta psi_t psi_t^T) M_{t-1} + beta psi_t e_t^T`.  Running it gives the
exact weights `w_{tj} = <phi(q_t), M_t[:, j]>` at a cost of `O(n^2 r)`, which is the
price of exposing `pi_hat` on the original positions.  `psi` is l2-normalized, as in
DeltaNet, so that `I - beta psi psi^T` is a contraction for `beta in (0, 2)`.

Note on notation: DESIGN.md writes the update as `H_{t-1}(I - beta phi phi^T)`, which
is the same map with `H` transposed (`d_v x r`).  This file follows the paper's
`eq:memory-delta` with `H in R^{r x d_v}`.

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **yes** | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **no** | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Configuration

| parameter | default |
|---|---|
| `beta` | `1.0` |
| `feature` | `'elu'` |

`beta`: write strength (not an inverse temperature; see sec:memory-state).

```python
ab.describe("delta_rnn")     # the same keys, with everything else it declares
```

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `beta`, `mass`, `state_rank`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = d
    return {"flops": 4 * n * r * dv + 2 * m * r * dv,
            "state_bytes": r * dv * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 25,165,824 | 268,435,456 | 0.0938× |
| `state_bytes` | 8,192 | 262,144 | 0.0312× |
| `bytes_moved` | 524,288 | 4,718,592 | 0.111× |

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

attn = ab.build("delta_rnn", beta=1.0, feature='elu')
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
DeltaRNNAttention(name='delta_rnn', category='recurrent', objective='modify', backend='reference', beta=1.0, feature='elu')
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[beta, mass, state_rank], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [5.3093, 8.0564, 8.2169, 7.149]
extras: ['beta', 'mass', 'state_rank']
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `delta_rnn` removed.

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`gqa`](gqa.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`lowrank_ot`](lowrank_ot.md), [`mla`](mla.md), [`otke`](otke.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

[Open the browser with `delta_rnn` selected](index.md?compare=delta_rnn) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **schlag2021linear (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="schlag2021linear"><code>schlag2021linear</code></span>
- **yang2024parallelizing (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="yang2024parallelizing"><code>yang2024parallelizing</code></span>
- **widrow1960adaptive (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="widrow1960adaptive"><code>widrow1960adaptive</code></span>
- **yang2025gated (n.d.).** *Atom: Low-Bit Quantization for Efficient and Accurate LLM Serving.*<br><span class="attn-cite" data-bibkey="yang2025gated"><code>yang2025gated</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="schlag2021linear">
<pre class="attn-bib__src"><code>@inproceedings{schlag2021linear,
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
<div class="attn-bib__entry" data-bibkey="yang2024parallelizing">
<pre class="attn-bib__src"><code>@inproceedings{yang2024parallelizing,
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
<div class="attn-bib__entry" data-bibkey="widrow1960adaptive">
<pre class="attn-bib__src"><code>@inproceedings{widrow1960adaptive,
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
<div class="attn-bib__entry" data-bibkey="yang2025gated">
<pre class="attn-bib__src"><code>@inproceedings{yang2025gated,
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

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/delta_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/delta_rnn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
