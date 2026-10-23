# KV-cache quantization

<p class="tx-regkey"><code>quantkv</code> &middot; <code>ab.build("quantkv")</code></p>

**KV-cache quantization: uniform affine (asymmetric) quantization of keys and values.**

category
:   [`compress`](../taxonomy.md#compress) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `approximate` — targets fidelity to dense softmax; judged by output error

class
:   `attnbank.QuantKVAttention`, [`attnbank/methods/compress/quantkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/quantkv.py)

monograph
:   `sec:compress`

## What it changes

Changes the stored context $(\mu, k, v)$ itself. This is the one category that routinely perturbs the integrand, so $\widehat v \neq v$ and the error has a value term.

It pursues **approximation**: it is trying to be dense softmax, so it is fair to judge it by output error, and it must supply a correctness anchor saying where it becomes exact.

## In the paper's notation

`sec:compress-quant` and `prop:compress-sensitivity`.  Keys and values stay on the
same domain; only their representation changes, so

```text
k_hat_j = Q_{b_k}(k_j),   v_hat_j = Q_{b_v}(v_j),
Q_b(x)  = round((x - lo) / Delta) * Delta + lo,   Delta = (hi - lo) / (2^b - 1),
```

with `lo, hi` the min and max over a *group*.  The key error enters as a log-weight
perturbation `delta_l_j = beta <q, delta k_j>` — an exponential tilt of `pi_q` —
and is amplified by `e^{osc(delta_l)} - 1` and by `||q||`; the value error enters
linearly and is attention-weighted.  That asymmetry is the content of
`prop:compress-sensitivity` and of `eq:error-joint`, and is what study S4 measures.

Grouping.  `axis='token'` shares a scale among `group` channels of one token (KIVI's
value grouping); `axis='channel'` shares it among `group` tokens of one channel
(KIVI's key grouping, which confines an outlier channel's coarse step to that channel).
Only the *token-varying* part of the key error matters, since softmax cancels a common
shift — which is exactly why per-channel grouping helps for keys.

Per-channel grouping pools statistics across tokens, so it is **not causal**; the
method raises rather than silently leaking.  `axis='token'` is causal and is the
default.

Exact limit: `bits_k = bits_v = None` (the identity quantizer).  16-bit affine
quantization is *not* bit-exact in fp32, so it would be a dishonest anchor.

The quantizer itself is `attnbank.components.quantization.affine_quantize`,
shared with `lowprec`; this file is only the decision to apply it to the *cache*.

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
| `bits_k` | `4` |
| `bits_v` | `4` |
| `axis` | `'token'` |
| `group` | `None` |

```python
ab.describe("quantkv")     # the same keys, with everything else it declares
```

## Correctness anchor

**Exact limit.** Built with `bits_k=None`, `bits_v=None` this method provably *is* dense softmax, and `tests/test_conformance.py` asserts it — output and weights, float32 and float64.

```pycon
>>> ab.describe("quantkv").limit_config
{'bits_k': None, 'bits_v': None}
```

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** It returns both the exact and the perturbed score matrices (`s`, `s_hat`) in `extras` — the perturbation *is* the object of study, and a fused kernel hands back neither.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `delta_k`, `delta_v`, `s`, `s_hat`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    bk = (self.bits_k or 8 * bytes_per_el) / 8
    bv = (self.bits_v or 8 * bytes_per_el) / 8
    state = n * (d * bk + dv * bv)
    return {"flops": 2 * m * n * (d + dv), "state_bytes": state,
            "bytes_moved": m * (d + dv) * bytes_per_el + state}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 268,435,456 | 268,435,456 | 1× |
| `state_bytes` | 65,536 | 262,144 | 0.25× |
| `bytes_moved` | 327,680 | 4,718,592 | 0.0694× |

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

attn = ab.build("quantkv", bits_k=4, bits_v=4)
out = attn(q, k, v, causal=True,
           return_weights=True)

print(attn)
print(out)
print("invariant W:", out.check_invariant(v))

# objective = approximate: it targets dense softmax, so it is judged by
# output error -- split into the distributional and value terms.
row = ab.compare(["quantkv"], q, k, v, causal=True,
                 configs={"quantkv": {'bits_k': 4, 'bits_v': 4}})[0]
print({key: round(row[key], 5) for key in
       ("output_error", "distributional_error", "value_error", "tv")})
```

```text
QuantKVAttention(name='quantkv', category='compress', objective='approximate', backend='reference', axis='token', bits_k=4, bits_v=4, group=None)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=[1, 2, 64, 32], extras=[delta_k, delta_v, s, s_hat], dtype=float32, device=cpu)
invariant W: 0.0
{'output_error': 0.15591, 'distributional_error': 0.08556, 'value_error': 0.12244, 'tv': 0.02758}
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `quantkv` removed.

changes the same object — [`compress`](../taxonomy.md#compress)
:   [`evict_h2o`](evict_h2o.md), [`evict_snapkv`](evict_snapkv.md), [`gqa`](gqa.md), [`linformer`](linformer.md), [`lowrank_kv`](lowrank_kv.md), [`merge_kv`](merge_kv.md), [`mla`](mla.md) &nbsp;·&nbsp; [browse the bank, compress only](index.md?category=compress)

judged the same way — `approximate`, beyond the ones above
:   [`blocktopk`](blocktopk.md), [`clustered`](clustered.md), [`lowprec`](lowprec.md), [`lsh_sample`](lsh_sample.md), [`nmf_oracle`](nmf_oracle.md) *(oracle)*, [`nystrom`](nystrom.md), [`performer`](performer.md), [`sink_window`](sink_window.md), [`svd_oracle`](svd_oracle.md) *(oracle)*, [`topk`](topk.md), [`trigrff`](trigrff.md), [`value_topk`](value_topk.md) *(oracle)*, [`window`](window.md) &nbsp;·&nbsp; [browse the bank, approximate only](index.md?objective=approximate)

[Open the browser with `quantkv` selected](index.md?compare=quantkv) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **Liu et al. (2024).** *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)<br><span class="attn-cite" data-bibkey="liu2024kivi"><code>liu2024kivi</code></span>
- **Hooper et al. (2024).** *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)<br><span class="attn-cite" data-bibkey="hooper2024kvquant"><code>hooper2024kvquant</code></span>
- **Liu et al. (2024).** *IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact.* Findings of the Association for Computational Linguistics: ACL 2024. [arXiv:2403.01241](https://arxiv.org/abs/2403.01241)<br><span class="attn-cite" data-bibkey="liu2024intactkv"><code>liu2024intactkv</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="liu2024kivi">
<pre class="attn-bib__src"><code>@inproceedings{liu2024kivi,
  title     = {{KIVI}: A Tuning-Free Asymmetric 2bit Quantization for {KV} Cache},
  author    = {Liu, Zirui and Yuan, Jiayi and Jin, Hongye and Zhong, Shaochen and Xu, Zhaozhuo and Braverman, Vladimir and Chen, Beidi and Hu, Xia},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  year      = {2024},
  eprint    = {2402.02750},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="hooper2024kvquant">
<pre class="attn-bib__src"><code>@inproceedings{hooper2024kvquant,
  title     = {{KVQuant}: Towards 10 Million Context Length {LLM} Inference with {KV} Cache Quantization},
  author    = {Hooper, Coleman and Kim, Sehoon and Mohammadzadeh, Hiva and Mahoney, Michael W. and Shao, Yakun Sophia and Keutzer, Kurt and Gholami, Amir},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {37},
  year      = {2024},
  eprint    = {2401.18079},
  archivePrefix = {arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="liu2024intactkv">
<pre class="attn-bib__src"><code>@inproceedings{liu2024intactkv,
  title = {{IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact}},
  author = {Ruikang Liu and Haoli Bai and Haokun Lin and Yuening Li and Han Gao and Zhengzhuo Xu and Lu Hou and Jun Yao and Chun Yuan},
  year = {2024},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  eprint = {2403.01241},
  archivePrefix = {arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/compress/quantkv.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/compress/quantkv.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
