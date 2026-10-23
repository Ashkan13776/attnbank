# Decayed linear recurrence

<p class="tx-regkey"><code>decay_rnn</code> &middot; <code>ab.build("decay_rnn")</code></p>

**Decayed linear recurrence: H_t = gamma H_{t-1} + psi(k_t) v_t^T.**

category
:   [`recurrent`](../taxonomy.md#recurrent) — one of the seven objects in $y(q)=\int v\,\mathrm{d}\pi_q$ a method can change

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.DecayRNNAttention`, [`attnbank/methods/recurrent/decay_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/decay_rnn.py)

monograph
:   `sec:memory-state`

## What it changes

Changes the operator. $y$ is produced by a finite-state recurrence over positions rather than by an integral against a measure — though it is still linear in $v$, which is why the weights can be recovered at all.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

## In the paper's notation

`sec:memory-state`, `tab:memory-recurrences` (rows "Retention", "GLA / Mamba-2").
`A_t = gamma Id` with a fixed `gamma < 1` is retention (RetNet); a data-dependent
scalar or diagonal gate gives Mamba-2, GLA, HGRN2, RWKV-5/6 and the mLSTM.  The weights
on the original positions are

```text
w_{tj} = gamma^{(t - j)} <phi(q_t), psi(k_j)>,    j <= t,
```

an exponential recency prior multiplying a rank-`r` content term.

Decay restores *boundedness* without restoring normalization: the effective mass
`sum_i prod_{u>i} gamma_u` is bounded by `1/(1 - gamma)` rather than growing with
`n`, so the output cannot blow up with context length, but it is still not an
expectation under a probability measure (set `normalize=True` to recover one).  This
is the distinction `sec:memory-state` draws between "restoring boundedness" and
"restoring the contract".

State size is `r x d_v` regardless of `gamma`, so sweeping `gamma` at fixed state
separates *recency* from *capacity* — which is what study S6 needs.

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
| `decay` | `0.9` |
| `feature` | `'elu'` |
| `normalize` | `False` |

```python
ab.describe("decay_rnn")     # the same keys, with everything else it declares
```

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Backends

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `decay`, `gate`, `mass`.

## Cost model

Analytic, per `(batch, head)`, declared by the method itself:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = d
    return {"flops": 2 * n * r * dv + 2 * m * r * dv + m * n,
            "state_bytes": r * dv * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
```

At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 17,825,792 | 268,435,456 | 0.0664× |
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

attn = ab.build("decay_rnn", decay=0.9, feature='elu')
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
DecayRNNAttention(name='decay_rnn', category='recurrent', objective='modify', backend='reference', decay=0.9, feature='elu', normalize=False)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[decay, gate, mass], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = False | first row sums: [43.946, 78.9989, 108.2692, 140.6155]
extras: ['decay', 'gate', 'mass']
```

## Related methods

Two relations, because the registry declares two axes: what a method changes, and what it is trying to be. Both lists are `ab.available(...)` with `decay_rnn` removed.

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`delta_rnn`](delta_rnn.md), [`linear_rnn`](linear_rnn.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`gqa`](gqa.md), [`latent_stochastic`](latent_stochastic.md), [`linear_elu`](linear_elu.md), [`lowrank_ot`](lowrank_ot.md), [`mla`](mla.md), [`sinkhorn`](sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

[Open the browser with `decay_rnn` selected](index.md?compare=decay_rnn) and tick up to three more to see them in columns. Every row there is one the registry declares, and the rows the selected methods agree on fold away.

## References

- **Sun et al. (2023).** *Retentive Network: A Successor to Transformer for Large Language Models.* [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)<br><span class="attn-cite" data-bibkey="sun2023retentive"><code>sun2023retentive</code></span>
- **Yang et al. (2024).** *Gated Linear Attention Transformers with Hardware-Efficient Training.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2312.06635](https://arxiv.org/abs/2312.06635)<br><span class="attn-cite" data-bibkey="yang2024gated"><code>yang2024gated</code></span>
- **Dao and Gu (2024).** *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* Proceedings of the 41st International Conference on Machine Learning (ICML). [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)<br><span class="attn-cite" data-bibkey="dao2024transformers"><code>dao2024transformers</code></span>
- **Peng et al. (2023).** *RWKV: Reinventing RNNs for the Transformer Era.* Findings of the Association for Computational Linguistics: EMNLP 2023. [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)<br><span class="attn-cite" data-bibkey="peng2023rwkv"><code>peng2023rwkv</code></span>
- **Beck et al. (2024).** *xLSTM: Extended Long Short-Term Memory.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2405.04517](https://arxiv.org/abs/2405.04517)<br><span class="attn-cite" data-bibkey="beck2024xlstm"><code>beck2024xlstm</code></span>

<details class="attn-bib no-copy">
<summary>BibTeX — 5 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="sun2023retentive">
<pre class="attn-bib__src"><code>@misc{sun2023retentive,
  title={Retentive Network: A Successor to Transformer for Large Language Models},
  author={Sun, Yutao and Dong, Li and Huang, Shaohan and Ma, Shuming and Xia, Yuqing and Xue, Jilong and Wang, Jianyong and Wei, Furu},
  year={2023},
  eprint={2307.08621},
  archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="yang2024gated">
<pre class="attn-bib__src"><code>@inproceedings{yang2024gated,
  title={Gated Linear Attention Transformers with Hardware-Efficient Training},
  author={Yang, Songlin and Wang, Bailin and Shen, Yikang and Panda, Rameswar and Kim, Yoon},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series={PMLR},
  volume={235},
  pages={56501--56523},
  year={2024},
  eprint={2312.06635},
  archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="dao2024transformers">
<pre class="attn-bib__src"><code>@inproceedings{dao2024transformers,
  title={Transformers are {SSM}s: Generalized Models and Efficient Algorithms Through Structured State Space Duality},
  author={Dao, Tri and Gu, Albert},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  series={PMLR},
  volume={235},
  pages={10041--10071},
  year={2024},
  eprint={2405.21060},
  archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="peng2023rwkv">
<pre class="attn-bib__src"><code>@inproceedings{peng2023rwkv,
  title={{RWKV}: Reinventing {RNN}s for the Transformer Era},
  author={Peng, Bo and Alcaide, Eric and Anthony, Quentin and Albalak, Alon and Arcadinho, Samuel and Biderman, Stella and others},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
  year={2023},
  eprint={2305.13048}, archivePrefix={arXiv}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="beck2024xlstm">
<pre class="attn-bib__src"><code>@inproceedings{beck2024xlstm,
  title={{xLSTM}: Extended Long Short-Term Memory},
  author={Beck, Maximilian and P{\&quot;o}ppel, Korbinian and Spanring, Markus and Auer, Andreas and Prudnikova, Oleksandra and Kopp, Michael and Klambauer, G{\&quot;u}nter and Brandstetter, Johannes and Hochreiter, Sepp},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  eprint={2405.04517}, archivePrefix={arXiv}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

*Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/decay_rnn.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/decay_rnn.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.*
