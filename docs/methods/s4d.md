# S4D

<p class="attn-standfirst" markdown="span">The diagonal state-space layer, as a long convolution over the values.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `s4d` &middot; `ab.build("s4d")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.S4DAttention` &middot; [`attnbank/methods/recurrent/s4d.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/s4d.py)

monograph
:   `sec:memory-state`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the operator. $y$ is produced by a finite-state recurrence over positions rather than by an integral against a measure — though it is still linear in $v$, which is why the weights can be recovered at all.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

</div>

## What it declares

Each row is a commitment the [conformance suite](../conformance.md) tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | <span class="attn-flag" data-v="yes">yes</span> | `causal=True` masks so query $i$ sees keys $0 \dots i + (n-m)$. |
| `expresses_weights` | <span class="attn-flag" data-v="no">no</span> | it cannot name $\widehat\pi_q$ on the original positions and is excluded from the error decomposition by declaration. |
| `row_stochastic` | <span class="attn-flag" data-v="no">no</span> | the weights are **not** a probability measure — signed, or not summing to one — so the output can leave the convex hull of the values. |
| `head_independent` | <span class="attn-flag" data-v="yes">yes</span> | each head is computed from its own $K, V$. |
| `trainable` | <span class="attn-flag" data-v="yes">yes</span> | it has learned parameters; the reference implementation initialises them from a seed, so the numbers here are an untrained operator. |

</div>

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 16,121,856 | 268,435,456 | 0.0601× |
| `state_bytes` | 4,096 | 262,144 | 0.0156× |
| `bytes_moved` | 266,240 | 4,718,592 | 0.0564× |

</div>

<p class="attn-caveat" markdown="span">Counted operations and resident bytes, not measured time — a cost model is not a benchmark. These are what the monograph's capacity arguments are made of; `examples/08_cost_and_capacity.py` walks the comparison through properly.</p>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    """Costed as the deployed layer runs: build the kernel, then one FFT convolution."""
    logn = max(1.0, math.log2(max(n, 2)))
    return {"flops": 6 * dv * n * self.n_state + 15 * dv * n * logn,
            # a complex state of N poles for each of the dv channels
            "state_bytes": 2 * self.n_state * dv * bytes_per_el,
            "bytes_moved": (n * dv + m * dv + 2 * self.n_state * dv) * bytes_per_el}
```

## Backends

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

`reference` is plain PyTorch — the correctness anchor and the default. Declaring a second backend is a claim that it equals the first, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `kernel`, `n_state`, `poles`, `reads_qk`, `state_size`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_state` | `16` |
| `init` | `'lin'` |
| `dt_min` | `0.001` |
| `dt_max` | `0.1` |
| `fft` | `False` |
| `seed` | `0` |

</div>

`n_state`: poles per channel (each standing for a conjugate pair).

`init` selects the pole placement, `'lin'` (S4D-Lin) or `'inv'`
(S4D-Inv); `dt_min` / `dt_max` bound the log-uniform timestep, which is what
sets the range of timescales the channels cover; `fft` switches to the
frequency-domain evaluation (see the module docstring).

`ab.describe("s4d")` returns the same keys alongside everything else this page states.

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("s4d", n_state=16, init='lin')
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

<div class="attn-out" markdown="1">

```text
S4DAttention(name='s4d', category='recurrent', objective='modify', backend='reference', dt_max=0.1, dt_min=0.001, fft=False, init='lin', n_state=16, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=None, v_hat=None, extras=[kernel, n_state, poles, reads_qk, state_size], dtype=float32, device=cpu)
invariant W: 0.0
expresses_weights = False -- no measure over the original positions exists, so weights is None
extras: ['kernel', 'n_state', 'poles', 'reads_qk', 'state_size']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

<p class="attn-notation__sec"><span class="attn-notation__from">monograph</span> <code>sec:memory-state</code>, <code>tab:memory-recurrences</code></p>

Each value channel carries its own
single-input single-output state-space model

```text
h_t^{(c)} = A_c h_{t-1}^{(c)} + B_c v_{t,c},     y_{t,c} = C_c h_t^{(c)},
```

with `A_c = diag(exp(dt_c lambda))` diagonal and stable.  Because `A`, `B`, `C`
and the step `dt` are **parameters and not functions of the input**, the recurrence
unrolls into a convolution whose kernel depends on the lag alone,

```text
y_{t,c} = sum_{l >= 0} K_c[l] v_{t-l, c},        K_c[l] = 2 Re sum_n C_{cn} Bbar_{cn} lambdabar_{cn}^l,
```

and the layer can be evaluated for the whole sequence at once.  That input-independence
is the property this corner of the library exists to exhibit, and the step that produced
the current generation of models — Mamba, GLA, the gated linear RNNs — was precisely
to give it up and let `A_t` depend on `x_t`.

**What is implemented, and what is not.**  This is **S4D** (Gu, Gupta, Goel and Re,
2022), the *diagonal* state-space model, not the full S4 of Gu, Goel and Re (2022),
whose state matrix is diagonal-plus-low-rank and whose kernel is evaluated through a
Cauchy kernel and a Woodbury correction on the generating function.  S4D keeps the
diagonal part, drops the low-rank correction — DSS (Gupta et al., 2022) showed that
costs nothing in accuracy — and has a closed-form kernel, which is what is coded here.
Calling this file `s4` would have named something it is not.

**It does not read the queries or the keys.**  An LTI layer mixes positions with a
filter indexed by the lag; there is nowhere for `q` and `k` to enter.  The library
keeps the attention signature so that the operator can be put side by side with softmax
on identical inputs, and the resulting gap — an SSM of this kind cannot do associative
recall at all — is the finding, not a defect of the wrapper.  `extras['reads_qk']` is
`False` and `tests/test_recurrent_ssm.py` asserts it.

**Weights.**  `expresses_weights = False`.  Every value channel is convolved with a
*different* kernel, so there is no single measure `pi_hat` over the `n` context
positions that reproduces the output: the operator mixes positions, but with a different
mixture per channel.  Manufacturing a weight matrix here would mean averaging kernels
across channels and reporting something the method does not compute.  The per-channel
kernels are returned in `extras['kernel']` instead, which is the honest object.

**Frequency domain.**  `fft=True` evaluates the same convolution with three FFTs,
which is how S4 is actually run.  It is not the default because an FFT mixes the whole
sequence and is therefore not bit-exactly causal — a later value perturbs an earlier
output in the last few bits — and the conformance suite checks causality exactly.
`tests/test_recurrent_ssm.py` checks the two paths agree to 1e-10 in fp64.

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`
:   16 further methods besides these declare `objective = modify`, which is how they are scored rather than what they do &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both come from `ab.available(...)` with `s4d` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `s4d` selected](index.md?compare=s4d) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- **Gu et al. (2022).** *Efficiently Modeling Long Sequences with Structured State Spaces.* International Conference on Learning Representations (ICLR). [arXiv:2111.00396](https://arxiv.org/abs/2111.00396)<br><span class="attn-cite" data-bibkey="gu2022efficiently"><code>gu2022efficiently</code></span>
- **Gu et al. (2022).** *On the Parameterization and Initialization of Diagonal State Space Models.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2206.11893](https://arxiv.org/abs/2206.11893)<br><span class="attn-cite" data-bibkey="gu2022parameterization"><code>gu2022parameterization</code></span>
- **Gupta et al. (2022).** *Diagonal State Spaces are as Effective as Structured State Spaces.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2203.14343](https://arxiv.org/abs/2203.14343)<br><span class="attn-cite" data-bibkey="gupta2022diagonal"><code>gupta2022diagonal</code></span>
- **Gu et al. (2020).** *HiPPO: Recurrent Memory with Optimal Polynomial Projections.* Advances in Neural Information Processing Systems (NeurIPS). [arXiv:2008.07669](https://arxiv.org/abs/2008.07669)<br><span class="attn-cite" data-bibkey="gu2020hippo"><code>gu2020hippo</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="gu2022efficiently">
<pre class="attn-bib__src"><code>@inproceedings{gu2022efficiently,
  title     = {Efficiently Modeling Long Sequences with Structured State Spaces},
  author    = {Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  eprint    = {2111.00396},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2111.00396}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="gu2022parameterization">
<pre class="attn-bib__src"><code>@inproceedings{gu2022parameterization,
  title     = {On the Parameterization and Initialization of Diagonal State Space Models},
  author    = {Gu, Albert and Gupta, Ankit and Goel, Karan and R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
  eprint    = {2206.11893},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2206.11893}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="gupta2022diagonal">
<pre class="attn-bib__src"><code>@inproceedings{gupta2022diagonal,
  title     = {Diagonal State Spaces are as Effective as Structured State Spaces},
  author    = {Gupta, Ankit and Gu, Albert and Berant, Jonathan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022},
  eprint    = {2203.14343},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2203.14343}
}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="gu2020hippo">
<pre class="attn-bib__src"><code>@inproceedings{gu2020hippo,
  title     = {HiPPO: Recurrent Memory with Optimal Polynomial Projections},
  author    = {Gu, Albert and Dao, Tri and Ermon, Stefano and Rudra, Atri and
               R{\'e}, Christopher},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2020},
  eprint    = {2008.07669},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2008.07669}
}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/s4d.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/s4d.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
