# LRU

<p class="attn-standfirst" markdown="span">A diagonal linear RNN with a stable parametrisation, and no state-space theory.</p>

<div class="attn-facts" data-cat="recurrent" markdown="1">

key
:   `lru` &middot; `ab.build("lru")`

category
:   [`recurrent`](../taxonomy.md#recurrent)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.LRUAttention` &middot; [`attnbank/methods/recurrent/lru.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/lru.py)

monograph
:   `sec:memory-state`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the operator. $y$ is produced by a finite-state recurrence over positions rather than by an integral against a measure — though it is still linear in $v$, which is why the weights can be recovered at all.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

</div>

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

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 16,908,288 | 268,435,456 | 0.063× |
| `state_bytes` | 64 | 262,144 | 0.000244× |
| `bytes_moved` | 270,336 | 4,718,592 | 0.0573× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    r = self.n_state
    return {"flops": 8 * n * r * dv + 8 * n * r + 8 * m * r * dv,
            "state_bytes": 2 * r * bytes_per_el,
            "bytes_moved": (n * dv + m * dv + 4 * r * dv) * bytes_per_el}
```

!!! note "A cost model is not a benchmark"
    These are counted operations and resident bytes, not measured time.
    They are what the monograph's capacity arguments are made of;
    `examples/08_cost_and_capacity.py` walks through the comparison properly.

## Backends

<div class="attn-tab" markdown="1">

| backend | what it is |
|---|---|
| `reference` | plain PyTorch; the correctness anchor and the default |

</div>

**`reference` only, and that is the correct answer.** A scan over positions with a fixed-size state is a different program from a masked softmax.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `b_bar`, `c_eff`, `d_skip`, `lam`, `n_state`, `radius`, `reads_qk`, `state_size`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_state` | `16` |
| `r_min` | `0.0` |
| `r_max` | `1.0` |
| `max_phase` | `6.283185307179586` |
| `feedthrough` | `True` |
| `seed` | `0` |

</div>

`n_state`: the size `N` of the (channel-shared) complex state.

`r_min` / `r_max` bound `|lambda|` at initialisation — the memory
horizon, since mode `n` decays with time constant `-1/log|lambda_n|` — and
`max_phase` bounds the initial phase.  The defaults are the paper's.

```python
ab.describe("lru")     # the same keys, with everything else it declares
```

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

<div class="attn-out" markdown="1">

```text
LRUAttention(name='lru', category='recurrent', objective='modify', backend='reference', feedthrough=True, max_phase=6.283185307179586, n_state=16, r_max=1.0, r_min=0.0, seed=0)
AttentionOutput(y=[1, 2, 64, 32], weights=None, v_hat=None, extras=[b_bar, c_eff, d_skip, lam, n_state, radius, reads_qk, state_size], dtype=float32, device=cpu)
invariant W: 0.0
expresses_weights = False -- no measure over the original positions exists, so weights is None
extras: ['b_bar', 'c_eff', 'd_skip', 'lam', 'n_state', 'radius', 'reads_qk', 'state_size']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

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

</div>

## Related methods

changes the same object — [`recurrent`](../taxonomy.md#recurrent)
:   [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`linear_rnn`](linear_rnn.md), [`mega`](mega.md), [`mlstm`](mlstm.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, recurrent only](index.md?category=recurrent)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`espformer`](espformer.md), [`gqa`](gqa.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`lowrank_ot`](lowrank_ot.md), [`mla`](mla.md), [`otke`](otke.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `lru` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `lru` selected](index.md?compare=lru) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *Resurrecting Recurrent Neural Networks for Long Sequences.*<br><span class="attn-cite" data-bibkey="orvieto2023resurrecting"><code>orvieto2023resurrecting</code></span>
- *Simplified State Space Layers for Sequence Modeling.*<br><span class="attn-cite" data-bibkey="smith2023simplified"><code>smith2023simplified</code></span>
- *Prefix Sums and Their Applications.*<br><span class="attn-cite" data-bibkey="blelloch1990prefix"><code>blelloch1990prefix</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 3 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="orvieto2023resurrecting">
<pre class="attn-bib__src"><code>@inproceedings{orvieto2023resurrecting,
  title     = {Resurrecting Recurrent Neural Networks for Long Sequences}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="smith2023simplified">
<pre class="attn-bib__src"><code>@inproceedings{smith2023simplified,
  title     = {Simplified State Space Layers for Sequence Modeling}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="blelloch1990prefix">
<pre class="attn-bib__src"><code>@techreport{blelloch1990prefix,
  title       = {Prefix Sums and Their Applications}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/recurrent/lru.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/recurrent/lru.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
