# ESPFormer

<p class="attn-standfirst" markdown="span">Doubly stochastic attention from *expected sliced transport plans*.</p>

<div class="attn-facts" data-cat="coupling" markdown="1">

key
:   `espformer` &middot; `ab.build("espformer")`

category
:   [`coupling`](../taxonomy.md#coupling)

objective
:   `modify` — defines a different operator; fidelity to softmax is not the goal

class
:   `attnbank.ESPFormerAttention` &middot; [`attnbank/methods/coupling/espformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/espformer.py)

monograph
:   `sec:ot`

</div>

## What it changes

<div class="attn-changes" markdown="1">

Every category in the bank changes one object in the reference operator $y(q)=\int v\,\mathrm{d}\pi_q$; this method changes the **normalisation**. $Z(q)$ stops being a per-query constant and becomes a global constraint, so the queries are no longer independent of one another.

It pursues **modification**: it defines a *different* operator. Asking how closely it approximates softmax is asking the wrong question — it is not trying to.

</div>

## What it declares

Each row is a commitment the conformance suite tests, not a description.

<div class="attn-decl" markdown="1">

| attribute | declared | what that commits it to |
|---|---|---|
| `supports_causal` | **no** | `causal=True` raises `CausalNotSupportedError` — a statement about the operator, not a missing feature. |
| `expresses_weights` | **yes** | it can name $\widehat\pi_q$ on the **original** $n$ positions, so it enters the error decomposition. |
| `row_stochastic` | **yes** | those weights are $\ge 0$ and sum to one: $\widehat\pi_q$ is a probability measure. |
| `head_independent` | **yes** | each head is computed from its own $K, V$. |
| `trainable` | **no** | no learned parameters — what you build is what the paper describes. |

</div>

## Correctness anchor

**None, and that is the declaration.** `objective = modify` means this is a different operator, not an approximation of softmax, so neither anchor applies. The conformance suite does not skip the check — it *asserts the exemption*, requiring `limit_config()` and `convergence_configs()` to both be `None` for a `modify` method.

## Cost model

Analytic, per `(batch, head)`. At $m = 1024$, $n = 1024$, $d = 64$, $dv = 64$ and `bytes_per_el=2`, against the reference operator at the same shape:

<div class="attn-tab" markdown="1">

| quantity | this method | `softmax` | ratio |
|---|---:|---:|---:|
| `flops` | 275,166,265,344 | 268,435,456 | 1.03e+03× |
| `state_bytes` | 262,144 | 262,144 | 1× |
| `bytes_moved` | 134,742,016 | 4,718,592 | 28.6× |

</div>

The numbers are the method's own `cost()`, evaluated — the declaration, not a measurement:

```python
def cost(self, m, n, d, dv, bytes_per_el=2):
    """What this file executes, not the quoted asymptotics.

    ``n_sl`` soft sorts (``m^2 + n^2`` each), ``n_sl`` triple products
    ``A^T (I B)`` (``mn(m + n)`` each), the cost matrix, and one ``m x n`` apply.
    The ``mn(m + n)`` term is the price of relaxing the sort: with hard
    permutations the same product is a relabelling in ``O(mn)`` and the total
    collapses to the paper's ``O(n_sl * n(n + d))``.
    """
    n_sl = self.n_slices if self.n_slices is not None else d
    return {"flops": 2 * n_sl * (m + n) * d + n_sl * (m * m + n * n)
                     + 2 * n_sl * m * n * (m + n) + 3 * m * n + 2 * m * n * dv,
            "state_bytes": n * (d + dv) * bytes_per_el,
            "bytes_moved": (m * d + n * (d + dv) + m * dv
                            + n_sl * m * n) * bytes_per_el}
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

**`reference` only, and that is the correct answer.** A fixed-point iteration over the whole matrix is not a kernel call, and the normalisation it computes is not per-query.

Declaring a backend is a claim that it equals the reference, checked by `tests/test_conformance.py::test_backend_equivalence`; declaring `sdpa` here would be pretending. See [Backends](../backends.md).

Diagnostics returned in `extras`: `col_marginal`, `gamma`, `marginal_error`, `n_slices`, `plan`, `row_marginal`, `row_sum_error`, `slice_cost`, `slice_weights`, `slices`, `target_col`, `target_row`.

## Configuration

<div class="attn-tab" markdown="1">

| parameter | default |
|---|---|
| `n_slices` | `None` |
| `tau` | `0.001` |
| `tau_slice` | `0.1` |
| `cost` | `'sqeuclidean'` |
| `seed` | `0` |

</div>

`n_slices`: `None` for the paper's axis-aligned `Theta = I` (`d` slices),
or an integer for that many random directions on the sphere.
`tau`: SoftSort temperature (the paper's `t = 1e-3`); `-> 0` is a hard sort.
`tau_slice`: inverse temperature of the slice weights (the paper's `tau = 0.1`),
applied to the cost of the unit-mass plan `U_theta`.  `0` averages the slices.
`cost`: `'sqeuclidean'` (Algorithm 1) or `'euclidean'` (the released code).

```python
ab.describe("espformer")     # the same keys, with everything else it declares
```

## Run it

```python
import torch, attnbank as ab

torch.manual_seed(0)
b, h, n, d = 1, 2, 64, 32
q, k, v = (torch.randn(b, h, n, d) for _ in range(3))

attn = ab.build("espformer", tau=0.001, tau_slice=0.1)
out = attn(q, k, v, causal=False,   # it declares supports_causal = False
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

<div class="attn-out" markdown="1">

```text
ESPFormerAttention(name='espformer', category='coupling', objective='modify', backend='reference', cost='sqeuclidean', n_slices=None, seed=0, tau=0.001, tau_slice=0.1)
AttentionOutput(y=[1, 2, 64, 32], weights=[1, 2, 64, 64], v_hat=None, extras=[col_marginal, gamma, marginal_error, n_slices, plan, row_marginal, row_sum_error, slice_cost, slice_weights, slices, target_col, target_row], dtype=float32, device=cpu)
invariant W: 0.0
row_stochastic = True | first row sums: [1.0, 1.0, 1.0, 1.0]
extras: ['col_marginal', 'gamma', 'marginal_error', 'n_slices', 'plan', 'row_marginal', 'row_sum_error', 'slice_cost', 'slice_weights', 'slices', 'target_col', 'target_row']
```

</div>

## In the paper's notation

<div class="attn-notation" markdown="1">

`sec:ot`, "Doubly stochastic attention in transformers".  Sinkformers reach the
column constraint by iterating `eq:ot-sinkhorn`; the iteration is sequential, and
its Hilbert-metric rate degrades exponentially in `||c||_inf / eps`.  ESPFormer
— Shahbazi et al. (2025), building on the expected sliced transport plan of Liu
et al. (2025) — reaches it in a *single* pass instead, by averaging one-dimensional
transport plans.

The construction, for one slice `theta` in `R^d`:

1. project the queries and the keys on `theta`:  `q_i^theta = <q_i, theta>`,
   `k_j^theta = <k_j, theta>`;
2. between two *uniform* measures on the line the optimal plan is the monotone
   rearrangement: sort both sides and match rank to rank.  With `A` and `B` the
   (descending) sorting permutations of the projected queries and keys,

```text
   U_theta = (1/m) A^T I B,
```

   where `I` is the `m x n` rank-to-rank coupling (`eq:ot-nwcorner` in spirit:
   the north-west-corner plan between the uniform marginals `1/m` and `1/n`, with
   rows rescaled to sum to one).  `I` *is* the identity when `m == n`, which is
   the form stated in the paper.  `U_theta` has row marginal `1/m` and column
   marginal `1/n`: it is a transport plan, not a row normalization;
3. the plans of the individual slices are averaged with weights that favour the
   cheap slices,

```text
   G = sum_l sigma_l U_{theta_l},
   sigma_l = softmax_l( -tau_slice * sum_ij C_ij [U_{theta_l}]_ij ),
   C_ij = ||beta q_i - beta k_j||^2 .
```

A convex combination of couplings in `Pi(nu, mubar)` is again in `Pi(nu, mubar)`,
so `G` is doubly stochastic exactly, with no iteration at all.  The attention
operator is `pi_hat = m G` (row-stochastic) and `y = pi_hat v`.

**Soft sorting.**  `argsort` has zero gradient almost everywhere, so `A` and `B`
are replaced by `SoftSort_t` (Prillo and Eisenschlos, 2020),
`SoftSort_t(x)[r, i] = softmax_i( -(x_i - sort(x)_r)^2 / t )`.  This is the one
place where the construction stops being exact: at finite `t` the relaxed `A`
is row-stochastic but not a permutation, so its *column* sums — and hence the row
sums of `G` — deviate from one.  The operator is renormalized here so that
`pi_hat` is a probability measure, and the deviation that was removed is reported
as `extras["row_sum_error"]`.  It goes to zero as `t -> 0`; the test suite checks
that it does.

**Two conventions, stated so that numbers transfer.**  The slice cost here is that of
the unit-mass plan `U_theta`, as in the paper; the released code omits the `1/n` and
so weights the slices with an inverse temperature larger by that factor (its default is
`10` against the paper's `0.1`).  And the paper's Algorithm 1 uses the *squared*
Euclidean cost while the released code uses the distance itself; both are available
through `cost=`, with the paper's as the default.  Neither choice changes the support
of the operator, only how sharply the cheap slices are preferred.  Under those two
conventions `extras["plan"]` reproduces the authors' released module bit for bit;
`tests/test_coupling_transport.py` writes out that computation and checks it.

**No key potential.**  `prop:ot-conditional` says the *entropic*-OT conditional is
softmax tilted by one number per key.  ESP's plan is a mixture of permutations, not
a Gibbs kernel `exp(s + f + g)`, so no such `psi` exists and `extras` deliberately
contains no `"potential"`: study S7's tilt analysis does not apply to it.  What it
does expose — `gamma`, `row_marginal`, `col_marginal`, `marginal_error` —
matches `sinkhorn`, so the marginal diagnostics are shared.

**Cost.**  Not sub-quadratic, and not claimed to be.  The paper reports
`O(N_sl * n(n + d))` for `N_sl` slices and credits the method with being *parallel*
over slices where Sinkhorn is sequential over iterations.  In this reference
implementation the dominant term is larger still: `A^T I B` is a product of dense
`m x m`, `m x n` and `n x n` matrices once the sort is relaxed, so forming the
`N_sl` plans costs `O(N_sl * mn(m + n))`.  A hard sort would make the product a
permutation composition and recover the quoted figure, at the price of the gradient
the soft sort exists to provide.  `cost()` reports what this file executes.

**Slices.**  The paper sets `Theta = I_{d x d}` — axis-aligned slices, so that the
mechanism adds no parameters at all; that is the default here (`n_slices=None`) and
the reason `trainable = False`.  Its Table 6 ablates *learnable* slicers, which help
when the number of slices is small; `n_slices=L` draws `L` directions uniformly on
the sphere through a seeded generator, which is that ablation's initialization but not
its training.

Not causal, for the reason every doubly stochastic operator is not: `prop:ot-causal`,
a lower-triangular matrix with equal row and column sums is diagonal.  Sorting is in
any case a global operation over the sequence.

</div>

## Related methods

changes the same object — [`coupling`](../taxonomy.md#coupling)
:   [`dustbin_sinkhorn`](dustbin_sinkhorn.md), [`lowrank_ot`](lowrank_ot.md), [`otke`](otke.md), [`sinkhorn`](sinkhorn.md), [`unbalanced_sinkhorn`](unbalanced_sinkhorn.md) &nbsp;·&nbsp; [browse the bank, coupling only](index.md?category=coupling)

judged the same way — `modify`, beyond the ones above
:   [`cosformer`](cosformer.md), [`decay_rnn`](decay_rnn.md), [`delta_rnn`](delta_rnn.md), [`gla`](gla.md), [`gqa`](gqa.md), [`hgrn2`](hgrn2.md), [`hyena`](hyena.md), [`isab`](isab.md), [`latent_stochastic`](latent_stochastic.md), [`latte`](latte.md), [`linear_elu`](linear_elu.md), [`linear_rnn`](linear_rnn.md), [`lru`](lru.md), [`mega`](mega.md), [`mla`](mla.md), [`mlstm`](mlstm.md), [`perceiver`](perceiver.md), [`polaformer`](polaformer.md), [`rebased`](rebased.md), [`retnet`](retnet.md), [`rwkv`](rwkv.md), [`s4d`](s4d.md), [`s5`](s5.md) &nbsp;·&nbsp; [browse the bank, modify only](index.md?objective=modify)

Both lists are `ab.available(...)` with `espformer` removed — the two axes the registry declares, nothing hand-picked. [Open the browser with `espformer` selected](index.md?compare=espformer) to set it beside up to three of them; the rows they agree on fold away.

## References

<div class="attn-refs" markdown="1">

- *ESPFormer: Doubly-Stochastic Attention with Expected Sliced Transport Plans.*<br><span class="attn-cite" data-bibkey="shahbazi2025espformer"><code>shahbazi2025espformer</code></span>
- *Expected Sliced Transport Plans.*<br><span class="attn-cite" data-bibkey="liu2025expected"><code>liu2025expected</code></span>
- *SoftSort: A Continuous Relaxation for the argsort Operator.*<br><span class="attn-cite" data-bibkey="prillo2020softsort"><code>prillo2020softsort</code></span>
- *Sinkformers: Transformers with Doubly Stochastic Attention.*<br><span class="attn-cite" data-bibkey="sander2022sinkformers"><code>sander2022sinkformers</code></span>

</div>

<details class="attn-bib no-copy">
<summary>BibTeX — 4 entries, verbatim from <code>attnbank/references.bib</code></summary>
<div class="attn-bib__entries">
<div class="attn-bib__entry" data-bibkey="shahbazi2025espformer">
<pre class="attn-bib__src"><code>@inproceedings{shahbazi2025espformer,
  title={{ESPFormer}: Doubly-Stochastic Attention with Expected Sliced Transport Plans}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="liu2025expected">
<pre class="attn-bib__src"><code>@inproceedings{liu2025expected,
  title={Expected Sliced Transport Plans}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="prillo2020softsort">
<pre class="attn-bib__src"><code>@inproceedings{prillo2020softsort,
  title={{SoftSort}: A Continuous Relaxation for the argsort Operator}</code></pre>
</div>
<div class="attn-bib__entry" data-bibkey="sander2022sinkformers">
<pre class="attn-bib__src"><code>@inproceedings{sander2022sinkformers,
  title         = {Sinkformers: Transformers with Doubly Stochastic Attention}</code></pre>
</div>
</div>
</details>

The bibliography is packaged with the library (`attnbank/references.bib`) and `tests/test_conformance.py` fails if a method cites a key that is not in it.

---

<div class="attn-colophon" markdown="1">

Generated by `docs/gen_methods.py` from the registry and from the module docstring of [`attnbank/methods/coupling/espformer.py`](https://github.com/Ashkan13776/attnbank/blob/main/attnbank/methods/coupling/espformer.py); the example above was executed to produce its output. Regenerate with `python docs/gen_methods.py`.

</div>
