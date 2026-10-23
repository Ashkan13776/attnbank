# Examples

<p class="gal-lead">Short, self-contained scripts -- and the output under each one is real. <code>docs/gen_gallery.py</code> runs every script and stores its stdout next to the code, and CI runs all 10 again on each build, so a broken example is a failing build rather than a stale page.</p>

<p class="gal-facts"><span>10 scripts</span><span>419 lines of Python</span><span>0 outputs written by hand</span></p>

```bash
git clone https://github.com/Ashkan13776/attnbank
python attnbank/examples/01_quickstart.py
```

## Start here

<div class="gal-grid" markdown>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">01</span> [Quickstart: one import, three calls](01_quickstart.md)

The whole library behind one name. `attend` for the one-liner, `build` for the object you keep, `compare` for the numbers you report.

</div>

<pre class="gal-snip"><code>method        output err  distributional     value
softmax         0.00e+00        0.00e+00  0.00e+00
flash           2.47e-07        1.97e-07  0.00e+00
topk            7.20e-01        7.20e-01  0.00e+00
window          1.74e+00        1.74e+00  0.00e+00
performer       2.20e+00        2.20e+00  0.00e+00</code></pre>

</div>

</div>

## The foundation

<p class="gal-blurb">The invariant every method satisfies, and the error split it makes possible. Everything else in the gallery measures with these two.</p>

<div class="gal-grid" markdown>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">02</span> [Invariant W: every method is a measure times a value field](02_the_invariant.md)

`y == weights @ (v_hat if v_hat is not None else value)` holds for every registered method -- that is the claim the library is built on, and here it is, checked over the whole bank in float64.

</div>

<pre class="gal-snip"><code>method              weights   v_hat   max |y - W v_hat|
abc                     yes       -            4.44e-16
atom                    yes     yes            0.00e+00
based                   yes       -            0.00e+00
bigbird                 yes       -            0.00e+00
blocktopk               yes       -            0.00e+00
clustered               yes       -            0.00e+00
cosformer               yes       -            0.00e+00
decay_rnn               yes       -            0.00e+00</code></pre>

</div>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">03</span> [Splitting output error into its two causes](03_error_decomposition.md)

An approximation can be wrong in two ways: it attends to the wrong places, or it aggregates the wrong values. `metrics.error_decomposition` separates them exactly.

</div>

<pre class="gal-snip"><code>method                         total  distributional     value
topk k=16                  1.031e+00       1.031e+00 0.000e+00
window w=16                2.358e+00       2.358e+00 0.000e+00
quantkv bits_k=16,bits_v=2 7.239e-01       2.753e-05 7.239e-01
quantkv bits_k=2,bits_v=16 6.133e-01       6.133e-01 3.566e-05
mla r=16                   1.563e+00       1.150e+00 9.908e-01</code></pre>

</div>

</div>

## The results

<p class="gal-blurb">Not tutorials: each of these is a claim, with the run that backs it printed underneath.</p>

<div class="gal-grid" markdown>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">04</span> [Top-k is not output-optimal](04_topk_is_not_output_optimal.md)

Selecting the `k` largest attention weights minimises the *omitted mass* `eps_q = pi_q(S^c)`. It does not minimise `||y - y_hat||`, because the bound is `eps_q * osc(v)` and a light token carrying an extreme value can matter more than a heavy token sitting near the mean.

</div>

<pre class="gal-snip"><code> budget k        topk  value_topk   ratio
        2   2.982e+00   2.632e+00    1.13x
        4   1.841e+00   1.619e+00    1.14x
        8   9.851e-01   8.258e-01    1.19x
       16   3.530e-01   2.905e-01    1.22x</code></pre>

</div>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">05</span> [Random features: the error that shrinks, and the error that does not](05_random_features_converge.md)

FAVOR+ (`performer`) estimates `exp(<q, k>)` with `r` positive random features. More features, less error -- and the conformance suite checks that this is monotone, because for a Monte-Carlo method "reduces to softmax at some finite setting" is not available as a correctness anchor.

</div>

<pre class="gal-snip"><code>     r   performer (approximate)   linear_elu (modify)
     8                    0.2903                0.0828
    32                    0.1580                0.0828
   128                    0.1000                0.0828
   512                    0.0613                0.0828
  2048                    0.0317                0.0828</code></pre>

</div>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">06</span> [Attention-matrix error is not output error](06_matrix_error_is_not_output_error.md)

Most of the literature reports `||P - P_hat||_F`. The quantity a model actually feels is `||(P - P_hat) V||`. They are not the same ranking: the Frobenius norm charges a method for mass it moved between two tokens with nearly identical values, which costs the output nothing.

</div>

<pre class="gal-snip"><code>ranked by Frobenius: topk &lt; blocktopk &lt; performer &lt; sink_window &lt; window &lt; nystrom
ranked by output:    topk &lt; performer &lt; blocktopk &lt; sink_window &lt; window &lt; nystrom</code></pre>

</div>

</div>

## The bank in practice

<p class="gal-blurb">Finding a method, and reading what it costs.</p>

<div class="gal-grid" markdown>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">07</span> [Finding your way around the bank](07_browse_the_bank.md)

Thirty-odd methods is too many to remember and too few to need a search engine. Four calls cover it: `categories` for the taxonomy, `list_methods` for the table, `available` for the names, `describe` for one method in full.

</div>

<pre class="gal-snip"><code>THE TAXONOMY -- what each category changes in y(q) = int v dpi_q

  exact        nothing changes but the evaluation: tiling, chunking, precision
  interaction  the interaction g(q, x) is replaced by a cheaper surrogate
  restrict     the support of pi_q is restricted to a subset S_q of positions
  compress     the stored context (mu, k, v) is replaced by a smaller one
  factorize    pi_q is factored through r latents: pi_q = a(q, .) b(., x)
  coupling     normalisation stops being per-query; queries are coupled
  recurrent    the operator itself becomes a finite-state recurrence</code></pre>

</div>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">08</span> [What a method actually saves](08_cost_and_capacity.md)

Every method declares an analytic cost model: `flops`, `state_bytes` (the resident KV cache) and `bytes_moved`. Decoding is bandwidth-bound, so `bytes_moved` -- not `flops` -- is the number that predicts latency, and `state_bytes` is the number that decides whether the context fits at all.

</div>

<pre class="gal-snip"><code>decode step, n = 4096, d = dv = 128, fp16

method                GFLOPs   state MB  vs dense  bytes moved MB
lru                   0.0676       0.00      0.00            1.02
s5                    0.0676       0.00      0.00            1.02
isab                  0.0183       0.00      0.00            2.00
lowrank_ot            0.0087       0.00      0.00            2.00
perceiver             0.0197       0.00      0.00            2.00
shifted_window        0.0000       0.00      0.00            0.00</code></pre>

</div>

</div>

## The contract

<p class="gal-blurb">What the declarations buy: a new method is tested against what it declares, and changing the backend cannot move a result.</p>

<div class="gal-grid" markdown>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">09</span> [Adding a method: declare, implement, conform](09_add_your_own_method.md)

Three steps. Subclass `attnbank.AttentionMethod`, declare what the method *is*, implement `_forward`. The declaration is not documentation -- the conformance suite reads it and tests the method against it, so a wrong declaration is a failing build.

</div>

<pre class="gal-snip"><code>example_sink_topk: 9 checks passed, 0 failed
  [pass] shapes -- AttentionOutput(y=[2, 2, 6, 6], weights=[2, 2, 6, 12], v_hat=None, extras=[omitted_mass], dtype=float64, device=cpu)
  [pass] invariant W -- max |y - W v| = 0.00e+00
  [pass] row stochastic -- row sums in [1.000000, 1.000000]
  [pass] no causal leakage -- queries 0..3 unchanged
  [pass] finite at large logits -- q, k scaled by 10</code></pre>

</div>

<div class="gal-card" markdown>
<div class="gal-card-text" markdown>

### <span class="gal-num">10</span> [Backends: the operator stays, the execution path changes](10_backends.md)

A method is two things: an *operator* (the mathematics, one readable file) and an *execution path*. The operator never changes with the backend -- and that is a tested claim, not a slogan.

</div>

<pre class="gal-snip"><code>attnbank backends on this machine (torch 2.10.0+cu128, cuda yes)
----------------------------------------------------------------
reference   available    plain PyTorch; always available; the correctness anchor
            version: 2.10.0+cu128
sdpa        available    torch SDPA; CUDA dispatches to FlashAttention-2 or mem-efficient
            version: 2.10.0+cu128
flash_attn  unavailable  the flash-attn package; CUDA, fp16/bf16 only</code></pre>

</div>

</div>

## Run them yourself

Every example runs from a clone, with no arguments. The whole gallery:

```bash
pytest tests/test_examples.py -q   # execute every script, as CI does
python docs/gen_gallery.py         # re-run them and regenerate these pages
```
