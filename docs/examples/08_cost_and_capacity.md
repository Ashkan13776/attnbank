# What a method actually saves

<p class="gal-meta"><code>examples/08_cost_and_capacity.py</code> &middot; <a href="https://github.com/Ashkan13776/attnbank/blob/main/examples/08_cost_and_capacity.py">source on GitHub</a></p>

Every method declares an analytic cost model: `flops`, `state_bytes` (the resident
KV cache) and `bytes_moved`. Decoding is bandwidth-bound, so `bytes_moved` -- not
`flops` -- is the number that predicts latency, and `state_bytes` is the number
that decides whether the context fits at all.

The conformance suite asserts that any method claiming to save capacity really does
have `state_bytes <= softmax` at a long context. Here is that table.

## The script

```{.python title="examples/08_cost_and_capacity.py"}
--8<-- "examples/08_cost_and_capacity.py:13:36"
```

## Output

<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code> ran the script and stored what it printed, and <code>tests/test_examples.py</code> runs it again in CI.</p>

```{.text .gal-stdout title="$ python examples/08_cost_and_capacity.py"}
--8<-- "docs/examples/output/08_cost_and_capacity.txt"
```
