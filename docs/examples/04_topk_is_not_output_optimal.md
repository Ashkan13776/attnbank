# Top-k is not output-optimal

<p class="gal-meta"><code>examples/04_topk_is_not_output_optimal.py</code> &middot; <a href="https://github.com/Ashkan13776/attnbank/blob/main/examples/04_topk_is_not_output_optimal.py">source on GitHub</a></p>

Selecting the `k` largest attention weights minimises the *omitted mass*
`eps_q = pi_q(S^c)`. It does not minimise `||y - y_hat||`, because the bound is
`eps_q * osc(v)` and a light token carrying an extreme value can matter more than a
heavy token sitting near the mean.

`value_topk` is the output-aware oracle: same budget, chosen to minimise output error
instead. The gap between the two lines below is the cost of selecting on scores alone.

## The script

```{.python title="examples/04_topk_is_not_output_optimal.py"}
--8<-- "examples/04_topk_is_not_output_optimal.py:13:32"
```

## Output

<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code> ran the script and stored what it printed, and <code>tests/test_examples.py</code> runs it again in CI.</p>

```{.text .gal-stdout title="$ python examples/04_topk_is_not_output_optimal.py"}
--8<-- "docs/examples/output/04_topk_is_not_output_optimal.txt"
```
