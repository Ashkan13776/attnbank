# Splitting output error into its two causes

<p class="gal-meta"><code>examples/03_error_decomposition.py</code> &middot; <a href="https://github.com/Ashkan13776/attnbank/blob/main/examples/03_error_decomposition.py">source on GitHub</a></p>

An approximation can be wrong in two ways: it attends to the wrong places, or it
aggregates the wrong values. `metrics.error_decomposition` separates them exactly:

```text
y - y_hat = (pi - pi_hat) v  +  pi_hat (v - v_hat)
            \_____________/    \________________/
            distributional          value
```

Quantising the KV cache moves error into the *value* term; restricting the support moves
it into the *distributional* term. Reading one number instead of two hides which.

## The script

```{.python title="examples/03_error_decomposition.py"}
--8<-- "examples/03_error_decomposition.py:15:45"
```

## Output

<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code> ran the script and stored what it printed, and <code>tests/test_examples.py</code> runs it again in CI.</p>

```{.text .gal-stdout title="$ python examples/03_error_decomposition.py"}
--8<-- "docs/examples/output/03_error_decomposition.txt"
```
