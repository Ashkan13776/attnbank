# Splitting output error into its two causes

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

```python title="examples/03_error_decomposition.py"
--8<-- "examples/03_error_decomposition.py"
```

## Output

```text
--8<-- "docs/examples/output/03_error_decomposition.txt"
```
