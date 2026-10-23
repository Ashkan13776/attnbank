# Attention-matrix error is not output error

Most of the literature reports `||P - P_hat||_F`. The quantity a model actually feels
is `||(P - P_hat) V||`. They are not the same ranking: the Frobenius norm charges a
method for mass it moved between two tokens with nearly identical values, which costs
the output nothing.

Rank the same methods by both and watch the order change.

## The script

```python title="examples/06_matrix_error_is_not_output_error.py"
--8<-- "examples/06_matrix_error_is_not_output_error.py"
```

## Output

```text
--8<-- "docs/examples/output/06_matrix_error_is_not_output_error.txt"
```
