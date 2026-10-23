# Top-k is not output-optimal

Selecting the `k` largest attention weights minimises the *omitted mass*
`eps_q = pi_q(S^c)`. It does not minimise `||y - y_hat||`, because the bound is
`eps_q * osc(v)` and a light token carrying an extreme value can matter more than a
heavy token sitting near the mean.

`value_topk` is the output-aware oracle: same budget, chosen to minimise output error
instead. The gap between the two lines below is the cost of selecting on scores alone.

## The script

```python title="examples/04_topk_is_not_output_optimal.py"
--8<-- "examples/04_topk_is_not_output_optimal.py"
```

## Output

```text
--8<-- "docs/examples/output/04_topk_is_not_output_optimal.txt"
```
