# What a method actually saves

Every method declares an analytic cost model: `flops`, `state_bytes` (the resident
KV cache) and `bytes_moved`. Decoding is bandwidth-bound, so `bytes_moved` -- not
`flops` -- is the number that predicts latency, and `state_bytes` is the number
that decides whether the context fits at all.

The conformance suite asserts that any method claiming to save capacity really does
have `state_bytes <= softmax` at a long context. Here is that table.

## The script

```python title="examples/08_cost_and_capacity.py"
--8<-- "examples/08_cost_and_capacity.py"
```

## Output

```text
--8<-- "docs/examples/output/08_cost_and_capacity.txt"
```
