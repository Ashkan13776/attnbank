# Invariant W: every method is a measure times a value field

`y == weights @ (v_hat if v_hat is not None else value)` holds for every registered
method -- that is the claim the library is built on, and here it is, checked over the
whole bank in float64.

A method that genuinely cannot express its measure on the original `n` positions sets
`expresses_weights = False` and returns `weights=None`; it is then excluded from the
error decomposition *by declaration*, which is visible in the output below.

## The script

```python title="examples/02_the_invariant.py"
--8<-- "examples/02_the_invariant.py"
```

## Output

```text
--8<-- "docs/examples/output/02_the_invariant.txt"
```
