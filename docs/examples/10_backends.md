# Backends: the operator stays, the execution path changes

A method is two things: an *operator* (the mathematics, one readable file) and an
*execution path*. The operator never changes with the backend -- and that is a tested
claim, not a slogan.

Most of the bank declares `reference` only, and that is the correct answer rather
than an omission: an operator that is not a dense softmax over a mask, or whose
`extras` need the full score matrix anyway, has no kernel to dispatch to.

## The script

```python title="examples/10_backends.py"
--8<-- "examples/10_backends.py"
```

## Output

```text
--8<-- "docs/examples/output/10_backends.txt"
```
