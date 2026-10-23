# Adding a method: declare, implement, conform

Three steps. Subclass `attnbank.AttentionMethod`, declare what the method *is*,
implement `_forward`. The declaration is not documentation -- the conformance suite
reads it and tests the method against it, so a wrong declaration is a failing build.

The method below keeps the `k` largest weights *and* every position in the first
`s` sinks, a sort of hand-rolled StreamingLLM. It is registered under a name that
will not collide with the shipped bank.

## The script

```python title="examples/09_add_your_own_method.py"
--8<-- "examples/09_add_your_own_method.py"
```

## Output

```text
--8<-- "docs/examples/output/09_add_your_own_method.txt"
```
