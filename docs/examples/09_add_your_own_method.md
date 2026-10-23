# Adding a method: declare, implement, conform

<p class="gal-meta"><code>examples/09_add_your_own_method.py</code> &middot; <a href="https://github.com/Ashkan13776/attnbank/blob/main/examples/09_add_your_own_method.py">source on GitHub</a></p>

Three steps. Subclass `attnbank.AttentionMethod`, declare what the method *is*,
implement `_forward`. The declaration is not documentation -- the conformance suite
reads it and tests the method against it, so a wrong declaration is a failing build.

The method below keeps the `k` largest weights *and* every position in the first
`s` sinks, a sort of hand-rolled StreamingLLM. It is registered under a name that
will not collide with the shipped bank.

## The script

```{.python title="examples/09_add_your_own_method.py"}
--8<-- "examples/09_add_your_own_method.py:13:75"
```

## Output

<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code> ran the script and stored what it printed, and <code>tests/test_examples.py</code> runs it again in CI.</p>

```{.text .gal-stdout title="$ python examples/09_add_your_own_method.py"}
--8<-- "docs/examples/output/09_add_your_own_method.txt"
```
