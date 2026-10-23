# Invariant W: every method is a measure times a value field

<p class="gal-meta"><code>examples/02_the_invariant.py</code> &middot; <a href="https://github.com/Ashkan13776/attnbank/blob/main/examples/02_the_invariant.py">source on GitHub</a></p>

`y == weights @ (v_hat if v_hat is not None else value)` holds for every registered
method -- that is the claim the library is built on, and here it is, checked over the
whole bank in float64.

A method that genuinely cannot express its measure on the original `n` positions sets
`expresses_weights = False` and returns `weights=None`; it is then excluded from the
error decomposition *by declaration*, which is visible in the output below.

## The script

```{.python title="examples/02_the_invariant.py"}
--8<-- "examples/02_the_invariant.py:13:31"
```

## Output

<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code> ran the script and stored what it printed, and <code>tests/test_examples.py</code> runs it again in CI.</p>

```{.text .gal-stdout title="$ python examples/02_the_invariant.py"}
--8<-- "docs/examples/output/02_the_invariant.txt"
```
