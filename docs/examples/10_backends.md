# Backends: the operator stays, the execution path changes

<p class="gal-meta"><code>examples/10_backends.py</code> &middot; <a href="https://github.com/Ashkan13776/attnbank/blob/main/examples/10_backends.py">source on GitHub</a></p>

A method is two things: an *operator* (the mathematics, one readable file) and an
*execution path*. The operator never changes with the backend -- and that is a tested
claim, not a slogan.

Most of the bank declares `reference` only, and that is the correct answer rather
than an omission: an operator that is not a dense softmax over a mask, or whose
`extras` need the full score matrix anyway, has no kernel to dispatch to.

## The script

```{.python title="examples/10_backends.py"}
--8<-- "examples/10_backends.py:13:63"
```

## Output

<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code> ran the script and stored what it printed, and <code>tests/test_examples.py</code> runs it again in CI.</p>

```{.text .gal-stdout title="$ python examples/10_backends.py"}
--8<-- "docs/examples/output/10_backends.txt"
```
