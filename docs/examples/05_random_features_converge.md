# Random features: the error that shrinks, and the error that does not

<p class="gal-meta"><code>examples/05_random_features_converge.py</code> &middot; <a href="https://github.com/Ashkan13776/attnbank/blob/main/examples/05_random_features_converge.py">source on GitHub</a></p>

FAVOR+ (`performer`) estimates `exp(<q, k>)` with `r` positive random features.
More features, less error -- and the conformance suite checks that this is monotone,
because for a Monte-Carlo method "reduces to softmax at some finite setting" is not
available as a correctness anchor.

`linear_elu` uses a *fixed* feature map instead. It does not converge to softmax at
any `r`, and it is not trying to: it declares `objective = modify`. Watch the two
columns separate.

## The script

```{.python title="examples/05_random_features_converge.py"}
--8<-- "examples/05_random_features_converge.py:14:32"
```

## Output

<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code> ran the script and stored what it printed, and <code>tests/test_examples.py</code> runs it again in CI.</p>

```{.text .gal-stdout title="$ python examples/05_random_features_converge.py"}
--8<-- "docs/examples/output/05_random_features_converge.txt"
```
