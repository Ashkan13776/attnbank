# Attention-matrix error is not output error

<p class="gal-meta"><code>examples/06_matrix_error_is_not_output_error.py</code> &middot; <a href="https://github.com/Ashkan13776/attnbank/blob/main/examples/06_matrix_error_is_not_output_error.py">source on GitHub</a></p>

Most of the literature reports `||P - P_hat||_F`. The quantity a model actually feels
is `||(P - P_hat) V||`. They are not the same ranking: the Frobenius norm charges a
method for mass it moved between two tokens with nearly identical values, which costs
the output nothing.

Rank the same methods by both and watch the order change.

## The script

```{.python title="examples/06_matrix_error_is_not_output_error.py"}
--8<-- "examples/06_matrix_error_is_not_output_error.py:12:38"
```

## Output

<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code> ran the script and stored what it printed, and <code>tests/test_examples.py</code> runs it again in CI.</p>

```{.text .gal-stdout title="$ python examples/06_matrix_error_is_not_output_error.py"}
--8<-- "docs/examples/output/06_matrix_error_is_not_output_error.txt"
```
