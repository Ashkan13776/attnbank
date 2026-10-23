# Finding your way around the bank

<p class="gal-meta"><code>examples/07_browse_the_bank.py</code> &middot; <a href="https://github.com/Ashkan13776/attnbank/blob/main/examples/07_browse_the_bank.py">source on GitHub</a></p>

Thirty-odd methods is too many to remember and too few to need a search engine. Four
calls cover it: `categories` for the taxonomy, `list_methods` for the table,
`available` for the names, `describe` for one method in full.

## The script

```{.python title="examples/07_browse_the_bank.py"}
--8<-- "examples/07_browse_the_bank.py:9:29"
```

## Output

<p class="gal-prov">Captured stdout, not prose: <code>docs/gen_gallery.py</code> ran the script and stored what it printed, and <code>tests/test_examples.py</code> runs it again in CI.</p>

```{.text .gal-stdout title="$ python examples/07_browse_the_bank.py"}
--8<-- "docs/examples/output/07_browse_the_bank.txt"
```
