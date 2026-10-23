# Random features: the error that shrinks, and the error that does not

FAVOR+ (`performer`) estimates `exp(<q, k>)` with `r` positive random features.
More features, less error -- and the conformance suite checks that this is monotone,
because for a Monte-Carlo method "reduces to softmax at some finite setting" is not
available as a correctness anchor.

`linear_elu` uses a *fixed* feature map instead. It does not converge to softmax at
any `r`, and it is not trying to: it declares `objective = modify`. Watch the two
columns separate.

## The script

```python title="examples/05_random_features_converge.py"
--8<-- "examples/05_random_features_converge.py"
```

## Output

```text
--8<-- "docs/examples/output/05_random_features_converge.txt"
```
