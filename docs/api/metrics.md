# Metrics

Error metrics matching §4 of the monograph. All of them take a reference measure
`pi [.., m, n]`, an approximate `pi_hat [.., m, n]`, values `v [.., n, dv]` and
optionally `v_hat`, and reduce per query to shape `[.., m]`.

A function named `*_bound` returns an **upper bound** on a quantity, and
`tests/test_metrics.py` checks over randomised inputs that it really bounds it.
Everything else is an identity or a definition.

::: attnbank.metrics
