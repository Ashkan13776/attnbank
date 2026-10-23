"""
Invariant W: every method is a measure times a value field
==========================================================

``y == weights @ (v_hat if v_hat is not None else value)`` holds for every registered
method -- that is the claim the library is built on, and here it is, checked over the
whole bank in float64.

A method that genuinely cannot express its measure on the original ``n`` positions sets
``expresses_weights = False`` and returns ``weights=None``; it is then excluded from the
error decomposition *by declaration*, which is visible in the output below.
"""
import torch

import attnbank as ab

torch.manual_seed(0)
q, k, v = (torch.randn(1, 2, 24, 16, dtype=torch.float64) for _ in range(3))

print(f"{'method':<18}{'weights':>9}{'v_hat':>8}{'max |y - W v_hat|':>20}")
for name in ab.available():
    out = ab.build(name)(q, k, v, return_weights=True)
    dev = ab.check_invariant(out, v)             # raises if the invariant is violated
    print(f"{name:<18}"
          f"{'yes' if out.weights is not None else '-':>9}"
          f"{'yes' if out.v_hat is not None else '-':>8}"
          f"{dev:>20.2e}")

print("\nnot a single failure -- otherwise check_invariant would have raised.")
print("The same property is asserted in CI for every method, in float32 and float64:")
print("  pytest tests/test_conformance.py -k invariant")
