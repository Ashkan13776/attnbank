"""
Backends: the operator stays, the execution path changes
========================================================

A method is two things: an *operator* (the mathematics, one readable file) and an
*execution path*. The operator never changes with the backend -- and that is a tested
claim, not a slogan.

Most of the bank declares ``reference`` only, and that is the correct answer rather
than an omission: an operator that is not a dense softmax over a mask, or whose
``extras`` need the full score matrix anyway, has no kernel to dispatch to.
"""
import warnings

import torch

import attnbank as ab

print("WHAT THIS MACHINE CAN RUN\n")
print(ab.backends())

print("\n\nWHO DECLARES WHAT\n")
matrix = ab.backend_matrix()
fast = {n: b for n, b in matrix.items() if len(b) > 1}
for name, declared in fast.items():
    print(f"  {name:<12}{', '.join(declared)}")
only = [n for n, b in matrix.items() if len(b) == 1]
print(f"\n  reference only ({len(only)}): {', '.join(only)}")
print("\n  Not a gap in the library -- a fact about the operators. `nmf_oracle` is an")
print("  oracle factorisation, `merge_kv` is a clustering step, `topk` needs the full")
print("  score matrix for the omitted mass it reports. None of them is a kernel call.")

print("\n\nEQUALITY, ON DEMAND\n")
torch.manual_seed(0)
q, k, v = (torch.randn(2, 4, 64, 32, dtype=torch.float64) for _ in range(3))
for name in fast:
    ref = ab.build(name, backend="reference")(q, k, v, causal=True).y
    for backend in matrix[name][1:]:
        if not ab.is_backend_available(backend):
            info = ab.backend_info(backend)
            print(f"  {name:<10}{backend:<12}skipped: {info.reason}")
            continue
        got = ab.build(name, backend=backend)(q, k, v, causal=True).y
        print(f"  {name:<10}{backend:<12}max |y - y_reference| = "
              f"{float((got - ref).abs().max()):.2e}")

print("\n\nNOTHING IS SILENT\n")
try:
    ab.build("topk", backend="sdpa")
except ab.BackendNotSupportedError as e:
    print("  explicit request for an undeclared backend ->", type(e).__name__)
    print("   ", str(e)[:150], "...")

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ab.build("softmax", backend="sdpa")(q, k, v, return_weights=True)
print("\n  asking a fused kernel for the attention weights ->",
      caught[0].category.__name__ if caught else "no warning")
print("   ", str(caught[0].message)[:150] if caught else "")

print("\n  The default is 'reference', and the eight studies pin it explicitly, so no")
print("  published number can move because a faster path appeared:",
      repr(ab.get_default_backend()))
