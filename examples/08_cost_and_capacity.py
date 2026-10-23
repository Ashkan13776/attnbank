"""
What a method actually saves
============================

Every method declares an analytic cost model: ``flops``, ``state_bytes`` (the resident
KV cache) and ``bytes_moved``. Decoding is bandwidth-bound, so ``bytes_moved`` -- not
``flops`` -- is the number that predicts latency, and ``state_bytes`` is the number
that decides whether the context fits at all.

The conformance suite asserts that any method claiming to save capacity really does
have ``state_bytes <= softmax`` at a long context. Here is that table.
"""
import attnbank as ab

M, N, D, DV = 1, 4096, 128, 128          # one decode step against a 4k context
dense = ab.build("softmax").cost(M, N, D, DV)

print(f"decode step, n = {N}, d = dv = {D}, fp16\n")
print(f"{'method':<18}{'GFLOPs':>10}{'state MB':>11}{'vs dense':>10}{'bytes moved MB':>16}")
rows = [(name, ab.build(name).cost(M, N, D, DV)) for name in ab.available()]
for name, c in sorted(rows, key=lambda r: r[1]["state_bytes"]):
    print(f"{name:<18}{c['flops'] / 1e9:>10.4f}{c['state_bytes'] / 2**20:>11.2f}"
          f"{c['state_bytes'] / dense['state_bytes']:>10.2f}"
          f"{c['bytes_moved'] / 2**20:>16.2f}")

print("""
The top of the table is the set of methods whose resident state does not grow with n:
windows and evictions keep a fixed number of tokens, recurrences keep a fixed-size
matrix, factorisations keep r latents. That is the whole content of 'finite state' --
and prop:memory-rank is the price: recall is bounded by what the state can hold
(study S6).

Note how little `flops` tells you. `blocktopk` has the fewest and one of the largest
caches; `softmax` and `flash` have identical cost models, because flash changes the
evaluation and not the operator. At decode it is `bytes_moved` that sets the clock
(eq:exact-decode-intensity, study S8).""")
