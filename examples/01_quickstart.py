"""
Quickstart: one import, three calls
===================================

The whole library behind one name. ``attend`` for the one-liner, ``build`` for the
object you keep, ``compare`` for the numbers you report.
"""
import torch

import attnbank as ab

torch.manual_seed(0)

# [B, H, m, d] queries against [B, H, n, d] keys and [B, H, n, dv] values.
# Causal masking aligns the LAST query with the LAST key, so m <= n and the same
# code path serves prefill (m == n) and decode (m == 1).
q, k, v = (torch.randn(2, 4, 256, 64) for _ in range(3))

# 1. The one-liner. `k=32` is top-k's budget: the tensors are positional, so the
#    short name is free for the method's own configuration.
y = ab.attend("topk", q, k, v, causal=True, k=32)
print("attend      ", tuple(y.shape))

# 2. The object, when you want the weights, the compressed values or the diagnostics.
attn = ab.build("topk", k=32)
out = attn(q, k, v, causal=True, return_weights=True)
print("build       ", out)
print("extras      ", sorted(out.extras))

# 3. The comparison, scored against dense softmax.
rows = ab.compare(["softmax", "flash", "topk", "window", "performer"], q, k, v,
                  causal=True,
                  configs={"topk": {"k": 32}, "window": {"w": 32}, "performer": {"r": 128}})
print(f"\n{'method':<12}{'output err':>12}{'distributional':>16}{'value':>10}")
for r in rows:
    print(f"{r['name']:<12}{r['output_error']:>12.2e}"
          f"{r.get('distributional_error', float('nan')):>16.2e}"
          f"{r.get('value_error', float('nan')):>10.2e}")
