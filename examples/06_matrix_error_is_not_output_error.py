"""
Attention-matrix error is not output error
==========================================

Most of the literature reports ``||P - P_hat||_F``. The quantity a model actually feels
is ``||(P - P_hat) V||``. They are not the same ranking: the Frobenius norm charges a
method for mass it moved between two tokens with nearly identical values, which costs
the output nothing.

Rank the same methods by both and watch the order change.
"""
import torch

import attnbank as ab
from attnbank import metrics

torch.manual_seed(0)
q, k, v = (torch.randn(1, 4, 128, 64) for _ in range(3))

rows = ab.compare(["topk", "window", "sink_window", "blocktopk", "performer", "nystrom"],
                  q, k, v,
                  configs={"topk": {"k": 16}, "window": {"w": 16},
                           "sink_window": {"sinks": 4, "w": 12},
                           "blocktopk": {"block": 8, "n_blocks": 2},
                           "performer": {"r": 64}, "nystrom": {"r": 16}})

by_fro = [r["name"] for r in sorted(rows, key=lambda r: r["matrix_fro"])]
by_out = [r["name"] for r in sorted(rows, key=lambda r: r["output_error"])]

print(f"{'method':<14}{'||P-P_hat||_F':>15}{'||(P-P_hat)V||':>16}{'output err':>12}")
for r in sorted(rows, key=lambda r: r["output_error"]):
    print(f"{r['name']:<14}{r['matrix_fro']:>15.4f}{r['matrix_AV_fro']:>16.4f}"
          f"{r['output_error']:>12.4f}")

print(f"\nranked by Frobenius: {' < '.join(by_fro)}")
print(f"ranked by output:    {' < '.join(by_out)}")
print("\nSpearman between the two rankings is what study S1 reports across configurations;")
print("`metrics.spearman` computes it, and `matrix_AV_fro` is the matrix-level norm that")
print("does control the output.")
