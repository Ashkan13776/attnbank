#!/usr/bin/env python3
"""Regenerate every number in the error-decomposition block of docs/index.md.

The page states three error splits and a closure residual, and both used to be
unreproducible. The splits -- 1.03 / 2.76 / 0.72 -- came out of no seed or
shape anyone could find. The closure, "8.5e-15 over 1.78 million query rows",
paired a real figure from study S4 with a row count from a different study.
Those are exactly the kind of claim this library exists to make checkable, so
they are computed here and the page quotes this script.

    python tools/landing_figures.py

The snippet printed on the page is the first half of this file verbatim: it
carries its own seed and shape, so a reader can paste it and get the table.
"""
import torch
import attnbank as ab
from attnbank import metrics

# ---- the snippet shown on the page --------------------------------------
torch.manual_seed(0)
q, k, v = (torch.randn(1, 4, 1024, 64) for _ in range(3))

rows = ab.compare(["topk", "quantkv", "performer"], q, k, v,
                  configs={"topk": {"k": 87},
                           "quantkv": {"bits_k": 16, "bits_v": 1},
                           "performer": {"r": 64}})

for r in rows:
    print(f'{r["name"]:<10} {r["category"]:<12} '
          f'out={r["output_error"]:.4f}  dist={r["distributional_error"]:.4f}  '
          f'val={r["value_error"]:.4f}  expr_w={r.get("expresses_weights", "-")}')

# ---- the closure check ---------------------------------------------------
# y - y_hat == (pi - pi_hat) v + pi_hat (v - v_hat), exactly, for every row.
ref = ab.build("softmax")(q, k, v, return_weights=True)
worst, nrows = 0.0, 0
for name, cfg in [("topk", {"k": 87}),
                  ("quantkv", {"bits_k": 16, "bits_v": 1}),
                  ("performer", {"r": 64})]:
    out = ab.build(name, **cfg)(q, k, v, return_weights=True)
    vh = v if out.v_hat is None else out.v_hat
    d = metrics.error_decomposition(ref.weights.double(), out.weights.double(),
                                    v.double(), vh.double())
    actual = (ref.y.double() - out.y.double()).norm(dim=-1)
    resid = (d["total"] - actual).abs()
    worst = max(worst, float(resid.max()))
    nrows += resid.numel()
    print(f"{name:<10} closure max |total-actual| = {float(resid.max()):.3e} "
          f"over {resid.numel()} query rows")
print(f"\nWORST RESIDUAL {worst:.3e} over {nrows} query rows")


# ---- the closure sweep: the residual and the row count on the page ------
import torch
import attnbank as ab
from attnbank import metrics

torch.manual_seed(0)
B, H, n, d = 2, 4, 512, 32
q, k, v = (torch.randn(B, H, n, d, dtype=torch.float64) for _ in range(3))

ref = ab.build("softmax")(q, k, v, return_weights=True)
pi, y = ref.weights, ref.y

worst, nrows, nmeth, skipped = 0.0, 0, 0, []
for info in ab.list_methods():
    name = info["name"]
    if not info["expresses_weights"]:
        skipped.append((name, "expresses_weights=False"))
        continue
    try:
        out = ab.build(name)(q, k, v, return_weights=True)
    except Exception as e:  # unsupported shape / backend
        skipped.append((name, type(e).__name__))
        continue
    if out.weights is None:
        skipped.append((name, "weights=None"))
        continue
    vh = v if out.v_hat is None else out.v_hat
    dec = metrics.error_decomposition(pi, out.weights, v, vh)
    resid = (dec["total"] - (y - out.y).norm(dim=-1)).abs()
    worst = max(worst, float(resid.max()))
    nrows += resid.numel()
    nmeth += 1
    print(f"{name:<16} {float(resid.max()):.3e}", flush=True)

print(f"\nMETHODS {nmeth}  ROWS {nrows}  WORST {worst:.4e}")
print(f"SKIPPED {len(skipped)}: {skipped}")
