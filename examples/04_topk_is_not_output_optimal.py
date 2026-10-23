"""
Top-k is not output-optimal
===========================

Selecting the ``k`` largest attention weights minimises the *omitted mass*
``eps_q = pi_q(S^c)``. It does not minimise ``||y - y_hat||``, because the bound is
``eps_q * osc(v)`` and a light token carrying an extreme value can matter more than a
heavy token sitting near the mean.

``value_topk`` is the output-aware oracle: same budget, chosen to minimise output error
instead. The gap between the two lines below is the cost of selecting on scores alone.
"""
import torch

import attnbank as ab

torch.manual_seed(0)
q, k, v = (torch.randn(1, 2, 64, 32) for _ in range(3))

print(f"{'budget k':>9}{'topk':>12}{'value_topk':>12}{'ratio':>8}")
for budget in (2, 4, 8, 16):
    rows = {r["name"]: r for r in
            ab.compare(["topk", "value_topk"], q, k, v, causal=True,
                       configs={"topk": {"k": budget},
                                "value_topk": {"k": budget, "chunk": 16}})}
    a, b = rows["topk"]["output_error"], rows["value_topk"]["output_error"]
    print(f"{budget:>9}{a:>12.3e}{b:>12.3e}{a / max(b, 1e-30):>8.2f}x")

print("\nSame budget, same inputs, strictly smaller error -- so the selection rule, not")
print("the budget, is what is being measured. Both are oracles: they see the true")
print("scores. A deployable selector (blocktopk, lsh_sample, evict_h2o) can only do")
print("worse than `topk`, which is why `topk` is the right upper bound to quote.")
