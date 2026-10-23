"""
Random features: the error that shrinks, and the error that does not
====================================================================

FAVOR+ (``performer``) estimates ``exp(<q, k>)`` with ``r`` positive random features.
More features, less error -- and the conformance suite checks that this is monotone,
because for a Monte-Carlo method "reduces to softmax at some finite setting" is not
available as a correctness anchor.

``linear_elu`` uses a *fixed* feature map instead. It does not converge to softmax at
any ``r``, and it is not trying to: it declares ``objective = modify``. Watch the two
columns separate.
"""
import torch

import attnbank as ab

torch.manual_seed(0)
q, k, v = (0.5 * torch.randn(1, 2, 64, 32, dtype=torch.float64) for _ in range(3))
ref = ab.build("softmax")(q, k, v).y

fixed = (ab.build("linear_elu")(q, k, v).y - ref).norm(dim=-1).mean()

print(f"{'r':>6}{'performer (approximate)':>26}{'linear_elu (modify)':>22}")
for r in (8, 32, 128, 512, 2048):
    err = torch.stack([(ab.build("performer", r=r, seed=s)(q, k, v).y - ref).norm(dim=-1).mean()
                       for s in range(4)]).mean()
    print(f"{r:>6}{err:>26.4f}{fixed:>22.4f}")

print("\nThe left column is an approximation converging; the right one is a different")
print("operator, measured against a target it never claimed. ab.describe('linear_elu')")
print("says so in the 'objective' field, and the suite exempts it from the anchor.")
