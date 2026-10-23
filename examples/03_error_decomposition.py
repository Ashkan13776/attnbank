"""
Splitting output error into its two causes
==========================================

An approximation can be wrong in two ways: it attends to the wrong places, or it
aggregates the wrong values. ``metrics.error_decomposition`` separates them exactly::

    y - y_hat = (pi - pi_hat) v  +  pi_hat (v - v_hat)
                \\_____________/    \\________________/
                distributional          value

Quantising the KV cache moves error into the *value* term; restricting the support moves
it into the *distributional* term. Reading one number instead of two hides which.
"""
import torch

from attnbank import build, metrics

torch.manual_seed(0)
q, k, v = (torch.randn(1, 4, 128, 64) for _ in range(3))

ref = build("softmax")(q, k, v, causal=True, return_weights=True)

print(f"{'method':<26}{'total':>10}{'distributional':>16}{'value':>10}")
for name, cfg in [("topk", {"k": 16}),
                  ("window", {"w": 16}),
                  ("quantkv", {"bits_k": 16, "bits_v": 2}),
                  ("quantkv", {"bits_k": 2, "bits_v": 16}),
                  ("mla", {"r": 16})]:
    out = build(name, **cfg)(q, k, v, causal=True, return_weights=True)
    d = metrics.error_decomposition(ref.weights, out.weights, v, out.v_hat)
    label = name + " " + ",".join(f"{key}={val}" for key, val in cfg.items())
    print(f"{label:<26}{d['total'].median():>10.3e}"
          f"{d['distributional'].median():>16.3e}{d['value'].median():>10.3e}")

print("""
Read the columns, not the totals. Restriction is purely distributional: it keeps the
values exactly and only moves mass. Quantising values is purely value error.
Quantising keys produces distributional error instead -- the same total can come from
two completely different mechanisms, and only the split tells you which.

Which of the two hurts more is a property of the activations, not of the method; on
real GPT-2 / Qwen2.5 activations it is the keys (study S4, prop:compress-sensitivity).
On the Gaussian inputs above the two are comparable, which is exactly why the paper
runs that study on captured activations rather than on torch.randn.""")
