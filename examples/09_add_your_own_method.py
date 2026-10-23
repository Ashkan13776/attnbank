"""
Adding a method: declare, implement, conform
============================================

Three steps. Subclass :class:`attnbank.AttentionMethod`, declare what the method *is*,
implement ``_forward``. The declaration is not documentation -- the conformance suite
reads it and tests the method against it, so a wrong declaration is a failing build.

The method below keeps the ``k`` largest weights *and* every position in the first
``s`` sinks, a sort of hand-rolled StreamingLLM. It is registered under a name that
will not collide with the shipped bank.
"""
import torch

import attnbank as ab

torch.manual_seed(0)


@ab.register
class SinkTopK(ab.AttentionMethod):
    """Top-k on the true scores, with the first ``sinks`` positions always retained."""

    name = "example_sink_topk"
    category = ab.Category.RESTRICT      # it changes the support of pi_q
    objective = ab.Objective.APPROXIMATE  # it targets dense softmax
    supports_causal = True
    expresses_weights = True              # it can name pi_hat on the original n
    row_stochastic = True                 # ... and those weights are a distribution
    trainable = False
    references = ("xiao2024efficient",)
    paper_section = "sec:sparse"

    def __init__(self, k: int | None = 8, sinks: int = 2, **kw):
        super().__init__(k=k, sinks=sinks, **kw)
        self.k, self.sinks = k, sinks

    def _forward(self, q, k, v, causal, return_weights):
        pi = ab.masked_softmax(ab.scores(q, k, self._scale(q.shape[-1]), causal))
        n = k.shape[-2]
        keep = torch.zeros_like(pi, dtype=torch.bool)
        keep[..., :min(self.sinks, n)] = True
        if self.k is not None:
            idx = pi.topk(min(self.k, n), dim=-1).indices
            keep.scatter_(-1, idx, True)
        else:
            keep |= pi > 0
        pi_hat = ab.renormalize(pi * keep)
        return ab.AttentionOutput(y=pi_hat @ v,
                                  weights=pi_hat if return_weights else None,
                                  extras={"omitted_mass": (pi * ~keep).sum(-1)})

    def cost(self, m, n, d, dv, bytes_per_el=2):
        kk = (self.k or n) + self.sinks
        return {"flops": 2 * m * n * d + 2 * m * kk * dv,
                "state_bytes": n * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + n * d + m * kk * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        # k = None keeps every visible key, so the method reduces to dense softmax.
        return {"k": None}


print(ab.describe("example_sink_topk"))
print()
print(ab.check_conformance("example_sink_topk"))
print()
print("It is now a first-class member of the bank:")
print("  ", ab.available("restrict"))
print("  ", ab.compare(["topk", "example_sink_topk"],
                       *(torch.randn(1, 2, 32, 16) for _ in range(3)), causal=True,
                       configs={"topk": {"k": 4}, "example_sink_topk": {"k": 4}})[1])
print("\nThe full eleven-point suite runs it automatically, because it is parametrized")
print("over the registry: `pytest tests/test_conformance.py`.")
