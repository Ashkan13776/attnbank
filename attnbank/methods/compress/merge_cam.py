"""CaM: the token leaving the cache is merged into its neighbours instead of dropped.

``sec:compress-evict``, ``sec:compress-merge``, ``prop:compress-noeviction``.  CaM sits
on a StreamingLLM cache -- ``sinks`` attention sinks plus a recent window of
``recent = budget - sinks`` -- and asks what to do with the token that falls off the
back of the window.  Eviction throws it away.  CaM writes its value into the next
``merge_window`` entries first:

    rho_j   = pi(j) / mean_{j < i <= j + recent} pi(i)        (clamped to [0, 1])
    g_j     ~ Bernoulli(rho_j)
    v_hat_i += g_j v_hat_j / merge_window,   j < i <= j + merge_window

``rho_j`` is the *prominence* of the departing token relative to the window it is
leaving, so a token nobody looked at is usually dropped and a heavy hitter is usually
carried.  Because ``v_hat_j`` on the right is the already-merged value, a token that is
carried and then itself evicted cascades further forward; running the loop in
increasing ``j`` on a mutable value field reproduces that exactly, as the released code
does by mutating ``value_states`` in place.

**What CaM does and does not preserve.**  Keys are never merged, so the measure is a
plain renormalised restriction: ``extras['eps']`` is the mass CaM deletes, and setting
``merge_window = 0`` -- the same method with the repair switched off -- deletes it
bit for bit and leaves ``v`` alone.  What CaM repairs is
the *value field*: ``v_hat != v``, and the evicted values survive inside the retained
ones.  This is therefore **not** the pushforward of ``merge_kv``, where the mass itself
is carried and ``pi_hat`` picks up the ``log |C_i|`` gauge bias; it is a restriction
with a repaired ``v_hat``, and ``eq:error_decomposition`` splits it that way -- CaM
leaves the distributional term exactly where eviction left it and attacks the value
term alone.  Saying so is the point: the two repairs are different objects, and
``tests/test_compress_evict.py`` measures both.

**Semantics: eviction, with a destructive twist.**  The merge pass is the prefill: the
whole stream is ingested, the cache is reduced to ``budget``, and *every* query is then
served from that one reduced, rewritten cache -- which is why the method declares
``supports_causal = False``.  The twist is that the damage is worse than hiding in a
way no member of ``restrict/`` can imitate: ``sink_window`` leaves ``v`` untouched, so
a later pass can still read the original values, whereas CaM has overwritten them.
The cache cannot be un-merged.

``mode`` defaults to ``'expected'``, which uses ``rho_j`` itself as the gate -- the
conditional expectation of the released code's Bernoulli draw.  The bank's conformance
suite requires a method to agree between fp32 and fp64 and between CPU and CUDA, and a
coin flip on a threshold does not; ``mode='sample'`` reproduces the released sampler
through a seeded ``torch.Generator`` for anyone who wants the variance.

Exact limit: ``budget = None`` (nothing leaves the window, nothing is merged).

Reference: Zhang et al. (2024) CaM; Xiao et al. (2024) StreamingLLM;
Bolya and Hoffman (2023) ToMe.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from ...components.restriction import restrict
from ...core import (
    AttentionMethod,
    Category,
    Objective,
    make_generator,
    masked_softmax,
    scores,
)
from ...registry import register


def streaming_support(n: int, budget: int, sinks: int, device=None):
    """StreamingLLM's cache as a boolean ``[n]``: the first ``sinks`` plus the last rest."""
    sinks = max(0, min(sinks, budget - 1))
    recent = max(1, budget - sinks)
    idx = torch.arange(n, device=device)
    return (idx < sinks) | (idx >= n - recent), sinks, recent


def cluster_of(retained: torch.Tensor):
    """Map every position to the first retained position at or after it.

    Returns ``(assign, retained_idx)``: ``assign`` ``[n]`` holds cluster ids ``0..r-1``
    numbered along the retained positions, ``retained_idx`` ``[r]`` the positions
    themselves.  This is the hard surrogate of CaM's soft merge -- the dominant
    recipient of each evicted token -- and is what ``metrics.merge_bound`` needs.
    """
    n = int(retained.shape[0])
    retained_idx = retained.nonzero().flatten()
    ids = torch.cumsum(retained.long(), 0) - 1                   # id of each retained pos
    assign = torch.empty(n, dtype=torch.long, device=retained.device)
    nxt = int(ids[-1]) if bool(retained[-1]) else 0
    for j in range(n - 1, -1, -1):
        if bool(retained[j]):
            nxt = int(ids[j])
        assign[j] = nxt
    return assign, retained_idx


@register
class CaMAttention(AttentionMethod):
    """Streaming eviction whose departing values are merged forward, not discarded."""

    name = "merge_cam"
    category = Category.COMPRESS
    objective = Objective.APPROXIMATE
    supports_causal = False          # one rewritten cache serves every query
    expresses_weights = True
    row_stochastic = True
    references = ("zhang2024cam", "xiao2024efficient", "bolya2023token")
    paper_section = "sec:compress"

    def __init__(self, budget: int | None = 6, sinks: int = 2, merge_window: int = 4,
                 obs: int = 4, mode: str = "expected", seed: int = 0, **kw):
        """``budget`` = sinks + recent window; ``merge_window`` = CaM's ``merge_budget``."""
        super().__init__(budget=budget, sinks=sinks, merge_window=merge_window, obs=obs,
                         mode=mode, seed=seed, **kw)
        self.budget, self.sinks, self.merge_window = budget, sinks, merge_window
        self.obs, self.mode, self.seed = obs, mode, seed

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        pi = masked_softmax(scores(q, k, self._scale(d), causal))
        full = torch.ones(*q.shape[:-2], m, n, dtype=torch.bool, device=q.device)
        if self.budget is None or self.budget >= n:
            every = torch.arange(n, device=q.device)
            return restrict(pi, full, v, return_weights,
                            {"budget": self.budget, "assign": every,
                             "sizes": torch.ones_like(every), "retained_idx": every,
                             "cluster_radius": torch.zeros_like(k[..., 0]), "v_bar": v})

        retained, sinks, recent = streaming_support(n, self.budget, self.sinks, q.device)
        obs = max(1, min(self.obs, m))
        a = pi[..., -obs:, :].mean(-2)                                    # [..., n]

        # prominence of j against the window it is about to leave, and the gate
        rho = torch.zeros_like(a)
        for j in range(n):
            tail = a[..., j + 1:min(j + recent, n)]
            if tail.shape[-1] == 0:
                continue
            rho[..., j] = a[..., j] / tail.mean(-1).clamp_min(1e-30)
        rho = rho.clamp(0.0, 1.0)
        if self.mode == "sample":
            g = make_generator(self.seed)
            u = torch.rand(rho.shape, generator=g, dtype=torch.float64)
            gate = (u.to(rho.device) < rho.double()).to(rho.dtype)
        else:
            gate = rho

        # the merge pass: increasing j on a mutable field, so carries cascade forward
        v_hat = v.clone()
        for j in range(n):
            if bool(retained[j]):
                continue
            hi = min(j + 1 + self.merge_window, n)
            if hi <= j + 1:
                continue
            share = (gate[..., j, None] * v_hat[..., j, :]) / self.merge_window
            v_hat[..., j + 1:hi, :] = v_hat[..., j + 1:hi, :] + share.unsqueeze(-2)

        assign, retained_idx = cluster_of(retained)
        sizes = torch.bincount(assign, minlength=int(retained_idx.numel()))
        anchor = k.index_select(-2, retained_idx.index_select(0, assign))
        dev = (k - anchor).norm(dim=-1)                                   # [..., n]
        radius = torch.zeros(*k.shape[:-2], int(retained_idx.numel()),
                             dtype=k.dtype, device=k.device).scatter_reduce(
            -1, assign.expand(*k.shape[:-1]), dev, "amax", include_self=False)

        support = retained.expand(*q.shape[:-2], m, n)
        out = restrict(pi, support, v_hat, return_weights,
                       {"budget": self.budget, "sinks": sinks, "recent": recent,
                        "merge_window": self.merge_window, "prominence": rho,
                        "merge_gate": gate, "assign": assign, "sizes": sizes,
                        "cluster_radius": radius, "retained_idx": retained_idx,
                        "v_bar": v_hat.index_select(-2, retained_idx)})
        return replace(out, v_hat=v_hat)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        b = max(1, min(self.budget or n, n))
        return {"flops": 2 * m * b * (d + dv) + n * dv,
                "state_bytes": b * (d + dv) * bytes_per_el,
                "bytes_moved": (m * d + m * b * (d + dv) + m * dv) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return {"budget": None}
