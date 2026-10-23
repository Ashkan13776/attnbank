"""Hedgehog: a feature map *fitted* to the softmax attention it is replacing.

``sec:kernel-det``, ``sec:kernel-limits``.  Every other entry in this directory picks
``phi`` in advance -- ``exp(omega^T u - ||u||^2/2)``, ``elu + 1``, ``relu``, ``x^2``,
``1 + z + z^2/2`` -- and then asks how close the induced ``pi_hat`` lands.  Zhang et
al. (2024) invert the question: fix the separable *form*, learn the map.  Their
diagnosis is that linear attentions fail on two measurable axes, spikiness and
monotonicity of the induced weights, and their feature map is built to restore both,

    phi_W(u) = softmax([ W u + b ; -(W u + b) ]) in R^{2r},

-- an exponential map (spiky) run through a softmax over the feature axis (bounded,
so it never overflows), with the negated copy appended so that the map is monotone in
``<q,k>`` in both directions.  Queries and keys get their own ``(W, b)``.  The
parameters are then obtained by *distillation*: the paper's Eq. 7 minimises the
cross-entropy between the induced measure and the true one,

    L = - sum_i sum_j pi_q_i(j) log pi_hat_q_i(j),

which is the KL from softmax to ``pi_hat`` up to a constant, so the fit is an explicit
projection of softmax onto the rank-``2r`` separable family.

This file performs that projection **inside the forward pass**, per ``(batch, head)``,
on the tensors it is given, with Adam in float64 from a seeded initialization.  It is
therefore an *oracle* in the sense of ``nmf_oracle`` and ``svd_oracle``: it reports
how well the best feature map of width ``2r`` that this objective can find does, which
upper-bounds what a deployed Hedgehog achieves after distillation on a corpus.  Two
consequences are declared rather than hidden.  ``trainable = True`` -- the defaults are
not a method, they are an initialization plus a fitting budget.  And
``supports_causal = False``: the fitted ``W`` depends on the whole attention matrix,
so an early query's output would move when a later key changed.  A deployed Hedgehog
distills once and then runs causally; this one cannot, and saying so is cheaper than
a mask that does not mean what it says.

The anchor is ``convergence_configs`` over ``steps``.  More optimization lowers the
cross-entropy, which lowers the KL, which lowers the output error -- monotonically in
the mean, and towards softmax, which no finite budget reaches because the family has
rank ``2r``.  ``steps = 0`` is the untrained random feature map and is the worst point
of the sequence by construction.

Reference: Zhang et al. (2024) Hedgehog; Katharopoulos et al. (2020) for the
normalized separable readout.
"""
from __future__ import annotations

from typing import Optional

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    make_generator,
    masked_softmax,
    normalize_nonnegative,
    scores,
)
from ...registry import register


def hedgehog_features(x, w, b):
    """``phi_W(x) = softmax([Wx + b, -(Wx + b)])``: ``[..., t, d] -> [..., t, 2r]``."""
    a = torch.einsum("...td,...rd->...tr", x, w) + b.unsqueeze(-2)
    return torch.softmax(torch.cat([a, -a], dim=-1), dim=-1)


def distill(q, k, target, r: int, steps: int, lr: float, seed: int):
    """Fit ``(W_q, b_q, W_k, b_k)`` to ``target`` by the cross-entropy of Eq. 7.

    One independent problem per ``(batch, head)``: the loss is a sum over leading
    axes and Adam's update is elementwise, so no batch or head sees another's data.
    Everything runs in float64 so that the fit -- and with it the operator -- does not
    depend on the precision the caller happened to use.
    """
    d = q.shape[-1]
    shape = (*q.shape[:-2], r, d)
    zeros = lambda: torch.zeros(*q.shape[:-2], r, dtype=torch.float64, device=q.device)
    draw = lambda off: (torch.randn(shape, generator=make_generator(seed + off),
                                    dtype=torch.float64) / d ** 0.5).to(q.device)
    params = [draw(0), zeros(), draw(1), zeros()]
    if steps > 0:
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.Adam(params, lr=lr)
        tiny = torch.finfo(torch.float64).tiny
        with torch.enable_grad():
            for _ in range(steps):
                pq = hedgehog_features(q, params[0], params[1])
                pk = hedgehog_features(k, params[2], params[3])
                g_hat = pq @ pk.transpose(-1, -2)
                log_pi = (torch.log(g_hat.clamp_min(tiny))
                          - torch.log(g_hat.sum(-1, keepdim=True).clamp_min(tiny)))
                loss = -(target * log_pi).sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
    return [p.detach() for p in params]


@register
class HedgehogAttention(AttentionMethod):
    """Separable feature map distilled against the softmax weights it replaces."""

    name = "hedgehog"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = False          # the fit sees the whole attention matrix
    expresses_weights = True
    row_stochastic = True            # phi > 0 (a softmax), so g_hat > 0
    trainable = True
    references = ("zhang2024hedgehog", "katharopoulos2020transformers")
    paper_section = "sec:kernel"

    def __init__(self, r: Optional[int] = None, steps: int = 100, lr: float = 0.05,
                 seed: int = 0, **kw):
        """``r`` feature directions (``2r`` features); ``None`` means the head dim ``d``."""
        super().__init__(r=r, steps=steps, lr=lr, seed=seed, **kw)
        self.r, self.steps, self.lr, self.seed = r, steps, lr, seed

    def _forward(self, q, k, v, causal, return_weights):
        d = q.shape[-1]
        r = self.r or d
        q64, k64 = q.detach().double(), k.detach().double()
        target = masked_softmax(scores(q64, k64, self._scale(d), causal=False))
        wq, bq, wk, bk = distill(q64, k64, target, r, self.steps, self.lr, self.seed)

        pq = hedgehog_features(q64, wq, bq).to(q.dtype)
        pk = hedgehog_features(k64, wk, bk).to(q.dtype)
        g_hat = pq @ pk.transpose(-1, -2)
        pi, norm = normalize_nonnegative(g_hat)
        log_pi = pi.double().clamp_min(torch.finfo(torch.float64).tiny).log()
        cross_entropy = -(target * log_pi).sum(-1)       # the objective, Eq. 7
        extras = {"g_hat": g_hat, "Z_hat": norm, "phi_q": pq, "phi_k": pk,
                  "cross_entropy": cross_entropy.to(q.dtype), "r": r,
                  "steps": self.steps}
        return AttentionOutput(y=pi @ v, weights=pi if return_weights else None,
                               extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = 2 * (self.r or d)
        return {"flops": 2 * (m + n) * r * d + 2 * n * r * dv + 2 * m * r * dv,
                "state_bytes": r * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv + r * (dv + 1)) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return None        # the family has rank 2r; no budget makes it softmax

    @classmethod
    def convergence_configs(cls):
        return [{"steps": 0}, {"steps": 20}, {"steps": 80}, {"steps": 320}]
