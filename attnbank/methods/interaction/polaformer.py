"""PolaFormer: polarity-aware linear attention, and the measure it cannot express.

``sec:kernel-det``.  Every nonnegative feature map in this directory throws away the
sign structure of the logit: ``relu`` deletes the negative coordinates outright, and
with them the opposite-signed part of ``<q, k>``.  Meng et al. (2025) keep it.  Split
each vector into its positive and negative parts, ``q = q+ - q-`` with
``q+ = max(q,0)``, so that

    <q, k> = (<q+,k+> + <q-,k->) - (<q+,k-> + <q-,k+>),

and build two *separate* nonnegative interactions from the two halves, each sharpened
by an elementwise power ``f(x) = x^p`` with ``p = 1 + alpha sigmoid(w)`` learned per
channel (their Eq. 9; ``p > 1`` has positive first and second derivatives, which is
their entropy-reduction condition):

    g_same(q,x) = <f(q+), f(k+)> + <f(q-), f(k-)>,
    g_opp (q,x) = <f(q-), f(k+)> + <f(q+), f(k-)>.

**Invariant W fails here, and that is the finding.**  The paper does not subtract the
two -- that would make the surrogate signed and the measure disappear.  It row-
normalizes each separately and sends them to *different halves of the value
channels*: with ``v = [v1, v2]``,

    y = [ pi_same v1 , pi_opp v2 ].

Each of ``pi_same`` and ``pi_opp`` is a genuine probability measure, but the output is
not an average of ``v`` against *one* measure on the ``n`` positions, so no single
``pi_hat`` exists and the method declares ``expresses_weights = False``.  It is the
clearest example in the bank of an operator that is nonnegative, normalized and
channel-wise convex, yet outside the framework's single-measure form -- the value
field is not shared across output coordinates.  Both measures are returned in
``extras`` so the pair can still be studied.

PolaFormer the vision backbone adds a learned per-channel temperature, a depthwise
convolution on the values and a gating branch; those are architecture, not ``g``, and
are left out, exactly as ``cosformer`` leaves out TransNormer's surroundings.  The
one exception is the per-channel temperature: at initialization it is a single
constant on both ``q`` and ``k``, which cancels in the row normalization, so omitting
it changes nothing at the defaults.

``power`` defaults to ``3``, the initialization ``1 + alpha sigmoid(0)`` at the
paper's ``alpha = 4``.  It is learned, so ``trainable = True``; ``power = 1`` is the
plain polarity split with no sharpening.

Reference: Meng et al. (2025) PolaFormer; Katharopoulos et al. (2020) for the
normalized separable readout; Qin et al. (2022) for the ``relu``-feature baseline it
repairs.
"""
from __future__ import annotations

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    causal_mask,
    normalize_nonnegative,
)
from ...registry import register


def polarity_features(x, power: float):
    """``(f(x+), f(x-))`` with ``f(u) = u^power`` applied to each nonnegative part."""
    return torch.relu(x) ** power, torch.relu(-x) ** power


@register
class PolaFormerAttention(AttentionMethod):
    """Same-signed and opposite-signed interactions, each normalized, on split values."""

    name = "polaformer"
    category = Category.INTERACTION
    objective = Objective.MODIFY
    supports_causal = True
    expresses_weights = False        # two measures, one per half of the value channels
    row_stochastic = False           # there is no single pi_hat to be stochastic
    trainable = True                 # the per-channel power is learned
    references = ("meng2025polaformer", "katharopoulos2020transformers", "qin2022cosformer")
    paper_section = "sec:kernel"

    def __init__(self, power: float = 3.0, **kw):
        super().__init__(power=power, **kw)
        self.power = power

    def _forward(self, q, k, v, causal, return_weights):
        m, n = q.shape[-2], k.shape[-2]
        qp, qn = polarity_features(q, self.power)
        kp, kn = polarity_features(k, self.power)
        kpt, knt = kp.transpose(-1, -2), kn.transpose(-1, -2)

        g_same = qp @ kpt + qn @ knt
        g_opp = qn @ kpt + qp @ knt
        support = causal_mask(m, n, device=q.device) if causal else None
        pi_same, z_same = normalize_nonnegative(g_same, support)
        pi_opp, z_opp = normalize_nonnegative(g_opp, support)

        half = (v.shape[-1] + 1) // 2        # odd dv: the same-signed half takes the extra
        v_same, v_opp = v[..., :half], v[..., half:]
        y = torch.cat([pi_same @ v_same, pi_opp @ v_opp], dim=-1)
        extras = {"weights_same_sign": pi_same, "weights_opposite_sign": pi_opp,
                  "Z_same": z_same, "Z_opposite": z_opp, "power": self.power,
                  "split": v_same.shape[-1]}
        return AttentionOutput(y=y, weights=None, extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        r = 2 * d                       # [f(x+), f(x-)], used in two pairings
        return {"flops": 4 * n * r * dv + 4 * m * r * dv + 2 * (m + n) * r,
                "state_bytes": 2 * r * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv
                                + 2 * r * (dv + 1)) * bytes_per_el}
