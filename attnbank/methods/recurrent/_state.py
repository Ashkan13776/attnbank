"""Shared pieces for ``recurrent/``: feature maps and the readout convention.

``sec:memory-state``, ``eq:memory-generalrec``.  Every method in this category has the
form

    H_t = A_t(H_{t-1}) + beta_t psi(k_t) v_t^T,     y_hat_t = H_t^T phi(q_t),

with ``H in R^{r x d_v}``.  Only the first row of ``tab:memory-recurrences`` (plain
linear attention with the denominator kept) computes a normalized expectation; the rest
drop the normalizer, and with it normalization invariance, constant preservation and
convex-hull confinement all at once.  That loss is the point of the category, so the
readout here is unnormalized by default and ``row_stochastic`` is ``False``.

Because ``H_t`` is *linear in the values*, every one of these operators can still write
``y_hat_t = sum_j w_{tj} v_j`` with the ``w`` returned as ``weights``: Invariant W holds
even though the weights are not a probability measure.
"""
from __future__ import annotations

import torch

FEATURES = {
    "elu": lambda x: torch.nn.functional.elu(x) + 1.0,
    "relu": torch.relu,
    "identity": lambda x: x,
}


def feature_map(name):
    if name not in FEATURES:
        raise ValueError(f"feature must be one of {sorted(FEATURES)}, got {name!r}")
    return FEATURES[name]


def readout_positions(m, n, causal, device):
    """For each query, the index of the state to read: its own position, or the final
    state when the mask is off (the whole context is in the state)."""
    if causal:
        return torch.arange(m, device=device) + (n - m)
    return torch.full((m,), n - 1, dtype=torch.long, device=device)
