"""Feature maps and the readout convention of ``eq:memory-generalrec``.

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
    """The named feature map ``phi``, or a ``ValueError`` listing the legal names.

    Examples
    --------
    >>> import torch
    >>> from attnbank.components import feature_map
    >>> feature_map("relu")(torch.tensor([-1.0, 2.0]))
    tensor([0., 2.])
    >>> bool((feature_map("elu")(torch.randn(16)) > 0).all())   # positive by design
    True
    """
    if name not in FEATURES:
        raise ValueError(f"feature must be one of {sorted(FEATURES)}, got {name!r}")
    return FEATURES[name]


def readout_positions(m, n, causal, device):
    """For each query, the index of the state to read.

    Its own position under causal masking; the final state otherwise, since then the
    whole context is already in the state.

    Examples
    --------
    >>> from attnbank.components import readout_positions
    >>> readout_positions(3, 5, causal=True, device="cpu")
    tensor([2, 3, 4])
    >>> readout_positions(3, 5, causal=False, device="cpu")
    tensor([4, 4, 4])
    """
    if causal:
        return torch.arange(m, device=device) + (n - m)
    return torch.full((m,), n - 1, dtype=torch.long, device=device)


__all__ = ["FEATURES", "feature_map", "readout_positions"]
