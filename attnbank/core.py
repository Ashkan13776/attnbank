"""Core interface for attnbank.

Every efficient-attention method is an ``AttentionMethod``. The interface mirrors
the paper's framework: attention is y(q) = E_{x ~ pi_q}[v(x)], and a method may
change the evaluation (exact), the interaction g, the accessible support, the
context representation (mu, k, v), the factorization of pi_q, or the operator.

Tensor convention (all methods): q [B, H, m, d], k [B, H, n, d], v [B, H, n, dv].
Causal masking aligns the last query with the last key (decode-friendly).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from torch import Tensor


class Category(str, Enum):
    EXACT = "exact"              # same operator, different execution
    INTERACTION = "interaction"  # replaces g by g_hat
    RESTRICT = "restrict"        # restricts support of pi_q to S_q
    COMPRESS = "compress"        # replaces (mu, k, v) by a cheaper representation
    FACTORIZE = "factorize"      # pi_q ~ a(q, .) b(., x) through a latent space
    COUPLING = "coupling"        # global constraints across queries (e.g. OT)
    RECURRENT = "recurrent"      # finite-state operators (linear RNN / SSM)


class Objective(str, Enum):
    ACCELERATE = "accelerate"    # exact up to floating point
    APPROXIMATE = "approximate"  # targets fidelity to the reference operator
    MODIFY = "modify"            # defines a different operator


@dataclass
class AttentionOutput:
    """Result of a forward pass.

    y:        output [B, H, m, dv]
    weights:  implied conditional measure pi_hat over the ORIGINAL n context
              positions [B, H, m, n], if the method can express it (else None)
    v_hat:    value field actually aggregated, on the original positions
              [B, H, n, dv], if it differs from v (value compression)
    extras:   method-specific diagnostics (e.g. support indices, omitted mass)
    """
    y: Tensor
    weights: Optional[Tensor] = None
    v_hat: Optional[Tensor] = None
    extras: dict = field(default_factory=dict)


TINY = 1e-30  # denominator floor; tests assert it is never actually reached


def causal_mask(m: int, n: int, device=None) -> Tensor:
    """Boolean [m, n] mask, True where key j is visible to query i (aligned at the end)."""
    i = torch.arange(m, device=device)[:, None] + (n - m)
    j = torch.arange(n, device=device)[None, :]
    return j <= i


def query_positions(m: int, n: int, device=None) -> Tensor:
    """Absolute context position of each query under the end-alignment convention: [m]."""
    return torch.arange(m, device=device) + (n - m)


def scores(q: Tensor, k: Tensor, scale: float, causal: bool) -> Tensor:
    """Scaled logits s = beta <q, k_j>, with -inf on masked entries. Shape [..., m, n]."""
    s = torch.einsum("...md,...nd->...mn", q, k) * scale
    if causal:
        mask = causal_mask(q.shape[-2], k.shape[-2], device=q.device)
        s = s.masked_fill(~mask, float("-inf"))
    return s


def masked_softmax(s: Tensor, support: Optional[Tensor] = None) -> Tensor:
    """Softmax of ``s`` restricted to ``support``, computed in log space.

    ``support`` is a boolean tensor broadcastable to ``s``. Rows with empty support
    return the uniform distribution over the row rather than ``nan`` (documented
    fallback of DESIGN.md Sec. 4, rule 10); such rows are reported by the caller.
    """
    if support is not None:
        s = s.masked_fill(~support, float("-inf"))
    dead = torch.isneginf(s).all(dim=-1, keepdim=True)
    s = torch.where(dead, torch.zeros_like(s), s)
    return torch.softmax(s, dim=-1)


def weights_from_log_magnitude(logmag: Tensor, sign: Optional[Tensor] = None,
                               support: Optional[Tensor] = None,
                               floor: float = 1e-6) -> tuple[Tensor, Tensor]:
    """Normalize a possibly *signed* surrogate interaction given in log magnitude.

    A surrogate ``g_hat_j = sign_j * exp(logmag_j)`` induces
    ``pi_hat_j = g_hat_j / sum_i g_hat_i``.  The common factor ``exp(max_j logmag_j)``
    cancels, so the maximum is subtracted before exponentiating; the maximum is taken
    over the *visible* keys only, which keeps the computation causal.

    When ``sign`` is given the normalizer can vanish or change sign (Sec. 3.2: this is
    exactly what goes wrong for trigonometric random features).  Its magnitude is then
    floored at ``floor`` with the sign preserved, so the result is finite rather than
    ``nan``; the returned ``Z`` is the *unfloored* relative normalizer
    ``Z_hat / exp(max logmag)``, so a caller can report how often it was non-positive.
    """
    if support is not None:
        logmag = logmag.masked_fill(~support, float("-inf"))
    dead = torch.isneginf(logmag).all(dim=-1, keepdim=True)
    logmag = torch.where(dead, torch.zeros_like(logmag), logmag)
    shift = logmag.amax(dim=-1, keepdim=True)
    w = torch.exp(logmag - shift)
    if sign is not None:
        w = w * sign
    z = w.sum(-1, keepdim=True)
    den = torch.where(z.abs() < floor, torch.sign(z + (z == 0).to(z.dtype)) * floor, z)
    return w / den, z.squeeze(-1)


def renormalize(w: Tensor) -> Tensor:
    """Divide by the row sum, flooring the denominator at ``TINY`` (never reached in tests)."""
    return w / w.sum(-1, keepdim=True).clamp_min(TINY)


def normalize_nonnegative(g: Tensor, support: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """Turn a nonnegative surrogate interaction into a probability measure.

    Entries outside ``support`` get zero mass.  A row whose total is zero -- possible
    when a feature map annihilates the query (``relu``) or underflows -- falls back to
    the uniform distribution on its support, the documented finite result required by
    DESIGN.md Sec. 4 rule 10.  The unnormalized row sums are returned alongside so that
    the fallback can be counted rather than hidden.
    """
    if support is not None:
        g = g * support
    z = g.sum(-1, keepdim=True)
    if support is None:
        uniform = torch.full_like(g, 1.0 / g.shape[-1])
    else:
        cnt = support.sum(-1, keepdim=True).clamp_min(1)
        uniform = support.to(g.dtype) / cnt
    pi = torch.where(z > 0, g / z.clamp_min(TINY), uniform)
    return pi, z.squeeze(-1)


def make_generator(seed: int, device=None) -> torch.Generator:
    """Seeded ``torch.Generator``. Randomness never touches the global RNG (DESIGN.md Sec. 4)."""
    g = torch.Generator(device="cpu" if device is None else device)
    g.manual_seed(int(seed))
    return g


class AttentionMethod(torch.nn.Module):
    """Base class. Subclasses set the class attributes and implement ``_forward``."""

    name: str = "base"
    category: Category = Category.EXACT
    objective: Objective = Objective.ACCELERATE
    supports_causal: bool = True
    expresses_weights: bool = True   # can it return pi_hat on the ORIGINAL n positions?
    row_stochastic: bool = True      # are those weights >= 0 with rows summing to 1?
    head_independent: bool = True    # False for methods that share K/V across query heads
    trainable: bool = False      # has learned parameters / needs training to be meaningful
    references: tuple = ()       # bibtex keys of the paper(s) implemented
    paper_section: str = ""      # \label of the section that discusses the method

    def __init__(self, scale: Optional[float] = None, **kwargs):
        super().__init__()
        self.scale = scale
        self.config = kwargs

    def _scale(self, d: int) -> float:
        return self.scale if self.scale is not None else 1.0 / math.sqrt(d)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, causal: bool = False,
                return_weights: bool = False) -> AttentionOutput:
        if causal and not self.supports_causal:
            raise ValueError(f"{self.name} does not support causal attention")
        return self._forward(q, k, v, causal=causal, return_weights=return_weights)

    def _forward(self, q, k, v, causal, return_weights) -> AttentionOutput:
        raise NotImplementedError

    # Analytic cost model per (batch, head). Override where meaningful.
    def cost(self, m: int, n: int, d: int, dv: int, bytes_per_el: int = 2) -> dict:
        """Return {'flops': ..., 'state_bytes': ..., 'bytes_moved': ...}."""
        return {}

    # ---- correctness anchors (DESIGN.md Sec. 2) -----------------------------
    @classmethod
    def limit_config(cls) -> Optional[dict]:
        """Config under which this method provably reduces to dense softmax, or ``None``.

        The conformance suite builds ``cls(**limit_config())`` and requires the output and
        the weights to match :class:`SoftmaxAttention`. ``None`` means no exact limit
        exists; the method must then supply :meth:`convergence_configs`, unless its
        ``objective`` is :attr:`Objective.MODIFY`, in which case it does not target
        softmax at all and neither anchor applies.
        """
        return None

    @classmethod
    def convergence_configs(cls) -> Optional[list[dict]]:
        """Configs whose mean error against softmax must decrease monotonically.

        Used for Monte-Carlo methods, which have no exact limit at finite resource.
        """
        return None

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name!r}, category={self.category.value}, objective={self.objective.value})"
