"""Polynomial sketch: truncate exp to degree P, then sketch each tensor power.

``sec:kernel-rf``, the deterministic half.  The softmax interaction is an entire
function of the inner product,

    g(q, x) = exp(beta <q, k_x>) = sum_{p >= 0} <qt, kt_x>^p / p!,
    qt = beta q,  kt = k,

so truncating the series at degree ``P`` gives a surrogate that is a *polynomial
kernel of degree P* -- exactly representable by the feature map
``x -> (1, x, x^{(x)2}/sqrt(2!), ..., x^{(x)P}/sqrt(P!))`` of dimension
``sum_p d^p``, which is why nobody writes it down.  The sketching literature does the
only sensible thing with that feature map: it never forms it.  TensorSketch
(Pham and Pagh, 2013) replaces the ``p``-fold tensor power by ``s`` buckets,

    TS_p(x) = FFT^{-1}( prod_{i=1}^{p} FFT(CountSketch_i(x)) ) in R^s,
    E <TS_p(x), TS_p(y)> = <x, y>^p,

and Ahle et al. (2019) assemble exactly this sum of sketched powers into an oblivious
sketch of the Gaussian / exponential kernel.  PolySketchFormer (Kacham et al., 2024)
is the attention instance of the same construction.  The shipped feature map is

    Psi(x) = [ 1, TS_1(x), TS_2(x)/sqrt(2!), ..., TS_P(x)/sqrt(P!) ] in R^{1 + P s},
    g_hat(q, x) = <Psi(qt), Psi(kt_x)>,

so this is a linear-attention method with an explicit ``1 + P s`` dimensional state --
the point of the whole exercise, since the unsketched map needs ``sum_p d^p``.
``sketch=None`` evaluates the same polynomial exactly through powers of the inner
product and is the control the sketch is measured against; at ``degree = 2`` that
control is ``based``, which is the same surrogate with the tensor-power feature map
written out in full. The difference between the two files is the sketch, and the
resource that converges here is the feature budget ``1 + P s`` rather than ``P`` alone.

Two error sources, and they are different in kind.  The truncation error is
**deterministic** and vanishes as ``P -> infinity``, which is why this is an
``approximate`` method and not a ``modify`` one.  The sketch error is Monte-Carlo and
vanishes as ``s -> infinity`` at fixed ``P``; its variance grows like
``(||qt|| ||kt||)^{2p}``, so a high degree needs normalised queries and keys to be
usable at all -- the reason PolySketchFormer normalises them.  ``convergence_configs``
raises both at once, because that is the knob a user actually has: the pair ``(P, s)``
is one feature budget and neither half converges alone.

What is lost.  An even-degree truncation of ``exp`` is strictly positive on the whole
real line, so with ``sketch=None`` and even ``degree`` the surrogate is nonnegative and
``pi_hat`` is a genuine probability measure -- that is what ``based`` relies on.  The
sketch destroys it: ``<Psi(q), Psi(k)>`` is an unbiased estimate of a positive number
and takes negative values, so ``pi_hat`` is a signed measure of unit mass.  The class
declares the weaker property, ``row_stochastic = False``, and reports the negative mass
in ``extras`` rather than assuming it away.

A sharper caveat, worth more than the two above.  Softmax is invariant to adding a
constant to a row of logits; a truncated Taylor series is not.  ``g_hat`` is a
polynomial, so it cannot be shifted before evaluation the way ``exp`` can, and in the
large-logit regime the surrogate is dominated by its top term and ``pi_hat`` tends to
the softmax of ``P log|s|`` rather than of ``s``.  Accuracy here is a statement about
the *scale* of the logits, not only about ``P``.

Numerics.  Feature maps and their inner products are evaluated in float64 whatever
the input dtype, as in ``nystrom``, so that the error measured is the estimator's.
Even so, a sketched degree-``P`` polynomial is an ill-conditioned object once the
logits are large: the terms grow like ``s^P`` and cancel, and float32 inputs then
agree with float64 ones only to about ``1e-3``.  That is the same large-logit regime
the caveat above is about, and in it the estimator has already stopped tracking
softmax -- the precision loss is a symptom, not the disease.

Reference: Pham and Pagh (2013); Ahle et al. (2019); Kacham et al. (2024).
"""
from __future__ import annotations

import math
from functools import lru_cache

import torch

from ...core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    causal_mask,
    make_generator,
    weights_from_log_magnitude,
)
from ...errors import ConfigError
from ...registry import register


@lru_cache(maxsize=128)
def _countsketch_f64(degree: int, d: int, s: int, seed: int):
    """The float64 CPU draw of ``degree`` CountSketch matrices, memoised on its inputs."""
    g = make_generator(seed)
    bucket = torch.randint(s, (degree, d), generator=g)
    sign = torch.randint(2, (degree, d), generator=g).double() * 2 - 1
    mats = torch.zeros(degree, s, d, dtype=torch.float64)
    rows = torch.arange(degree).unsqueeze(-1).expand(degree, d)
    cols = torch.arange(d).unsqueeze(0).expand(degree, d)
    mats[rows, bucket, cols] = sign
    return mats


def countsketch_matrices(degree: int, d: int, s: int, seed: int, device,
                         dtype=torch.float64):
    """``[degree, s, d]``: one CountSketch per tensor factor, ``+-1`` once per column."""
    cached = _countsketch_f64(degree, d, s, seed)
    out = cached.to(device=device, dtype=dtype)
    # never hand back the cached tensor itself; a caller could mutate every later draw
    return out.clone() if out is cached else out


def tensor_sketch(x, mats):
    """TensorSketch of ``x^{(x)p}`` with ``p = len(mats)``: ``[..., t, s]``.

    The ``p``-fold tensor power is sketched as the circular convolution of ``p``
    CountSketches of ``x``, evaluated as a product of real FFTs (Pham and Pagh, 2013).
    Unbiased: ``E <TS_p(x), TS_p(y)> = <x, y>^p``.
    """
    s = mats.shape[-2]
    prod = None
    for mat in mats:
        f = torch.fft.rfft(x @ mat.T, n=s, dim=-1)
        prod = f if prod is None else prod * f
    return torch.fft.irfft(prod, n=s, dim=-1)


@register
class PolySketchAttention(AttentionMethod):
    """TensorSketch of the degree-P Taylor feature map; ``sketch=None`` is the control."""

    name = "polysketch"
    category = Category.INTERACTION
    objective = Objective.APPROXIMATE
    supports_causal = True           # g_hat is pointwise in (q, k); the mask restricts it
    expresses_weights = True
    row_stochastic = False           # the sketch (and odd degrees) make g_hat signed
    references = ("pham2013fast", "ahle2019oblivious", "kacham2024polysketchformer")
    paper_section = "sec:kernel"

    def __init__(self, degree: int = 4, sketch: int | None = 128, seed: int = 0, **kw):
        """``degree``: truncation order ``P``.  ``sketch``: buckets ``s`` per tensor
        power, or ``None`` to evaluate the polynomial kernel exactly."""
        super().__init__(degree=degree, sketch=sketch, seed=seed, **kw)
        if degree < 0:
            raise ConfigError(f"{self.name}: degree must be >= 0, got {degree!r}; it is "
                              "the order at which the exponential series is truncated.")
        if sketch is not None and sketch < 1:
            raise ConfigError(f"{self.name}: sketch must be a positive number of buckets "
                              f"or None (no sketch), got {sketch!r}.")
        self.degree, self.sketch, self.seed = degree, sketch, seed

    def features(self, x):
        """``Psi(x)``: ``[..., t, 1 + P s]``, the sketched polynomial feature map.

        Defined only when the method was built with a ``sketch``: the unsketched map
        has ``sum_p d^p`` coordinates and is never formed.
        """
        if self.sketch is None:
            raise ConfigError(f"{self.name}(sketch=None) has no explicit feature map: the "
                              "exact degree-P map needs sum_p d^p coordinates. Build it "
                              "with sketch=<buckets> to get one.")
        mats = countsketch_matrices(max(self.degree, 1), x.shape[-1], self.sketch,
                                    self.seed, x.device, x.dtype)
        parts = [torch.ones_like(x[..., :1])]
        for p in range(1, self.degree + 1):
            parts.append(tensor_sketch(x, mats[:p]) / math.sqrt(math.factorial(p)))
        return torch.cat(parts, dim=-1)

    def _surrogate(self, q, k, beta):
        """``g_hat`` on the original positions: ``[..., m, n]``, in float64."""
        qt, kt = q.double() * beta, k.double()
        if self.sketch is not None:
            return torch.einsum("...mf,...nf->...mn", self.features(qt), self.features(kt))
        ip = torch.einsum("...md,...nd->...mn", qt, kt)
        term = torch.ones_like(ip)
        out = torch.ones_like(ip)
        for p in range(1, self.degree + 1):
            term = term * ip / p
            out = out + term
        return out

    def _forward(self, q, k, v, causal, return_weights):
        m, n, d = q.shape[-2], k.shape[-2], q.shape[-1]
        g_hat = self._surrogate(q, k, self._scale(d))
        support = causal_mask(m, n, device=q.device) if causal else None
        pi, z_rel = weights_from_log_magnitude(torch.log(g_hat.abs()),
                                               sign=torch.sign(g_hat), support=support)
        pi = pi.to(q.dtype)
        y = pi @ v
        extras = {"degree": self.degree, "sketch": self.sketch,
                  "Z_rel": z_rel.to(q.dtype), "min_weight": pi.amin(-1),
                  "neg_mass": pi.clamp(max=0).sum(-1).abs(),
                  "feature_dim": 1 + self.degree * self.sketch if self.sketch else None}
        return AttentionOutput(y=y, weights=pi if return_weights else None, extras=extras)

    def cost(self, m, n, d, dv, bytes_per_el=2):
        if self.sketch is None:                       # exact kernel: the dense path
            return {"flops": 2 * m * n * d + 2 * m * n * (self.degree + 1),
                    "state_bytes": n * (d + dv) * bytes_per_el,
                    "bytes_moved": (m * d + n * (d + dv) + m * dv) * bytes_per_el}
        s = self.sketch
        df = 1 + self.degree * s
        sketching = (m + n) * self.degree * (2 * d * s + 5 * s * math.log2(2 * s))
        return {"flops": sketching + 2 * (m + n) * df * (dv + 1),
                "state_bytes": df * (dv + 1) * bytes_per_el,
                "bytes_moved": (m * d + n * (d + dv) + m * dv
                                + df * (dv + 1)) * bytes_per_el}

    @classmethod
    def limit_config(cls):
        return None       # no finite degree is exact; see convergence_configs

    @classmethod
    def convergence_configs(cls):
        return [{"degree": 2, "sketch": 8}, {"degree": 4, "sketch": 32},
                {"degree": 6, "sketch": 128}, {"degree": 8, "sketch": 512}]
