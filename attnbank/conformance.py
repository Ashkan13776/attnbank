"""Invariant W and the conformance suite, as a runtime API.

The library's one invariant is

    y == weights @ (v_hat if v_hat is not None else value)

Every registered method satisfies it, and ``tests/test_conformance.py`` proves it in
CI over the whole registry in float32 and float64.  This module exposes the same checks
at runtime, so the guarantee is available to anyone building on top of ``attnbank``:

* :func:`check_invariant` -- one output, one number, an exception if it is too large.
* :func:`check_conformance` -- one method, a printable report over the checks that can
  be run quickly; the full eleven-point suite lives in ``tests/test_conformance.py``.

Why it matters: :mod:`attnbank.metrics` splits output error into a *distributional*
term ``(pi - pi_hat) v`` and a *value* term ``pi_hat (v - v_hat)``.  That split is
meaningless unless a method's reported weights and values actually reproduce its
output.  Invariant W is what makes the decomposition a measurement rather than a
story, and it is the reason methods that cannot express their measure on the original
``n`` positions declare ``expresses_weights = False`` instead of returning something
approximate.

Examples
--------
>>> import torch, attnbank as ab
>>> q, k, v = (torch.randn(1, 2, 8, 16, dtype=torch.float64) for _ in range(3))
>>> out = ab.build("quantkv", bits_k=4, bits_v=4)(q, k, v, return_weights=True)
>>> ab.check_invariant(out, v) < 1e-9
True
>>> print(ab.check_conformance("topk"))                 # doctest: +ELLIPSIS
topk: ... checks passed...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from .core import AttentionOutput
from .errors import ConformanceError

__all__ = ["check_invariant", "check_conformance", "ConformanceReport", "Check"]

#: default tolerance per dtype for Invariant W (DESIGN.md Sec. 1)
INVARIANT_ATOL = {torch.float64: 1e-10, torch.float32: 1e-5, torch.float16: 1e-2,
                  torch.bfloat16: 1e-1}


def check_invariant(output: AttentionOutput, value: Optional[Tensor] = None,
                    atol: Optional[float] = None) -> float:
    """Verify ``y == weights @ (v_hat or value)``; return the largest deviation.

    Parameters
    ----------
    output:
        an :class:`~attnbank.AttentionOutput` produced with ``return_weights=True``.
    value:
        the ``value`` tensor the method was called with. Required unless the method
        returned ``v_hat`` (in which case ``v_hat`` is the field that was aggregated).
    atol:
        tolerance; defaults to ``1e-10`` in float64 and ``1e-5`` in float32, the
        tolerances in DESIGN.md Sec. 1.

    Returns
    -------
    float
        ``max |y - weights @ v_hat|``, or ``0.0`` when the method declares
        ``expresses_weights = False`` and so returned no weights.

    Raises
    ------
    ~attnbank.errors.ConformanceError
        if the deviation exceeds ``atol``, naming the deviation and the tolerance.

    Examples
    --------
    >>> import torch, attnbank as ab
    >>> q, k, v = (torch.randn(1, 1, 6, 4, dtype=torch.float64) for _ in range(3))
    >>> out = ab.build("merge_kv", n_clusters=3)(q, k, v, return_weights=True)
    >>> ab.check_invariant(out, v) < 1e-10
    True
    """
    if output.weights is None:
        return 0.0
    field_ = output.v_hat if output.v_hat is not None else value
    if field_ is None:
        raise ValueError(
            "check_invariant needs the `value` tensor: this output carries weights but "
            "no v_hat, so the aggregated field is the original value. Call "
            "check_invariant(out, value).")
    recon = output.weights @ field_
    dev = float((output.y - recon).abs().max())
    tol = atol if atol is not None else INVARIANT_ATOL.get(output.y.dtype, 1e-5)
    if not dev <= tol:
        raise ConformanceError(
            f"Invariant W violated: max |y - weights @ "
            f"{'v_hat' if output.v_hat is not None else 'value'}| = {dev:.3e} > {tol:.0e}. "
            "Either the weights do not describe what the method aggregated, or the "
            "method should declare expresses_weights = False and return weights=None. "
            "See DESIGN.md Sec. 1.")
    return dev


@dataclass
class Check:
    """One conformance check: its name, whether it passed, and a one-line detail."""

    name: str
    passed: Optional[bool]          # None = not applicable to this method
    detail: str = ""

    @property
    def mark(self) -> str:
        return {True: "pass", False: "FAIL", None: "n/a "}[self.passed]

    def __str__(self) -> str:
        return f"  [{self.mark}] {self.name}" + (f" -- {self.detail}" if self.detail else "")


@dataclass
class ConformanceReport:
    """The result of :func:`check_conformance`: a list of :class:`Check`, printable."""

    name: str
    config: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no check failed (checks that do not apply do not count)."""
        return all(c.passed is not False for c in self.checks)

    @property
    def failures(self) -> list:
        return [c for c in self.checks if c.passed is False]

    def raise_for_status(self) -> "ConformanceReport":
        """Raise :class:`~attnbank.errors.ConformanceError` if anything failed."""
        if not self.ok:
            raise ConformanceError(
                f"{self.name} failed {len(self.failures)} conformance check(s):\n"
                + "\n".join(str(c) for c in self.failures))
        return self

    def __str__(self) -> str:
        n_pass = sum(c.passed is True for c in self.checks)
        n_na = sum(c.passed is None for c in self.checks)
        cfg = f"({', '.join(f'{key}={val!r}' for key, val in self.config.items())}) " \
            if self.config else ""
        head = (f"{self.name}: {cfg}{n_pass} checks passed, {len(self.failures)} failed"
                + (f", {n_na} not applicable" if n_na else ""))
        return "\n".join([head] + [str(c) for c in self.checks])

    def __repr__(self) -> str:
        return str(self)


def check_conformance(name: str, strict: bool = False, **config) -> ConformanceReport:
    """Run the fast half of the conformance suite on one method, right now.

    The checks: output shapes, Invariant W, row stochasticity where declared, causal
    non-leakage, finiteness under large logits, input immutability, reproducibility,
    the cost-model keys, and the exact limit when the method declares one.  The full
    eleven-point suite -- including float64/CUDA agreement, batch and head
    independence, Monte-Carlo convergence and the bibliography check -- is
    ``tests/test_conformance.py``, parametrized over the whole registry.

    Parameters
    ----------
    name:
        a registry key.
    strict:
        raise instead of returning a report with failures.
    **config:
        passed to :func:`attnbank.build`.

    Returns
    -------
    ConformanceReport
        printable; ``report.ok`` is the one-bit answer.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.check_conformance("performer", r=64).ok
    True
    >>> ab.check_conformance("window", w=4).failures
    []
    """
    from .registry import build, get_class

    cls = get_class(name)
    meth = build(name, **config)
    rep = ConformanceReport(name=name, config=dict(config))
    b, h, m, n, d, dv = 2, 2, 6, 12, 8, 6
    gen = torch.Generator().manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    q, k, v = mk(b, h, m, d), mk(b, h, n, d), mk(b, h, n, dv)
    q0, k0, v0 = q.clone(), k.clone(), v.clone()

    out = meth(q, k, v, return_weights=True)

    shapes_ok = (tuple(out.y.shape) == (b, h, m, dv)
                 and (out.weights is None or tuple(out.weights.shape) == (b, h, m, n))
                 and (out.v_hat is None or tuple(out.v_hat.shape) == (b, h, n, dv)))
    rep.checks.append(Check("shapes", shapes_ok, repr(out)))

    if not cls.expresses_weights:
        rep.checks.append(Check("invariant W", None,
                                "expresses_weights = False, by declaration"))
    else:
        try:
            dev = check_invariant(out, v)
            rep.checks.append(Check("invariant W", True, f"max |y - W v| = {dev:.2e}"))
        except ConformanceError as e:
            rep.checks.append(Check("invariant W", False, str(e).split("\n")[0]))

    if cls.row_stochastic and out.weights is not None:
        s = out.weights.sum(-1)
        ok = bool((out.weights >= -1e-12).all()
                  and torch.allclose(s, torch.ones_like(s), atol=1e-9))
        rep.checks.append(Check("row stochastic", ok,
                                f"row sums in [{float(s.min()):.6f}, {float(s.max()):.6f}]"))
    else:
        rep.checks.append(Check("row stochastic", None,
                                "row_stochastic = False, by declaration"))

    if cls.supports_causal:
        base = meth(q, k, v, causal=True).y
        t = n - 2
        k2, v2 = k.clone(), v.clone()
        k2[..., t:, :] += 1.0
        v2[..., t:, :] += 1.0
        pert = meth(q, k2, v2, causal=True).y
        safe = max(0, t - (n - m))
        ok = safe == 0 or bool(torch.equal(base[..., :safe, :], pert[..., :safe, :]))
        rep.checks.append(Check("no causal leakage", ok, f"queries 0..{safe - 1} unchanged"))
    else:
        rep.checks.append(Check("no causal leakage", None,
                                "supports_causal = False, by declaration"))

    big = meth(10.0 * q, 10.0 * k, v, return_weights=True)
    finite = all(t is None or bool(torch.isfinite(t).all())
                 for t in (big.y, big.weights, big.v_hat))
    rep.checks.append(Check("finite at large logits", finite, "q, k scaled by 10"))

    rep.checks.append(Check("inputs not mutated",
                            bool(torch.equal(q, q0) and torch.equal(k, k0)
                                 and torch.equal(v, v0))))

    rep.checks.append(Check("reproducible",
                            bool(torch.equal(build(name, **config)(q, k, v).y,
                                             build(name, **config)(q, k, v).y)),
                            "two builds agree bit for bit"))

    cost = meth.cost(m, n, d, dv)
    cost_ok = (set(cost) == {"flops", "state_bytes", "bytes_moved"}
               and all(c > 0 and c == c and c != float("inf") for c in cost.values()))
    rep.checks.append(Check("cost model", cost_ok, str(cost) if cost_ok else repr(cost)))

    lim = cls.limit_config()
    if lim is not None and not config:
        sq = 12
        qs, ks, vs = mk(b, h, sq, d), mk(b, h, sq, d), mk(b, h, sq, dv)
        got = build(name, **lim)(qs, ks, vs).y
        ref = build("softmax")(qs, ks, vs).y
        err = float((got - ref).abs().max())
        rep.checks.append(Check("exact limit", err <= 1e-5,
                                f"limit_config={lim}, max |y - softmax| = {err:.2e}"))
    elif lim is None and cls.convergence_configs() is not None:
        rep.checks.append(Check("exact limit", None,
                                f"Monte-Carlo: {len(cls.convergence_configs())} "
                                "convergence_configs instead"))
    else:
        rep.checks.append(Check("exact limit", None,
                                "objective = modify; it does not target softmax"
                                if lim is None else "skipped (explicit config given)"))

    return rep.raise_for_status() if strict else rep
