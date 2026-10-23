"""Execution backends: what actually runs, whether it is here, and what it would cost.

A method in this bank is two things that are easy to confuse: an *operator* (the
mathematics, one file, written for clarity) and an *execution path* (how that operator
is evaluated). The operator never changes with the backend. The execution path does.

    import attnbank as ab

    print(ab.backends())                          # what this machine can run, and why not
    ab.build("softmax", backend="sdpa")           # ask for one explicitly
    ab.backend_matrix()                           # which method supports which

Four backends are known to the library:

``reference``
    Plain PyTorch, always available, CPU and CUDA, every dtype. It is the correctness
    anchor: every other backend is tested against it, so it is also the default. Slow
    on purpose -- DESIGN.md Sec. 6 asks these implementations to be readable.
``sdpa``
    ``torch.nn.functional.scaled_dot_product_attention``. No extra dependency, and on
    CUDA it dispatches to FlashAttention-2 or the memory-efficient kernel; pass
    ``sdpa_kernel="flash" | "mem_efficient" | "math"`` to pin the sub-backend instead of
    letting torch choose.
``flash_attn``
    The ``flash-attn`` package (Dao et al.). CUDA only, fp16/bf16 only.
``xformers`` / ``triton``
    Detected and reported, but **no method declares them**. For ``xformers`` the reason
    is that ``sdpa`` already routes to the same memory-efficient kernel in torch >= 2.2;
    for ``triton``, that the bank ships no hand-written kernel. ``ab.backends()`` says
    so rather than leaving you to guess.

Declaring a backend is a *claim that it equals the reference*, and
``tests/test_conformance.py::test_backend_equivalence`` is where that claim is checked --
for every method, every declared backend, both causal settings, fp32 and fp64, plus the
awkward shapes. A method whose output is not literally a dense softmax over a mask, or
whose ``extras`` need the full score matrix anyway, declares ``reference`` only. That is
the right answer for most of the bank, and it is stated rather than implied.

Examples
--------
>>> import attnbank as ab
>>> ab.get_default_backend()
'reference'
>>> ab.is_backend_available("reference")
True
>>> ab.get_class("softmax").backends
('reference', 'sdpa', 'flash_attn')
>>> ab.get_class("nmf_oracle").backends
('reference',)
"""
from __future__ import annotations

import importlib.util
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .errors import AttnBankError

__all__ = ["BackendInfo", "BackendTable", "BackendNotSupportedError",
           "BackendFallbackWarning", "backends", "backend_info", "is_backend_available",
           "get_default_backend", "set_default_backend", "default_backend",
           "resolve_backend", "REFERENCE"]

#: the backend every method supports and every other backend is measured against
REFERENCE = "reference"

_DEFAULT = REFERENCE


class BackendNotSupportedError(AttnBankError, ValueError):
    """An explicitly requested backend is not declared by the method, or not installed."""


class BackendFallbackWarning(UserWarning):
    """A requested backend was unavailable and the call fell back. Never silent."""


@dataclass(frozen=True)
class BackendInfo:
    """One backend: whether it runs here, why not, and the install line that fixes it.

    Examples
    --------
    >>> import attnbank as ab
    >>> info = ab.backend_info("reference")
    >>> info.name, info.available
    ('reference', True)
    >>> print(ab.backend_info("triton").reason)          # doctest: +ELLIPSIS
    no method declares it: the bank ships no hand-written kernel...
    """

    name: str
    available: bool
    reason: str = ""
    install: str = ""
    version: str = ""
    description: str = ""

    def __str__(self) -> str:
        mark = "available" if self.available else "unavailable"
        out = f"{self.name:<12}{mark:<13}{self.description}"
        if not self.available:
            out += f"\n{'':<12}reason: {self.reason}"
            if self.install:
                out += f"\n{'':<12}fix:    {self.install}"
        elif self.version:
            out += f"\n{'':<12}version: {self.version}"
        return out


class BackendTable(list):
    """A ``list`` of :class:`BackendInfo` that prints as a report.

    Examples
    --------
    >>> import attnbank as ab
    >>> table = ab.backends()
    >>> table["reference"].available
    True
    >>> "sdpa" in [b.name for b in table]
    True
    """

    def __getitem__(self, key):
        if isinstance(key, str):
            for info in self:
                if info.name == key:
                    return info
            raise KeyError(f"unknown backend {key!r}; known: {[b.name for b in self]}")
        return super().__getitem__(key)

    @property
    def names(self) -> list[str]:
        return [b.name for b in self]

    @property
    def usable(self) -> list[str]:
        """Just the backends that run on this machine."""
        return [b.name for b in self if b.available]

    def __str__(self) -> str:
        head = (f"attnbank backends on this machine "
                f"(torch {torch.__version__}, "
                f"cuda {'yes' if torch.cuda.is_available() else 'no'})")
        body = "\n".join(str(b) for b in self)
        return (f"{head}\n{'-' * min(100, len(head))}\n{body}\n\n"
                f"default: {get_default_backend()!r}.  "
                "ab.build(name, backend=...) to choose; ab.backend_matrix() for which "
                "method supports which.")

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

def _spec(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def _version(module: str) -> str:
    try:
        from importlib.metadata import version

        return version(module)
    except Exception:  # pragma: no cover
        return ""


def _detect_reference() -> BackendInfo:
    return BackendInfo(
        REFERENCE, True, version=torch.__version__,
        description="plain PyTorch; always available; the correctness anchor")


def _detect_sdpa() -> BackendInfo:
    ok = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    return BackendInfo(
        "sdpa", ok,
        reason="" if ok else "torch.nn.functional.scaled_dot_product_attention is missing",
        install="" if ok else "pip install 'torch>=2.2'",
        version=torch.__version__ if ok else "",
        description="torch SDPA; CUDA dispatches to FlashAttention-2 or mem-efficient")


def _detect_flash_attn() -> BackendInfo:
    have = _spec("flash_attn")
    cuda = torch.cuda.is_available()
    if have and cuda:
        return BackendInfo("flash_attn", True, version=_version("flash-attn"),
                           description="the flash-attn package; CUDA, fp16/bf16 only")
    reason = ("the flash_attn package is not installed" if not have
              else "flash_attn is installed but no CUDA device is visible")
    return BackendInfo("flash_attn", False, reason=reason,
                       install=("pip install 'attnbank[flash]'   # needs CUDA and a C++ toolchain"
                                if not have else "run on a machine with a CUDA device"),
                       description="the flash-attn package; CUDA, fp16/bf16 only")


def _detect_xformers() -> BackendInfo:
    have = _spec("xformers")
    return BackendInfo(
        "xformers", False,
        reason=("no method declares it: torch >= 2.2 already routes `sdpa` to the same "
                "memory-efficient kernel, so a second path would add a dependency and a "
                "second thing to keep equal to the reference"
                + ("  (the package IS installed here)" if have else "")),
        install="pip install 'attnbank[xformers]'" if not have else "",
        version=_version("xformers") if have else "",
        description="memory-efficient attention; detected, deliberately unused")


def _detect_triton() -> BackendInfo:
    have = _spec("triton")
    return BackendInfo(
        "triton", False,
        reason=("no method declares it: the bank ships no hand-written kernel, because "
                "these implementations are read against the paper (DESIGN.md Sec. 6)"
                + ("  (the package IS installed here)" if have else "")),
        install="pip install 'attnbank[triton]'" if not have else "",
        version=_version("triton") if have else "",
        description="hand-written kernels; detected, deliberately unused")


_DETECTORS = {REFERENCE: _detect_reference, "sdpa": _detect_sdpa,
              "flash_attn": _detect_flash_attn, "xformers": _detect_xformers,
              "triton": _detect_triton}

#: backends a method is allowed to declare (the other two are reported, not dispatched)
DISPATCHABLE = (REFERENCE, "sdpa", "flash_attn")

_cache: dict[str, BackendInfo] = {}


def backend_info(name: str, refresh: bool = False) -> BackendInfo:
    """Detection result for one backend, cached.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.backend_info("reference").available
    True
    >>> ab.backend_info("triton").available          # detected, deliberately unused
    False
    """
    if name not in _DETECTORS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(_DETECTORS)}")
    if refresh or name not in _cache:
        _cache[name] = _DETECTORS[name]()
    return _cache[name]


def backends(refresh: bool = False) -> BackendTable:
    """What this machine can run, and for anything it cannot, why and how to fix it.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.backends()["reference"].available
    True
    >>> print(ab.backends())                         # doctest: +ELLIPSIS
    attnbank backends on this machine ...
    reference   available ...
    """
    return BackendTable(backend_info(n, refresh) for n in _DETECTORS)


def is_backend_available(name: str) -> bool:
    """``True`` if ``name`` can actually run here.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.is_backend_available("reference")
    True
    """
    return backend_info(name).available


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def get_default_backend() -> str:
    """The backend used when a build does not name one. ``'reference'`` out of the box.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.get_default_backend()
    'reference'
    """
    return _DEFAULT


def set_default_backend(name: str) -> str:
    """Set the process-wide default backend; returns the previous one.

    The studies behind the monograph pin this to ``'reference'`` on import
    (``attnbank/studies/__init__.py``) so that adding a backend can never move a
    published number.

    Examples
    --------
    >>> import attnbank as ab
    >>> previous = ab.set_default_backend("sdpa")
    >>> ab.get_default_backend()
    'sdpa'
    >>> ab.set_default_backend(previous)
    'sdpa'
    """
    global _DEFAULT
    if name not in DISPATCHABLE:
        raise BackendNotSupportedError(
            f"{name!r} cannot be a default backend. Dispatchable backends are: "
            f"{', '.join(DISPATCHABLE)}. ab.backends() explains the others.")
    if not is_backend_available(name):
        info = backend_info(name)
        raise BackendNotSupportedError(
            f"{name!r} is not available here: {info.reason}."
            + (f" Fix: {info.install}" if info.install else ""))
    previous, _DEFAULT = _DEFAULT, name
    return previous


class default_backend:
    """Context manager form of :func:`set_default_backend`.

    Examples
    --------
    >>> import attnbank as ab
    >>> with ab.default_backend("sdpa"):
    ...     ab.build("softmax").backend
    'sdpa'
    >>> ab.get_default_backend()
    'reference'
    """

    def __init__(self, name: str):
        self.name = name
        self._previous: Optional[str] = None

    def __enter__(self) -> str:
        self._previous = set_default_backend(self.name)
        return self.name

    def __exit__(self, *exc) -> None:
        global _DEFAULT
        _DEFAULT = self._previous


def resolve_backend(requested: Optional[str], supported: Sequence[str],
                    method: str) -> str:
    """Pick the backend to run, loudly.

    * ``requested is None`` -- use the process default if the method supports it and it
      is installed, otherwise ``reference``. Falling back from the *default* is quiet,
      because the default is a preference, not an instruction.
    * ``requested`` is not declared by the method -- raise
      :class:`BackendNotSupportedError` listing what it does declare. An explicit
      request is an instruction; silently doing something else would make a benchmark
      meaningless.
    * ``requested`` is declared but not installed here -- warn with
      :class:`BackendFallbackWarning` and use ``reference``. Announced, never silent.

    Examples
    --------
    >>> from attnbank.backends import resolve_backend
    >>> resolve_backend(None, ("reference", "sdpa"), "softmax")
    'reference'
    >>> resolve_backend("sdpa", ("reference", "sdpa"), "softmax")
    'sdpa'
    >>> resolve_backend("sdpa", ("reference",), "nmf_oracle")
    Traceback (most recent call last):
        ...
    attnbank.backends.BackendNotSupportedError: 'nmf_oracle' does not support backend 'sdpa'...
    """
    supported = tuple(supported)
    if requested is None:
        wanted = get_default_backend()
        if wanted in supported and is_backend_available(wanted):
            return wanted
        return REFERENCE
    if requested not in supported:
        extra = ""
        if requested in _DETECTORS and requested not in DISPATCHABLE:
            extra = f" {backend_info(requested).reason.capitalize()}."
        elif requested not in _DETECTORS:
            extra = f" Known backends: {', '.join(sorted(_DETECTORS))}."
        raise BackendNotSupportedError(
            f"{method!r} does not support backend {requested!r}. It declares: "
            f"{', '.join(repr(b) for b in supported)}.{extra} "
            "ab.backend_matrix() shows the whole matrix; a method that declares "
            "'reference' only has no equivalent kernel, which is a fact about the "
            "operator, not a gap in the library.")
    if not is_backend_available(requested):
        info = backend_info(requested)
        warnings.warn(
            f"backend {requested!r} was requested for {method!r} but is not available "
            f"here ({info.reason}); falling back to {REFERENCE!r}. "
            + (f"Fix: {info.install}. " if info.install else "")
            + "This is announced because a silent downgrade would make a benchmark "
              "meaningless.",
            BackendFallbackWarning, stacklevel=3)
        return REFERENCE
    return requested
