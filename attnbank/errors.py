"""Exceptions that tell you what to do next.

Every error raised by :mod:`attnbank` carries three things: *what* went wrong, *where*
(which argument, which method), and *what to try instead* -- a near-miss name, the list
of methods that do support the feature you asked for, or the shape that was expected.

The classes subclass both :class:`AttnBankError` and the built-in exception a caller
would naturally catch (``KeyError``, ``ValueError``), so ``except KeyError`` around
``ab.build(...)`` keeps working.

Examples
--------
>>> import attnbank as ab
>>> ab.build("topkk")                                    # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
    ...
attnbank.errors.UnknownMethodError: unknown method 'topkk'. Did you mean: 'topk'?
"""
from __future__ import annotations

import difflib
from typing import Iterable, Optional, Sequence

__all__ = [
    "AttnBankError",
    "UnknownMethodError",
    "ShapeError",
    "CausalNotSupportedError",
    "ConfigError",
    "ConformanceError",
    "suggest",
]


class AttnBankError(Exception):
    """Base class for every error attnbank raises deliberately.

    ``except attnbank.AttnBankError`` catches all of them; the subclasses also inherit
    from the built-in exception you would otherwise expect, so narrower handlers that
    predate this module keep working.
    """

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return self.args[0] if len(self.args) == 1 and isinstance(self.args[0], str) \
            else super().__str__()


class UnknownMethodError(AttnBankError, KeyError):
    """A method name is not in the registry. Carries near-miss suggestions."""


class ShapeError(AttnBankError, ValueError):
    """A tensor has the wrong rank or a dimension that disagrees with another tensor."""


class CausalNotSupportedError(AttnBankError, ValueError):
    """``causal=True`` was asked of a method that declares ``supports_causal = False``."""


class ConfigError(AttnBankError, TypeError):
    """A configuration keyword is not accepted by the method being built."""


class ConformanceError(AttnBankError, AssertionError):
    """A method violated the design contract (most often Invariant W)."""


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def suggest(name: str, candidates: Iterable[str], n: int = 3) -> list[str]:
    """Near matches for ``name`` among ``candidates``, best first.

    Uses :func:`difflib.get_close_matches` with a forgiving cutoff and then adds
    substring matches, which catches ``"flash_attention"`` -> ``"flash"``.

    Examples
    --------
    >>> from attnbank.errors import suggest
    >>> suggest("topkk", ["topk", "value_topk", "window"])
    ['topk']
    >>> suggest("flash_attention", ["flash", "splitk"])
    ['flash']
    """
    candidates = list(candidates)
    hits = difflib.get_close_matches(name, candidates, n=n, cutoff=0.6)
    lowered = name.lower()
    for c in candidates:
        if len(hits) >= n:
            break
        if c not in hits and (c in lowered or lowered in c):
            hits.append(c)
    return hits


def unknown_method_message(name: str, candidates: Sequence[str],
                           hint: Optional[str] = None) -> str:
    """The body of :class:`UnknownMethodError`: near misses first, full list last."""
    near = suggest(name, candidates)
    parts = [f"unknown method {name!r}."]
    if near:
        parts.append("Did you mean: " + ", ".join(repr(c) for c in near) + "?")
    if hint:
        parts.append(hint)
    parts.append(f"ab.available() lists all {len(candidates)} registered methods; "
                 "ab.list_methods() prints them grouped by category.")
    return " ".join(parts)


def _shape_str(shape) -> str:
    return "[" + ", ".join(str(int(s)) for s in shape) + "]"


def shape_message(argument: str, expected: str, got, extra: Optional[str] = None) -> str:
    """``"query must be [B, H, m, d]; got [2, 16, 8] (3 dims, expected 4)"``."""
    msg = f"{argument} must be {expected}; got {_shape_str(got)}"
    if extra:
        msg += f" ({extra})"
    return msg
