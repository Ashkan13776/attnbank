"""Backwards-compatible alias for :mod:`attnbank.components.restriction`.

The renormalized restriction moved to ``attnbank/components/`` when the bank grew a
shared-component layer; this module keeps the old import path working, including the
one used by ``studies/S2_restriction.py``.
"""
from ...components.restriction import (  # noqa: F401
    reference_pi, restrict, topk_support, visible)

__all__ = ["reference_pi", "visible", "restrict", "topk_support"]
