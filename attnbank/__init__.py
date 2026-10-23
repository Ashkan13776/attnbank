"""attnbank -- every efficient attention method, side by side, organised by what it changes.

Attention is a query-conditioned expectation, ``y(q) = E_{x ~ pi_q}[v(x)]``.  Efficiency
has to come from somewhere, and there are only so many places: a method either evaluates
that expression differently or changes one of the objects in it.  This library implements
the methods in that taxonomy behind one interface, and holds every one of them to a single
invariant::

    y == weights @ (v_hat if v_hat is not None else value)

Quick start
-----------
>>> import torch, attnbank as ab
>>> q, k, v = (torch.randn(2, 4, 128, 64) for _ in range(3))
>>> y = ab.attend("topk", q, k, v, causal=True, k=32)        # the one-liner
>>> y.shape
torch.Size([2, 4, 128, 64])

>>> attn = ab.build("performer", r=128)                       # the object, for reuse
>>> out = attn(q, k, v, return_weights=True)
>>> out.y.shape, out.weights.shape
(torch.Size([2, 4, 128, 64]), torch.Size([2, 4, 128, 128]))

>>> rows = ab.compare(["softmax", "topk", "performer"], q, k, v,
...                   configs={"topk": {"k": 32}, "performer": {"r": 128}})
>>> rows[0]["output_error"] < 1e-6
True

Finding your way around
-----------------------
``ab.list_methods()`` prints the bank grouped by category, ``ab.describe(name)`` prints
one method in detail, ``ab.categories()`` explains the taxonomy, and
``ab.check_conformance(name)`` re-runs the contract on demand.

The companion monograph is *What Does Efficient Attention Approximate? A Unified View of
Interactions, Memory, and Error*; ``DESIGN.md`` is the binding specification.
"""
from __future__ import annotations

__version__ = "0.2.0"

from . import components, metrics
from .backends import (
    BackendFallbackWarning,
    BackendInfo,
    BackendNotSupportedError,
    backend_info,
    backends,
    default_backend,
    get_default_backend,
    is_backend_available,
    set_default_backend,
)
from .conformance import check_conformance, check_invariant
from .core import (
    AttentionMethod,
    AttentionOutput,
    Category,
    Objective,
    causal_mask,
    check_qkv,
    make_generator,
    masked_softmax,
    normalize_nonnegative,
    query_positions,
    renormalize,
    scores,
    weights_from_log_magnitude,
)
from .errors import (
    AttnBankError,
    CausalNotSupportedError,
    ConfigError,
    ConformanceError,
    ShapeError,
    UnknownMethodError,
)
from .info import MethodInfo, MethodTable
from .registry import (
    attend,
    available,
    backend_matrix,
    build,
    categories,
    compare,
    describe,
    get_class,
    list_methods,
    method_table,
    register,
)

__all__ = [
    "__version__",
    # ---- the flat surface: start here -------------------------------------
    "attend", "build", "compare", "available", "list_methods", "describe",
    "categories", "method_table", "get_class", "register",
    # ---- execution backends ------------------------------------------------
    "backends", "backend_info", "backend_matrix", "is_backend_available",
    "get_default_backend", "set_default_backend", "default_backend", "BackendInfo",
    # ---- the contract ------------------------------------------------------
    "check_invariant", "check_conformance",
    # ---- types -------------------------------------------------------------
    "AttentionMethod", "AttentionOutput", "Category", "Objective",
    "MethodInfo", "MethodTable",
    # ---- errors ------------------------------------------------------------
    "AttnBankError", "UnknownMethodError", "ShapeError", "CausalNotSupportedError",
    "ConfigError", "ConformanceError", "BackendNotSupportedError",
    "BackendFallbackWarning",
    # ---- building blocks for new methods -----------------------------------
    "causal_mask", "query_positions", "scores", "masked_softmax", "renormalize",
    "normalize_nonnegative", "make_generator", "weights_from_log_magnitude",
    "check_qkv",
    # ---- submodules --------------------------------------------------------
    "components", "metrics",
]
