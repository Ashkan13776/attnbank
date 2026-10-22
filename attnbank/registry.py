"""Method registry: ``@register`` a class, ``build(name, **cfg)`` an instance."""
from __future__ import annotations

from typing import Optional, Sequence

from torch import Tensor

from .core import AttentionMethod, Category

_REGISTRY: dict[str, type[AttentionMethod]] = {}


def register(cls: type[AttentionMethod]) -> type[AttentionMethod]:
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise KeyError(f"duplicate method name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, **cfg) -> AttentionMethod:
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**cfg)


def get_class(name: str) -> type[AttentionMethod]:
    """The registered class (not an instance), for inspecting class attributes."""
    _ensure_loaded()
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available(category: Optional[Category | str] = None) -> list[str]:
    _ensure_loaded()
    if category is None:
        return sorted(_REGISTRY)
    category = Category(category)
    return sorted(n for n, c in _REGISTRY.items() if c.category == category)


def method_table() -> list[dict]:
    """One row per method, for the paper's table (Appendix C).

    Columns: name, category, objective, causal, expresses_weights, row_stochastic,
    head_independent, trainable, paper_section, limit_config, n_convergence_configs,
    references.
    """
    _ensure_loaded()
    rows = []
    for n, c in sorted(_REGISTRY.items()):
        lim = c.limit_config()
        conv = c.convergence_configs()
        rows.append(dict(
            name=n,
            category=c.category.value,
            objective=c.objective.value,
            causal=c.supports_causal,
            expresses_weights=c.expresses_weights,
            row_stochastic=c.row_stochastic,
            head_independent=c.head_independent,
            trainable=c.trainable,
            paper_section=c.paper_section,
            limit_config=lim,
            n_convergence_configs=0 if conv is None else len(conv),
            references=",".join(c.references),
        ))
    return rows


# ---------------------------------------------------------------------------
# Convenience API (DESIGN.md Sec. 2)
# ---------------------------------------------------------------------------

def attend(name: str, q: Tensor, k: Tensor, v: Tensor, causal: bool = False,
           **cfg) -> Tensor:
    """One-liner: build ``name`` with ``cfg`` and return only ``y``.

    Config keys that collide with the positional names are spelled with a trailing
    underscore, e.g. ``attend("topk", q, k, v, k_=64)`` sets the budget ``k=64``.
    """
    cfg = {(key[:-1] if key.endswith("_") else key): val for key, val in cfg.items()}
    return build(name, **cfg)(q, k, v, causal=causal).y


def compare(names: Sequence[str], q: Tensor, k: Tensor, v: Tensor, causal: bool = False,
            configs: Optional[dict[str, dict]] = None,
            reference: str = "softmax", raw: bool = False) -> list[dict]:
    """Run several methods on the same inputs and score each against the reference.

    Returns one dict per method with the per-query error statistics the paper uses:
    output error (absolute and relative), the two terms of ``eq:error_decomposition``,
    total variation, and the matrix norms of Sec. 4.2 -- all reduced to a median over
    queries, batch and heads.  The default return is JSON-serializable; pass
    ``raw=True`` to add the per-query tensors and the method's ``extras`` under
    ``"_raw"``.

    Methods that cannot express weights on the original positions
    (``expresses_weights = False``) report only the output error; the distributional /
    value split is undefined for them by design, not by accident.
    """
    from . import metrics

    configs = configs or {}
    ref_out = build(reference, **configs.get(reference, {}))(
        q, k, v, causal=causal, return_weights=True)
    pi, y = ref_out.weights, ref_out.y

    def med(t: Tensor) -> float:
        return float(t.double().median())

    rows = []
    for name in names:
        cls = get_class(name)
        out = build(name, **configs.get(name, {}))(q, k, v, causal=causal, return_weights=True)
        err = metrics.output_error(y, out.y)
        row: dict = {
            "name": name,
            "category": cls.category.value,
            "objective": cls.objective.value,
            "output_error": med(err),
            "relative_error": med(metrics.output_error(y, out.y, relative=True)),
        }
        if raw:
            row["_raw"] = {"y_hat": out.y, "output_error": err, "weights": out.weights,
                           "v_hat": out.v_hat, "extras": out.extras}
        if out.weights is not None:
            dec = metrics.error_decomposition(pi, out.weights, v, out.v_hat)
            row["distributional_error"] = med(dec["distributional"])
            row["value_error"] = med(dec["value"])
            row["tv"] = med(metrics.total_variation(pi, out.weights))
            mat = metrics.attention_matrix_errors(pi, out.weights, v)
            row.update({f"matrix_{key}": med(val) for key, val in mat.items()})
        rows.append(row)
    return rows


_loaded = False


def _ensure_loaded():
    global _loaded
    if not _loaded:
        _loaded = True
        from . import reference  # noqa: F401
        from . import methods  # noqa: F401
