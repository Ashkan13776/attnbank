"""The flat, top-level API: one import, a handful of obvious functions.

    import attnbank as ab

    ab.attend("topk", q, k, v, k=64)        # the one-liner
    ab.build("topk", k=64)                  # the object, for reuse
    ab.compare(["softmax", "topk"], q, k, v)
    ab.list_methods()                       # a readable table
    ab.describe("topk")                     # one method, in detail

Registration is a decorator: ``@register`` on an :class:`~attnbank.AttentionMethod`
subclass puts it in the bank, and from then on it appears in :func:`available`,
:func:`list_methods`, :func:`method_table` and -- the part that matters -- in the
conformance suite, which is parametrized over the registry.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union

from torch import Tensor

from .core import AttentionMethod, Category, Objective, check_qkv
from .errors import UnknownMethodError, unknown_method_message

__all__ = ["register", "build", "get_class", "available", "categories", "describe",
           "list_methods", "method_table", "attend", "compare"]

_REGISTRY: dict[str, type[AttentionMethod]] = {}


def register(cls: type[AttentionMethod]) -> type[AttentionMethod]:
    """Class decorator: add ``cls`` to the bank under ``cls.name``.

    Raises ``KeyError`` on a duplicate name, so two methods can never silently shadow
    each other.  Registration is all it takes to be picked up by the conformance
    suite -- there is no second list to update.

    Examples
    --------
    >>> import attnbank as ab
    >>> @ab.register                                     # doctest: +SKIP
    ... class MyAttention(ab.AttentionMethod):
    ...     name = "mine"
    ...     category = ab.Category.RESTRICT
    """
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise KeyError(f"duplicate method name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, **config) -> AttentionMethod:
    """Instantiate a registered method. The factory, in the ``from_pretrained`` spirit.

    Parameters
    ----------
    name:
        a registry key; see :func:`available`. An unknown name raises
        :class:`~attnbank.errors.UnknownMethodError` with near matches.
    **config:
        the method's own hyper-parameters -- ``k`` for ``topk``, ``r`` for
        ``performer``, ``bits_k`` / ``bits_v`` for ``quantkv``. ``ab.describe(name)``
        prints the accepted keys with their defaults.

    Returns
    -------
    AttentionMethod
        a ``torch.nn.Module``; call it as ``attn(query, key, value, causal=...)``.

    Examples
    --------
    >>> import torch, attnbank as ab
    >>> attn = ab.build("topk", k=32)
    >>> attn.name, attn.category.value
    ('topk', 'restrict')
    >>> q, k, v = (torch.randn(2, 4, 16, 32) for _ in range(3))
    >>> out = attn(q, k, v, causal=True, return_weights=True)
    >>> out.y.shape, out.weights.shape
    (torch.Size([2, 4, 16, 32]), torch.Size([2, 4, 16, 16]))
    >>> ab.build("topk", kk=4)                           # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    attnbank.errors.ConfigError: 'topk' got an unexpected config key 'kk'. Did you mean: 'k'?
    """
    cls = get_class(name)
    _check_config(cls, config)
    return cls(**config)


def _accepted_config(cls: type[AttentionMethod]) -> list[str]:
    """Named ``__init__`` parameters along the MRO, plus the base class's ``scale``."""
    import inspect as _inspect

    names = {"scale"}
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            sig = _inspect.signature(init)
        except (TypeError, ValueError):  # pragma: no cover
            continue
        for pname, p in sig.parameters.items():
            if pname == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            names.add(pname)
        if klass is AttentionMethod:
            break
    return sorted(names)


def _check_config(cls: type[AttentionMethod], config: dict) -> None:
    """Reject config keys the method does not accept, suggesting the near misses."""
    from .errors import ConfigError, suggest

    accepted = _accepted_config(cls)
    unknown = [key for key in config if key not in accepted]
    if not unknown:
        return
    bad = unknown[0]
    near = suggest(bad, accepted)
    hint = (" Did you mean: " + ", ".join(repr(c) for c in near) + "?") if near else ""
    raise ConfigError(
        f"{cls.name!r} got an unexpected config key {bad!r}.{hint}"
        f" It accepts: {', '.join(accepted)}."
        f" Run ab.describe({cls.name!r}) to see the defaults.")


def get_class(name: str) -> type[AttentionMethod]:
    """The registered class (not an instance), for reading class attributes.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.get_class("nystrom").row_stochastic
    False
    """
    _ensure_loaded()
    if name not in _REGISTRY:
        raise UnknownMethodError(unknown_method_message(name, sorted(_REGISTRY)))
    return _REGISTRY[name]


def available(category: Optional[Union[Category, str]] = None,
              objective: Optional[Union[Objective, str]] = None) -> list[str]:
    """Sorted method names, optionally filtered.

    Parameters
    ----------
    category:
        one of ``exact, interaction, restrict, compress, factorize, coupling,
        recurrent`` (see :func:`categories`).
    objective:
        one of ``accelerate, approximate, modify``.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.available("coupling")
    ['lowrank_ot', 'sinkhorn']
    >>> ab.available("exact", objective="accelerate")
    ['flash', 'softmax', 'splitk']
    """
    _ensure_loaded()
    names = sorted(_REGISTRY)
    if category is not None:
        cat = _coerce(Category, category, "category")
        names = [n for n in names if _REGISTRY[n].category == cat]
    if objective is not None:
        obj = _coerce(Objective, objective, "objective")
        names = [n for n in names if _REGISTRY[n].objective == obj]
    return names


def categories() -> dict[str, str]:
    """The taxonomy: ``{category: what methods in it change}``.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.categories()["coupling"]
    'normalisation stops being per-query; queries are coupled'
    """
    from .core import CATEGORY_DOC

    return {c.value: CATEGORY_DOC[c] for c in Category}


def describe(name: str) -> "Any":
    """Everything one method declares, as a printable :class:`~attnbank.info.MethodInfo`.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.describe("quantkv").expresses_weights
    True
    >>> ab.describe("linformer").row_stochastic
    False
    """
    from .info import MethodInfo

    return MethodInfo.from_class(get_class(name))


def list_methods(category: Optional[Union[Category, str]] = None,
                 objective: Optional[Union[Objective, str]] = None,
                 causal: Optional[bool] = None) -> "Any":
    """A readable table of the bank, grouped by category.

    The return value is a :class:`~attnbank.info.MethodTable`: a ``list`` of row dicts
    that renders as a table when printed, so it is equally good at the REPL and in a
    script.

    Parameters
    ----------
    category, objective:
        as in :func:`available`.
    causal:
        keep only methods that do (``True``) or do not (``False``) support causal
        masking.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.list_methods("recurrent").names
    ['decay_rnn', 'delta_rnn', 'linear_rnn']
    >>> table = ab.list_methods("coupling")
    >>> str(table).splitlines()[0]
    'coupling -- normalisation stops being per-query; queries are coupled'
    >>> [row["objective"] for row in table]
    ['modify', 'modify']
    """
    from .info import MethodInfo, MethodTable

    rows = []
    for name in available(category, objective):
        info = MethodInfo.from_class(_REGISTRY[name])
        if causal is not None and info.supports_causal is not causal:
            continue
        row = info.to_dict()
        row["causal"] = row.pop("supports_causal")
        rows.append(row)
    return MethodTable(rows)


def method_table() -> list[dict]:
    """One machine-readable row per method -- the paper's Appendix C table.

    Columns: ``name, category, objective, causal, expresses_weights, row_stochastic,
    head_independent, trainable, paper_section, limit_config, n_convergence_configs,
    references``.  Stable and JSON-friendly; use :func:`list_methods` when a human is
    reading.

    Examples
    --------
    >>> import attnbank as ab
    >>> {r["name"]: r["limit_config"] for r in ab.method_table()}["topk"]
    {'k': None}
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
# The one-liners
# ---------------------------------------------------------------------------

_QKV_ALIASES = (("query", "q"), ("key", "k"), ("value", "v"))


def _resolve_qkv(query, key, value, config: dict, where: str):
    """Accept ``query/key/value`` or the short ``q/k/v``, and free ``k`` for configs.

    ``k`` is the one genuinely ambiguous name: it used to mean the key tensor and it is
    also ``topk``'s budget.  The rule is unambiguous in practice -- a ``Tensor`` passed
    as ``k=`` is the key tensor, anything else is configuration -- and the preferred
    spelling, ``attend(name, q, k, v, k=64)``, needs no rule at all because the tensors
    are positional.  The historical trailing-underscore spelling (``k_=64``) still
    works.
    """
    bound = {"query": query, "key": key, "value": value}
    for long, short in _QKV_ALIASES:
        if short not in config:
            continue
        candidate = config[short]
        if bound[long] is None and isinstance(candidate, Tensor):
            bound[long] = config.pop(short)
    # historical spelling: a trailing underscore escaped a collision with q / k / v
    config = {(key_[:-1] if key_.endswith("_") and key_[:-1] in ("q", "k", "v") else key_): val
              for key_, val in config.items()}
    missing = [long for long, _ in _QKV_ALIASES if bound[long] is None]
    if missing:
        raise TypeError(
            f"{where} is missing {', '.join(missing)}. Call it as "
            f"{where.split('(')[0]}(name, query, key, value, **config) -- the tensors "
            "are positional, which is what frees `k=` to mean a method's own budget.")
    check_qkv(bound["query"], bound["key"], bound["value"], where=where)
    return bound["query"], bound["key"], bound["value"], config


def attend(name: str, query: Optional[Tensor] = None, key: Optional[Tensor] = None,
           value: Optional[Tensor] = None, causal: bool = False, **config) -> Tensor:
    """Run a method once and return only ``y``. The shortest useful call in the library.

    Parameters
    ----------
    name:
        a registry key; see :func:`available`.
    query, key, value:
        ``[B, H, m, d]``, ``[B, H, n, d]``, ``[B, H, n, dv]``. Positional, so the
        familiar ``attend("topk", q, k, v, ...)`` is unchanged.
    causal:
        mask so query ``i`` sees keys ``0 .. i + (n - m)``.
    **config:
        passed to :func:`build`. Because the tensors are positional, ``k=64`` means
        the method's budget, not the key tensor.

    Returns
    -------
    Tensor
        ``y``, of shape ``[B, H, m, dv]``. Use :func:`build` when you also want
        ``weights``, ``v_hat`` or ``extras``.

    Examples
    --------
    >>> import torch, attnbank as ab
    >>> q, k, v = (torch.randn(2, 4, 16, 32) for _ in range(3))
    >>> ab.attend("topk", q, k, v, causal=True, k=8).shape
    torch.Size([2, 4, 16, 32])
    >>> torch.allclose(ab.attend("flash", q, k, v), ab.attend("softmax", q, k, v),
    ...                atol=1e-6)
    True
    """
    query, key, value, config = _resolve_qkv(query, key, value, config, "ab.attend(...)")
    return build(name, **config)(query, key, value, causal=causal).y


def compare(names: Sequence[str], query: Optional[Tensor] = None,
            key: Optional[Tensor] = None, value: Optional[Tensor] = None,
            causal: bool = False, configs: Optional[dict[str, dict]] = None,
            reference: str = "softmax", raw: bool = False,
            **qkv_aliases) -> list[dict]:
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

    Parameters
    ----------
    names:
        methods to run, in order.
    query, key, value:
        the usual ``[B, H, *, *]`` tensors; positional.
    causal:
        passed to every method.
    configs:
        ``{method_name: {**config}}``; methods not mentioned use their defaults.
    reference:
        the method everything is scored against. ``"softmax"`` unless you are
        deliberately comparing against something else.
    raw:
        attach the per-query tensors under ``"_raw"``.

    Examples
    --------
    >>> import torch, attnbank as ab
    >>> q, k, v = (torch.randn(1, 2, 16, 16) for _ in range(3))
    >>> rows = ab.compare(["softmax", "topk", "window"], q, k, v,
    ...                   configs={"topk": {"k": 4}, "window": {"w": 4}})
    >>> [r["name"] for r in rows]
    ['softmax', 'topk', 'window']
    >>> rows[0]["output_error"] < 1e-6
    True
    >>> sorted(k for k in rows[1] if "error" in k)
    ['distributional_error', 'output_error', 'relative_error', 'value_error']
    """
    from . import metrics

    query, key, value, leftover = _resolve_qkv(query, key, value, dict(qkv_aliases),
                                               "ab.compare(...)")
    if leftover:
        raise TypeError(
            f"ab.compare() got unexpected keyword argument(s) "
            f"{', '.join(repr(key_) for key_ in leftover)}. Per-method settings go in "
            "configs={'topk': {'k': 64}}, not in the call itself.")
    q, k, v = query, key, value
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
            row.update({f"matrix_{key_}": med(val) for key_, val in mat.items()})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _coerce(enum_cls, value, what: str):
    """``Category("restrict")`` with an error that lists the legal values."""
    try:
        return enum_cls(value)
    except ValueError:
        legal = ", ".join(repr(m.value) for m in enum_cls)
        raise ValueError(
            f"unknown {what} {value!r}; expected one of {legal}") from None


_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        from . import reference  # noqa: F401
        from . import methods  # noqa: F401
