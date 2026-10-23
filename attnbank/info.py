"""Discoverability: readable descriptions of what is in the bank.

Two objects, both of which print themselves:

* :class:`MethodInfo` -- everything one method declares, in a block you can read.
  Reached through :func:`attnbank.describe` or ``AttentionMethod.describe()``.
* :class:`MethodTable` -- a list of rows that renders as a table grouped by category.
  Reached through :func:`attnbank.list_methods`.

Both keep their machine-readable form: ``MethodInfo.to_dict()`` and ``MethodTable``
itself (a ``list`` of ``dict``) are JSON-serialisable, so nothing here is only pretty.
"""
from __future__ import annotations

import inspect as _inspect
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .core import CATEGORY_DOC, AttentionMethod, Category, Objective

__all__ = ["MethodInfo", "MethodTable"]

_FLAG = {True: "yes", False: "no"}


def _summary(obj) -> str:
    """First sentence of a docstring, whitespace-collapsed."""
    doc = _inspect.getdoc(obj) or ""
    first = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    return " ".join(first.split())


@dataclass(frozen=True)
class MethodInfo:
    """Everything a registered method declares, in one printable object.

    Examples
    --------
    >>> import attnbank as ab
    >>> info = ab.describe("topk")
    >>> info.category, info.objective
    ('restrict', 'approximate')
    >>> info.limit_config
    {'k': None}
    >>> print(info)                                      # doctest: +ELLIPSIS
    topk  --  restrict / approximate
    ...
    """

    name: str
    class_name: str
    category: str
    objective: str
    supports_causal: bool
    expresses_weights: bool
    row_stochastic: bool
    head_independent: bool
    trainable: bool
    references: tuple = ()
    paper_section: str = ""
    backends: tuple = ("reference",)
    limit_config: Optional[dict] = None
    n_convergence_configs: int = 0
    summary: str = ""
    parameters: dict = field(default_factory=dict)
    doc: str = ""

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_class(cls, klass: type[AttentionMethod]) -> "MethodInfo":
        """Read the declaration attributes and ``__init__`` signature off a class."""
        conv = klass.convergence_configs()
        params: dict[str, Any] = {}
        try:
            sig = _inspect.signature(klass.__init__)
        except (TypeError, ValueError):  # pragma: no cover - builtins
            sig = None
        if sig is not None:
            for pname, p in sig.parameters.items():
                if pname in ("self", "kw", "kwargs") or p.kind is p.VAR_KEYWORD:
                    continue
                params[pname] = (None if p.default is p.empty else p.default)
        return cls(
            name=klass.name,
            class_name=klass.__name__,
            category=klass.category.value,
            objective=klass.objective.value,
            supports_causal=bool(klass.supports_causal),
            expresses_weights=bool(klass.expresses_weights),
            row_stochastic=bool(klass.row_stochastic),
            head_independent=bool(klass.head_independent),
            trainable=bool(klass.trainable),
            references=tuple(klass.references),
            paper_section=klass.paper_section,
            backends=tuple(klass.backends),
            limit_config=klass.limit_config(),
            n_convergence_configs=0 if conv is None else len(conv),
            summary=_summary(klass),
            parameters=params,
            doc=_inspect.getdoc(klass) or "",
        )

    # ---- machine-readable ---------------------------------------------------
    def to_dict(self) -> dict:
        """A plain ``dict`` (JSON-serialisable apart from ``references``, a tuple)."""
        d = asdict(self)
        d["references"] = list(self.references)
        d["backends"] = list(self.backends)
        return d

    # ---- human-readable -----------------------------------------------------
    def __str__(self) -> str:
        cat = Category(self.category)
        w = 78
        title = f"{self.name}  --  {self.category} / {self.objective}"
        lines = [title, "=" * len(title)]
        if self.summary:
            lines += textwrap.wrap(self.summary, w)
        lines += ["", f"changes      {CATEGORY_DOC[cat]}",
                  f"objective    {Objective(self.objective).description}"]
        anchor = ("limit_config = " + repr(self.limit_config)) if self.limit_config is not None \
            else (f"{self.n_convergence_configs} convergence configs"
                  if self.n_convergence_configs else
                  "none (objective = modify; it does not target softmax)")
        lines += [
            f"causal       {_FLAG[self.supports_causal]}",
            f"weights      {_FLAG[self.expresses_weights]}"
            + ("" if self.expresses_weights else
               "   (excluded from the error decomposition, by declaration)"),
            f"stochastic   {_FLAG[self.row_stochastic]}",
            "heads        " + ("independent" if self.head_independent
                               else "coupled (shares K/V across query heads)"),
            f"trainable    {_FLAG[self.trainable]}",
            f"anchor       {anchor}",
            f"backends     {', '.join(self.backends)}"
            + ("   (reference only: no equivalent kernel exists for this operator)"
               if tuple(self.backends) == ("reference",) else ""),
        ]
        if self.parameters:
            args = ", ".join(f"{key}={val!r}" for key, val in self.parameters.items())
            lines += ["", f"build        ab.build({self.name!r}, {args})"]
        if self.paper_section:
            lines += [f"section      {self.paper_section}"]
        if self.references:
            lines += [f"references   {', '.join(self.references)}"]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return str(self)


class MethodTable(list):
    """A ``list`` of row dicts that prints as a table grouped by category.

    It is a real list, so ``[r["name"] for r in table]`` and ``json.dumps(table)``
    work; the only addition is the rendering.

    Examples
    --------
    >>> import attnbank as ab
    >>> table = ab.list_methods("restrict")
    >>> sorted(r["name"] for r in table)[:3]
    ['blocktopk', 'lsh_sample', 'sink_window']
    >>> print(table)                                     # doctest: +ELLIPSIS
    restrict -- the support of pi_q is restricted...
    ...
    """

    #: columns rendered, in order, as ``(key, header, width)``
    COLUMNS: Sequence[tuple] = (("name", "method", 16), ("objective", "objective", 11),
                                ("causal", "causal", 6), ("expresses_weights", "weights", 7),
                                ("row_stochastic", "stoch.", 6),
                                ("backends", "backends", 8))

    def __init__(self, rows: Iterable[dict] = ()):
        super().__init__(rows)

    @property
    def names(self) -> list[str]:
        """Just the method names, in table order."""
        return [r["name"] for r in self]

    def filter(self, **constraints) -> "MethodTable":
        """Rows matching every ``column=value`` constraint.

        Examples
        --------
        >>> import attnbank as ab
        >>> ab.list_methods().filter(objective="modify", causal=True).names[:2]
        ['cosformer', 'decay_rnn']
        """
        return MethodTable(r for r in self
                           if all(r.get(key) == val for key, val in constraints.items()))

    def __str__(self) -> str:
        if not self:
            return "(no methods match)"
        cells = {}
        widths = {}
        for key, header, minw in self.COLUMNS:
            vals = [_cell(r.get(key)) for r in self]
            cells[key] = vals
            widths[key] = max(minw, len(header), *(len(x) for x in vals))
        head = "  ".join(h.ljust(widths[key]) for key, h, _ in self.COLUMNS)
        head += "  summary"
        rule = "-" * min(100, len(head) + 30)

        out: list[str] = []
        order = [c.value for c in Category]
        for cat in sorted({r["category"] for r in self},
                          key=lambda c: order.index(c) if c in order else 99):
            doc = CATEGORY_DOC.get(Category(cat), "")
            out += ["", f"{cat} -- {doc}", rule, head, rule]
            for i, r in enumerate(self):
                if r["category"] != cat:
                    continue
                row = "  ".join(cells[key][i].ljust(widths[key])
                                for key, _, _ in self.COLUMNS)
                summary = r.get("summary", "")
                out.append((row + "  " + summary).rstrip())
        n = len(self)
        out += ["", f"{n} method{'s' if n != 1 else ''}.  "
                    "ab.describe(name) for one of them; ab.build(name, **config) to use it."]
        return "\n".join(out[1:])

    def __repr__(self) -> str:
        return str(self)


def _cell(val) -> str:
    if isinstance(val, (list, tuple)):
        return ",".join(str(x) for x in val)
    if val is True:
        return "yes"
    if val is False:
        return "-"
    if val is None:
        return ""
    return str(val)
