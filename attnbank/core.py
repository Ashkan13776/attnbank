"""Core interface for attnbank.

Every efficient-attention method is an ``AttentionMethod``. The interface mirrors
the paper's framework: attention is y(q) = E_{x ~ pi_q}[v(x)], and a method may
change the evaluation (exact), the interaction g, the accessible support, the
context representation (mu, k, v), the factorization of pi_q, or the operator.

Tensor convention (all methods): ``query [B, H, m, d]``, ``key [B, H, n, d]``,
``value [B, H, n, dv]``.  Causal masking aligns the last query with the last key
(decode-friendly), so ``m <= n`` and query ``i`` sees keys ``0 .. i + (n - m)``.

The public argument names are ``query, key, value`` -- the transformers convention,
which also frees ``k`` to mean a method's own budget (``ab.attend("topk", q, k, v,
k=64)``).  They remain positional, so ``attn(q, k, v)`` is unchanged, and the short
spellings ``q=``, ``k=``, ``v=`` are still accepted as keywords for tensors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor

from .backends import REFERENCE, resolve_backend
from .errors import CausalNotSupportedError, ShapeError, shape_message


class Category(str, Enum):
    """What object in ``y(q) = int v dpi_q`` a method changes -- the paper's taxonomy.

    Examples
    --------
    >>> from attnbank import Category
    >>> Category.RESTRICT.value
    'restrict'
    >>> import attnbank as ab
    >>> ab.available(Category.COUPLING)
    ['lowrank_ot', 'sinkhorn']
    """

    EXACT = "exact"              # same operator, different execution
    INTERACTION = "interaction"  # replaces g by g_hat
    RESTRICT = "restrict"        # restricts support of pi_q to S_q
    COMPRESS = "compress"        # replaces (mu, k, v) by a cheaper representation
    FACTORIZE = "factorize"      # pi_q ~ a(q, .) b(., x) through a latent space
    COUPLING = "coupling"        # global constraints across queries (e.g. OT)
    RECURRENT = "recurrent"      # finite-state operators (linear RNN / SSM)

    @property
    def description(self) -> str:
        """One line of prose describing what this category changes.

        Examples
        --------
        >>> from attnbank import Category
        >>> Category.RESTRICT.description
        'the support of pi_q is restricted to a subset S_q of positions'
        """
        return CATEGORY_DOC[self]


class Objective(str, Enum):
    """What a method is *trying* to be -- the paper's third axis.

    Conflating these is the most persistent confusion in the literature: a method that
    **approximates** softmax is judged by fidelity to it, a method that **modifies** it
    is a different operator and fidelity is not its goal.

    Examples
    --------
    >>> import attnbank as ab
    >>> ab.available(objective=ab.Objective.MODIFY)[:3]
    ['cosformer', 'decay_rnn', 'delta_rnn']
    """

    ACCELERATE = "accelerate"    # exact up to floating point
    APPROXIMATE = "approximate"  # targets fidelity to the reference operator
    MODIFY = "modify"            # defines a different operator

    @property
    def description(self) -> str:
        """One line of prose. ``Objective.MODIFY.description`` -> ``'defines a ...'``."""
        return OBJECTIVE_DOC[self]


CATEGORY_DOC: dict = {}
OBJECTIVE_DOC: dict = {
    Objective.ACCELERATE: "same operator, exact up to floating point; only the "
                          "evaluation changes",
    Objective.APPROXIMATE: "targets fidelity to dense softmax; judged by output error",
    Objective.MODIFY: "defines a different operator; fidelity to softmax is not the goal",
}

CATEGORY_DOC.update({
    Category.EXACT: "nothing changes but the evaluation: tiling, chunking, precision",
    Category.INTERACTION: "the interaction g(q, x) is replaced by a cheaper surrogate",
    Category.RESTRICT: "the support of pi_q is restricted to a subset S_q of positions",
    Category.COMPRESS: "the stored context (mu, k, v) is replaced by a smaller one",
    Category.FACTORIZE: "pi_q is factored through r latents: pi_q = a(q, .) b(., x)",
    Category.COUPLING: "normalisation stops being per-query; queries are coupled",
    Category.RECURRENT: "the operator itself becomes a finite-state recurrence",
})


@dataclass
class AttentionOutput:
    """Result of a forward pass. Four fields, exactly -- nothing else (DESIGN.md Sec. 2).

    Attributes
    ----------
    y:
        output, ``[B, H, m, dv]``.
    weights:
        the implied conditional measure ``pi_hat`` over the **original** ``n`` context
        positions, ``[B, H, m, n]``, or ``None`` when the method declares
        ``expresses_weights = False``.
    v_hat:
        the value field actually aggregated, on the original positions,
        ``[B, H, n, dv]``, or ``None`` when it equals ``value`` (no value compression).
    extras:
        method-specific diagnostics -- support indices, omitted mass, normalizer
        statistics. Keys are documented by each method; nothing in the library's own
        contract depends on them.

    Invariant W holds for every registered method::

        y == weights @ (v_hat if v_hat is not None else value)

    Examples
    --------
    >>> import torch, attnbank as ab
    >>> q, k, v = (torch.randn(1, 2, 4, 8) for _ in range(3))
    >>> out = ab.build("topk", k=2)(q, k, v, return_weights=True)
    >>> out                                              # doctest: +ELLIPSIS
    AttentionOutput(y=[1, 2, 4, 8], weights=[1, 2, 4, 4], v_hat=None, extras=[...],
                    dtype=float32, device=cpu)
    >>> torch.allclose(out.y, out.weights @ v, atol=1e-6)
    True
    """

    y: Tensor
    weights: Optional[Tensor] = None
    v_hat: Optional[Tensor] = None
    extras: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        def shp(t: Optional[Tensor]) -> str:
            if t is None:
                return "None"
            return "[" + ", ".join(str(int(x)) for x in t.shape) + "]"

        extras = "[" + ", ".join(sorted(map(str, self.extras))) + "]" if self.extras             else "{}"
        tail = ""
        if self.y is not None:
            tail = f", dtype={str(self.y.dtype).removeprefix('torch.')}, " \
                   f"device={self.y.device}"
        return (f"AttentionOutput(y={shp(self.y)}, weights={shp(self.weights)}, "
                f"v_hat={shp(self.v_hat)}, extras={extras}{tail})")

    def check_invariant(self, value: Optional[Tensor] = None,
                        atol: Optional[float] = None) -> float:
        """Verify ``y == weights @ (v_hat or value)`` and return the max deviation.

        Delegates to :func:`attnbank.conformance.check_invariant`; see there for the
        tolerances and the exception raised on failure.
        """
        from .conformance import check_invariant

        return check_invariant(self, value, atol=atol)


TINY = 1e-30  # denominator floor; tests assert it is never actually reached


def causal_mask(m: int, n: int, device=None) -> Tensor:
    """Boolean ``[m, n]`` mask, ``True`` where key ``j`` is visible to query ``i``.

    Queries are aligned with the **last** keys, so query ``i`` sees ``0 .. i + (n - m)``.
    The same code path therefore serves prefill (``m == n``) and decode (``m < n``).

    Examples
    --------
    >>> from attnbank import causal_mask
    >>> causal_mask(2, 3).int()
    tensor([[1, 1, 0],
            [1, 1, 1]], dtype=torch.int32)
    """
    i = torch.arange(m, device=device)[:, None] + (n - m)
    j = torch.arange(n, device=device)[None, :]
    return j <= i


def query_positions(m: int, n: int, device=None) -> Tensor:
    """Absolute context position of each query under the end-alignment convention: ``[m]``.

    Examples
    --------
    >>> from attnbank import query_positions
    >>> query_positions(3, 5)
    tensor([2, 3, 4])
    """
    return torch.arange(m, device=device) + (n - m)


def scores(q: Tensor, k: Tensor, scale: float, causal: bool) -> Tensor:
    """Scaled logits ``s = beta <q, k_j>``, ``-inf`` on masked entries, ``[..., m, n]``.

    Examples
    --------
    >>> import torch
    >>> from attnbank import scores
    >>> q, k = torch.ones(1, 1, 2, 4), torch.ones(1, 1, 3, 4)
    >>> scores(q, k, scale=0.5, causal=True)[0, 0]
    tensor([[2., 2., -inf],
            [2., 2., 2.]])
    """
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

    Examples
    --------
    >>> import torch
    >>> from attnbank import masked_softmax
    >>> s = torch.tensor([[0.0, 0.0, 0.0]])
    >>> masked_softmax(s, support=torch.tensor([[True, True, False]]))
    tensor([[0.5000, 0.5000, 0.0000]])
    >>> masked_softmax(s, support=torch.tensor([[False, False, False]]))  # no nan
    tensor([[0.3333, 0.3333, 0.3333]])
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

    Examples
    --------
    >>> import torch
    >>> from attnbank import weights_from_log_magnitude
    >>> pi, z = weights_from_log_magnitude(torch.tensor([[0.0, 0.0, -1e30]]))
    >>> pi
    tensor([[0.5000, 0.5000, 0.0000]])
    >>> float(pi.sum())
    1.0
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
    """Divide by the row sum, flooring the denominator at ``TINY``.

    The floor is never actually reached in the test suite; it exists so that a
    pathological row returns a finite number instead of ``nan`` (DESIGN.md Sec. 4).

    Examples
    --------
    >>> import torch
    >>> from attnbank import renormalize
    >>> renormalize(torch.tensor([[1.0, 3.0]]))
    tensor([[0.2500, 0.7500]])
    """
    return w / w.sum(-1, keepdim=True).clamp_min(TINY)


def normalize_nonnegative(g: Tensor, support: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    """Turn a nonnegative surrogate interaction into a probability measure.

    Entries outside ``support`` get zero mass.  A row whose total is zero -- possible
    when a feature map annihilates the query (``relu``) or underflows -- falls back to
    the uniform distribution on its support, the documented finite result required by
    DESIGN.md Sec. 4 rule 10.  The unnormalized row sums are returned alongside so that
    the fallback can be counted rather than hidden.

    Examples
    --------
    >>> import torch
    >>> from attnbank import normalize_nonnegative
    >>> pi, z = normalize_nonnegative(torch.tensor([[1.0, 3.0, 0.0]]))
    >>> pi
    tensor([[0.2500, 0.7500, 0.0000]])
    >>> normalize_nonnegative(torch.zeros(1, 3))[0]        # uniform, never nan
    tensor([[0.3333, 0.3333, 0.3333]])
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
    """Seeded ``torch.Generator``; randomness never touches the global RNG.

    Every stochastic method draws through one of these, which is why two builds of the
    same method agree bit for bit and why studies reproduce (DESIGN.md Sec. 4).

    Examples
    --------
    >>> import torch
    >>> from attnbank import make_generator
    >>> a = torch.randn(3, generator=make_generator(0))
    >>> b = torch.randn(3, generator=make_generator(0))
    >>> bool((a == b).all())
    True
    """
    g = torch.Generator(device="cpu" if device is None else device)
    g.manual_seed(int(seed))
    return g


def check_qkv(query: Tensor, key: Tensor, value: Tensor,
              where: str = "") -> None:
    """Validate the ``[B, H, n, d]`` convention, naming the argument that is wrong.

    Raises :class:`~attnbank.errors.ShapeError` with the offending argument, what was
    expected and what arrived -- never a bare broadcasting failure from deep inside a
    method.

    Examples
    --------
    >>> import torch
    >>> from attnbank.core import check_qkv
    >>> check_qkv(torch.randn(1, 2, 4, 8), torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 3))
    >>> check_qkv(torch.randn(4, 8), torch.randn(1, 2, 6, 8), torch.randn(1, 2, 6, 3))
    Traceback (most recent call last):
        ...
    attnbank.errors.ShapeError: query must be [B, H, m, d]; got [4, 8] (2 dims, expected 4)
    """
    suffix = f" (in {where})" if where else ""
    for arg, t, spec in (("query", query, "[B, H, m, d]"), ("key", key, "[B, H, n, d]"),
                         ("value", value, "[B, H, n, dv]")):
        if not isinstance(t, Tensor):
            raise ShapeError(f"{arg} must be a torch.Tensor{suffix}; got "
                             f"{type(t).__name__}. Tensors are {spec}.")
        if t.dim() != 4:
            raise ShapeError(shape_message(arg + suffix, spec, t.shape,
                                           f"{t.dim()} dims, expected 4"))
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != value.shape[:2]:
        raise ShapeError(
            f"query, key and value must share the leading [B, H]{suffix}; got "
            f"query{list(query.shape[:2])}, key{list(key.shape[:2])}, "
            f"value{list(value.shape[:2])}")
    if key.shape[-2] != value.shape[-2]:
        raise ShapeError(
            shape_message("value" + suffix, f"[B, H, n, dv] with n = {key.shape[-2]} "
                          "(the number of keys)", value.shape,
                          f"n = {value.shape[-2]} values for {key.shape[-2]} keys"))
    if query.shape[-1] != key.shape[-1]:
        raise ShapeError(
            shape_message("key" + suffix, f"[B, H, n, d] with d = {query.shape[-1]} "
                          "(the query head dimension)", key.shape,
                          f"d = {key.shape[-1]}"))
    if query.shape[-2] > key.shape[-2]:
        raise ShapeError(
            f"m = {query.shape[-2]} queries for n = {key.shape[-2]} keys{suffix}. "
            "Causal masking aligns the LAST query with the LAST key, so m <= n is "
            "required; pass the full key/value context alongside the query block.")


class AttentionMethod(torch.nn.Module):
    """Base class for every method in the bank.

    A subclass declares what it is through class attributes and implements one method,
    :meth:`_forward`.  Everything else -- validation, the causal guard, the registry
    entry, the conformance suite -- follows from the declaration.

    Declaration attributes
    ----------------------
    ``name``, ``category``, ``objective``, ``supports_causal``, ``expresses_weights``,
    ``row_stochastic``, ``head_independent``, ``trainable``, ``references``,
    ``paper_section``.  See DESIGN.md Sec. 2 for the exact meaning of each; the
    conformance suite tests the method against what it declares, so an attribute is a
    commitment, not documentation.

    Examples
    --------
    >>> import torch, attnbank as ab
    >>> attn = ab.build("topk", k=4)
    >>> attn
    TopKAttention(name='topk', category='restrict', objective='approximate',
                  backend='reference', k=4)
    >>> q, k, v = (torch.randn(1, 2, 8, 16) for _ in range(3))
    >>> attn(q, k, v, causal=True).y.shape
    torch.Size([1, 2, 8, 16])
    """

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

    #: execution paths this method claims are *equal to the reference*. Declaring a
    #: backend here is a claim the conformance suite checks
    #: (``test_backend_equivalence``), so ``("reference",)`` -- the default -- is the
    #: correct and common answer: most operators in the bank are not a dense softmax
    #: over a mask, or need the full score matrix for their ``extras`` anyway.
    backends: tuple = (REFERENCE,)

    def __init__(self, scale: Optional[float] = None, backend: Optional[str] = None,
                 **kwargs):
        super().__init__()
        self.scale = scale
        self.config = kwargs
        #: what the caller asked for (``None`` = "use the default")
        self.requested_backend = backend
        #: what will actually run. See :func:`attnbank.backends.resolve_backend`.
        self.backend = resolve_backend(backend, type(self).backends, self.name)

    # ---- backend dispatch ---------------------------------------------------
    def _impl(self):
        """The bound ``_forward_<backend>`` for the resolved backend, else ``_forward``."""
        return getattr(self, f"_forward_{self.backend}", self._forward)

    def _fallback_to_reference(self, why: str):
        """Announce a per-call fallback and hand back the reference implementation.

        Used when a kernel simply cannot produce what this call asked for -- most often
        ``return_weights=True``, which no fused kernel materialises. ``transformers``
        does the same thing for ``output_attentions`` with ``attn_implementation="sdpa"``;
        the point is that it is said out loud.
        """
        import warnings

        from .backends import BackendFallbackWarning

        warnings.warn(
            f"{self.name!r}: backend {self.backend!r} cannot {why}; this call falls back "
            f"to {REFERENCE!r}. The result is identical -- only the speed is not.",
            BackendFallbackWarning, stacklevel=3)
        return self._forward

    def _scale(self, d: int) -> float:
        return self.scale if self.scale is not None else 1.0 / math.sqrt(d)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, causal: bool = False,
                return_weights: bool = False) -> AttentionOutput:
        """Run the method.

        Parameters
        ----------
        query:
            ``[B, H, m, d]``. Also accepted positionally as ``q``.
        key:
            ``[B, H, n, d]``.
        value:
            ``[B, H, n, dv]``.
        causal:
            mask so that query ``i`` sees keys ``0 .. i + (n - m)``. Raises
            :class:`~attnbank.errors.CausalNotSupportedError` -- which names the
            methods that do support it -- when the method declares otherwise.
        return_weights:
            also materialise ``pi_hat`` on the original ``n`` positions. Costs
            ``O(m n)`` memory, so it is off by default; the conformance suite and
            :func:`attnbank.compare` turn it on.

        Returns
        -------
        AttentionOutput
            with ``y``, and ``weights`` / ``v_hat`` when the method can express them.
        """
        check_qkv(query, key, value, where=f"{self.name}.forward")
        if causal and not self.supports_causal:
            raise CausalNotSupportedError(self._causal_message())
        return self._impl()(query, key, value, causal=causal,
                            return_weights=return_weights)

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

    # ---- presentation -------------------------------------------------------
    def extra_repr(self) -> str:  # pragma: no cover - torch.nn.Module hook
        return ", ".join(f"{key}={val!r}" for key, val in sorted(self.config.items()))

    def __repr__(self) -> str:
        """``TopKAttention(name='topk', category='restrict', objective=..., k=4)``."""
        cfg = ", ".join(f"{key}={val!r}" for key, val in sorted(self.config.items())
                        if not key.startswith("_"))
        head = (f"name={self.name!r}, category={self.category.value!r}, "
                f"objective={self.objective.value!r}, backend={self.backend!r}")
        return f"{type(self).__name__}({head}{', ' + cfg if cfg else ''})"

    @classmethod
    def describe(cls) -> "MethodInfo":
        """A printable summary: what it changes, what it declares, where to read more.

        Examples
        --------
        >>> import attnbank as ab
        >>> print(ab.get_class("nystrom").describe())    # doctest: +ELLIPSIS
        nystrom ...
        """
        from .info import MethodInfo

        return MethodInfo.from_class(cls)


    def _causal_message(self) -> str:
        """``causal=True`` was refused: say so, and name the methods that accept it."""
        from .registry import available, get_class

        try:
            same = [n for n in available(self.category) if get_class(n).supports_causal]
            allc = [n for n in available() if get_class(n).supports_causal]
        except Exception:  # pragma: no cover - registry unavailable
            same, allc = [], []
        if same:
            alt = (f" Methods in the same category ({self.category.value}) that do: "
                   + ", ".join(repr(n) for n in same) + ".")
        elif allc:
            alt = (f" No method in the {self.category.value} category supports it; "
                   f"{len(allc)} others do, e.g. "
                   + ", ".join(repr(n) for n in allc[:5])
                   + ". ab.list_methods(causal=True) lists them all.")
        else:  # pragma: no cover
            alt = ""
        return (f"{self.name!r} does not support causal attention: it declares "
                f"supports_causal = False, which is a statement about the operator, "
                f"not a missing feature.{alt} Pass causal=False to use it as it is.")


if TYPE_CHECKING:  # pragma: no cover
    from .info import MethodInfo
