"""The shared ``sdpa`` execution path (``backend="sdpa"``).

``torch.nn.functional.scaled_dot_product_attention`` is the production kernel that ships
with PyTorch: on CUDA it dispatches to FlashAttention-2 or to the memory-efficient
kernel, on CPU to a fused math path. It computes exactly

    y = softmax(beta <q, k> + mask) v,

which is the operator of ``softmax``, ``flash``, and -- after their own preprocessing --
``gqa`` and ``mla``. Those four declare the backend; nothing else in the bank does,
because nothing else in the bank *is* a dense softmax over a mask once its ``extras``
are accounted for.

Two details this helper exists to get right.

**Alignment.** ``is_causal=True`` masks with the diagonal aligned at the *top left*. The
library's convention (DESIGN.md Sec. 2) aligns the last query with the last key, so
query ``i`` sees ``0 .. i + (n - m)``. Those agree only when ``m == n``. The helper
always passes an explicit boolean mask built by :func:`attnbank.causal_mask`, so decode
(``m < n``) is correct rather than accidentally right.

**Kernel choice.** ``sdpa_kernel=`` pins the sub-backend -- ``"flash"``,
``"mem_efficient"``, ``"math"``, or ``None`` to let torch choose. Pinning matters for a
benchmark: "SDPA" is three different kernels with three different performance and
numerics profiles. An unavailable sub-backend raises rather than silently choosing
another.
"""
from __future__ import annotations

import contextlib
from typing import Optional

import torch
from torch import Tensor

from ..core import causal_mask

__all__ = ["sdpa_attend", "SDPA_KERNELS"]

#: sub-backends selectable through ``sdpa_kernel=``
SDPA_KERNELS = ("flash", "mem_efficient", "math", "cudnn")


def _kernel_context(kernel: Optional[str]):
    if kernel is None:
        return contextlib.nullcontext()
    if kernel not in SDPA_KERNELS:
        raise ValueError(
            f"sdpa_kernel must be one of {SDPA_KERNELS} or None, got {kernel!r}")
    from torch.nn.attention import SDPBackend
    from torch.nn.attention import sdpa_kernel as _ctx

    backend = {"flash": SDPBackend.FLASH_ATTENTION,
               "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
               "math": SDPBackend.MATH,
               "cudnn": getattr(SDPBackend, "CUDNN_ATTENTION", SDPBackend.MATH)}[kernel]
    return _ctx(backend)


def sdpa_attend(query: Tensor, key: Tensor, value: Tensor, scale: float,
                causal: bool = False, kernel: Optional[str] = None) -> Tensor:
    """``softmax(scale * q k^T + mask) v`` through PyTorch's fused kernel.

    Parameters
    ----------
    query, key, value:
        ``[B, H, m, d]``, ``[B, H, n, d]``, ``[B, H, n, dv]``.
    scale:
        the ``beta`` multiplying the logits; passed through rather than recomputed, so
        a method's own ``scale=`` override is honoured.
    causal:
        apply the library's end-aligned causal mask (**not** ``is_causal=True``; see
        the module docstring).
    kernel:
        pin a sub-backend, or ``None`` to let torch choose.

    Returns
    -------
    Tensor
        ``[B, H, m, dv]``, equal to the reference implementation within the tolerance
        asserted by ``tests/test_conformance.py::test_backend_equivalence``.

    Examples
    --------
    >>> import torch, attnbank as ab
    >>> from attnbank.components import sdpa_attend
    >>> q, k, v = (torch.randn(1, 2, 6, 8, dtype=torch.float64) for _ in range(3))
    >>> y = sdpa_attend(q, k, v, scale=8 ** -0.5, causal=True)
    >>> ref = ab.build("softmax", backend="reference")(q, k, v, causal=True).y
    >>> bool((y - ref).abs().max() < 1e-12)
    True
    """
    mask = None
    if causal:
        mask = causal_mask(query.shape[-2], key.shape[-2], device=query.device)
    with _kernel_context(kernel):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, scale=scale)
