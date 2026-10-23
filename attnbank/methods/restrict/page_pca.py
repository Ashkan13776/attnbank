"""Page-local key PCA: the key-only spectral page summary.

``sec:sparse-query`` ("Blocks and pages").  The same page selector as ``oval`` -- keys
are paged, each page stores ``(mu, B, R = X B)`` with ``B`` orthonormal of rank ``r``,
and a page is ranked by the estimated log-mass

    l_B(q) = beta mu^T q + lse(beta R (B^T q)),

the top ``n_pages`` complete pages being fetched and attended exactly.  The only
difference is the basis: here ``B`` holds the leading eigenvectors of ``X^T X``, the
directions along which the page's *keys* vary most.  Under isotropic queries that is
exactly the subspace minimizing the expected squared logit residual
``E_q ||(I - P_B) X q||^2``, so it is the right answer to the question "which ``r``
directions reconstruct this page's logits best" -- which is the question LOCKS and
Loki ask.

It is in the bank as the thing ``oval`` is argued against: ``oval`` claims that logit
fidelity is the wrong target because a logit error only matters when it moves mass
between tokens with different values, and this file is the ``eta = 0`` endpoint of
that argument made separately runnable.  The two are identical in stored size, in
decode-time scoring cost, and in every part of the pipeline except the eigenproblem,
which is the comparison the claim needs;  ``ab.build("oval", eta=0.0)`` is this method
and ``tests/test_restrict_learned.py`` asserts it.

Loki applies one *global* PCA basis to the whole key cache and selects tokens; LOCKS
fits a basis per page and selects pages.  This file is the per-page form, because that
is what makes it comparable to ``oval`` and to ``blocktopk``.

Reference: Hwang (2026) LOCKS; Singhania et al. (2024) Loki.
"""
from __future__ import annotations

from ...registry import register
from .oval import _SpectralPageSelector


@register
class PagePCAAttention(_SpectralPageSelector):
    """Page selection by a rank-r PCA basis of the page's centered keys."""

    name = "page_pca"
    references = ("hwang2026locks", "singhania2024loki", "tang2024quest")
    eta = 0.0
