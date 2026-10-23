"""Shared components: the parts a method *reuses*, so a method file shows what it *adds*.

Five groups of building blocks, each the single home of one recurring behaviour:

``attnbank.components.masking``
    the causal convention, visibility masks, log-space softmax, renormalisation
    (re-exported from :mod:`attnbank.core`, which is where new methods usually find them).
``attnbank.components.online_softmax``
    the state / merge / readout monoid of ``prop:exact-monoid``, shared by ``flash``
    (one pass over key blocks) and ``splitk`` (two-level reduction).
``attnbank.components.quantization``
    quantise--dequantise in every flavour the bank needs: symmetric INT, asymmetric
    affine with grouping, the FP8 grids, and the Hadamard rotation. Shared by
    ``lowprec`` (inside the kernel) and ``quantkv`` (on the cache).
``attnbank.components.restriction``
    the renormalised restriction of ``prop:sparse-identity``. Every method in
    ``restrict/`` -- and the evictions in ``compress/`` -- differ only in how they
    choose ``S_q``; the operator afterwards is this.
``attnbank.components.recurrence``
    feature maps and the readout convention of ``eq:memory-generalrec``.
``attnbank.components.sdpa``
    the shared ``torch.nn.functional.scaled_dot_product_attention`` execution path used
    by every ``_forward_sdpa``, including the end-alignment convention that
    ``is_causal=True`` gets wrong when ``m < n``.

Nothing here is a method. Everything here is tested through the methods that use it and
through the conformance suite.

Examples
--------
>>> from attnbank import components
>>> sorted(components.__all__)[:4]
['FEATURES', 'affine_quantize', 'causal_mask', 'feature_map']
"""
from __future__ import annotations

from ..core import (
                    causal_mask,
                    masked_softmax,
                    normalize_nonnegative,
                    query_positions,
                    renormalize,
                    scores,
                    weights_from_log_magnitude,
)
from .online_softmax import merge_states, online_state, readout
from .quantization import affine_quantize, hadamard, quantize_fp8, quantize_int
from .recurrence import FEATURES, feature_map, readout_positions
from .restriction import reference_pi, restrict, topk_support, visible
from .sdpa import sdpa_attend

__all__ = [
    # masking / normalisation (home: attnbank.core)
    "causal_mask", "query_positions", "scores", "masked_softmax", "renormalize",
    "normalize_nonnegative", "weights_from_log_magnitude",
    # online softmax monoid
    "online_state", "merge_states", "readout",
    # quantisation
    "quantize_int", "quantize_fp8", "affine_quantize", "hadamard",
    # restriction
    "reference_pi", "visible", "restrict", "topk_support",
    # recurrence
    "FEATURES", "feature_map", "readout_positions",
    # execution
    "sdpa_attend",
]
