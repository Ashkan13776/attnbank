"""attnbank: a modular bank of attention mechanisms organized by what they change.

    import attnbank as ab
    attn = ab.build("softmax")
    out = attn(q, k, v, causal=True, return_weights=True)
    ab.attend("topk", q, k, v, k_=64)
    ab.compare(["softmax", "topk", "performer"], q, k, v, causal=True)
"""
from .core import (AttentionMethod, AttentionOutput, Category, Objective, causal_mask,
                   query_positions, scores, masked_softmax, renormalize, normalize_nonnegative, make_generator,
                   weights_from_log_magnitude)
from .registry import (available, attend, build, compare, get_class, method_table,
                       register)
from . import metrics

__all__ = ["AttentionMethod", "AttentionOutput", "Category", "Objective", "causal_mask",
           "query_positions", "scores", "masked_softmax", "renormalize", "normalize_nonnegative", "make_generator",
           "weights_from_log_magnitude",
           "available", "attend", "build", "compare", "get_class", "method_table",
           "register", "metrics"]
