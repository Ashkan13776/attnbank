"""Methods in category 'compress': the stored (mu, k, v) is replaced by a cheaper one."""
from . import (  # noqa: F401
    evict_h2o,
    evict_snapkv,
    gqa,
    linformer,
    lowrank_kv,
    merge_kv,
    mla,
    quantkv,
)

__all__ = ["evict_h2o", "evict_snapkv", "gqa", "linformer", "lowrank_kv", "merge_kv",
           "mla", "quantkv"]
