"""Last-loop KV sharing for looped models: the shared rule, the K2 training forward and exact replay."""
from .flash import flash_available
from .policy import config_keys, kd_policy
from .replay import exact_document, exact_documents
from .rule import AttentionPlan, Layout, k2_forward, loop_count, native_forward, pad_token

__all__ = ["AttentionPlan", "Layout", "config_keys", "exact_document", "exact_documents", "flash_available",
           "k2_forward", "kd_policy", "loop_count", "native_forward", "pad_token"]
