"""Hydra bounded-residency attention package.

This package currently exposes the extracted Triton attention kernels and
runtime policy layer from the research repo. The public API is deliberately
small while the benchmark evidence is being consolidated.
"""

from .api import flash_attn_blackwell, hydra, hydra_attention
from .csr import build_dense_causal_csr, build_sliding_window_csr
from .kernel_decode import launch_attn_fwd_decode
from .policy import (
    RuntimePolicy,
    last_policy_decision,
    policy_history,
    set_runtime_policy,
)

__version__ = "0.1.0"


__all__ = [
    "hydra",
    "hydra_attention",
    "flash_attn_blackwell",
    "build_dense_causal_csr",
    "build_sliding_window_csr",
    "launch_attn_fwd_decode",
    "RuntimePolicy",
    "set_runtime_policy",
    "last_policy_decision",
    "policy_history",
]
