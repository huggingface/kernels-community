from ._ops import ops

from .flash_attn_interface import (
    reserve_rng_state,
    fwd_kvcache,
    flash_attn_with_kvcache,
    bwd,
    fwd,
    varlen_fwd,
    varlen_bwd,
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
)

fwd_explicit = ops.fwd

__all__ = [
    "fwd_kvcache",
    "flash_attn_with_kvcache",
    "fwd",
    "bwd",
    "varlen_fwd",
    "varlen_bwd",
    "fwd_explicit",
    "reserve_rng_state",
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
]

# tvm-ffi builds must import without torch; register the torch custom ops
# only when it is installed.
try:
    import torch as _torch  # noqa: F401
except ImportError:
    pass
else:
    from . import _torch_interface  # noqa: F401
