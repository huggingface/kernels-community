import types

from .causal_conv1d_interface import (
    causal_conv1d_fn,
    causal_conv1d_ref,
    causal_conv1d_update,
)
from .causal_conv1d_varlen import causal_conv1d_varlen_states
from .cpp_functions import (
    causal_conv1d_bwd_function,
    causal_conv1d_fwd_function,
    causal_conv1d_update_function,
)

# Module-like handle exposing the raw C++ ops, matching the legacy
# ``causal_conv1d_cuda`` import used by mamba-ssm internals.
causal_conv1d_cuda = types.SimpleNamespace(
    causal_conv1d_fwd=causal_conv1d_fwd_function,
    causal_conv1d_bwd=causal_conv1d_bwd_function,
    causal_conv1d_update=causal_conv1d_update_function,
)

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "causal_conv1d_ref",
    "causal_conv1d_varlen_states",
    "causal_conv1d_cuda",
]
