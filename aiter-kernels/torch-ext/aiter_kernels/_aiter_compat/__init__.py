"""Compatibility shim layer for aiter cross-tree dependencies.

The upstream ``aiter/ops/triton/**`` Triton ops import a handful of helpers
from outside the triton subtree:

- ``aiter.dtypes`` / ``aiter.utility.dtypes`` — dtype constants keyed by GPU arch.
- ``aiter.jit.utils.torch_guard.torch_compile_guard`` — torch.compile/torch.library
  registration helper.
- ``aiter.utility.triton.triton_metadata_redirect.AOTMetadataContext`` — AOT
  metadata-path redirector.
- ``aiter.jit.utils.chip_info.get_gfx`` — GPU arch query.

In a Hub-kernel build we don't ship the rest of upstream aiter — we vendor
just enough here that the Triton ops import cleanly. The runtime hot paths
(the actual Triton kernels) do not rely on these helpers; they're used for
torch.library schema registration, AOT caching, and dtype lookups that we
re-derive from torch / Triton directly.
"""

from . import dtypes
from . import chip_info
from . import torch_guard
from . import triton_metadata_redirect

__all__ = [
    "dtypes",
    "chip_info",
    "torch_guard",
    "triton_metadata_redirect",
]
