import os
import torch

# kernels-community: must run before the extension is loaded, see `_nccl_symbols.py`
from . import _nccl_symbols  # noqa: F401
from ._runtime import find_nccl_root

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass


# kernels-community: upstream checks NCCL and initializes the JIT here, at import time. The Hub kernel
# defers both to the first buffer construction (see `_runtime.py`), so that importing never needs a GPU,
# a CUDA toolkit or a particular NCCL version.
from ._runtime import check_nccl as check_nccl_so, ensure_initialized as init_jit


# Import APIs after initialization
from . import comm
from .comm import destroy_all_managed_nccl_comm, get_physical_domain_size, get_logical_domain_size
from .buffers.allocator import BufferAllocator
from .buffers.base import BufferBase
from .buffers.ep import EPBuffer, EPHandle
from .buffers.engram import EngramBuffer
from .buffers.bucket import BucketBuffer, BucketSession
from .buffers.pp import PPBuffer
# noinspection PyUnresolvedReferences
from .utils.event import EventOverlap, EventHandle

# noinspection PyUnresolvedReferences
from ._C import (
    get_num_allocation_alignment,
    get_num_rdma_alignment,
    get_num_tma_alignment,
    topk_idx_t,
)

__version__ = '2.5.0'
