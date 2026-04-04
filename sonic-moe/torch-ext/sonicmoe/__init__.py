# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import os as _os
import sys as _sys

# Inject vendored quack-kernels into the module search path so that
# `import quack` resolves to the bundled copy when no system install exists.
_vendor_dir = _os.path.join(_os.path.dirname(__file__), "_vendor")
if _vendor_dir not in _sys.path:
    _sys.path.insert(0, _vendor_dir)

__version__ = "0.1.1"

from .enums import KernelBackendMoE
from .functional import enable_quack_gemm, moe_general_routing_inputs, moe_TC_softmax_topk_layer
from .moe import MoE
