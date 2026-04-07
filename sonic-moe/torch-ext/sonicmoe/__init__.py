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

# Lazy imports: defer heavy dependencies (cutlass, cuda, triton) so that
# `import sonicmoe` succeeds in environments without GPU libraries
# (e.g. the nix build sandbox get-kernel-check).
from .enums import KernelBackendMoE

_LAZY_IMPORTS = {
    "MoE": ".moe",
    "enable_quack_gemm": ".functional",
    "moe_general_routing_inputs": ".functional",
    "moe_TC_softmax_topk_layer": ".functional",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
