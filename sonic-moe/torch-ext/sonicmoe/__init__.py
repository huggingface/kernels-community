# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

from functools import lru_cache

__version__ = "0.1.1"

from .enums import KernelBackendMoE

_LAZY_IMPORTS = {
    "MoE": ".moe",
    "enable_quack_gemm": ".functional",
    "moe_general_routing_inputs": ".functional",
    "moe_TC_softmax_topk_layer": ".functional",
}

@lru_cache(maxsize=None)
def _load_attr(name: str):
    import importlib
    module_path = _LAZY_IMPORTS[name]
    mod = importlib.import_module(module_path, __name__)
    return getattr(mod, name)

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        return _load_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "KernelBackendMoE",
    "MoE",
    "enable_quack_gemm",
    "moe_general_routing_inputs",
    "moe_TC_softmax_topk_layer",
]
