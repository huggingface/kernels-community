import importlib.util
import sys
import types
from pathlib import Path

import torch

# In CI (where `kernels` is installed and `LOCAL_KERNELS` points to the Nix build),
# load strictly through `kernels.get_kernel(...)`. Only fall back to the local
# `torch-ext` directory when running pytest directly without `kernels` installed.
if importlib.util.find_spec("kernels") is None:
    _TORCH_EXT = Path(__file__).resolve().parent.parent / "torch-ext"
    if str(_TORCH_EXT) not in sys.path:
        sys.path.insert(0, str(_TORCH_EXT))

    if not (_TORCH_EXT / "gemma_quant_kernels" / "_ops.py").exists() and "gemma_quant_kernels._ops" not in sys.modules:
        _ops_mod = types.ModuleType("gemma_quant_kernels._ops")
        _ops_mod.add_op_namespace_prefix = lambda op_name: f"gemma_quant_kernels::{op_name}"  # type: ignore[attr-defined]
        _ops_mod.ops = getattr(torch.ops, "gemma_quant_kernels", None)  # type: ignore[attr-defined]
        sys.modules["gemma_quant_kernels._ops"] = _ops_mod
