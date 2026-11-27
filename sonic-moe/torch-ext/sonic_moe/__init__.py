# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

__version__ = "0.1.2.post1"

from . import _deps  # noqa: F401 (resolves kernel dependencies at load time)

from .enums import KernelBackendMoE
from .functional import moe_general_routing_inputs, moe_TC_softmax_topk_layer
from .moe import MoE

__all__ = [
    "KernelBackendMoE",
    "MoE",
    "moe_general_routing_inputs",
    "moe_TC_softmax_topk_layer",
]
