# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER Triton kernels for AMD ROCm.

Repackaged from the ``aiter/ops/triton/**`` subtree of the
`ROCm/aiter <https://github.com/ROCm/aiter>`_ project as a self-contained
Hugging Face Hub kernel. Each subpackage under :mod:`aiter_kernels` maps 1:1
to the equivalent upstream module under ``aiter.ops.triton``.

Flash Attention is **not** included here — it lives in
``kernels-community/aiter-flash-attn`` and is synced separately.
"""

from __future__ import annotations

from . import _aiter_compat  # noqa: F401  (must be importable before ops)

from . import activation
from . import causal_conv1d
from . import causal_conv1d_update_single_token
from . import fusions
from . import gated_delta_net
from . import gather_kv_b_proj
from . import gemm
from . import gluon
from . import gmm
from . import kv_cache
from . import moe
from . import normalization
from . import quant
from . import rope
# Top-level re-exports for drop-in parity with the standalone ``aiter-rope`` repo.
from .rope import RotateStyle, apply_rotary_transformers
from . import softmax
from . import topk
from . import utils

# ``comms`` pulls in iris and is optional — make it import-safe.
try:
    from . import comms

    # Re-export communication primitives at this level for convenience
    from .comms import (
        IrisCommContext,
        reduce_scatter,
        all_gather,
        reduce_scatter_rmsnorm_quant_all_gather,
        IRIS_COMM_AVAILABLE,
    )

    _COMMS_AVAILABLE = True
except ImportError:
    _COMMS_AVAILABLE = False
    IRIS_COMM_AVAILABLE = False
    comms = None  # type: ignore[assignment]


__kernel_metadata__ = {
    "license": "mit",
}


__all__ = [
    "__kernel_metadata__",
    "RotateStyle",
    "apply_rotary_transformers",
    "activation",
    "causal_conv1d",
    "causal_conv1d_update_single_token",
    "comms",
    "fusions",
    "gated_delta_net",
    "gather_kv_b_proj",
    "gemm",
    "gluon",
    "gmm",
    "kv_cache",
    "moe",
    "normalization",
    "quant",
    "rope",
    "softmax",
    "topk",
    "utils",
]

if _COMMS_AVAILABLE:
    __all__.extend(
        [
            "IrisCommContext",
            "reduce_scatter",
            "all_gather",
            "reduce_scatter_rmsnorm_quant_all_gather",
            "IRIS_COMM_AVAILABLE",
        ]
    )
