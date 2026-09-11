# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .matmul import matmul_2d
from .batched import matmul_batched
from .grouped import matmul_grouped
from .moe import (
    weighted_reduce,
    moe_fused_batched,
    moe_fused_grouped,
    moe_unfused_batched,
    moe_unfused_grouped,
    moe_torch_grouped,
)
# imported for its import-time side effect: registers the dgrad formulas on the ops, so an
# ordinary forward call differentiates. Exports nothing.
from . import backward  # noqa: F401
from .recipes import Epilogue, Quantization, get_supported_act_fns
from .swizzle import swizzle_mx_scales, unswizzle_mx_scales
from .scheduling import compute_grouped_scheduling
from .quant import (
    fp8_act_quant_block_dynamic,
    fp8_act_quant_tensor_wide,
    mxfp4_act_quant,
    mxfp8_act_quant,
    nvfp4_act_quant,
    nvfp4_quantize_two_level,
)

__all__ = [
    # 2D matmul
    "matmul_2d",
    # Batched matmul + MoE forwards
    "matmul_batched",
    "moe_fused_batched",
    "moe_unfused_batched",
    # Grouped matmul + MoE forwards
    "matmul_grouped",
    "moe_fused_grouped",
    "moe_unfused_grouped",
    "moe_torch_grouped",
    # Grouped scheduling (for MoE and grouped matmul)
    "compute_grouped_scheduling",
    "weighted_reduce",
    # MX/NVFP4 scale layout (apply to weight scales at load time)
    "swizzle_mx_scales",
    "unswizzle_mx_scales",
    # Quantization helpers (weights at load time; activations offline)
    "fp8_act_quant_block_dynamic",
    "fp8_act_quant_tensor_wide",
    "mxfp4_act_quant",
    "mxfp8_act_quant",
    "nvfp4_act_quant",
    "nvfp4_quantize_two_level",
    # Epilogue and Quantization configs
    "Epilogue",
    "Quantization",
    "get_supported_act_fns",
]
