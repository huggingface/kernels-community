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
from .moe import moe_fused_batched, moe_fused_grouped
from . import backward
from .formats import get_supported_act_fns
from .swizzle import swizzle_mx_scales, unswizzle_mx_scales
from .quant import mxfp4_act_quant, mxfp8_act_quant, nvfp4_act_quant

__all__ = [
    # the three GEMM dispatchers (the weight format is read off the tensors; `activation_format`
    # names the activations', the gate|up fusion rides as kwargs)
    "matmul_2d",
    "matmul_batched",
    "matmul_grouped",
    # the fused MoE forwards over them
    "moe_fused_batched",
    "moe_fused_grouped",
    # importing it registers the dgrad formulas on the ops, so a forward call differentiates
    "backward",
    "get_supported_act_fns",
    # load-time helpers: the swizzled scale layout the SM100 scaled-MMA reads, and the row-wise
    # quantizers a loader uses to quantize weights into the group formats
    "swizzle_mx_scales",
    "unswizzle_mx_scales",
    "mxfp8_act_quant",
    "mxfp4_act_quant",
    "nvfp4_act_quant",
]
