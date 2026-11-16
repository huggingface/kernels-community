"""Pallas Mosaic TPU kernels from Tokamax (openxla/tokamax).

This kernel carries a subset of Tokamax: the TPU kernel bodies, without
Tokamax's `Op`, autotuning and benchmarking layers, so it needs nothing
beyond JAX at runtime. Each op is exposed as its kernel module, e.g.
`tokamax.ragged_dot.gmm` or `tokamax.gmm_v2.gmm_v2`. The kernels take and
return JAX arrays; Torch kernels can call them through `torch_tpu`'s
`jax_op`.
"""

from ._src import quantization
from ._src.ops.csa_gather import pallas_mosaic_tpu_kernel as csa_gather
from ._src.ops.deepseek_rope import pallas_mosaic_tpu_kernel as deepseek_rope
from ._src.ops.dsv4_compressor import pallas_mosaic_tpu_kernel as dsv4_compressor
from ._src.ops.experimental.fused_fp8_matmul import (
    pallas_mosaic_tpu_kernel as fused_fp8_matmul,
)
from ._src.ops.experimental.fused_moe_decode import (
    fused_moe_decode_kernel as fused_moe_decode,
)
from ._src.ops.experimental.mla import pallas_mosaic_tpu_kernel as mla
from ._src.ops.experimental.mla_swa import pallas_mosaic_tpu_kernel as mla_swa
from ._src.ops.experimental.tpu.splash_attention import (
    ring_attention_kernel,
    splash_attention_kernel,
    splash_attention_mask,
)
from ._src.ops.gmm_v2 import gmm_v2, tgmm_v2
from ._src.ops.grouped_lowrank_proj import (
    pallas_mosaic_tpu_kernel as grouped_lowrank_proj,
)
from ._src.ops.masked_dense_mla import pallas_mosaic_tpu_kernel as masked_dense_mla
from ._src.ops.mhc import fused_post_pre_kernel as mhc_fused_post_pre
from ._src.ops.mhc import post_kernel as mhc_post
from ._src.ops.mhc import pre_kernel as mhc_pre
from ._src.ops.proj_and_save_state import (
    pallas_mosaic_tpu_kernel as proj_and_save_state,
)
from ._src.ops.ragged_dot import pallas_mosaic_tpu_kernel as ragged_dot
from ._src.ops.ragged_gather import pallas_mosaic_tpu_kernel as ragged_gather
from ._src.ops.ragged_gather_reduce import (
    pallas_mosaic_tpu_kernel as ragged_gather_reduce,
)
from ._src.ops.router_topk import pallas_mosaic_tpu_kernel as router_topk
from ._src.ops.sparse_mla import pallas_mosaic_tpu_kernel as sparse_mla
from ._src.ops.topk import pallas_mosaic_tpu_kernel as topk
from ._src.version import TOKAMAX_VERSION as __version__
from ._vendor.qwix import QArray, dequantize, quantize

__all__ = [
    "QArray",
    "__version__",
    "csa_gather",
    "deepseek_rope",
    "dequantize",
    "dsv4_compressor",
    "fused_fp8_matmul",
    "fused_moe_decode",
    "gmm_v2",
    "grouped_lowrank_proj",
    "masked_dense_mla",
    "mhc_fused_post_pre",
    "mhc_post",
    "mhc_pre",
    "mla",
    "mla_swa",
    "proj_and_save_state",
    "quantization",
    "quantize",
    "ragged_dot",
    "ragged_gather",
    "ragged_gather_reduce",
    "ring_attention_kernel",
    "router_topk",
    "sparse_mla",
    "splash_attention_kernel",
    "splash_attention_mask",
    "tgmm_v2",
    "topk",
]
