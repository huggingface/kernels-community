"""Gluon kernels: paths stock Triton cannot express (explicit warp roles, per-partition register budgets, partial
accumulators in tensor memory). They are written to Triton 3.9's Gluon API and Blackwell (sm_10x); older Triton
exports ``None``."""
from ..compat import is_triton_greater_or_equal

if is_triton_greater_or_equal("3.9"):
    from .grouped_block_fp8 import w8a8_block_dynamic_fp8_matmul_grouped_gluon
    from .grouped_mx_w4a16 import mx_weight_only_matmul_grouped_gluon
    from .grouped_tensor_fp8 import w8a8_tensor_dynamic_fp8_matmul_grouped_gluon
else:
    w8a8_block_dynamic_fp8_matmul_grouped_gluon = mx_weight_only_matmul_grouped_gluon = None
    w8a8_tensor_dynamic_fp8_matmul_grouped_gluon = None
