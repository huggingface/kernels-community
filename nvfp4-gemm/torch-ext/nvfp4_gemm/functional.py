import torch

from ._ops import add_op_namespace_prefix, ops
from ._pack import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, PackedWeight, swizzled_sf_shape


@torch.library.register_fake(add_op_namespace_prefix("scaled_fp4_quant"))
def _quant_fake(input, input_global_scale, is_sf_swizzled_layout):
    n = input.size(-1)
    m = input.numel() // n
    out = input.new_empty((m, n // 2), dtype=torch.uint8)
    if is_sf_swizzled_layout:
        sf_m, sf_n = swizzled_sf_shape(m, n)
        sf = input.new_empty((sf_m, sf_n), dtype=torch.int32)
    else:
        sf = input.new_empty((m, n // 16), dtype=torch.uint8)
    return out, sf


@torch.library.register_fake(add_op_namespace_prefix("cutlass_scaled_fp4_mm"))
def _mm_fake(a, b, a_sf, b_sf, alpha):
    return a.new_empty((a.size(0), b.size(0)), dtype=torch.bfloat16)


@torch.library.register_fake(add_op_namespace_prefix("nvfp4_gemv"))
def _gemv_fake(a, b, b_sf, alpha):
    return a.new_empty((a.size(0), b.size(0)), dtype=torch.bfloat16)


def _as_fp8(sf: torch.Tensor) -> torch.Tensor:
    return sf.view(torch.float8_e4m3fn) if sf.dtype == torch.int32 else sf


def gemm(pw: PackedWeight, x: torch.Tensor) -> torch.Tensor:
    """y[..., n] = x[..., k] @ W.T with dynamic NVFP4 activation quantization
    (W4A4 on the Blackwell block-scaled tensor cores)."""
    if x.shape[-1] != pw.k:
        raise ValueError(f"expected {pw.k} input features, got {x.shape[-1]}")
    x2 = x.reshape(-1, pw.k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    # Decode path: W4A16 GEMV keeps activations in bf16 (no activation
    # quantization). The kernel re-reads the weight per output row, so it is
    # only worthwhile for tiny m; prefill stays on the W4A4 CUTLASS path.
    if x2.size(0) <= 2 and getattr(pw, "sf_rowmajor", None) is not None:
        alpha = (1.0 / pw.global_scale).to(torch.float32).reshape(1)
        y = ops.nvfp4_gemv(x2.to(torch.bfloat16), pw.qweight, pw.sf_rowmajor, alpha)
        return y.reshape(*x.shape[:-1], pw.n)
    a_amax = x2.abs().amax().to(torch.float32)
    a_gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax.clamp(min=1e-12)).reshape(1)
    aq, a_sf = ops.scaled_fp4_quant(x2.to(torch.bfloat16), a_gs, True)
    alpha = (1.0 / (a_gs * pw.global_scale)).to(torch.float32)
    # The quant op returns swizzled scales packed 4-per-int32; the GEMM takes
    # the same bytes typed as fp8-e4m3.
    y = ops.cutlass_scaled_fp4_mm(aq, pw.qweight, _as_fp8(a_sf), _as_fp8(pw.sf), alpha)
    return y.reshape(*x.shape[:-1], pw.n)


def gemm_swiglu(pw: PackedWeight, x: torch.Tensor) -> torch.Tensor:
    """silu(x @ W_gate.T) * (x @ W_up.T) for a SwiGLU-packed matrix."""
    if not pw.swiglu:
        raise ValueError("packed matrix was not packed with pack_swiglu")
    y = gemm(pw, x)
    half = pw.n // 2
    return torch.nn.functional.silu(y[..., :half]) * y[..., half:]
