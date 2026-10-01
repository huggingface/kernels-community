from dataclasses import dataclass

import torch

from ._ops import ops

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0
SF_VEC_SIZE = 16  # elements sharing one FP8 scale


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def swizzled_sf_shape(rows: int, cols: int) -> tuple[int, int]:
    """Shape of the swizzled scale-factor tensor (dtype int32) produced by
    `scaled_fp4_quant(..., is_sf_swizzled_layout=True)` for a [rows, cols]
    input. Mirrors `computeSwizzledSFShape` in nvfp4_utils.cuh: 128-row tiles,
    4 FP8 scales packed per int32 column."""
    return _round_up(rows, 128), _round_up(cols // SF_VEC_SIZE, 4) // 4


def global_scale_for(w: torch.Tensor) -> torch.Tensor:
    """Per-tensor second-level scale: FP8_MAX * FP4_MAX / amax."""
    amax = w.abs().amax().to(torch.float32)
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / amax.clamp(min=1e-12)).reshape(1)


@dataclass
class PackedWeight:
    """NVFP4 representation of a bf16 matrix [n, k]."""

    qweight: torch.Tensor  # uint8 [n, k/2], two e2m1 values per byte
    sf: torch.Tensor  # int32, swizzled FP8-E4M3 block scales
    global_scale: torch.Tensor  # fp32 [1]
    n: int
    k: int
    swiglu: bool = False
    # uint8 [n, k/16] row-major FP8-E4M3 block scales, consumed by the W4A16
    # GEMV decode path (None when only the swizzled layout is available).
    sf_rowmajor: torch.Tensor | None = None

    def to(self, device) -> "PackedWeight":
        return PackedWeight(
            qweight=self.qweight.to(device),
            sf=self.sf.to(device),
            global_scale=self.global_scale.to(device),
            n=self.n,
            k=self.k,
            swiglu=self.swiglu,
            sf_rowmajor=(
                None if self.sf_rowmajor is None else self.sf_rowmajor.to(device)
            ),
        )


def pack(
    w: torch.Tensor, device: str | torch.device = "cuda", swiglu: bool = False
) -> PackedWeight:
    """Quantize a bf16/fp16 matrix [n, k] to NVFP4 (requires sm100+)."""
    if w.dim() != 2 or not w.is_floating_point():
        raise ValueError("pack() expects a 2-D floating-point weight")
    n, k = w.shape
    if k % SF_VEC_SIZE != 0:
        raise ValueError(f"in_features must be a multiple of {SF_VEC_SIZE}, got {k}")
    wd = w.detach().to(device, torch.bfloat16).contiguous()
    gs = global_scale_for(wd).to(device)
    qweight, sf = ops.scaled_fp4_quant(wd, gs, True)
    # Row-major scales for the W4A16 GEMV decode path; the qweight output is
    # identical to the swizzled call's, so it is discarded.
    _, sf_rowmajor = ops.scaled_fp4_quant(wd, gs, False)
    return PackedWeight(
        qweight=qweight,
        sf=sf,
        global_scale=gs,
        n=n,
        k=k,
        swiglu=swiglu,
        sf_rowmajor=sf_rowmajor,
    )


def pack_swiglu(
    w_gate: torch.Tensor, w_up: torch.Tensor, device: str | torch.device = "cuda"
) -> PackedWeight:
    """Pack gate and up projections [n, k] as one [2n, k] matrix; the GEMM
    output has gate values in columns [0, n) and up in [n, 2n)."""
    if w_gate.shape != w_up.shape:
        raise ValueError("gate and up matrices must have the same shape")
    return pack(torch.cat([w_gate, w_up], dim=0), device=device, swiglu=True)


def quantize_reference(w: torch.Tensor) -> torch.Tensor:
    """Pure-torch NVFP4 emulation (no GPU kernel): quantize + dequantize to
    the same numerics the kernel produces, for tests and quality analysis."""
    orig_dtype = w.dtype
    wf = w.float()
    gs = global_scale_for(wf)  # fp8_max*fp4_max/amax
    n, k = wf.shape
    g = wf.reshape(n, k // SF_VEC_SIZE, SF_VEC_SIZE)
    # Block scale: amax/FP4_MAX * global_scale, stored in FP8-E4M3.
    block_amax = g.abs().amax(-1, keepdim=True)
    sf = (block_amax / FLOAT4_E2M1_MAX) * gs
    sf = sf.to(torch.float8_e4m3fn).float().clamp(min=1e-12)
    scaled = g * (gs / sf)
    # e2m1 grid: {0, .5, 1, 1.5, 2, 3, 4, 6} with signs.
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    q = grid[(scaled.abs().unsqueeze(-1) - grid).abs().argmin(-1)] * scaled.sign()
    return (q * sf / gs).reshape(n, k).to(orig_dtype)
