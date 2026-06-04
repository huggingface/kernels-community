"""Shared test helpers: FP8/FP4 weight constructors and dequant references.

The ``finegrained_fp8`` package is on ``sys.path`` via ``conftest.py`` (pytest
auto-loads it before any test module)."""

import torch

from finegrained_fp8.utils import FP4_SCALE_GROUP_K  # type: ignore


# ── Device + capability ───────────────────────────────────────────────────────

TEST_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else None
)
# Triton's ``fp8e4nv`` (``float8_e4m3fn``) requires SM89+ on CUDA. SM80 (A100)
# fails kernel compilation, so every FP8/FP4 test in this suite needs this gate.
SUPPORTS_FP8 = TEST_DEVICE == "xpu" or (
    TEST_DEVICE == "cuda" and torch.cuda.get_device_capability() >= (8, 9)
)
SUPPORTS_FP4 = TEST_DEVICE == "xpu" or (
    TEST_DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 10
)
# SM89 (Ada Lovelace, e.g. L4) has looser FP8 numerics vs the dequant+matmul
# reference than Hopper/Blackwell — wide tolerances here, tight everywhere else.
IS_SM89 = TEST_DEVICE == "cuda" and torch.cuda.get_device_capability() == (8, 9)
# SM90 (Hopper, e.g. H100) is the only architecture the benchmark baselines
# are calibrated against — every other SM has its own latency profile.
IS_SM90 = TEST_DEVICE == "cuda" and torch.cuda.get_device_capability() == (9, 0)
DTYPE_TAG = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
}
DTYPE_TO_TOL = (
    {
        torch.bfloat16: (0.2, 0.05),
        torch.float16: (0.2, 0.05),
        torch.float32: (0.2, 0.05),
    }
    if IS_SM89
    else {
        torch.bfloat16: (1e-4, 1e-2),
        torch.float16: (1e-4, 1e-2),
        torch.float32: (1e-4, 1e-4),
    }
)


def accelerator_module():
    if TEST_DEVICE == "cuda":
        return torch.cuda
    if TEST_DEVICE == "xpu":
        return torch.xpu
    raise RuntimeError("No supported accelerator available")


# ── FP8 ───────────────────────────────────────────────────────────────────────

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = torch.finfo(FP8_DTYPE).min


def make_fp8_weights(
    out_features,
    in_features,
    device,
    block_size,
    num_experts=None,
    scale_dtype=torch.float32,
):
    """Random FP8 weights with block-wise inv-scales. ``num_experts`` toggles 2D
    (linear) vs 3D (MoE experts). Non-aligned ``out_features``/``in_features``
    are padded to block boundaries before quantizing, then trimmed back.

    ``scale_dtype`` selects the inv-scale storage:
    - ``torch.float32`` (default): standard per-block fp32 scales.
    - ``torch.float8_e8m0fnu``: UE8M0 (DSv4-Flash style) — inv-scales snapped
      up to the nearest power-of-2 and re-quantization done against the snapped
      scale so dequant exactly matches storage.
    """
    is_2d = num_experts is None
    E = 1 if is_2d else num_experts
    W = torch.randn(E, out_features, in_features, dtype=torch.float32, device=device)
    block_n, block_k = (
        block_size if block_size is not None else (out_features, in_features)
    )
    pad_n = (-out_features) % block_n
    pad_k = (-in_features) % block_k
    W = torch.nn.functional.pad(W, [0, pad_k, 0, pad_n])
    Np, Kp = out_features + pad_n, in_features + pad_k
    R = W.reshape(E, Np // block_n, block_n, Kp // block_k, block_k)
    max_abs = R.abs().amax(dim=(-3, -1))
    safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    inv_scales = (safe / FP8_MAX).to(torch.float32)

    if scale_dtype == torch.float8_e8m0fnu:
        # Snap inv_scales up to next power-of-2 (UE8M0 encoding: 2^(exp-127)).
        bits = inv_scales.contiguous().view(torch.int32)
        exp_ceil = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        exp_ceil = exp_ceil.clamp(1, 254)
        inv_scales = (exp_ceil << 23).view(torch.float32)

    scale = 1.0 / inv_scales
    Wq = (R * scale.unsqueeze(-1).unsqueeze(-3)).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    Wq = Wq.reshape(E, Np, Kp)[:, :out_features, :in_features].contiguous()

    if scale_dtype == torch.float8_e8m0fnu:
        inv_scales = exp_ceil.to(torch.uint8).view(torch.float8_e8m0fnu)

    return (Wq.squeeze(0), inv_scales.squeeze(0)) if is_2d else (Wq, inv_scales)


def make_static_activation_scale(A: torch.Tensor) -> torch.Tensor:
    """Calibration-style per-tensor static activation scale: ``amax / 448``,
    floored at 1e-12 to avoid div-by-zero on all-zero inputs."""
    return (A.abs().amax() / 448.0).clamp(min=1e-12).to(torch.float32)


def quant_dequant_a_fp8(
    A: torch.Tensor, block_k: int, scale: torch.Tensor | None = None
) -> torch.Tensor:
    """Mirror the FP8 kernel's inline act-quant in fp32.

    Dynamic (``scale=None``): per-row, per-block_k fp32 scale = ``amax / 448``;
    quantize and multiply back. Static (scalar ``scale``): use the given
    calibration scale for the whole tensor.
    """
    if scale is not None:
        A_fp8 = (A.float() / scale).to(FP8_DTYPE)
        return A_fp8.float() * scale
    M, K = A.shape
    groups = A.float().reshape(M, K // block_k, block_k)
    s = groups.abs().amax(dim=-1) / 448.0
    A_fp8 = (groups / s.unsqueeze(-1).clamp(min=1e-12)).to(FP8_DTYPE)
    return (A_fp8.float() * s.unsqueeze(-1)).reshape(M, K)


def dequant_b_fp8(
    B: torch.Tensor, Bs: torch.Tensor, block_n: int, block_k: int
) -> torch.Tensor:
    """Per-block FP8 weight dequant via broadcasted block scales. ``Bs`` may be
    ``float32`` or ``float8_e8m0fnu`` (UE8M0; decoded as ``2^(exp - 127)``)."""
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = (Bs.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    N, K = B.shape
    scales_full = Bs.repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1
    )[:N, :K]
    return B.float() * scales_full


# ── FP4 (E2M1 + UE8M0) ────────────────────────────────────────────────────────

# E2M1 decode LUT indexed by raw 4-bit code: high bit is sign, low 3 bits select
# magnitude from {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
_E2M1_DECODE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def make_fp4_weights(out_features, in_features, device, num_experts=None):
    """Random FP4-packed (E2M1) weights with UE8M0 scales of all-ones.
    ``num_experts`` toggles between 2D (linear) and 3D (MoE experts)."""
    assert in_features % FP4_SCALE_GROUP_K == 0, (
        f"K ({in_features}) must be divisible by {FP4_SCALE_GROUP_K} for FP4"
    )
    packed_k = in_features // 2
    n_groups = in_features // FP4_SCALE_GROUP_K
    if num_experts is None:
        w_shape = (out_features, packed_k)
        s_shape = (out_features, n_groups)
    else:
        w_shape = (num_experts, out_features, packed_k)
        s_shape = (num_experts, out_features, n_groups)
    weights = torch.randint(-8, 8, w_shape, dtype=torch.int8, device=device)
    scales = torch.ones(s_shape, dtype=torch.float8_e8m0fnu, device=device)
    return weights.contiguous(), scales.contiguous()


def quant_dequant_a_fp4(A: torch.Tensor) -> torch.Tensor:
    """Mirror the FP4 kernel's inline act-quant round-trip in fp32: per-row,
    per-32-K-group UE8M0 power-of-2 scale (ceil to next pow2 of ``|amax|/448``);
    quantize and multiply back by the same pow2."""
    M, K = A.shape
    groups = A.float().reshape(M, K // FP4_SCALE_GROUP_K, FP4_SCALE_GROUP_K)
    bits = (groups.abs().amax(dim=-1) / 448.0).contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    s_pow2 = (exp.clamp(1, 254) << 23).view(torch.float32)
    A_fp8 = (groups / s_pow2.unsqueeze(-1).clamp(min=1e-12)).to(FP8_DTYPE)
    return (A_fp8.float() * s_pow2.unsqueeze(-1)).reshape(M, K)


def dequant_b_fp4(B: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
    """Decode packed E2M1 ``int8`` weights, apply UE8M0 K-group scales."""
    N, K_half = B.shape
    K = K_half * 2
    lut = torch.tensor(_E2M1_DECODE, dtype=torch.float32, device=B.device)
    codes = B.to(torch.uint8).long()
    decoded = torch.stack(
        [lut[codes & 0x0F], lut[(codes >> 4) & 0x0F]], dim=-1
    ).reshape(N, K)
    bs_fp32 = (Bs.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    return decoded * bs_fp32.repeat_interleave(FP4_SCALE_GROUP_K, dim=-1)


# ── Unified matmul reference ──────────────────────────────────────────────────


def ref_matmul(A, B, Bs, block_size, output_dtype=torch.float32, activation_scale=None):
    """Pure-PyTorch reference for ``matmul``: quant+dequant both sides, fp32
    matmul, cast to output. Dispatches on ``B.dtype``. ``activation_scale``
    selects the static-quant path (single calibration scalar on FP8)."""
    if B.dtype == torch.int8:
        A_deq = quant_dequant_a_fp4(A)
        B_deq = dequant_b_fp4(B, Bs)
    else:
        N, K = B.shape
        block_n, block_k = block_size if block_size is not None else (N, K)
        A_deq = quant_dequant_a_fp8(A, block_k, scale=activation_scale)
        B_deq = dequant_b_fp8(B, Bs, block_n, block_k)
    return (A_deq @ B_deq.T).to(output_dtype)
