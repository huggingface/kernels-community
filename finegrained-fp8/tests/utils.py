"""Shared test helpers: FP8/FP4 weight constructors and dequant references.

The ``finegrained_fp8`` package is on ``sys.path`` via ``conftest.py`` (pytest
auto-loads it before any test module)."""

import torch

from finegrained_fp8.utils import MX_SCALE_GROUP_K, NIBBLES_PER_BYTE  # type: ignore


# ── Device + capability ───────────────────────────────────────────────────────

TEST_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if hasattr(torch, "xpu") and torch.xpu.is_available()
    else None
)
# FP8 kernels require Hopper (SM90) or newer on CUDA. SM89 (Ada Lovelace) can
# technically compile fp8e4nv but its numerics drift from the dequant+matmul
# reference by enough to need separate tolerances — not worth maintaining for
# a non-target deployment SM.
SUPPORTS_FP8 = TEST_DEVICE == "xpu" or (
    TEST_DEVICE == "cuda" and torch.cuda.get_device_capability() >= (9, 0)
)
# The MX paths (MXFP4/MXFP8) need no extra gate: ``tl.dot_scaled`` runs natively on
# Blackwell (SM100+) and via software emulation (the DecomposeScaledBlocked pass —
# upcast to bf16/fp16, then a standard MMA) everywhere else. Their only real
# requirement is FP8 for the E4M3 activation quant, already covered by SUPPORTS_FP8.
# SM90 (Hopper, e.g. H100) is the only architecture the benchmark baselines
# are calibrated against — every other SM has its own latency profile.
IS_SM90 = TEST_DEVICE == "cuda" and torch.cuda.get_device_capability() == (9, 0)
DTYPE_TAG = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
}
DTYPE_TO_TOL = {
    torch.bfloat16: (1e-4, 1e-2),
    torch.float16: (1e-4, 1e-2),
    torch.float32: (1e-4, 1e-4),
}


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
    scale_layout="block",
):
    """Random FP8 weights with block-wise inv-scales. ``num_experts`` toggles 2D
    (linear) vs 3D (MoE experts). Non-aligned ``out_features``/``in_features``
    are padded to block boundaries before quantizing, then trimmed back.

    ``scale_dtype`` selects the inv-scale storage:
    - ``torch.float32`` (default): standard per-block fp32 scales.
    - ``torch.float8_e8m0fnu``: UE8M0 (DSv4-Flash style) — inv-scales snapped
      up to the nearest power-of-2 and re-quantization done against the snapped
      scale so dequant exactly matches storage.

    ``scale_layout`` (3D/MoE only) picks the shape the grouped/batched kernels
    consume: ``block`` keeps the ``[E, n_blocks, k_blocks]`` grid; ``per_tensor_1d``
    / ``per_tensor_111`` collapse to one scale per expert (``[E]`` / ``[E, 1, 1]``).
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

    if is_2d:  # linear weights always carry block-layout scales
        return Wq.squeeze(0), inv_scales.squeeze(0)
    return Wq, _apply_scale_layout(inv_scales, scale_layout)


def _apply_scale_layout(Bs, layout):
    """Reshape 3D block inv-scales ``[E, n_blocks, k_blocks]`` into the layout the
    MoE kernels expect. ``block`` is a no-op; ``per_tensor_1d`` / ``per_tensor_111``
    take the first block's scale per expert (problems that use these set
    ``block_size=None``, so there is exactly one block to take)."""
    if layout == "block":
        return Bs
    per_tensor = Bs[:, 0, 0].contiguous()  # [E]
    if layout == "per_tensor_1d":
        return per_tensor
    if layout == "per_tensor_111":
        return per_tensor.view(-1, 1, 1).contiguous()
    raise ValueError(f"Unsupported scale layout: {layout}")


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


# ── MXFP4 (packed E2M1 + UE8M0 group-32) ──────────────────────────────────────

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


def make_fp4_weights(out_features, in_features, device, block_size, num_experts=None):
    """Random MXFP4 weights: packed E2M1 codes with all-ones UE8M0 K-group scales.
    ``block_size`` is ``(block_n, block_k)`` like ``make_fp8_weights``; E2M1 scales
    are per-row, so ``block_n`` must be 1 and ``block_k`` (``MX_SCALE_GROUP_K`` = 32
    for MXFP4) sets the K-group. ``num_experts`` toggles between 2D (linear) and 3D
    (MoE experts)."""
    block_n, block_k = block_size
    assert block_n == 1, f"E2M1 scales are per-row; block_n must be 1, got {block_n}"
    assert in_features % block_k == 0, (
        f"K ({in_features}) must be divisible by block_k ({block_k})"
    )
    packed_k = in_features // NIBBLES_PER_BYTE
    n_groups = in_features // block_k
    if num_experts is None:
        w_shape = (out_features, packed_k)
        s_shape = (out_features, n_groups)
    else:
        w_shape = (num_experts, out_features, packed_k)
        s_shape = (num_experts, out_features, n_groups)
    weights = torch.randint(-8, 8, w_shape, dtype=torch.int8, device=device)
    scales = torch.ones(s_shape, dtype=torch.float8_e8m0fnu, device=device)
    return weights.contiguous(), scales.contiguous()


def quant_dequant_a_mx(A: torch.Tensor) -> torch.Tensor:
    """Mirror the MX kernels' (W4A8 FP4 / W8A8 MXFP8) inline activation-quant
    round-trip in fp32: per-row, per-32-K-group UE8M0 power-of-2 scale (ceil to next
    pow2 of ``|amax|/448``); quantize to E4M3 and multiply back by the same pow2."""
    M, K = A.shape
    groups = A.float().reshape(M, K // MX_SCALE_GROUP_K, MX_SCALE_GROUP_K)
    bits = (groups.abs().amax(dim=-1) / 448.0).contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    s_pow2 = (exp.clamp(1, 254) << 23).view(torch.float32)
    A_fp8 = (groups / s_pow2.unsqueeze(-1).clamp(min=1e-12)).to(FP8_DTYPE)
    return (A_fp8.float() * s_pow2.unsqueeze(-1)).reshape(M, K)


def dequant_b_fp4(B: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
    """Decode packed E2M1 ``int8`` weights, apply UE8M0 K-group scales. The K-group
    size is inferred from shapes (``K / n_groups``) to match whatever ``block_k``
    ``make_fp4_weights`` produced."""
    N, K_half = B.shape
    K = K_half * NIBBLES_PER_BYTE
    lut = torch.tensor(_E2M1_DECODE, dtype=torch.float32, device=B.device)
    codes = B.to(torch.uint8).long()
    decoded = torch.stack(
        [lut[codes & 0x0F], lut[(codes >> 4) & 0x0F]], dim=-1
    ).reshape(N, K)
    bs_fp32 = (Bs.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    block_k = K // Bs.shape[-1]
    return decoded * bs_fp32.repeat_interleave(block_k, dim=-1)


# ── MXFP8 (E4M3 weights + UE8M0 group-32) ──────────────────────────────────────
# MXFP8 weights ARE block-wise FP8 weights at 1x32 granularity with UE8M0 scales —
# build them via ``make_fp8_weights(..., (1, MX_SCALE_GROUP_K), scale_dtype=e8m0)``
# and dequant with ``dequant_b_fp8(..., 1, MX_SCALE_GROUP_K)``. Only the activation
# quant is MXFP8-specific (UE8M0 group-32, shared with FP4 via ``quant_dequant_a_mx``).


def _is_mxfp4(B, Bs):
    """Packed E2M1 weights (``int8``, two nibbles/byte → unpacked K = ``2 * K_half``)
    with UE8M0 group-32 scales ``[N, K // 32]``."""
    return (
        B.dtype == torch.int8
        and Bs.dtype == torch.float8_e8m0fnu
        and Bs.ndim == 2
        and tuple(Bs.shape)
        == (B.shape[0], B.shape[1] * NIBBLES_PER_BYTE // MX_SCALE_GROUP_K)
    )


def _is_mxfp8(B, Bs):
    """E4M3 weights with UE8M0 group-32 scales ``[N, K // 32]``."""
    return (
        B.dtype == FP8_DTYPE
        and Bs.dtype == torch.float8_e8m0fnu
        and Bs.ndim == 2
        and tuple(Bs.shape) == (B.shape[0], B.shape[1] // MX_SCALE_GROUP_K)
    )


# ── Unified matmul reference ──────────────────────────────────────────────────


def ref_matmul(A, B, Bs, block_size, output_dtype=torch.float32, activation_scale=None):
    """Pure-PyTorch reference for ``matmul``: dequant both sides, fp32 matmul, cast.
    Mirrors the dispatcher's routing (``B.dtype`` / scale layout). The MX paths (FP4
    and MXFP8) share the UE8M0 group-32 activation quant; ``activation_scale`` selects
    the static-quant FP8 path."""
    if _is_mxfp4(B, Bs):  # MXFP4: MX activation, packed E2M1 weights
        A_deq, B_deq = quant_dequant_a_mx(A), dequant_b_fp4(B, Bs)
    elif _is_mxfp8(B, Bs):  # MXFP8: MX activation, E4M3 weights at 1x32
        A_deq, B_deq = quant_dequant_a_mx(A), dequant_b_fp8(B, Bs, 1, MX_SCALE_GROUP_K)
    else:  # block / tensor / static FP8
        N, K = B.shape
        block_n, block_k = block_size if block_size is not None else (N, K)
        A_deq = quant_dequant_a_fp8(A, block_k, scale=activation_scale)
        B_deq = dequant_b_fp8(B, Bs, block_n, block_k)
    return (A_deq @ B_deq.T).to(output_dtype)
