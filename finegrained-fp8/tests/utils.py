"""Shared test helpers: quantized-weight constructors and dequant references.

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


def maybe_compile(fn, enabled):
    """Wrap ``fn`` in ``torch.compile`` (max-autotune, fullgraph) when ``enabled``,
    resetting the compiler and clearing the allocator cache first; otherwise return
    ``fn`` unchanged."""
    if not enabled:
        return fn
    torch.compiler.reset()
    accelerator_module().empty_cache()
    return torch.compile(fn, mode="max-autotune", fullgraph=True)


# ── Quantized weights ─────────────────────────────────────────────────────────

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
FP8_MIN = torch.finfo(FP8_DTYPE).min
E2M1_MAX = 6.0  # largest magnitude on the E2M1 (FP4) grid

# E2M1 decode LUT indexed by raw 4-bit code: high bit is sign, low 3 bits select
# magnitude from {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
_E2M1_DECODE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip
# Midpoints between adjacent E2M1 magnitudes: round |x| to the nearest grid index.
_E2M1_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _ue8m0_exp(x):
    """UE8M0 exponent of the next power-of-2 >= ``x``, clamped to the encodable range
    [1, 254]; the decoded scale is ``2^(exp - 127)``."""
    bits = x.contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return exp.clamp(1, 254)


def make_weights(
    out_features,
    in_features,
    device,
    block_size,
    *,
    weight_dtype=FP8_DTYPE,
    scale_dtype=torch.float32,
    scale_layout="block",
    num_experts=None,
):
    """Random quantized weights (+ inv-scales) for ``weight_dtype`` via one flow:
    draw fp32 weights, take a per-block ``amax``, derive an inv-scale (optionally a
    UE8M0 power-of-2), scale the weights into the target grid's range, and quantize.
    ``weight_dtype`` is the stored element type — the same axis the kernel ``matmul``
    dispatcher routes on (``B.dtype``), not the recipe label.

    ``num_experts`` toggles 2D (linear) vs 3D (MoE experts); non-aligned dims are
    padded to block boundaries before quantizing, then trimmed back.

    - ``torch.float8_e4m3fn`` (FP8 / MXFP8): one E4M3 byte per value; ``scale_dtype``
      picks fp32 or UE8M0 block inv-scales, laid out per ``scale_layout`` (block /
      per_tensor_1d / per_tensor_111, 3D only).
    - ``torch.int8`` (packed E2M1 = FP4): two codes per byte, always with per-row
      UE8M0 group scales — ``block_n`` must be 1, and ``scale_dtype`` / ``scale_layout``
      are forced to UE8M0 / block.
    """
    is_fp4 = weight_dtype == torch.int8  # packed E2M1 (FP4); else E4M3 (FP8)
    is_2d = num_experts is None
    E = 1 if is_2d else num_experts

    if is_fp4:
        block_n, block_k = block_size
        assert block_n == 1, (
            f"E2M1 scales are per-row; block_n must be 1, got {block_n}"
        )
        assert in_features % block_k == 0, (
            f"K ({in_features}) must be divisible by block_k ({block_k})"
        )
        scale_dtype, scale_layout, qmax = torch.float8_e8m0fnu, "block", E2M1_MAX
    else:
        block_n, block_k = (
            block_size if block_size is not None else (out_features, in_features)
        )
        qmax = FP8_MAX

    W = torch.randn(E, out_features, in_features, dtype=torch.float32, device=device)
    pad_n, pad_k = (-out_features) % block_n, (-in_features) % block_k
    W = torch.nn.functional.pad(W, [0, pad_k, 0, pad_n])
    Np, Kp = out_features + pad_n, in_features + pad_k
    R = W.reshape(E, Np // block_n, block_n, Kp // block_k, block_k)
    max_abs = R.abs().amax(dim=(-3, -1))
    safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    inv_scales = (safe / qmax).to(torch.float32)

    if scale_dtype in (torch.float8_e8m0fnu, torch.uint8):
        # Snap inv_scales up to the next power-of-2 (UE8M0 encoding: 2^(exp-127)).
        exp_ceil = _ue8m0_exp(inv_scales)
        inv_scales = (exp_ceil << 23).view(torch.float32)

    # Scale weights into the grid's range, then trim padding back off.
    scaled = (R * (1.0 / inv_scales).unsqueeze(-1).unsqueeze(-3)).reshape(E, Np, Kp)
    scaled = scaled[:, :out_features, :in_features]

    if is_fp4:
        # Round |scaled| to the nearest E2M1 magnitude index, re-apply the sign, then
        # pack two 4-bit codes per byte (low nibble = even K, per ``dequant_b_fp4``).
        boundaries = torch.tensor(_E2M1_BOUNDARIES, device=device)
        codes = torch.bucketize(scaled.abs(), boundaries).to(torch.uint8)
        codes |= (scaled < 0).to(torch.uint8) << 3
        Wq = (codes[..., 0::2] | (codes[..., 1::2] << 4)).view(torch.int8).contiguous()
    else:
        Wq = scaled.clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE).contiguous()

    if scale_dtype in (torch.float8_e8m0fnu, torch.uint8):
        # UE8M0 exponent bytes, exposed as float8_e8m0fnu or as raw uint8 (a common on-disk
        # encoding, e.g. MiniMax-M3-MXFP8) — both are accepted by the detectors / kernels.
        inv_scales = exp_ceil.to(torch.uint8)
        if scale_dtype == torch.float8_e8m0fnu:
            inv_scales = inv_scales.view(torch.float8_e8m0fnu)

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


# ── Dequant / activation-quant references ─────────────────────────────────────


def quant_dequant_a(
    A: torch.Tensor,
    block_k: int,
    *,
    scale: torch.Tensor | None = None,
    pow2_scale: bool = False,
) -> torch.Tensor:
    """Mirror a kernel's inline activation quant→dequant round-trip in fp32.

    Static (scalar ``scale``): quantize the whole tensor against the given
    calibration scale. Dynamic (``scale=None``): per-row, per-``block_k`` scale =
    ``amax / 448``, snapped up to the next power of two when ``pow2_scale`` (UE8M0,
    the MX recipe) else kept fp32; quantize to E4M3 and scale back.
    """
    if scale is not None:
        A_fp8 = (A.float() / scale).to(FP8_DTYPE)
        return A_fp8.float() * scale
    M, K = A.shape
    groups = A.float().reshape(M, K // block_k, block_k)
    s = groups.abs().amax(dim=-1, keepdim=True) / 448.0
    if pow2_scale:
        s = (_ue8m0_exp(s) << 23).view(torch.float32)
    A_fp8 = (groups / s.clamp(min=1e-12)).to(FP8_DTYPE)
    return (A_fp8.float() * s).reshape(M, K)


def dequant_b(
    B: torch.Tensor, Bs: torch.Tensor, block_n: int, block_k: int
) -> torch.Tensor:
    """Per-block weight dequant. ``B`` is E4M3 (one value/byte) or packed E2M1
    (``int8``, two codes/byte = FP4); ``Bs`` is fp32 or UE8M0 (``2^(exp - 127)``)
    block inv-scales, broadcast over ``block_n`` rows × ``block_k`` cols."""
    if B.dtype == torch.int8:  # packed E2M1 (FP4): two codes per byte
        N, K_half = B.shape
        lut = torch.tensor(_E2M1_DECODE, dtype=torch.float32, device=B.device)
        codes = B.to(torch.uint8).long()
        B_vals = torch.stack(
            [lut[codes & 0x0F], lut[(codes >> 4) & 0x0F]], dim=-1
        ).reshape(N, K_half * NIBBLES_PER_BYTE)
    else:
        B_vals = B.float()
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = (Bs.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    N, K = B_vals.shape
    scales_full = Bs.repeat_interleave(block_n, dim=0).repeat_interleave(
        block_k, dim=1
    )[:N, :K]
    return B_vals * scales_full


# ── Unified matmul reference ──────────────────────────────────────────────────


def ref_matmul(A, B, Bs, block_size, output_dtype=torch.float32, activation_scale=None):
    """Pure-PyTorch reference for ``matmul``: dequant both sides, fp32 matmul, cast.
    ``block_size`` is the weight scale block: ``(1, 32)`` marks the MX recipes (per-row
    UE8M0 group-32 → power-of-2 activation quant), any other shape / ``None`` is FP8.
    ``dequant_b`` decodes E2M1 vs E4M3 weights by ``B.dtype``; ``activation_scale``
    (FP8 only) selects static quant."""
    is_mx = block_size == (1, MX_SCALE_GROUP_K)
    block_n, block_k = block_size if block_size is not None else B.shape
    A_deq = quant_dequant_a(A, block_k, scale=activation_scale, pow2_scale=is_mx)
    B_deq = dequant_b(B, Bs, block_n, block_k)
    return (A_deq @ B_deq.T).to(output_dtype)


# ── shared weight-recipe registry (test_ops scenarios + test_moe fused problems) ──
#
# One row per support-matrix line: make(N, K, E) -> (B, Bs); dequant(B, Bs) -> fp32
# (E, N, K); act_quant[recipe] -> the host quant fn the ops themselves call (None = the
# family default applied to a raw A); dq_act dequantizes its output for the torch oracle.

from finegrained_fp8.utils import (  # type: ignore  # noqa: E402
    NVFP4_SCALE_GROUP_K,
    fp8_act_quant_block_dynamic,
    fp8_act_quant_tensor_wide,
    mxfp4_act_quant,
    mxfp8_act_quant,
    nvfp4_act_quant,
)

_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1_LUT = _E2M1_LUT + [-v for v in _E2M1_LUT]


def dq_scale(s: torch.Tensor) -> torch.Tensor:
    """Group scale to fp32 by dtype: E4M3 (NVFP4) reads directly, UE8M0 (uint8 or
    float8_e8m0fnu) is 2^(exp-127), fp32 passes through."""
    if s.dtype == torch.float8_e4m3fn:
        return s.float()
    if s.dtype in (torch.uint8, torch.float8_e8m0fnu):
        return torch.pow(2.0, s.view(torch.uint8).float() - 127)
    return s.float()


def dq_e2m1(q: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 (two nibbles per byte along the last dim, low first) to fp32."""
    lut = torch.tensor(_E2M1_LUT, device=q.device)
    u = q.view(torch.uint8)
    lo, hi = lut[(u & 0xF).long()], lut[(u >> 4).long()]
    return torch.stack([lo, hi], dim=-1).reshape(*q.shape[:-1], q.shape[-1] * 2)


def dq_grouped(q: torch.Tensor, s: torch.Tensor, group: int) -> torch.Tensor:
    """Dequantize group-scaled values (packed E2M1 or E4M3) along the last dim."""
    v = dq_e2m1(q) if q.dtype in (torch.int8, torch.uint8) else q.float()
    return v * torch.repeat_interleave(dq_scale(s), group, dim=-1)


def dq_block_fp8(B, Bs, block_n: int, block_k: int) -> torch.Tensor:
    """(E, N, K) E4M3 + (E, N//bn, K//bk) block inv-scales -> fp32."""
    s = torch.repeat_interleave(
        torch.repeat_interleave(dq_scale(Bs), block_n, dim=-2), block_k, dim=-1
    )
    return B.float() * s


def _make_nvfp4(N, K, E):
    w = torch.randn(E, N, K, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.05
    qs = [nvfp4_act_quant(w[e].contiguous()) for e in range(E)]
    return (
        torch.stack([q for q, _ in qs]).view(torch.int8),
        torch.stack([s for _, s in qs]),
    )


def _make_full(dtype):
    def make(N, K, E):
        return torch.randn(E, N, K, device=TEST_DEVICE, dtype=dtype) * 0.05, None

    return make


def _make_mx(weight_dtype, scale_dtype):
    def make(N, K, E):
        return make_weights(
            N,
            K,
            TEST_DEVICE,
            [1, MX_SCALE_GROUP_K],
            weight_dtype=weight_dtype,
            scale_dtype=scale_dtype,
            num_experts=E,
        )

    return make


_MX_ACT = {None: mxfp8_act_quant, "mxfp8": mxfp8_act_quant, "mxfp4": mxfp4_act_quant}

WEIGHTS = {
    "fp8_128x128": dict(
        make=lambda N, K, E: make_weights(N, K, TEST_DEVICE, [128, 128], num_experts=E),
        dequant=lambda B, Bs: dq_block_fp8(B, Bs, 128, 128),
        input_recipes=(None, "fp8"),
        output_recipes=(None, "fp8"),
        act_quant={
            None: lambda A: fp8_act_quant_block_dynamic(A, 128),
            "fp8": lambda A: fp8_act_quant_block_dynamic(A, 128),
        },
        dq_act=lambda q, s: q.float() * torch.repeat_interleave(s.float(), 128, dim=-1),
    ),
    "fp8_tensor": dict(
        make=lambda N, K, E: make_weights(
            N, K, TEST_DEVICE, None, scale_layout="per_tensor_111", num_experts=E
        ),
        dequant=lambda B, Bs: B.float() * Bs.float().reshape(-1, 1, 1),
        input_recipes=(None, "fp8"),
        output_recipes=(None,),
        act_quant={
            None: lambda A: fp8_act_quant_tensor_wide(A, A.shape[-1]),
            "fp8": lambda A: fp8_act_quant_tensor_wide(A, A.shape[-1]),
        },
        dq_act=lambda q, s: q.float() * s.float().reshape(-1, 1),
    ),
    "mxfp8": dict(
        make=_make_mx(torch.float8_e4m3fn, torch.float8_e8m0fnu),
        dequant=lambda B, Bs: dq_grouped(B, Bs, MX_SCALE_GROUP_K),
        input_recipes=(None, "mxfp8", "mxfp4"),
        output_recipes=(None, "mxfp8", "mxfp4"),
        act_quant=_MX_ACT,
        dq_act=lambda q, s: dq_grouped(q, s, MX_SCALE_GROUP_K),
    ),
    # UE8M0 scales stored as raw uint8 (e.g. MiniMax-M3-MXFP8 checkpoints) — must still
    # detect as MXFP8 and route to the MX path, not fall back to block-dynamic.
    "mxfp8_u8": dict(
        make=_make_mx(torch.float8_e4m3fn, torch.uint8),
        dequant=lambda B, Bs: dq_grouped(B, Bs, MX_SCALE_GROUP_K),
        input_recipes=(None,),
        output_recipes=(None, "mxfp8"),
        act_quant=_MX_ACT,
        dq_act=lambda q, s: dq_grouped(q, s, MX_SCALE_GROUP_K),
    ),
    "mxfp4": dict(
        make=_make_mx(torch.int8, torch.float8_e8m0fnu),
        dequant=lambda B, Bs: dq_grouped(B, Bs, MX_SCALE_GROUP_K),
        input_recipes=(None, "mxfp8", "mxfp4"),
        output_recipes=(None, "mxfp8", "mxfp4"),
        act_quant=_MX_ACT,
        dq_act=lambda q, s: dq_grouped(q, s, MX_SCALE_GROUP_K),
    ),
    "nvfp4": dict(
        make=_make_nvfp4,
        dequant=lambda B, Bs: dq_grouped(B, Bs, NVFP4_SCALE_GROUP_K),
        input_recipes=(None, "nvfp4"),
        output_recipes=(None, "nvfp4"),
        act_quant={None: nvfp4_act_quant, "nvfp4": nvfp4_act_quant},
        dq_act=lambda q, s: dq_grouped(q, s, NVFP4_SCALE_GROUP_K),
    ),
    "bf16": dict(
        make=_make_full(torch.bfloat16),
        dequant=lambda B, Bs: B.float(),
        input_recipes=(None,),
        output_recipes=(None,),
        act_quant={None: None},
        dq_act=None,
    ),
    "fp16": dict(
        make=_make_full(torch.float16),
        dequant=lambda B, Bs: B.float(),
        input_recipes=(None,),
        output_recipes=(None,),
        act_quant={None: None},
        dq_act=None,
    ),
}

# quant fns for requant-output verification, by output recipe name
REQUANT_FN = {
    "fp8": None,  # per-(row, N-block) scale — verified via dequant closeness only
    "mxfp8": mxfp8_act_quant,
    "mxfp4": mxfp4_act_quant,
    "nvfp4": nvfp4_act_quant,
}
REQUANT_GROUP = {
    "mxfp8": MX_SCALE_GROUP_K,
    "mxfp4": MX_SCALE_GROUP_K,
    "nvfp4": NVFP4_SCALE_GROUP_K,
}
