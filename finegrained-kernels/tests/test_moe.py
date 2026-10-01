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
"""Fused-vs-unfused MoE forward parity. The two forwards share the base ops, so this
file tests exactly what ``test_ops`` cannot: the fused epilogue math (GLU + intermediate
requant) against the host-side unfused path (``simulate_unfused`` rounds each fused step
through the activation dtype so they agree to reduce order), plus the moe orchestration
itself — ``weighted_reduce``, scheduling reuse across the two GEMMs, EP-sentinel
skipping at the reduce, and ``format`` forwarding. Op-level coverage (formats,
epilogues, requant, routing variants against an independent torch oracle) lives in
``test_ops.py``; the weight formats come from the shared ``WEIGHTS`` registry."""

from dataclasses import dataclass
from typing import Callable, Optional, Union

import pytest
import torch

from utils import (  # type: ignore
    DTYPE_TAG,
    DTYPE_TO_TOL,
    SUPPORTS_SWIZZLED_SCALES,
    TEST_DEVICE,
    WEIGHTS,
)

from finegrained_kernels import moe, swizzle_mx_scales  # type: ignore
from finegrained_kernels.epilogue import fused_glu  # type: ignore


@dataclass(frozen=True)
class MoEProblem:
    """End-to-end fused-MoE shape: ``num_tokens`` routed ``num_top_k`` ways through
    ``num_experts`` experts, hidden ``hidden_dim``, per-gate ``intermediate_dim``.
    ``weights`` names a ``WEIGHTS`` registry row; ``activation_format`` is forwarded to both
    forwards — ``None`` follows the weight format, ``"bf16"`` is weight-only (bf16 acts)."""

    weights: str
    num_tokens: int = 4
    num_experts: int = 8
    hidden_dim: int = 512
    intermediate_dim: int = 256
    num_top_k: int = 8
    sentinel_fraction: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    activation_format: Optional[str] = None
    swiglu_alpha: Optional[float] = None
    swiglu_limit: Optional[float] = None
    act_fn: Union[str, Callable] = "silu"  # a callable runs on the host between the GEMMs
    swizzled: bool = False  # pre-swizzled (5D SWIZZLE_32_4_4) weight scales — the deployment layout
    input_globals: bool = False  # calibrated NVFP4 activation input_scale per projection
    expert_globals: bool = False  # the down's calibrated input_scale differs per expert
    static: bool = False  # calibrated (static) activation scale, one per expert
    post_expert_norm: Optional[str] = None  # a model's per-expert output norm, by name

    @property
    def id(self):
        if self.swiglu_alpha is not None and self.swiglu_limit is not None:
            act = "_swiglu"
        elif self.swiglu_alpha is not None:
            act = "_swiglu_alpha"
        elif self.swiglu_limit is not None:
            act = "_swiglu_limit"
        elif (act_name := getattr(self.act_fn, "__name__", self.act_fn)) != "silu":
            act = f"_{act_name}"
        else:
            act = ""
        fmt = "" if self.activation_format is None else f"_format_{self.activation_format}"
        return (
            f"{self.weights}_T{self.num_tokens}_E{self.num_experts}_H{self.hidden_dim}"
            f"_I{self.intermediate_dim}_top{self.num_top_k}_{DTYPE_TAG[self.dtype]}"
            f"{act}{fmt}{'_swizzled' if self.swizzled else ''}"
            f"{'_inputglobals' if self.input_globals else ''}"
            f"{'_expertglobals' if self.expert_globals else ''}"
            f"{'_static' if self.static else ''}"
            f"{'_' + self.post_expert_norm if self.post_expert_norm else ''}"
            f"{'_sentinel' if self.sentinel_fraction > 0 else ''}"
        )


MOE_PROBLEMS = [
    # ── one decode-size + one small-batch shape per weight family ──
    MoEProblem(weights="mxfp4", num_tokens=1),
    MoEProblem(weights="mxfp4"),
    MoEProblem(weights="mxfp4", dtype=torch.float16),
    MoEProblem(weights="mxfp8", num_tokens=1),
    MoEProblem(weights="mxfp8"),
    # UE8M0 scales stored as raw uint8 (e.g. MiniMax-M3-MXFP8 checkpoints) — must still
    # detect as MXFP8 and route to the MX path, not fall back to block-dynamic.
    MoEProblem(weights="mxfp8_u8"),
    MoEProblem(weights="fp8_128x128", num_tokens=1),
    MoEProblem(weights="fp8_128x128"),
    # calibrated (static) activation quant: the scales replace the runtime ones in both GEMMs of
    # both chains, and the intermediate stays bf16 — an epilogue requant has no calibrated scale
    # to emit. Each expert is its own quantized module, so each carries its own scale, and
    # ``num_top_k`` stays below ``num_experts`` so each calibrates on its own tokens.
    MoEProblem(weights="fp8_128x128", num_tokens=64, num_top_k=2, static=True),
    # per-TENSOR weights under the same scheme: what Mistral-4 carries (qscheme_act="TENSOR",
    # weight_block_size None, top-4). Ministral-3 is its dense counterpart, on the 2D op.
    MoEProblem(weights="fp8_tensor", num_tokens=64, num_top_k=4, static=True),
    # and at decode, where the batched op runs BLOCK_SIZE_M=1 and the tuner's swap arm reshapes
    # the single token — the gate|up span doubles there, which the prefill cell above never sees
    MoEProblem(weights="fp8_tensor", num_tokens=1, num_top_k=4, static=True),
    # past the row count where a per-expert scale stops reading raw rows per tile and lays them
    # out quantized instead (``quantize_routed_rows_per_expert``): 4096 routed rows, which the
    # 64-token cells above never reach, so only this one runs the kernel's pre-quantized arm
    MoEProblem(weights="fp8_tensor", num_tokens=1024, num_top_k=4, static=True),
    # block-FP8 with UE8M0 (power-of-two) scales — the whole-model UE8M0 contract: acts,
    # weights, and the fused intermediate requant all power-of-two (DeepSeek-V4 attn / B200).
    MoEProblem(weights="fp8_128x128_ue8m0", num_tokens=1),
    MoEProblem(weights="fp8_128x128_ue8m0"),
    MoEProblem(weights="nvfp4"),
    # ── calibrated NVFP4 activation input_scale per projection (the checkpoint contract):
    # gate_up quantizes hidden against its global, the intermediate requant normalizes
    # against the down's, the down consumes it — asymmetric threading breaks parity hard ──
    MoEProblem(weights="nvfp4", input_globals=True),
    MoEProblem(weights="nvfp4", num_tokens=1, input_globals=True),
    # ── pre-swizzled weight scales (swizzle once at load — the deployment contract): the fused
    # grouped chain then runs scatter-free gate_up -> swizzled Cs -> down's 5D-As fast path, and
    # batched decode reads the descriptor scale load. Values unchanged, so parity holds as-is. ──
    MoEProblem(weights="mxfp8", swizzled=True),
    MoEProblem(weights="mxfp8", num_tokens=1, swizzled=True),
    MoEProblem(weights="mxfp4", swizzled=True),
    MoEProblem(weights="mxfp4", num_tokens=1, swizzled=True),
    MoEProblem(weights="nvfp4", swizzled=True),
    MoEProblem(weights="nvfp4", num_tokens=1, swizzled=True),
    # the full deployment stack for a calibrated NVFP4 checkpoint: pre-swizzled artifact +
    # per-projection input globals, at decode batch (the bench's GLM-NVFP4 decode cell)
    MoEProblem(weights="nvfp4", num_tokens=1, swizzled=True, input_globals=True),
    # ── input_scale per expert (the modelopt layout, once the gate|up stack's two calibrated
    # halves are merged into one global per expert at load): the requant epilogue normalizes each
    # row by its expert's down global and the down folds the same one back ──
    MoEProblem(weights="nvfp4", input_globals=True, expert_globals=True),
    MoEProblem(weights="nvfp4", num_tokens=1, input_globals=True, expert_globals=True),
    MoEProblem(weights="nvfp4", swizzled=True, input_globals=True, expert_globals=True),
    # ── a model's per-expert output norm, folded into the reduce: one cell per normalization
    # form (the form decides what the row's mean square is taken on) plus a decode shape ──
    MoEProblem(weights="mxfp8", post_expert_norm="input_scaled_rms_norm"),
    MoEProblem(weights="mxfp8", num_tokens=1, post_expert_norm="input_scaled_rms_norm"),
    MoEProblem(weights="nvfp4", input_globals=True, post_expert_norm="rms_norm"),
    MoEProblem(weights="mxfp4", post_expert_norm="centered_rms_norm"),
    # with EP sentinels: those rows are never written, so the fused reduce has to mask their
    # rsqrt the way it masks their values (0 * NaN is NaN)
    MoEProblem(weights="mxfp8", sentinel_fraction=0.25, post_expert_norm="rms_norm"),
    # ── full precision: scale-less BF16 weights resolve to format None and the fused
    # gate_up hands the down a bare (unscaled) intermediate ──
    # a caller-provided activation (the torch GLU itself) runs on the host between the GEMMs — the
    # path any activation outside ``get_supported_act_fns()`` takes
    MoEProblem(weights="mxfp8", act_fn=fused_glu),
    MoEProblem(weights="mxfp4", activation_format="bf16", act_fn=fused_glu),  # weight-only bf16 hand-off
    MoEProblem(weights="bf16", num_tokens=1),
    MoEProblem(weights="bf16"),
    # ── contraction dims on the 64 grid but off the 128 grid (gpt-oss H=I=2880): only
    # BK=64 divides, so the W4A4 chain runs the no-swap BK=64 dot_scaled rows ──
    MoEProblem(weights="mxfp4", hidden_dim=320, intermediate_dim=320),
    # ── explicit format forwarding: W4A8 chain on mxfp4 weights (default is W4A4) ──
    MoEProblem(weights="mxfp4", activation_format="mxfp8"),
    # ── weight-only weight-only: bf16 acts × mxfp4 weights, dedicated dequant-then-bf16-dot
    # kernels (the gpt-oss / matmul_ogs format); intermediate stays bf16 (no requant) ──
    MoEProblem(weights="mxfp4", activation_format="bf16", num_tokens=1),
    MoEProblem(weights="mxfp4", activation_format="bf16"),
    # ── clamped/scaled SwiGLU (GPT-OSS / MiniMax-M3); glu is format-independent ──
    MoEProblem(weights="mxfp8", swiglu_alpha=1.702, swiglu_limit=7.0),
    # alpha / limit are independent glu branches — cover each alone
    MoEProblem(weights="mxfp8", swiglu_alpha=1.702),
    MoEProblem(weights="mxfp8", swiglu_limit=7.0),
    # ── GeGLU / ReGLU (activation orthogonal to format, one MXFP8 shape each) ──
    MoEProblem(weights="mxfp8", act_fn="gelu"),
    MoEProblem(weights="mxfp8", act_fn="relu"),
    # ── expert parallelism: non-local experts sentinel-masked ──
    MoEProblem(weights="mxfp8", num_tokens=8, sentinel_fraction=0.875),
    MoEProblem(weights="fp8_128x128", num_tokens=8, sentinel_fraction=0.875),
    # int32 pointer-offset overflow guard for the fused paths: the last experts'
    # gate_up offsets exceed 2^31 elements (127 * 2*2048 * 6144 = 3.196e9); a regressed
    # int64 cast corrupts the high-routed tokens vs the torch reference. E is a power of
    # two (the fused-grouped scheduling kernels require it).
    MoEProblem(
        weights="fp8_128x128",
        num_tokens=512,
        num_experts=128,
        hidden_dim=6144,
        intermediate_dim=2048,
        num_top_k=4,
    ),
]


def _make_moe_weights(problem: MoEProblem):
    """gate_up ``(E, 2I, H)`` and down ``(E, H, I)`` weights + block inv-scales + per-tensor globals
    (``None`` for single-level formats) for the format. ``swizzled`` swizzles once here (the
    deployment contract, not per call): the gate_up scale is the ONE gate-interleaved artifact
    (6D — the shape carries the layout) and every forward consumes it directly — fused kernels
    read block pairs, the unfused plain GEMM remaps its block index in-kernel."""
    make = WEIGHTS[problem.weights]["make"]
    gate_up, gate_up_s, gate_up_g = make(
        2 * problem.intermediate_dim, problem.hidden_dim, problem.num_experts
    )
    down, down_s, down_g = make(
        problem.hidden_dim, problem.intermediate_dim, problem.num_experts
    )
    if problem.swizzled:
        if not SUPPORTS_SWIZZLED_SCALES:
            pytest.skip("the SWIZZLE_32_4_4 scale layout is the tcgen05 fast path (CUDA-only)")
        gate_up_s = swizzle_mx_scales(gate_up_s)
        down_s = swizzle_mx_scales(down_s)
    return gate_up, gate_up_s, gate_up_g, down, down_s, down_g


def _make_moe_inputs(problem: MoEProblem):
    """Random ``(hidden, top_k_index, top_k_weights)`` for the fused-MoE problem shape."""
    hidden = torch.randn(
        problem.num_tokens, problem.hidden_dim, device=TEST_DEVICE, dtype=problem.dtype
    )
    top_k_index = torch.randint(
        0,
        problem.num_experts,
        (problem.num_tokens, problem.num_top_k),
        device=TEST_DEVICE,
        dtype=torch.int32,
    )
    if problem.sentinel_fraction > 0:
        # EP: mark a random subset of routed slots non-local with an out-of-range id
        # (== num_experts), which the fused path must skip.
        flat = top_k_index.reshape(-1)
        n_sentinel = int(round(flat.numel() * problem.sentinel_fraction))
        idx = torch.randperm(flat.numel(), device=flat.device)[:n_sentinel]
        flat[idx] = problem.num_experts
    top_k_weights = torch.rand(
        problem.num_tokens, problem.num_top_k, device=TEST_DEVICE, dtype=problem.dtype
    )
    return hidden, top_k_index, top_k_weights


def _input_globals(problem: MoEProblem, hidden):
    """The calibrated NVFP4 activation globals per projection, as a checkpoint provides them:
    the gate_up's is the hidden's own amax rule (one value — the hidden is quantized once, before
    routing), the down's a fixed plausible value (the intermediate's amax isn't known pre-run; any
    positive one is self-consistent) sitting above its calibrated amax/(6·448) for these shapes so
    the normalized values only SHRINK — real two-level math rather than a saturated regime. Both
    forwards get the same pair, so a one-sided thread breaks parity. ``expert_globals`` fans the
    down's out per expert (what modelopt writes): the gate_up requant normalizes each row by ITS
    expert's value and the down folds the same one back."""
    if not problem.input_globals:
        return None, None
    gate_up_in_g = (hidden.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float().reshape(1)
    down_in_g = torch.full((1,), 1e3, device=hidden.device, dtype=torch.float32)
    if problem.expert_globals:
        fan = torch.linspace(
            0.5, 2.0, problem.num_experts, device=hidden.device, dtype=torch.float32
        )
        down_in_g = (down_in_g * fan).contiguous()
    return gate_up_in_g, down_in_g


def _static_scales(problem: MoEProblem, hidden, gate_up, gate_up_s, top_k_index):
    """The calibrated activation scales, derived the way a calibration pass derives them: ``amax /
    448`` of what each GEMM actually sees — the routed hidden states for gate_up, the GLU
    intermediate for down — one per expert, each being its own quantized module, over the tokens
    routed to it. ``None`` on a dynamic problem, which quantizes at runtime instead."""
    if not problem.static:
        return None, None
    weight = WEIGHTS[problem.weights]["dequant"](gate_up, gate_up_s).float()
    inter = fused_glu(
        torch.einsum("th,enh->etn", hidden.float(), weight),
        problem.act_fn,
        problem.swiglu_alpha,
        problem.swiglu_limit,
    )
    routed = torch.stack([(top_k_index == e).any(dim=1) for e in range(problem.num_experts)])

    def calibrate(seen):  # (E, T, C) over every token -> (E,) over the ones each expert sees
        return (seen.abs() * routed[..., None]).amax(dim=(1, 2)).div(448.0).clamp(min=1e-12).float()

    scales = [calibrate(hidden.float().expand(problem.num_experts, -1, -1)), calibrate(inter)]
    return tuple(scale.contiguous() for scale in scales)


def _common_kwargs(problem: MoEProblem, hidden, top_k_index, gate_up, gate_up_s, gate_up_g, down_g):
    """The kwargs every forward takes: the weights' second-level globals, the calibrated activation
    globals and scales, the GLU knobs, and a model's per-expert output norm. The fused chain
    folds a named norm into its reduce while the unfused reference normalizes the routed rows in
    a pass of their own, so handing both the same weight is what makes that a parity check."""
    gate_up_in_g, down_in_g = _input_globals(problem, hidden)
    gate_up_act_s, down_act_s = _static_scales(problem, hidden, gate_up, gate_up_s, top_k_index)
    norm_weight = (
        torch.randn(problem.hidden_dim, device=TEST_DEVICE, dtype=problem.dtype) * 0.3
        if problem.post_expert_norm
        else None
    )
    return dict(
        gate_up_proj_weight_global_scale=gate_up_g,
        down_proj_weight_global_scale=down_g,
        gate_up_proj_input_global_scale=gate_up_in_g,
        down_proj_input_global_scale=down_in_g,
        gate_up_proj_activation_scale=gate_up_act_s,
        down_proj_activation_scale=down_act_s,
        act_fn=problem.act_fn,
        swiglu_alpha=problem.swiglu_alpha,
        swiglu_limit=problem.swiglu_limit,
        activation_format=problem.activation_format,
        post_expert_norm=problem.post_expert_norm,
        post_expert_norm_weight=norm_weight,
    )


def _assert_fused_correctness(out, ref, problem: MoEProblem):
    """Shape, dtype, and value checks against the unfused reference."""
    assert out.shape == (problem.num_tokens, problem.hidden_dim)
    assert out.dtype == problem.dtype
    atol, rtol = DTYPE_TO_TOL[problem.dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def _run_pair(problem: MoEProblem, fused_fn, unfused_fn):
    torch.manual_seed(0)
    gate_up, gate_up_s, gate_up_g, down, down_s, down_g = _make_moe_weights(problem)
    hidden, top_k_index, top_k_weights = _make_moe_inputs(problem)
    # The decoupled API takes pure block scales + the globals as separate args (nvfp4 weights are
    # two-level; other formats have a bare block scale + None global).
    common = _common_kwargs(problem, hidden, top_k_index, gate_up, gate_up_s, gate_up_g, down_g)
    ref = unfused_fn(
        hidden, top_k_index, top_k_weights, gate_up, down, gate_up_s, down_s, **common
    )
    out = fused_fn(
        hidden,
        top_k_index,
        top_k_weights,
        gate_up,
        down,
        gate_up_s,
        down_s,
        simulate_unfused=True,
        **common,
    )
    _assert_fused_correctness(out, ref, problem)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", MOE_PROBLEMS, ids=lambda p: p.id)
def test_fused_batched(problem):
    """Fused two-kernel MoE (gate_up + activation + requant + down + top-k reduce) via
    ``moe_fused_batched`` vs the unfused reference. ``simulate_unfused`` rounds each
    fused step through the activation dtype so the two agree to reduce order. NVFP4
    decode runs the software/swap arms (the native mxf4nvf4 M=128 staging is
    dot_scaled-only); the ops validate the pairing."""
    _run_pair(problem, moe.moe_fused_batched, moe.moe_unfused_batched)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("problem", MOE_PROBLEMS, ids=lambda p: p.id)
def test_fused_grouped(problem):
    """Fused grouped MoE (gather gate_up + activation + requant + grouped down + top-k
    reduce) via ``moe_fused_grouped`` vs the same unfused reference, with
    ``simulate_unfused`` rounding each fused step through the activation dtype."""
    _run_pair(problem, moe.moe_fused_grouped, moe.moe_unfused_grouped)


_PRODUCTION_ARM_PROBLEMS = [
    MoEProblem(weights="mxfp8"),
    MoEProblem(weights="fp8_128x128"),
    MoEProblem(weights="nvfp4", input_globals=True),
    MoEProblem(weights="mxfp8", num_tokens=8, sentinel_fraction=0.875),
]


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="Accelerator not available")
@pytest.mark.parametrize("fused_fn, unfused_fn", [
    (moe.moe_fused_grouped, moe.moe_unfused_grouped),
    (moe.moe_fused_batched, moe.moe_unfused_batched),
], ids=["grouped", "batched"])
@pytest.mark.parametrize("problem", _PRODUCTION_ARM_PROBLEMS, ids=lambda p: p.id)
def test_fused_production_arm(problem, fused_fn, unfused_fn):
    """The DEPLOYED fused forward (``simulate_unfused=False`` — fp32-accumulate epilogue,
    no per-step rounding) against the unfused reference at a loose tolerance. Every parity
    cell above flips the ``SIMULATE_UNFUSED`` kernel constexpr; this is the only value
    check the production arm itself gets end-to-end (including ``weighted_reduce``'s
    sentinel-skip path under EP)."""
    torch.manual_seed(0)
    gate_up, gate_up_s, gate_up_g, down, down_s, down_g = _make_moe_weights(problem)
    hidden, top_k_index, top_k_weights = _make_moe_inputs(problem)
    common = _common_kwargs(problem, hidden, top_k_index, gate_up, gate_up_s, gate_up_g, down_g)
    ref = unfused_fn(
        hidden, top_k_index, top_k_weights, gate_up, down, gate_up_s, down_s, **common
    )
    out = fused_fn(
        hidden, top_k_index, top_k_weights, gate_up, down, gate_up_s, down_s, **common
    )
    assert out.shape == ref.shape and out.dtype == ref.dtype
    denom = ref.float().norm().clamp(min=1e-6)
    rel = (out.float() - ref.float()).norm() / denom
    assert rel < 0.05, f"production fused arm diverges from unfused: rel={rel:.3e}"


_TORCH_BASELINE_PROBLEMS = [
    MoEProblem(weights="mxfp8", num_tokens=64),
    MoEProblem(weights="mxfp4", num_tokens=64),
    MoEProblem(weights="nvfp4", num_tokens=64),
    MoEProblem(weights="nvfp4", num_tokens=64, input_globals=True),
]


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
@pytest.mark.parametrize("problem", _TORCH_BASELINE_PROBLEMS, ids=lambda p: p.id)
def test_torch_grouped_baseline(problem):
    """``moe_torch_grouped`` (the cuBLAS ``scaled_grouped_mm`` baseline the bench compares
    against) vs the unfused reference — its correctness otherwise rides only the bench's
    unasserted parity print, and a wrong baseline silently distorts every figure. Weight
    scales go through torchao's rearrange (the baseline's own layout contract)."""
    pytest.importorskip("torchao")
    from torchao.prototype.moe_training.kernels.mxfp8 import (
        triton_mx_block_rearrange_per_group_3d,
    )

    torch.manual_seed(0)
    gate_up, gate_up_s, gate_up_g, down, down_s, down_g = _make_moe_weights(problem)
    hidden, top_k_index, top_k_weights = _make_moe_inputs(problem)
    common = _common_kwargs(problem, hidden, top_k_index, gate_up, gate_up_s, gate_up_g, down_g)
    ref = moe.moe_unfused_grouped(
        hidden, top_k_index, top_k_weights, gate_up, down, gate_up_s, down_s, **common
    )

    def preblock(ws):  # the baseline's own layout: torchao's rearrange, done once offline
        return triton_mx_block_rearrange_per_group_3d(ws.view(torch.uint8)).view(ws.dtype)

    out = moe.moe_torch_grouped(
        hidden, top_k_index, top_k_weights, gate_up, down,
        preblock(gate_up_s), preblock(down_s), **common,
    )
    assert out.shape == ref.shape
    denom = ref.float().norm().clamp(min=1e-6)
    rel = (out.float() - ref.float()).norm() / denom
    assert rel < 0.05, f"torch baseline diverges from unfused reference: rel={rel:.3e}"


def _run_compiled_across_shapes(fused_fn):
    """TWO different mxfp4 problems through ONE ``torch.compile(fullgraph=True)`` function
    with no compiler reset in between: the recompile marks the weight shapes
    automatic-dynamic, and the family predicates must still return real bools — a lazy
    SymBool reaching ``is_x(gate) != is_x(down)`` builds a nested symbolic Eq that crashes
    dynamo's ``evaluate_expr`` (the gpt-oss compile failure). ``fullgraph`` so any graph
    break fails loud; the shape pair keeps both contraction dims on different grids. Each
    output is value-checked against the same forward run eager — finite-only would pass a
    compiled path that returns wrong-but-finite numbers (e.g. a decomposed opaque op)."""
    torch.compiler.reset()
    compiled = torch.compile(fused_fn, fullgraph=True)
    for problem in (
        MoEProblem(weights="mxfp4", num_tokens=1),
        MoEProblem(
            weights="mxfp4", num_tokens=1, hidden_dim=320, intermediate_dim=320
        ),
    ):
        torch.manual_seed(0)
        gate_up, gate_up_s, gate_up_g, down, down_s, down_g = _make_moe_weights(problem)
        hidden, top_k_index, top_k_weights = _make_moe_inputs(problem)
        kw = dict(gate_up_proj_weight_global_scale=gate_up_g, down_proj_weight_global_scale=down_g)
        out = compiled(
            hidden, top_k_index, top_k_weights, gate_up, down, gate_up_s, down_s, **kw
        )
        ref = fused_fn(
            hidden, top_k_index, top_k_weights, gate_up, down, gate_up_s, down_s, **kw
        )
        atol, rtol = DTYPE_TO_TOL[problem.dtype]
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol, msg=problem.id)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
def test_fused_batched_compiles_across_shapes():
    """``moe_fused_batched`` through the shared two-shape compile check (see
    ``_run_compiled_across_shapes`` for the dynamo failure class it guards)."""
    _run_compiled_across_shapes(moe.moe_fused_batched)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE is None, reason="accelerator (CUDA/XPU) required")
def test_fused_grouped_compiles_across_shapes():
    """``moe_fused_grouped`` through the same two-shape compile check — the grouped chain
    additionally puts ``compute_grouped_scheduling`` (an opaque custom op) inside the
    graph, which the batched sibling never exercises."""
    _run_compiled_across_shapes(moe.moe_fused_grouped)

