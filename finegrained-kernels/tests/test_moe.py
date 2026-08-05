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
skipping at the reduce, and ``recipe`` forwarding. Op-level coverage (recipes,
epilogues, requant, routing variants against an independent torch oracle) lives in
``test_ops.py``; the weight recipes come from the shared ``WEIGHTS`` registry."""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch

from utils import (  # type: ignore
    DTYPE_TAG,
    DTYPE_TO_TOL,
    TEST_DEVICE,
    WEIGHTS,
)

from finegrained_kernels import moe, swizzle_mx_scales  # type: ignore


@dataclass(frozen=True)
class MoEProblem:
    """End-to-end fused-MoE shape: ``num_tokens`` routed ``num_top_k`` ways through
    ``num_experts`` experts, hidden ``hidden_dim``, per-gate ``intermediate_dim``.
    ``weight_recipe`` names a ``WEIGHTS`` registry row; ``act_recipe`` is forwarded to both
    forwards — ``"weights"`` follows the weight recipe, ``None`` is weight-only (bf16 acts)."""

    weight_recipe: str
    num_tokens: int = 4
    num_experts: int = 8
    hidden_dim: int = 512
    intermediate_dim: int = 256
    num_top_k: int = 8
    sentinel_fraction: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    act_recipe: Optional[str] = "weights"
    swiglu_alpha: Optional[float] = None
    swiglu_limit: Optional[float] = None
    act_fn: str = "silu"
    swizzled: bool = False  # pre-swizzled (5D SWIZZLE_32_4_4) weight scales — the deployment layout
    input_globals: bool = False  # calibrated NVFP4 activation input_scale per projection

    @property
    def id(self):
        if self.swiglu_alpha is not None and self.swiglu_limit is not None:
            act = "_swiglu"
        elif self.swiglu_alpha is not None:
            act = "_swiglu_alpha"
        elif self.swiglu_limit is not None:
            act = "_swiglu_limit"
        elif self.act_fn != "silu":
            act = f"_{self.act_fn}"
        else:
            act = ""
        recipe = "" if self.act_recipe == "weights" else f"_recipe_{self.act_recipe or 'bf16'}"
        return (
            f"{self.weight_recipe}_T{self.num_tokens}_E{self.num_experts}_H{self.hidden_dim}"
            f"_I{self.intermediate_dim}_top{self.num_top_k}_{DTYPE_TAG[self.dtype]}"
            f"{act}{recipe}{'_swizzled' if self.swizzled else ''}"
            f"{'_inputglobals' if self.input_globals else ''}"
            f"{'_sentinel' if self.sentinel_fraction > 0 else ''}"
        )


MOE_PROBLEMS = [
    # ── one decode-size + one small-batch shape per weight family ──
    MoEProblem(weight_recipe="mxfp4", num_tokens=1),
    MoEProblem(weight_recipe="mxfp4"),
    MoEProblem(weight_recipe="mxfp4", dtype=torch.float16),
    MoEProblem(weight_recipe="mxfp8", num_tokens=1),
    MoEProblem(weight_recipe="mxfp8"),
    # UE8M0 scales stored as raw uint8 (e.g. MiniMax-M3-MXFP8 checkpoints) — must still
    # detect as MXFP8 and route to the MX path, not fall back to block-dynamic.
    MoEProblem(weight_recipe="mxfp8_u8"),
    MoEProblem(weight_recipe="fp8_128x128", num_tokens=1),
    MoEProblem(weight_recipe="fp8_128x128"),
    # block-FP8 with UE8M0 (power-of-two) scales — the whole-model UE8M0 contract: acts,
    # weights, and the fused intermediate requant all power-of-two (DeepSeek-V4 attn / B200).
    MoEProblem(weight_recipe="fp8_128x128_ue8m0", num_tokens=1),
    MoEProblem(weight_recipe="fp8_128x128_ue8m0"),
    MoEProblem(weight_recipe="nvfp4"),
    # ── calibrated NVFP4 activation input_scale per projection (the checkpoint contract):
    # gate_up quantizes hidden against its global, the intermediate requant normalizes
    # against the down's, the down consumes it — asymmetric threading breaks parity hard ──
    MoEProblem(weight_recipe="nvfp4", input_globals=True),
    MoEProblem(weight_recipe="nvfp4", num_tokens=1, input_globals=True),
    # ── pre-swizzled weight scales (swizzle once at load — the deployment contract): the fused
    # grouped chain then runs scatter-free gate_up -> swizzled Cs -> down's 5D-As fast path, and
    # batched decode reads the descriptor scale load. Values unchanged, so parity holds as-is. ──
    MoEProblem(weight_recipe="mxfp8", swizzled=True),
    MoEProblem(weight_recipe="mxfp8", num_tokens=1, swizzled=True),
    MoEProblem(weight_recipe="mxfp4", swizzled=True),
    MoEProblem(weight_recipe="mxfp4", num_tokens=1, swizzled=True),
    MoEProblem(weight_recipe="nvfp4", swizzled=True),
    MoEProblem(weight_recipe="nvfp4", num_tokens=1, swizzled=True),
    # the full deployment stack for a calibrated NVFP4 checkpoint: pre-swizzled artifact +
    # per-projection input globals, at decode batch (the bench's GLM-NVFP4 decode cell)
    MoEProblem(weight_recipe="nvfp4", num_tokens=1, swizzled=True, input_globals=True),
    # ── full precision: scale-less BF16 weights resolve to recipe None and the fused
    # gate_up hands the down a bare (unscaled) intermediate ──
    MoEProblem(weight_recipe="bf16", num_tokens=1),
    MoEProblem(weight_recipe="bf16"),
    # ── contraction dims on the 64 grid but off the 128 grid (gpt-oss H=I=2880): only
    # BK=64 divides, so the W4A4 chain runs the no-swap BK=64 dot_scaled rows ──
    MoEProblem(weight_recipe="mxfp4", hidden_dim=320, intermediate_dim=320),
    # ── explicit recipe forwarding: W4A8 chain on mxfp4 weights (default is W4A4) ──
    MoEProblem(weight_recipe="mxfp4", act_recipe="mxfp8"),
    # ── weight-only weight-only: bf16 acts × mxfp4 weights, dedicated dequant-then-bf16-dot
    # kernels (the gpt-oss / matmul_ogs recipe); intermediate stays bf16 (no requant) ──
    MoEProblem(weight_recipe="mxfp4", act_recipe=None, num_tokens=1),
    MoEProblem(weight_recipe="mxfp4", act_recipe=None),
    # ── clamped/scaled SwiGLU (GPT-OSS / MiniMax-M3); glu is recipe-independent ──
    MoEProblem(weight_recipe="mxfp8", swiglu_alpha=1.702, swiglu_limit=7.0),
    # alpha / limit are independent glu branches — cover each alone
    MoEProblem(weight_recipe="mxfp8", swiglu_alpha=1.702),
    MoEProblem(weight_recipe="mxfp8", swiglu_limit=7.0),
    # ── GeGLU / ReGLU (activation orthogonal to recipe, one MXFP8 shape each) ──
    MoEProblem(weight_recipe="mxfp8", act_fn="gelu"),
    MoEProblem(weight_recipe="mxfp8", act_fn="relu"),
    # ── expert parallelism: non-local experts sentinel-masked ──
    MoEProblem(weight_recipe="mxfp8", num_tokens=8, sentinel_fraction=0.875),
    MoEProblem(weight_recipe="fp8_128x128", num_tokens=8, sentinel_fraction=0.875),
    # int32 pointer-offset overflow guard for the fused paths: the last experts'
    # gate_up offsets exceed 2^31 elements (127 * 2*2048 * 6144 = 3.196e9); a regressed
    # int64 cast corrupts the high-routed tokens vs the torch reference. E is a power of
    # two (the fused-grouped scheduling kernels require it).
    MoEProblem(
        weight_recipe="fp8_128x128",
        num_tokens=512,
        num_experts=128,
        hidden_dim=6144,
        intermediate_dim=2048,
        num_top_k=4,
    ),
]


def _make_moe_weights(problem: MoEProblem):
    """gate_up ``(E, 2I, H)`` and down ``(E, H, I)`` weights + block inv-scales + per-tensor globals
    (``None`` for single-level recipes) for the recipe. ``swizzled`` swizzles once here (the
    deployment contract, not per call): the gate_up scale is the ONE gate-interleaved artifact
    (6D — the shape carries the layout) and every forward consumes it directly — fused kernels
    read block pairs, the unfused plain GEMM remaps its block index in-kernel."""
    make = WEIGHTS[problem.weight_recipe]["make"]
    gate_up, gate_up_s, gate_up_g = make(
        2 * problem.intermediate_dim, problem.hidden_dim, problem.num_experts
    )
    down, down_s, down_g = make(
        problem.hidden_dim, problem.intermediate_dim, problem.num_experts
    )
    if problem.swizzled:
        gate_up_s = swizzle_mx_scales(gate_up_s, gate=True)
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
    # The decoupled API takes pure block scales + the per-tensor globals as separate args (nvfp4
    # weights are two-level; other recipes have a bare block scale + None global). The activation
    # input globals are a calibrated value for the gate_up (hidden's own amax rule) and a fixed
    # plausible one for the down (the intermediate's amax isn't known pre-run; any positive value
    # is self-consistent) — both forwards get the same pair, so a one-sided thread breaks parity.
    gate_up_in_g = down_in_g = None
    if problem.input_globals:
        gate_up_in_g = (hidden.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float().reshape(1)
        # above the intermediate's calibrated amax/(6*448) for these shapes, so the normalized
        # values only SHRINK (no e4m3 block-scale clipping) and the cell exercises real two-level
        # math rather than a saturated regime
        down_in_g = torch.full((1,), 1e3, device=hidden.device, dtype=torch.float32)
    common = dict(
        gate_up_proj_global_scale=gate_up_g,
        down_proj_global_scale=down_g,
        gate_up_input_global_scale=gate_up_in_g,
        down_input_global_scale=down_in_g,
        act_fn=problem.act_fn,
        swiglu_alpha=problem.swiglu_alpha,
        swiglu_limit=problem.swiglu_limit,
        recipe=problem.act_recipe,
    )
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
    MoEProblem(weight_recipe="mxfp8"),
    MoEProblem(weight_recipe="fp8_128x128"),
    MoEProblem(weight_recipe="nvfp4", input_globals=True),
    MoEProblem(weight_recipe="mxfp8", num_tokens=8, sentinel_fraction=0.875),
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
    gate_up_in_g = down_in_g = None
    if problem.input_globals:
        gate_up_in_g = (hidden.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float().reshape(1)
        down_in_g = torch.full((1,), 1e3, device=hidden.device, dtype=torch.float32)
    common = dict(
        gate_up_proj_global_scale=gate_up_g,
        down_proj_global_scale=down_g,
        gate_up_input_global_scale=gate_up_in_g,
        down_input_global_scale=down_in_g,
        act_fn=problem.act_fn,
        swiglu_alpha=problem.swiglu_alpha,
        swiglu_limit=problem.swiglu_limit,
        recipe=problem.act_recipe,
    )
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
    MoEProblem(weight_recipe="mxfp8", num_tokens=64),
    MoEProblem(weight_recipe="mxfp4", num_tokens=64),
    MoEProblem(weight_recipe="nvfp4", num_tokens=64),
    MoEProblem(weight_recipe="nvfp4", num_tokens=64, input_globals=True),
]


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
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
    gate_up_in_g = down_in_g = None
    if problem.input_globals:
        gate_up_in_g = (hidden.abs().amax() / (6.0 * 448.0)).clamp(min=1e-30).float().reshape(1)
        down_in_g = torch.full((1,), 1e3, device=hidden.device, dtype=torch.float32)
    common = dict(
        gate_up_proj_global_scale=gate_up_g,
        down_proj_global_scale=down_g,
        gate_up_input_global_scale=gate_up_in_g,
        down_input_global_scale=down_in_g,
        act_fn=problem.act_fn,
        swiglu_alpha=problem.swiglu_alpha,
        swiglu_limit=problem.swiglu_limit,
        recipe=problem.act_recipe,
    )
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
        MoEProblem(weight_recipe="mxfp4", num_tokens=1),
        MoEProblem(
            weight_recipe="mxfp4", num_tokens=1, hidden_dim=320, intermediate_dim=320
        ),
    ):
        torch.manual_seed(0)
        gate_up, gate_up_s, gate_up_g, down, down_s, down_g = _make_moe_weights(problem)
        hidden, top_k_index, top_k_weights = _make_moe_inputs(problem)
        kw = dict(gate_up_proj_global_scale=gate_up_g, down_proj_global_scale=down_g)
        out = compiled(
            hidden, top_k_index, top_k_weights, gate_up, down, gate_up_s, down_s, **kw
        )
        ref = fused_fn(
            hidden, top_k_index, top_k_weights, gate_up, down, gate_up_s, down_s, **kw
        )
        atol, rtol = DTYPE_TO_TOL[problem.dtype]
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol, msg=problem.id)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_fused_batched_compiles_across_shapes():
    """``moe_fused_batched`` through the shared two-shape compile check (see
    ``_run_compiled_across_shapes`` for the dynamo failure class it guards)."""
    _run_compiled_across_shapes(moe.moe_fused_batched)


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_fused_grouped_compiles_across_shapes():
    """``moe_fused_grouped`` through the same two-shape compile check — the grouped chain
    additionally puts ``compute_grouped_scheduling`` (an opaque custom op) inside the
    graph, which the batched sibling never exercises."""
    _run_compiled_across_shapes(moe.moe_fused_grouped)
