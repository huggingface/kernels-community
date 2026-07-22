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

from finegrained_fp8 import moe  # type: ignore


@dataclass(frozen=True)
class MoEProblem:
    """End-to-end fused-MoE shape: ``num_tokens`` routed ``num_top_k`` ways through
    ``num_experts`` experts, hidden ``hidden_dim``, per-gate ``intermediate_dim``.
    ``weights`` names a ``WEIGHTS`` registry row; ``recipe`` (optional) is forwarded to
    both forwards — ``None`` follows the weight recipe on each side."""

    weights: str
    num_tokens: int = 4
    num_experts: int = 8
    hidden_dim: int = 512
    intermediate_dim: int = 256
    num_top_k: int = 8
    sentinel_fraction: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    recipe: Optional[str] = None
    swiglu_alpha: Optional[float] = None
    swiglu_limit: Optional[float] = None
    act_fn: str = "silu"

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
        recipe = f"_recipe_{self.recipe}" if self.recipe else ""
        return (
            f"{self.weights}_T{self.num_tokens}_E{self.num_experts}_H{self.hidden_dim}"
            f"_I{self.intermediate_dim}_top{self.num_top_k}_{DTYPE_TAG[self.dtype]}"
            f"{act}{recipe}{'_sentinel' if self.sentinel_fraction > 0 else ''}"
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
    # block-FP8 with UE8M0 (power-of-two) scales — the whole-model UE8M0 contract: acts,
    # weights, and the fused intermediate requant all power-of-two (DeepSeek-V4 attn / B200).
    MoEProblem(weights="fp8_128x128_ue8m0", num_tokens=1),
    MoEProblem(weights="fp8_128x128_ue8m0"),
    MoEProblem(weights="nvfp4"),
    # ── full precision: scale-less BF16 weights resolve to recipe None and the fused
    # gate_up hands the down a bare (unscaled) intermediate ──
    MoEProblem(weights="bf16", num_tokens=1),
    MoEProblem(weights="bf16"),
    # ── contraction dims on the 64 grid but off the 128 grid (gpt-oss H=I=2880): only
    # BK=64 divides, so the W4A4 chain runs the no-swap BK=64 dot_scaled rows ──
    MoEProblem(weights="mxfp4", hidden_dim=320, intermediate_dim=320),
    # ── explicit recipe forwarding: W4A8 chain on mxfp4 weights (default is W4A4) ──
    MoEProblem(weights="mxfp4", recipe="mxfp8"),
    # ── clamped/scaled SwiGLU (GPT-OSS / MiniMax-M3); glu is recipe-independent ──
    MoEProblem(weights="mxfp8", swiglu_alpha=1.702, swiglu_limit=7.0),
    # alpha / limit are independent glu branches — cover each alone
    MoEProblem(weights="mxfp8", swiglu_alpha=1.702),
    MoEProblem(weights="mxfp8", swiglu_limit=7.0),
    # ── GeGLU / ReGLU (activation orthogonal to recipe, one MXFP8 shape each) ──
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
    (``None`` for single-level recipes) for the recipe."""
    make = WEIGHTS[problem.weights]["make"]
    gate_up, gate_up_s, gate_up_g = make(
        2 * problem.intermediate_dim, problem.hidden_dim, problem.num_experts
    )
    down, down_s, down_g = make(
        problem.hidden_dim, problem.intermediate_dim, problem.num_experts
    )
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
    # weights are two-level; other recipes have a bare block scale + None global).
    common = dict(
        gate_up_proj_global_scale=gate_up_g,
        down_proj_global_scale=down_g,
        act_fn=problem.act_fn,
        swiglu_alpha=problem.swiglu_alpha,
        swiglu_limit=problem.swiglu_limit,
        recipe=problem.recipe,
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


@pytest.mark.kernels_ci
@pytest.mark.skipif(TEST_DEVICE != "cuda", reason="CUDA required")
def test_fused_batched_compiles_across_shapes():
    """TWO different mxfp4 problems through ONE compiled function with no compiler
    reset in between: the recompile marks the weight shapes automatic-dynamic, and the
    family predicates must still return real bools — a lazy SymBool reaching
    ``is_x(gate) != is_x(down)`` builds a nested symbolic Eq that crashes dynamo's
    ``evaluate_expr`` (the gpt-oss compile failure). ``fullgraph`` so any graph break
    fails loud; the shape pair keeps both contraction dims on different grids."""
    torch.compiler.reset()
    compiled = torch.compile(moe.moe_fused_batched, fullgraph=True)
    for problem in (
        MoEProblem(weights="mxfp4", num_tokens=1),
        MoEProblem(
            weights="mxfp4", num_tokens=1, hidden_dim=320, intermediate_dim=320
        ),
    ):
        torch.manual_seed(0)
        gate_up, gate_up_s, gate_up_g, down, down_s, down_g = _make_moe_weights(problem)
        hidden, top_k_index, top_k_weights = _make_moe_inputs(problem)
        out = compiled(
            hidden, top_k_index, top_k_weights, gate_up, down, gate_up_s, down_s,
            gate_up_proj_global_scale=gate_up_g, down_proj_global_scale=down_g,
        )
        assert torch.isfinite(out.float()).all(), problem.id
