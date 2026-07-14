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

"""MoE forwards — thin orchestrations over the base ``matmul_grouped`` / ``matmul_batched`` ops.

The base ops carry the gate|up ``Epilogue`` (SwiGLU + FP8/MX requant) and the gather/scatter row
maps, so both the fused and unfused MoE forwards are pure sequencing here — no MoE-specific
kernels live in this module:

  fused:   gate_up (``Epilogue(gate=True)`` + ``Quantization(output_recipe=...)``) -> down -> ``weighted_reduce``. The
           SwiGLU + intermediate requant happen inside the gate_up kernel epilogue.
  unfused: gate_up (plain GEMM) -> host ``apply_glu`` -> down (plain GEMM) -> ``weighted_reduce``.
           The activation + requant happen between two plain GEMMs; the GEMMs self-quantize their
           raw inputs (``As=None``). Same math as the fused path, split across kernels.

grouped (prefill) shares one on-device routing pass (``compute_grouped_scheduling``): gate_up
gathers hidden by routed row and leaves its output expert-ordered; down reads it in place and
scatters to routed rows. batched (decode) dispatches per token: ``gather_idx`` reads each routed
row from the unexpanded hidden in-kernel (no copy), and EP-sentinel rows (``id >= num_experts``)
are left uninit by the GEMM and skipped in ``weighted_reduce``. ``moe_fused_*`` / ``moe_unfused_*``
are the neutral dispatchers over block-dynamic FP8 (128x128 block scales) and MXFP4/MXFP8 (UE8M0
group-32)."""

import torch

from .grouped import matmul_grouped
from .batched import matmul_batched
from .utils import (
    Epilogue,
    Quantization,
    apply_glu,
    compute_grouped_scheduling,
    fp8_act_quant_block_dynamic,
    mxfp_act_quant,
    weighted_reduce,
    is_mxfp,
    weight_block_size,
    is_mxfp4,
)


def _validate_moe(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale):
    """gate_up and down must share the recipe (both MX or both block-dynamic FP8 — the
    intermediate handed between them carries one quant format). Returns whether the recipe
    is MX (the fused dispatchers branch on it); the fp8 quantization block is derived from
    the scale shapes (``weight_block_size``), never passed."""
    is_mx = is_mxfp(gate_up_proj, gate_up_proj_scale)
    if is_mx != is_mxfp(down_proj, down_proj_scale):
        raise ValueError(
            "gate_up_proj and down_proj must use the same recipe (both MX or both block-dynamic FP8)."
        )
    return is_mx


def _gather_idx(top_k_index: torch.Tensor) -> torch.Tensor:
    """The batched routed-row gather: routed row ``s`` (``= t*K + k``) reads token ``s // num_top_k``
    of the unexpanded hidden. ``matmul_batched`` applies it in-kernel, so no ``(S, H)`` copy."""
    num_tokens, num_top_k = top_k_index.shape
    return (
        torch.arange(
            num_tokens * num_top_k, device=top_k_index.device, dtype=torch.int32
        )
        // num_top_k
    )


def _torch_weighted_reduce(down_out, top_k_index, top_k_weights, num_experts):
    """Naive (unfused) routing-weighted top-k reduce in plain torch — NOT the fused
    ``weighted_reduce`` kernel. Materializes the (bf16) weighted contribs, masks EP-sentinel rows
    (``id >= num_experts``, left uninit in ``down_out``) to 0, and torch-sums to ``(num_tokens, H)``
    (fp32 accumulate, activation-dtype out). This is the independent reference the fused
    ``weighted_reduce`` is checked against; the fused path's ``simulate_unfused`` reproduces its
    bf16-contrib rounding."""
    num_tokens, num_top_k = top_k_index.shape
    keep = (top_k_index.reshape(-1) < num_experts).reshape(-1, 1)
    contrib = torch.where(
        keep, down_out * top_k_weights.reshape(-1, 1), torch.zeros_like(down_out)
    )
    return contrib.view(num_tokens, num_top_k, down_out.size(1)).sum(dim=1)


# ── Fused grouped (prefill) ───────────────────────────────────────────────────


def w8a8_block_dynamic_fp8_moe_grouped(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H) FP8
    down_proj: torch.Tensor,  # (E, H, I) FP8
    gate_up_proj_scale: torch.Tensor,
    down_proj_scale: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Block-dynamic FP8 fused grouped MoE: gather gate_up+SiLU → FP8 intermediate →
    grouped down → routing-weighted top-k reduce. Returns ``(num_tokens, hidden_dim)``."""
    num_top_k = top_k_index.size(-1)
    NUM_EXPERTS = gate_up_proj.size(0)
    # activation scale groups follow the weight's K block, read off the scale shape
    _, block_k = weight_block_size(gate_up_proj, gate_up_proj_scale)

    expert_start, gather_idx, scatter_idx = compute_grouped_scheduling(
        top_k_index, NUM_EXPERTS, num_top_k
    )
    hidden_q, hidden_scale = fp8_act_quant_block_dynamic(hidden_states, block_k)

    # Phase 1: gate_up + SiLU + FP8 requant -> expert-ordered fp8 intermediate. Gather hidden by
    # routed row (gather_idx); leave the output expert-ordered (scatter_idx=None — the down
    # projection reads it in place, no scatter between the two GEMMs).
    inter, inter_scale = matmul_grouped(
        hidden_q,
        gate_up_proj,
        As=hidden_scale,
        Bs=gate_up_proj_scale,
        expert_start=expert_start,
        epilogue=Epilogue(
            gate=True,
            act_fn=act_fn,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            simulate_unfused=simulate_unfused,
        ),
        quantization=Quantization(output_recipe="fp8"),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    # Phase 2: grouped down over the expert-ordered intermediate (gather_idx=None), scattering to
    # routed rows (scatter_idx).
    down_out = matmul_grouped(
        inter,
        down_proj,
        As=inter_scale,
        Bs=down_proj_scale,
        expert_start=expert_start,
        output_dtype=hidden_states.dtype,
        scatter_idx=scatter_idx,
    )
    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the unfused
    # path's torch reduce (which materializes bf16 contribs); production accumulates in fp32.
    return weighted_reduce(
        down_out, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused
    )


def mxfp_dynamic_moe_grouped(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale: torch.Tensor,
    down_proj_scale: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """MXFP4/MXFP8 fused grouped MoE (UE8M0 group-32); gate_up and down must share the same MX
    format. Same structure as the block-dynamic path but with a tunable tile and MXFP8 intermediate."""
    gate_up_is_fp4 = is_mxfp4(gate_up_proj, gate_up_proj_scale)
    down_is_fp4 = is_mxfp4(down_proj, down_proj_scale)
    if gate_up_is_fp4 != down_is_fp4:
        raise ValueError(
            "gate_up_proj and down_proj must use the same MX format (both MXFP4 or both MXFP8)."
        )

    num_top_k = top_k_index.size(-1)
    NUM_EXPERTS = gate_up_proj.size(0)

    expert_start, gather_idx, scatter_idx = compute_grouped_scheduling(
        top_k_index, NUM_EXPERTS, num_top_k
    )
    hidden_q, hidden_scale = mxfp_act_quant(hidden_states)

    # Phase 1: gate_up + SiLU + MXFP8 requant -> expert-ordered intermediate. Gather hidden by
    # routed row (gather_idx); leave the output expert-ordered (scatter_idx=None).
    inter, inter_scale = matmul_grouped(
        hidden_q,
        gate_up_proj,
        As=hidden_scale,
        Bs=gate_up_proj_scale,
        expert_start=expert_start,
        epilogue=Epilogue(
            gate=True,
            act_fn=act_fn,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            simulate_unfused=simulate_unfused,
        ),
        quantization=Quantization(output_recipe="mxfp8"),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    # Phase 2: grouped down over the expert-ordered intermediate (gather_idx=None), scattering to
    # routed rows (scatter_idx).
    down_out = matmul_grouped(
        inter,
        down_proj,
        As=inter_scale,
        Bs=down_proj_scale,
        expert_start=expert_start,
        output_dtype=hidden_states.dtype,
        scatter_idx=scatter_idx,
    )
    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the unfused
    # path's torch reduce (which materializes bf16 contribs); production accumulates in fp32.
    return weighted_reduce(
        down_out, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused
    )


def moe_fused_grouped(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Fused grouped-MoE dispatcher — routes to the recipe matching the weight dtype /
    scale layout, mirroring ``moe_fused_batched``. Implemented: block-dynamic FP8 and MXFP8 /
    MXFP4 (UE8M0 group-32). ``simulate_unfused`` (testing) rounds each step through the
    activation dtype so the output matches the unfused reference to reduce order."""
    if _validate_moe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv
    ):
        return mxfp_dynamic_moe_grouped(
            hidden_states,
            top_k_index,
            top_k_weights,
            gate_up_proj,
            down_proj,
            gate_up_proj_scale_inv,
            down_proj_scale_inv,
            act_fn,
            swiglu_alpha,
            swiglu_limit,
            simulate_unfused,
        )

    return w8a8_block_dynamic_fp8_moe_grouped(
        hidden_states,
        top_k_index,
        top_k_weights,
        gate_up_proj,
        down_proj,
        gate_up_proj_scale_inv,
        down_proj_scale_inv,
        act_fn,
        swiglu_alpha,
        swiglu_limit,
        simulate_unfused,
    )


# ── Fused batched (decode) ────────────────────────────────────────────────────


def w8a8_block_dynamic_fp8_moe_batched(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H) FP8
    down_proj: torch.Tensor,  # (E, H, I) FP8
    gate_up_proj_scale: torch.Tensor,
    down_proj_scale: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Block-dynamic FP8 fused batched MoE: gate_up+SiLU → FP8 intermediate → batched down →
    routing-weighted top-k reduce. Returns ``(num_tokens, hidden_dim)``."""
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_ids = top_k_index.reshape(-1)
    gather_idx = _gather_idx(top_k_index)

    # Phase 1: gate_up + SiLU + FP8 requant -> per-row fp8 intermediate (op quantizes the raw
    # activations offline). gather_idx reads each routed row from the unexpanded hidden in-kernel
    # (no copy); the intermediate scale is the down projection's pre-quantized As.
    inter, inter_scale = matmul_batched(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale,
        expert_ids=expert_ids,
        epilogue=Epilogue(
            gate=True,
            act_fn=act_fn,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            simulate_unfused=simulate_unfused,
        ),
        quantization=Quantization(output_recipe="fp8"),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    # Phase 2: batched down over the pre-quantized intermediate (already routed-order, no gather).
    down_out = matmul_batched(
        inter,
        down_proj,
        As=inter_scale,
        Bs=down_proj_scale,
        expert_ids=expert_ids,
        output_dtype=hidden_states.dtype,
    )
    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the unfused
    # path's torch reduce (which materializes bf16 contribs); production accumulates in fp32.
    return weighted_reduce(
        down_out, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused
    )


def mxfp_dynamic_moe_batched(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale: torch.Tensor,
    down_proj_scale: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """MXFP4/MXFP8 fused batched MoE (UE8M0 group-32); gate_up and down must share the same MX
    format. Same structure as the block-dynamic path but with a tunable tile, an inline UE8M0
    activation quant, and an MXFP8 group-32 intermediate."""
    gate_up_is_fp4 = is_mxfp4(gate_up_proj, gate_up_proj_scale)
    down_is_fp4 = is_mxfp4(down_proj, down_proj_scale)
    if gate_up_is_fp4 != down_is_fp4:
        raise ValueError(
            "gate_up_proj and down_proj must use the same MX format (both MXFP4 or both MXFP8)."
        )

    NUM_EXPERTS = gate_up_proj.size(0)
    expert_ids = top_k_index.reshape(-1)
    gather_idx = _gather_idx(top_k_index)

    # Phase 1: gate_up + SiLU + MXFP8 requant -> per-row fp8 intermediate + UE8M0 group-32 scale
    # (the op quantizes the raw activations inline). gather_idx reads each routed row from the
    # unexpanded hidden in-kernel (no copy).
    inter, inter_scale = matmul_batched(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale,
        expert_ids=expert_ids,
        epilogue=Epilogue(
            gate=True,
            act_fn=act_fn,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            simulate_unfused=simulate_unfused,
        ),
        quantization=Quantization(output_recipe="mxfp8"),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    # Phase 2: batched down over the pre-quantized MXFP8 intermediate (routed-order, no gather).
    down_out = matmul_batched(
        inter,
        down_proj,
        As=inter_scale,
        Bs=down_proj_scale,
        expert_ids=expert_ids,
        output_dtype=hidden_states.dtype,
    )
    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the unfused
    # path's torch reduce (which materializes bf16 contribs); production accumulates in fp32.
    return weighted_reduce(
        down_out, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused
    )


def moe_fused_batched(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
) -> torch.Tensor:
    """Fused batched-MoE dispatcher — routes to the recipe matching the weight dtype /
    scale layout, mirroring ``moe_fused_grouped``. Implemented: block-dynamic FP8 and MXFP8 /
    MXFP4 (UE8M0 group-32). ``simulate_unfused`` (testing) rounds each step through the
    activation dtype so the output matches the unfused reference to reduce order."""
    if _validate_moe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv
    ):
        return mxfp_dynamic_moe_batched(
            hidden_states,
            top_k_index,
            top_k_weights,
            gate_up_proj,
            down_proj,
            gate_up_proj_scale_inv,
            down_proj_scale_inv,
            act_fn,
            swiglu_alpha,
            swiglu_limit,
            simulate_unfused,
        )

    return w8a8_block_dynamic_fp8_moe_batched(
        hidden_states,
        top_k_index,
        top_k_weights,
        gate_up_proj,
        down_proj,
        gate_up_proj_scale_inv,
        down_proj_scale_inv,
        act_fn,
        swiglu_alpha,
        swiglu_limit,
        simulate_unfused,
    )


# ── Unfused (plain GEMMs + host GLU) ──────────────────────────────────────────


def moe_unfused_grouped(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Unfused grouped MoE: gate_up (plain grouped GEMM, gather hidden) → host ``apply_glu`` →
    down (plain grouped GEMM, scatter to routed rows) → routing-weighted reduce. Same math as
    ``moe_fused_grouped`` but the SwiGLU + intermediate requant happen between two plain GEMMs
    rather than inside the gate_up epilogue; each GEMM self-quantizes its raw input (no ``Quantization`` given).
    Both recipes (block-dynamic FP8, MXFP4/MXFP8) route through the shared ``matmul_grouped``."""
    _validate_moe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv
    )

    num_top_k = top_k_index.size(-1)
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_start, gather_idx, scatter_idx = compute_grouped_scheduling(
        top_k_index, NUM_EXPERTS, num_top_k
    )

    # gate_up as a plain GEMM (no gate epilogue) over gathered hidden -> expert-ordered (S, 2I).
    gate_up_out = matmul_grouped(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale_inv,
        expert_start=expert_start,
        gather_idx=gather_idx,
    )
    gate, up = gate_up_out.chunk(2, dim=-1)
    inter = apply_glu(gate, up, act_fn, swiglu_alpha, swiglu_limit)
    # down over the expert-ordered intermediate (self-quantized), scattering to routed rows.
    down_out = matmul_grouped(
        inter,
        down_proj,
        Bs=down_proj_scale_inv,
        expert_start=expert_start,
        scatter_idx=scatter_idx,
    )
    return _torch_weighted_reduce(down_out, top_k_index, top_k_weights, NUM_EXPERTS)


def moe_unfused_batched(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    """Unfused batched MoE: gate_up (plain batched GEMM, gather hidden) → host ``apply_glu`` →
    down (plain batched GEMM) → routing-weighted reduce. Same math as ``moe_fused_batched`` but
    the SwiGLU + intermediate requant happen between two plain GEMMs; each GEMM self-quantizes its
    raw input (``As=None``). Both recipes route through the shared ``matmul_batched``."""
    _validate_moe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv
    )

    NUM_EXPERTS = gate_up_proj.size(0)
    expert_ids = top_k_index.reshape(-1)
    gather_idx = _gather_idx(top_k_index)

    # gate_up as a plain GEMM (no gate epilogue) over gathered hidden -> (S, 2I).
    gate_up_out = matmul_batched(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale_inv,
        expert_ids=expert_ids,
        gather_idx=gather_idx,
    )
    gate, up = gate_up_out.chunk(2, dim=-1)
    inter = apply_glu(gate, up, act_fn, swiglu_alpha, swiglu_limit)
    # down over the intermediate (self-quantized), routed-order output.
    down_out = matmul_batched(
        inter,
        down_proj,
        Bs=down_proj_scale_inv,
        expert_ids=expert_ids,
    )
    return _torch_weighted_reduce(down_out, top_k_index, top_k_weights, NUM_EXPERTS)
