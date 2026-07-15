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
are left uninit by the GEMM and skipped in ``weighted_reduce``. ``moe_fused_*`` / ``moe_unfused_*`` are recipe-neutral:
the base ops dispatch on the weight dtypes / scale layout (block-dynamic FP8, MXFP4/MXFP8,
NVFP4), and the fused forwards take an optional ``recipe`` naming the block's activation
quantization."""

import torch

from .grouped import matmul_grouped
from .batched import matmul_batched
from .utils import (
    Epilogue,
    Quantization,
    apply_glu,
    compute_grouped_scheduling,
    weighted_reduce,
    is_mx,
    is_mxfp8,
    is_mxfp4,
    is_nvfp4,
)


def _validate_moe(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale):
    """gate_up and down must share the recipe (both MX or both block-dynamic FP8 — the
    intermediate handed between them carries one quant format). Returns whether the recipe
    is MX (the fused dispatchers branch on it); the fp8 quantization block is derived from
    the scale shapes (``weight_block_size``), never passed."""
    gate_up_is_mx = is_mx(gate_up_proj, gate_up_proj_scale)
    if gate_up_is_mx != is_mx(down_proj, down_proj_scale):
        raise ValueError(
            "gate_up_proj and down_proj must use the same recipe (both MX or both block-dynamic FP8)."
        )
    return gate_up_is_mx


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


# ── Fused (gate_up epilogue owns SwiGLU + intermediate requant) ──────────────


def _block_recipe(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale, recipe):
    """The MoE block's activation recipe: validates the weight pairing; an explicit
    ``recipe`` is respected as-is, ``None`` follows the weight recipe (fp8 / mxfp8 /
    mxfp4 / nvfp4 — mxfp4 weights default to mxfp4 activations, the all-fp4 W4A4
    chain; unquantized BF16/FP16 weights carry no scales and stay ``None``, the
    full-precision path)."""
    _validate_moe(gate_up_proj, gate_up_proj_scale, down_proj, down_proj_scale)
    if is_mxfp4(gate_up_proj, gate_up_proj_scale) != is_mxfp4(
        down_proj, down_proj_scale
    ):
        raise ValueError(
            "gate_up_proj and down_proj must use the same MX format (both MXFP4 or both MXFP8)."
        )
    if recipe is not None:
        return recipe
    if gate_up_proj_scale is None:
        return None
    if is_nvfp4(gate_up_proj, gate_up_proj_scale):
        return "nvfp4"
    if is_mxfp4(gate_up_proj, gate_up_proj_scale):
        return "mxfp4"
    return "mxfp8" if is_mxfp8(gate_up_proj, gate_up_proj_scale) else "fp8"


def moe_fused_grouped(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H)
    down_proj: torch.Tensor,  # (E, H, I)
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    recipe: str | None = None,
) -> torch.Tensor:
    """Fused grouped MoE (prefill): gather gate_up + SiLU + requant epilogue → quantized
    expert-ordered intermediate → grouped down → routing-weighted top-k reduce. Returns
    ``(num_tokens, hidden_dim)``. The base ops dispatch on the weight dtypes / scale
    layout (block-dynamic FP8, MXFP8/MXFP4, NVFP4); ``recipe`` names the activation
    quantization for the whole block — activations and the fused intermediate requant
    carry it ("mxfp4"/"nvfp4" run all-fp4 W4A4 chains), ``None`` picks the weight
    family's recipe, and the ops validate the pairing. ``simulate_unfused`` (testing) rounds each step through
    the activation dtype so the output matches the unfused reference to reduce order."""
    recipe = _block_recipe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, recipe
    )
    num_top_k = top_k_index.size(-1)
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_start, gather_idx, scatter_idx = compute_grouped_scheduling(
        top_k_index, NUM_EXPERTS, num_top_k
    )

    # Phase 1: gate_up + SiLU + requant in the block recipe -> expert-ordered quantized
    # intermediate (the op quantizes the raw hidden itself). Gather hidden by routed row
    # (gather_idx); leave the output expert-ordered (scatter_idx=None — the down
    # projection reads it in place, no scatter between the two GEMMs).
    inter, inter_scale = matmul_grouped(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale_inv,
        expert_start=expert_start,
        epilogue=Epilogue(
            gate=True,
            act_fn=act_fn,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            simulate_unfused=simulate_unfused,
        ),
        quantization=Quantization(input_recipe=recipe, output_recipe=recipe),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    # Phase 2: grouped down over the expert-ordered pre-quantized intermediate (its dtypes
    # carry the recipe; gather_idx=None), scattering to routed rows (scatter_idx).
    down_out = matmul_grouped(
        inter,
        down_proj,
        As=inter_scale,
        Bs=down_proj_scale_inv,
        expert_start=expert_start,
        output_dtype=hidden_states.dtype,
        scatter_idx=scatter_idx,
    )
    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the
    # unfused path's torch reduce (which materializes bf16 contribs); production
    # accumulates in fp32.
    return weighted_reduce(
        down_out, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused
    )


def moe_fused_batched(
    hidden_states: torch.Tensor,  # (T, H)
    top_k_index: torch.Tensor,  # (T, K) int
    top_k_weights: torch.Tensor,  # (T, K)
    gate_up_proj: torch.Tensor,  # (E, 2I, H)
    down_proj: torch.Tensor,  # (E, H, I)
    gate_up_proj_scale_inv: torch.Tensor,
    down_proj_scale_inv: torch.Tensor,
    act_fn: str = "silu",
    swiglu_alpha: float | None = None,
    swiglu_limit: float | None = None,
    simulate_unfused: bool = False,
    recipe: str | None = None,
) -> torch.Tensor:
    """Fused batched MoE (decode): gate_up + SiLU + requant epilogue → per-row quantized
    intermediate → batched down → routing-weighted top-k reduce. Returns
    ``(num_tokens, hidden_dim)``. The base ops dispatch on the weight dtypes / scale
    layout (block-dynamic FP8, MXFP8/MXFP4, NVFP4 — decode runs the software/swap arms
    below the native mxf4nvf4 M=128 staging); ``recipe`` names the activation
    quantization for the whole block — activations and the fused intermediate requant
    carry it ("mxfp4" runs the all-fp4 W4A4 chain), ``None`` picks the weight family's
    recipe, and the ops validate the pairing. ``simulate_unfused`` (testing) rounds each
    step through the activation dtype so the output matches the unfused reference to
    reduce order."""
    recipe = _block_recipe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, recipe
    )
    NUM_EXPERTS = gate_up_proj.size(0)
    expert_ids = top_k_index.reshape(-1)
    gather_idx = _gather_idx(top_k_index)

    # Phase 1: gate_up + SiLU + requant in the block recipe -> per-row quantized
    # intermediate (the op quantizes the raw activations). gather_idx reads each routed
    # row from the unexpanded hidden in-kernel (no copy).
    inter, inter_scale = matmul_batched(
        hidden_states,
        gate_up_proj,
        Bs=gate_up_proj_scale_inv,
        expert_ids=expert_ids,
        epilogue=Epilogue(
            gate=True,
            act_fn=act_fn,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            simulate_unfused=simulate_unfused,
        ),
        quantization=Quantization(input_recipe=recipe, output_recipe=recipe),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    # Phase 2: batched down over the pre-quantized intermediate (its dtypes carry the
    # recipe; already routed-order, no gather).
    down_out = matmul_batched(
        inter,
        down_proj,
        As=inter_scale,
        Bs=down_proj_scale_inv,
        expert_ids=expert_ids,
        output_dtype=hidden_states.dtype,
    )
    # Phase 3: routing-weighted top-k reduce -> (num_tokens, hidden_dim). simulate_unfused
    # rounds each weighted contrib to the activation dtype before summing, matching the
    # unfused path's torch reduce (which materializes bf16 contribs); production
    # accumulates in fp32.
    return weighted_reduce(
        down_out, top_k_index, top_k_weights, NUM_EXPERTS, simulate_unfused
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
    recipe: str | None = None,
) -> torch.Tensor:
    """Unfused grouped MoE: gate_up (plain grouped GEMM, gather hidden) → host ``apply_glu`` →
    down (plain grouped GEMM, scatter to routed rows) → routing-weighted reduce. Same math as
    ``moe_fused_grouped`` but the SwiGLU + intermediate quant happen between two plain GEMMs
    rather than inside the gate_up epilogue; each GEMM quantizes its raw input in ``recipe``
    (``None`` follows the weight recipe, mirroring the fused forward — mxfp4 weights run the
    all-fp4 W4A4 chain). All recipes route through the shared ``matmul_grouped``."""
    recipe = _block_recipe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, recipe
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
        quantization=Quantization(input_recipe=recipe),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    gate, up = gate_up_out.chunk(2, dim=-1)
    inter = apply_glu(gate, up, act_fn, swiglu_alpha, swiglu_limit)
    # down over the expert-ordered intermediate (quantized in the same recipe), scattering
    # to routed rows.
    down_out = matmul_grouped(
        inter,
        down_proj,
        Bs=down_proj_scale_inv,
        expert_start=expert_start,
        quantization=Quantization(input_recipe=recipe),
        output_dtype=hidden_states.dtype,
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
    recipe: str | None = None,
) -> torch.Tensor:
    """Unfused batched MoE: gate_up (plain batched GEMM, gather hidden) → host ``apply_glu`` →
    down (plain batched GEMM) → routing-weighted reduce. Same math as ``moe_fused_batched`` but
    the SwiGLU + intermediate quant happen between two plain GEMMs; each GEMM quantizes its raw
    input in ``recipe`` (``None`` follows the weight recipe, mirroring the fused forward). All
    recipes route through the shared ``matmul_batched``."""
    recipe = _block_recipe(
        gate_up_proj, gate_up_proj_scale_inv, down_proj, down_proj_scale_inv, recipe
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
        quantization=Quantization(input_recipe=recipe),
        output_dtype=hidden_states.dtype,
        gather_idx=gather_idx,
    )
    gate, up = gate_up_out.chunk(2, dim=-1)
    inter = apply_glu(gate, up, act_fn, swiglu_alpha, swiglu_limit)
    # down over the intermediate (quantized in the same recipe), routed-order output.
    down_out = matmul_batched(
        inter,
        down_proj,
        Bs=down_proj_scale_inv,
        expert_ids=expert_ids,
        quantization=Quantization(input_recipe=recipe),
        output_dtype=hidden_states.dtype,
    )
    return _torch_weighted_reduce(down_out, top_k_index, top_k_weights, NUM_EXPERTS)
