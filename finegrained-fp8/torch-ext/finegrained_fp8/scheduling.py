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


import contextvars
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal


import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from ._ops import add_op_namespace_prefix
from .bayesian_autotuner import bayesian_autotune

from .compat import *  # noqa: F401,F403
from .recipes import *  # noqa: F401,F403
from .swizzle import *  # noqa: F401,F403
from .tile_layout import *  # noqa: F401,F403
from .quant import *  # noqa: F401,F403
from .scales import *  # noqa: F401,F403
from .mma import *  # noqa: F401,F403



# Flat-slot tile per program for the O(S) routing kernels (count + scatter). These are small
# latency-bound atomic kernels that want many programs: a sweep over {256..4096} x prefill shapes
# put 256 best (or within ~1%) for both, with 1024 up to ~1.5x slower. The grid derives from it
# so the two can't drift. Power of 2.
_ROUTING_BLOCK_SIZE = 256



@triton.jit
def _exclusive_offsets_kernel(
    ExpertFreq, ExpertStart, Counters, NUM_EXPERTS: tl.constexpr
):
    """Exclusive cumsum of per-expert token counts → ``expert_start`` (leading 0, trailing
    S), and zero the scatter counters — one launch."""
    offs = tl.arange(0, NUM_EXPERTS)
    incl = tl.cumsum(tl.load(ExpertFreq + offs), 0)
    tl.store(ExpertStart, 0)
    tl.store(ExpertStart + 1 + offs, incl)
    tl.store(Counters + offs, tl.zeros([NUM_EXPERTS], tl.int32))



@triton.jit
def _scatter_kernel(
    ExpertIds,
    Perm,
    PermToken,
    ExpertStart,
    Counters,
    S,
    NUM_TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Counting-sort scatter: each flat slot atomically claims the next slot of its expert
    (``expert_start[e] + counter[e]++``). O(S), replaces an O(S·logS) argsort. Within-expert
    order is arbitrary (atomic race) — fine, the per-token reduce is order-invariant. Slots whose
    expert is non-local (EP sentinel id ``>= NUM_EXPERTS``) are skipped — matches ``_count_kernel``,
    and avoids the atomic/store landing at an out-of-range (invalid) global address."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_id = tl.load(ExpertIds + offs, mask=offs < S, other=NUM_EXPERTS)
    valid = expert_id < NUM_EXPERTS
    dest = tl.load(ExpertStart + expert_id, mask=valid, other=0) + tl.atomic_add(
        Counters + expert_id, 1, mask=valid, sem="relaxed"
    )
    tl.store(Perm + dest, offs, mask=valid)
    tl.store(PermToken + dest, offs // NUM_TOP_K, mask=valid)



@triton.jit
def _count_kernel(
    ExpertIds, ExpertFreq, S, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    """Per-expert token count via atomics — replaces ``torch.histc`` (no float cast), fixed
    ``(NUM_EXPERTS,)`` output stays CUDA-graph friendly. ``ExpertFreq`` is pre-zeroed."""
    offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < S
    expert_id = tl.load(ExpertIds + offs, mask=mask, other=NUM_EXPERTS)
    tl.atomic_add(
        ExpertFreq + expert_id, 1, mask=mask & (expert_id < NUM_EXPERTS), sem="relaxed"
    )



def compute_grouped_scheduling(
    expert_ids: torch.Tensor, num_experts: int, num_top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """On-device routing: expert-sorted index (no copy of the activations) via two Triton
    launches — exclusive offsets + an atomic counting-sort scatter (replaces host ``argsort``).
    Run it once per layer and pass the results to every grouped GEMM of that layer. Returns
    ``(expert_start, gather_idx, scatter_idx)``:

    - ``expert_start`` — ``(E+1,)`` cumulative sorted-row starts padded with S; the tiling
      schedule the kernels build their register-resident tile layout from.
    - ``gather_idx`` — each sorted position's source row of hidden (``perm // num_top_k``,
      many-to-one for top_k > 1: the gather that reads hidden without replication). Pass as the
      GEMM's input map (``None`` = ``A`` already expert-sorted, e.g. the down projection).
    - ``scatter_idx`` — each sorted position's token-major routed destination row ``(t*K + j)``,
      the ``perm = torch.sort(expert_ids)`` indices (kernels un-permute by SCATTERING at store
      time, never materializing ``inv_perm``). Pass as the output map (``None`` = leave the output
      expert-sorted, e.g. the gate_up projection's intermediate).

    E must be a power of 2 (the scheduling kernels hold the per-expert vectors in one
    ``tl.arange`` block)."""
    # the scheduling kernels hold the (E,) frequency/offset vectors in one tl.arange
    # block, which requires a power of 2 — fail here with a clear message instead of a
    # Triton compile error from an internal kernel
    assert num_experts & (num_experts - 1) == 0, (
        f"num_experts ({num_experts}) must be a power of 2"
    )
    gather_idx, scatter_idx, expert_start = _compute_grouped_scheduling(
        expert_ids, num_experts, num_top_k
    )
    return expert_start, gather_idx, scatter_idx



@compile_time_only_triton_op(
    add_op_namespace_prefix("compute_grouped_scheduling"), mutates_args=(), opaque=True
)
def _compute_grouped_scheduling(
    expert_ids: torch.Tensor, num_experts: int, num_top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = expert_ids.device
    expert_ids = expert_ids.int().contiguous()  # routing kernels index with unit stride
    num_routed_tokens = expert_ids.numel()  # S = num_tokens * num_top_k
    expert_freq = torch.zeros(num_experts, dtype=torch.int32, device=device)
    expert_start = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
    counters = torch.empty(num_experts, dtype=torch.int32, device=device)
    perm = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
    perm_token = torch.empty(num_routed_tokens, dtype=torch.int32, device=device)
    with device_context(device):
        compile_time_only_triton_wrap(_count_kernel)[
            (triton.cdiv(num_routed_tokens, _ROUTING_BLOCK_SIZE),)
        ](
            expert_ids,
            expert_freq,
            num_routed_tokens,
            NUM_EXPERTS=num_experts,
            BLOCK_SIZE=_ROUTING_BLOCK_SIZE,
        )
        compile_time_only_triton_wrap(_exclusive_offsets_kernel)[(1,)](
            expert_freq,
            expert_start,
            counters,
            NUM_EXPERTS=num_experts,
        )
        compile_time_only_triton_wrap(_scatter_kernel)[
            (triton.cdiv(num_routed_tokens, _ROUTING_BLOCK_SIZE),)
        ](
            expert_ids,
            perm,
            perm_token,
            expert_start,
            counters,
            num_routed_tokens,
            NUM_TOP_K=num_top_k,
            NUM_EXPERTS=num_experts,
            BLOCK_SIZE=_ROUTING_BLOCK_SIZE,
        )
    return perm_token, perm, expert_start



@triton.jit
def resolve_grouped_tile(
    tile_id,
    num_n_tiles,
    exp_start,
    freqs,
    tile_start_excl,
    e_offs,
    GatherIdx,
    ScatterIdx,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """One persistent grouped tile: split the flat ``tile_id`` into (M-tile, N-tile), map
    the M-tile to its expert + rows via ``resolve_tile_inline`` (on the register-resident
    layout ``build_tile_layout`` builds once per program, passed in), and apply the virtual
    sort — rows load from ``in_row`` and store to ``out_row``, mapped by ``GatherIdx`` /
    ``ScatterIdx`` when present (``None`` = expert-sorted, the position itself; the ``None``
    check folds at trace time, so no separate has-gather/has-scatter flag is needed).

    Returns ``(pid_n, expert_id, expert_id64, in_row, out_row, row_mask, offs_bn)`` — both
    expert-id widths: ``expert_id`` (int32, e.g. TMA descriptor row indices, bounded by the
    expert count) and ``expert_id64`` (int64, for byte-offset pointer arithmetic — ``expert
    * stride`` overflows int32 at full expert counts). Shared by the base grouped GEMMs and
    the fused kernels; callers ``_``-ignore whichever width they don't use."""
    pid_m = tile_id // num_n_tiles
    pid_n = tile_id % num_n_tiles
    expert_id, offs_global_m, row_mask = resolve_tile_inline(
        pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M
    )
    if GatherIdx is not None:
        in_row = tl.load(GatherIdx + offs_global_m, mask=row_mask, other=0)
    else:
        in_row = offs_global_m
    if ScatterIdx is not None:
        out_row = tl.load(ScatterIdx + offs_global_m, mask=row_mask, other=0)
    else:
        out_row = offs_global_m
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    return pid_n, expert_id, expert_id.to(tl.int64), in_row, out_row, row_mask, offs_bn
