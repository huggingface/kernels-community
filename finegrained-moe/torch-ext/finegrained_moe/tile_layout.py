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



@triton.jit
def build_tile_layout(
    ExpertStart, NUM_EXPERTS: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
):
    """Load ``expert_start`` once and derive the per-BM tile layout vectors (kept in
    registers for the whole persistent loop): per-expert first sorted row, token count,
    exclusive tile-start cumsum, and the total M-tile count. ``ExpertStart`` is
    ``(NUM_EXPERTS + 1,)`` with a trailing ``S`` sentinel (``expert_start[E] == S``)."""
    e_offs = tl.arange(0, NUM_EXPERTS)
    exp_start = tl.load(ExpertStart + e_offs)
    exp_end = tl.load(ExpertStart + e_offs + 1)
    freqs = exp_end - exp_start
    tiles_per_e = (freqs + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    tile_start_excl = (
        tl.cumsum(tiles_per_e, 0) - tiles_per_e
    )  # first tile index of expert e
    total_m_tiles = tl.sum(tiles_per_e, 0)
    return exp_start, freqs, tile_start_excl, total_m_tiles, e_offs



@triton.jit
def resolve_tile_inline(
    pid_m, exp_start, freqs, tile_start_excl, e_offs, BLOCK_SIZE_M: tl.constexpr
):
    """Map an M-tile id to its owning expert + the tile's sorted row range, from the
    register-resident layout (no global loads). Returns ``(expert_id, sorted_indices,
    row_mask)``."""
    # Bucketize via the exclusive tile cumsum: #experts whose tile-start <= pid_m, minus 1.
    expert_id = tl.sum((tile_start_excl <= pid_m).to(tl.int32), 0) - 1
    sel = (
        e_offs == expert_id
    )  # scalar-index the E-vectors via mask-sum (no dynamic index)
    e_start = tl.sum(tl.where(sel, exp_start, 0), 0)
    e_tile_start = tl.sum(tl.where(sel, tile_start_excl, 0), 0)
    freq = tl.sum(tl.where(sel, freqs, 0), 0)
    within = pid_m - e_tile_start
    m_start = e_start + within * BLOCK_SIZE_M
    offs = tl.arange(0, BLOCK_SIZE_M)
    row_mask = offs < freq - within * BLOCK_SIZE_M
    sorted_indices = tl.max_contiguous(m_start + offs, BLOCK_SIZE_M)
    return expert_id, sorted_indices, row_mask
