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

"""Read byte-window4 CSF scale tiles without materializing a scale plane."""

import triton
import triton.language as tl


@triton.jit
def load_nvfp4_csf_scale(
    Fixed,
    Exceptions,
    RowOffsets,
    expert,
    rows,
    cols,
    ROWS,
    COLS,
    STEPS: tl.constexpr,
    mask=True,
):
    slab_bytes = 16 * (1 + COLS // 2)
    slab = expert.to(tl.int64) * (ROWS // 16) + rows // 16
    base = tl.load(Fixed + slab * slab_bytes + rows % 16, mask=mask, other=0).to(
        tl.uint32
    )
    packed = tl.load(
        Fixed + slab * slab_bytes + 16 + (rows % 16) * (COLS // 2) + cols // 2,
        mask=mask,
        other=0,
    ).to(tl.uint32)
    value = base + ((packed >> (4 * (cols % 2))) & 15)
    if STEPS > 0:
        query = rows * COLS + cols
        row_index = expert.to(tl.int64) * ROWS + rows
        start = tl.load(RowOffsets + row_index, mask=mask, other=0).to(tl.int64)
        end = tl.load(RowOffsets + row_index + 1, mask=mask, other=0).to(tl.int64)
        lo = start + tl.zeros_like(query)
        hi = end + tl.zeros_like(query)
        # Search only one row's exceptions; the checkpoint's whole-plane u24 positions
        # remain unchanged. CPU preparation supplies the exact worst-case row depth.
        for _ in range(STEPS):
            active = lo < hi
            mid = (lo + hi) // 2
            word = tl.load(Exceptions + mid, mask=active & mask, other=0)
            before = (word & 0xFFFFFF) < query
            lo = tl.where(active & before, mid + 1, lo)
            hi = tl.where(active & ~before, mid, hi)
        word = tl.load(Exceptions + lo, mask=(lo < end) & mask, other=0)
        found = (lo < end) & ((word & 0xFFFFFF) == query)
        value = tl.where(found, word >> 24, value)
    return value.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
