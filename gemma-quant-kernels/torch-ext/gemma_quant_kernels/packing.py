# Copyright 2026 Google LLC and The HuggingFace Inc. team. All rights reserved.
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

import torch


def unpack_int4(packed: torch.Tensor, original_width: int) -> torch.Tensor:
    """Unpack uint8 [..., ceil(cols/2)] -> signed int8 [..., cols] in [-8, 7] (offset-binary +8)."""
    low = (packed & 0x0F).to(torch.int8) - 8
    high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    unpacked = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :original_width]


def unpack_int2(packed: torch.Tensor, original_width: int) -> torch.Tensor:
    """Unpack uint8 [..., ceil(cols/4)] -> signed int8 [..., cols] in [-2, 1] (offset-binary +2)."""
    v0 = (packed & 0x03).to(torch.int8) - 2
    v1 = ((packed >> 2) & 0x03).to(torch.int8) - 2
    v2 = ((packed >> 4) & 0x03).to(torch.int8) - 2
    v3 = ((packed >> 6) & 0x03).to(torch.int8) - 2
    unpacked = torch.stack([v0, v1, v2, v3], dim=-1).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :original_width]


def pack_int4(weights: torch.Tensor) -> torch.Tensor:
    """Pack signed int8 in [-8, 7] of shape [..., cols] -> uint8 [..., ceil(cols/2)]."""
    orig_shape = weights.shape
    cols = orig_shape[-1]
    w_2d = weights.reshape(-1, cols)
    if cols % 2 != 0:
        w_2d = torch.nn.functional.pad(w_2d, (0, 1), value=0)
    u = (w_2d.to(torch.int16) + 8).to(torch.uint8)
    packed = (u[:, 0::2] & 0x0F) | ((u[:, 1::2] & 0x0F) << 4)
    return packed.reshape(*orig_shape[:-1], -1)


def pack_int2(weights: torch.Tensor) -> torch.Tensor:
    """Pack signed int8 in [-2, 1] of shape [..., cols] -> uint8 [..., ceil(cols/4)]."""
    orig_shape = weights.shape
    cols = orig_shape[-1]
    w_2d = weights.reshape(-1, cols)
    rem = cols % 4
    if rem != 0:
        w_2d = torch.nn.functional.pad(w_2d, (0, 4 - rem), value=0)
    u = (w_2d.to(torch.int16) + 2).to(torch.uint8)
    packed = (
        (u[:, 0::4] & 0x03)
        | ((u[:, 1::4] & 0x03) << 2)
        | ((u[:, 2::4] & 0x03) << 4)
        | ((u[:, 3::4] & 0x03) << 6)
    )
    return packed.reshape(*orig_shape[:-1], -1)
