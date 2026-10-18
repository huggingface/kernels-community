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

"""Timing shared by autotuning and benchmarks."""

from dataclasses import replace

import torch
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.testing import do_bench as _do_bench
from triton.testing import do_bench_cudagraph


def do_bench(fn, quantiles=None, *, cudagraph=False, **kwargs):
    """Time execution in milliseconds, excluding cache eviction.

    Event timing clears L2 before each measured interval. Graph timing has no
    eviction: cold-weight callers must rotate a working set larger than L2 inside
    ``fn``; otherwise it measures warm replay.
    """
    if not cudagraph:
        return _do_bench(fn, quantiles=quantiles, **kwargs)
    # Triton's graph timer warms up on its own stream; finish input producers first.
    torch.cuda.current_stream().synchronize()
    return do_bench_cudagraph(fn, quantiles=quantiles, **kwargs)


def weight_input(nargs):
    """Main GEMM weight and its logical read bytes, excluding inactive expert planes."""
    weight = nargs["B"]
    active_bytes = weight.numel() * weight.element_size()
    expert_ids = nargs.get("ExpertIds")
    expert_start = nargs.get("ExpertStart")
    if expert_ids is not None:
        experts = weight.shape[0]
        active = expert_ids[(expert_ids >= 0) & (expert_ids < experts)].unique().numel()
        active_bytes = active_bytes * active // experts
    elif expert_start is not None:
        experts = weight.shape[0]
        if expert_start.numel() < experts + 1:
            return weight, 0
        active = (expert_start[1:experts + 1] > expert_start[:experts]).sum().item()
        active_bytes = active_bytes * active // experts
    return weight, active_bytes


def rotation_count(active_bytes, device):
    """Cycle more than twice L2, or decline an impractically large captured graph."""
    if active_bytes <= 0:
        return None
    count = 2 * torch.cuda.get_device_properties(device).L2_cache_size // active_bytes + 1
    return count if count <= 1024 else None


def weight_banks(nargs, weight, active_bytes):
    """Rotate readonly weight values; return None when cold event timing is safer.

    Activations, scales and outputs stay shared. Raw tensor aliases of the weight
    are unsupported; host descriptors may alias views wholly inside its storage span.
    """
    if not isinstance(weight, torch.Tensor) or not weight.is_cuda:
        return None
    if active_bytes > weight.numel() * weight.element_size():
        return None
    # Broadcast/overlapping or holed layouts cannot count logical bytes as unique reads.
    if torch._debug_has_internal_overlap(weight) != 0:
        return None
    names = [name for name, value in nargs.items() if value is weight]
    if len(names) != 1:
        return None
    name = names[0]
    storage = weight.untyped_storage()._cdata
    descriptors = {}
    span_bytes = weight.numel() * weight.element_size()
    for key, value in nargs.items():
        if key != name and isinstance(value, torch.Tensor):
            if value.untyped_storage()._cdata == storage:
                return None
        if isinstance(value, TensorDescriptor):
            if not isinstance(value.base, torch.Tensor):
                return None
            if value.base.untyped_storage()._cdata != storage:
                continue
            base = value.base
            offset = base.data_ptr() - weight.data_ptr()
            extent = (1 + sum((size - 1) * stride for size, stride in zip(base.shape, base.stride())))
            if (offset < 0 or offset % 16 or offset % base.element_size()
                    or offset + extent * base.element_size() > span_bytes):
                return None
            descriptors[key] = (value, offset)
    count = rotation_count(active_bytes, weight.device)
    if count is None:
        return None
    free, _ = torch.cuda.mem_get_info(weight.device)
    free += (torch.cuda.memory_reserved(weight.device)
             - torch.cuda.memory_allocated(weight.device))
    if (count - 1) * span_bytes > free:
        return None
    banks = [nargs]
    try:
        with torch.no_grad():
            for _ in range(count - 1):
                clone = torch.empty_strided(weight.shape, weight.stride(),
                                            dtype=weight.dtype, device=weight.device)
                clone.copy_(weight)
                bank = dict(nargs, **{name: clone})
                rebound = {}
                for key, (descriptor, offset) in descriptors.items():
                    if id(descriptor) not in rebound:
                        base = descriptor.base
                        view = torch.empty(0, dtype=base.dtype, device=base.device).set_(
                            clone.untyped_storage(), offset // base.element_size(),
                            base.shape, base.stride(),
                        )
                        rebound[id(descriptor)] = replace(
                            descriptor, base=view, block_shape=list(descriptor.block_shape),
                        )
                    bank[key] = rebound[id(descriptor)]
                banks.append(bank)
    except torch.OutOfMemoryError:
        return None
    return banks
