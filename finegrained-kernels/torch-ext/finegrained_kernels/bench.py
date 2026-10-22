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

import torch
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.testing import do_bench as _do_bench
from triton.testing import do_bench_cudagraph

from .descriptors import maybe_descriptor


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


def gemm_weight(nargs):
    """``cold_weight`` for the GEMMs: their weight ``B`` and the bytes one call reads from it (only
    the routed experts' planes of an expert stack)."""
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


def free_device_bytes(device):
    """Memory a new bank can take: free device memory plus the caching allocator's unused reserve."""
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def routed_banks(nargs, experts, count):
    """Banks over the one expert stack: bank 0 is the call, bank ``k`` relabels its ``n`` active
    experts to the ``k``-th block of ``n`` experts it does not route to, so every bank reads other
    experts' weights and no weight is copied. None when there is no routing or the ``count``
    blocks do not fit the stack."""
    ids, start = nargs.get("ExpertIds"), nargs.get("ExpertStart")
    if ids is not None:
        active = ids[(ids >= 0) & (ids < experts)].unique()
    elif start is not None and start.numel() >= experts + 1:
        counts = start[1:experts + 1] - start[:experts]
        active = (counts > 0).nonzero().flatten()
    else:
        return None
    n = active.numel()
    if n == 0 or count * n > experts:
        return None
    idle = torch.arange(experts, device=active.device)
    idle = idle[~torch.isin(idle, active)]
    banks = [nargs]
    for k in range(1, count):
        block = idle[(k - 1) * n:k * n]
        if ids is not None:
            relabel = torch.arange(experts, device=ids.device, dtype=ids.dtype)
            relabel[active] = block.to(ids.dtype)
            routed = (ids >= 0) & (ids < experts)
            banks.append(dict(nargs, ExpertIds=torch.where(routed, relabel[ids.clamp(0, experts - 1)], ids)))
        else:
            # same rows, now owned by the block's experts in the same order
            moved = torch.zeros_like(counts)
            moved[block] = counts[active]
            bank_start = start.clone()
            bank_start[1:experts + 1] = (start[0] + moved.cumsum(0)).to(start.dtype)
            banks.append(dict(nargs, ExpertStart=bank_start))
    return banks


def weight_banks(nargs, weight, active_bytes):
    """Rotate the forward GEMMs' readonly B and its full-weight host descriptor.

    A routed call rotates which experts it reads (``routed_banks``); a dense one, or an expert stack
    too small for disjoint blocks, rotates copies of B. Other operands stay shared. The opted-in
    GEMMs pass B independently of outputs and scales; BSDescriptor may be an unread alias of BDescriptor.
    """
    if not weight.is_cuda or torch._debug_has_internal_overlap(weight) != 0:
        return None
    span_bytes = weight.numel() * weight.element_size()
    count = rotation_count(active_bytes, weight.device)
    if count is None:
        return None
    if weight.ndim == 3 and (banks := routed_banks(nargs, weight.shape[0], count)) is not None:
        return banks
    descriptor = nargs.get("BDescriptor")
    if isinstance(descriptor, TensorDescriptor):
        base = descriptor.base
        # Existing wrappers describe all of B, sometimes through a byte/shape view.
        if (not isinstance(base, torch.Tensor) or not weight.is_contiguous()
                or not base.is_contiguous() or base.data_ptr() != weight.data_ptr()
                or base.numel() * base.element_size() != span_bytes):
            return None
    if (count - 1) * span_bytes > free_device_bytes(weight.device):
        return None
    banks = [nargs]
    try:
        with torch.no_grad():
            for _ in range(count - 1):
                clone = torch.empty_strided(weight.shape, weight.stride(),
                                            dtype=weight.dtype, device=weight.device)
                clone.copy_(weight)
                bank = dict(nargs, B=clone)
                if isinstance(descriptor, TensorDescriptor):
                    base = clone.view(descriptor.base.dtype).view(descriptor.base.shape)
                    bank["BDescriptor"] = maybe_descriptor(base, list(descriptor.block_shape))
                    if nargs.get("BSDescriptor") is descriptor:
                        bank["BSDescriptor"] = bank["BDescriptor"]
                banks.append(bank)
    except torch.OutOfMemoryError:
        return None
    return banks
