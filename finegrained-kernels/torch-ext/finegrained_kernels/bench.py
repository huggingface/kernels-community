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


def _active_experts(nargs, experts):
    """The experts one call routes to (from ``ExpertIds`` or ``ExpertStart``), or None without routing."""
    ids, start = nargs.get("ExpertIds"), nargs.get("ExpertStart")
    if ids is not None:
        return ids[(ids >= 0) & (ids < experts)].unique()
    if start is not None:
        if start.numel() < experts + 1:
            return start.new_empty(0)  # no usable offsets: nothing to rotate
        return (start[1:experts + 1] > start[:experts]).nonzero().flatten()
    return None


def _active_bytes(nargs, weight):
    """The bytes one call reads from ``weight``: only the routed experts' planes of an expert stack."""
    nbytes = weight.numel() * weight.element_size()
    active = _active_experts(nargs, weight.shape[0])
    return nbytes if active is None else nbytes * active.numel() // weight.shape[0]


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
    active = _active_experts(nargs, experts)
    if active is None:
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
            counts = start[1:experts + 1] - start[:experts]
            moved = torch.zeros_like(counts)
            moved[block] = counts[active]
            bank_start = start.clone()
            bank_start[1:experts + 1] = (start[0] + moved.cumsum(0)).to(start.dtype)
            banks.append(dict(nargs, ExpertStart=bank_start))
    return banks


def _copied_banks(nargs, weight, count):
    """Banks over copies of ``B`` and its full-weight host descriptor (BSDescriptor may be an unread
    alias of BDescriptor). None when B's descriptor covers a different span or memory is short."""
    span_bytes = weight.numel() * weight.element_size()
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


def gemm_cold_banks(nargs):
    """``cold_banks`` for the GEMMs: launch-argument sets that together read more than twice L2 of
    distinct weight ``B``. A routed call reroutes to other experts of the one stack (``routed_banks``);
    a dense call, or a stack too small for disjoint expert sets, cycles copies of B. Other operands
    stay shared. None (cold event timing) when B cannot be rotated."""
    weight = nargs["B"]
    if not weight.is_cuda or torch._debug_has_internal_overlap(weight) != 0:
        return None
    count = rotation_count(_active_bytes(nargs, weight), weight.device)
    if count is None:
        return None
    if weight.ndim == 3 and (banks := routed_banks(nargs, weight.shape[0], count)) is not None:
        return banks
    return _copied_banks(nargs, weight, count)
