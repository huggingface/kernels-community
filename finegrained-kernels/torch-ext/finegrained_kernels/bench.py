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
from triton.testing import do_bench_cudagraph as _do_bench_cudagraph

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
    return _do_bench_cudagraph(fn, quantiles=quantiles, **kwargs)


def active_experts(ids, start, experts):
    """``(active, idle)``: the experts one call routes to, from its expert ids or its per-expert row
    offsets, and the others. None without routing."""
    if ids is not None:
        active = ids[(ids >= 0) & (ids < experts)].unique()
    elif start is not None:
        if start.numel() < experts + 1:
            active = start.new_empty(0)  # no usable offsets: nothing to rotate
        else:
            active = (start[1:experts + 1] > start[:experts]).nonzero().flatten()
    else:
        return None
    idle = torch.arange(experts, device=active.device)
    return active, idle[~torch.isin(idle, active)]


def rotation_count(active_bytes, device):
    """Calls to cycle so that together they read more than twice L2. Capped at 1024 launches per
    captured graph, so a weight under ~2*L2/1024 stays partly warm (a negligible kernel)."""
    if active_bytes <= 0:
        return 1
    return min(2 * torch.cuda.get_device_properties(device).L2_cache_size // active_bytes + 1, 1024)


def free_device_bytes(device):
    """Memory a new bank can take: free device memory plus the caching allocator's unused reserve."""
    free, _ = torch.cuda.mem_get_info(device)
    return free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)


def routed_banks(nargs, active, idle, count):
    """Up to ``count`` banks over the one expert stack: bank 0 is the call, bank ``k`` relabels its
    ``n`` active experts to the ``k``-th block of ``n`` idle ones, so every bank reads other experts'
    weights and no weight is copied. As many as the stack holds (``experts // n``)."""
    ids, start = nargs.get("ExpertIds"), nargs.get("ExpertStart")
    experts = active.numel() + idle.numel()
    banks = [nargs]
    n = active.numel()
    if n == 0:
        return banks
    count = min(count, experts // n)
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


def _weight_copies(nargs, weight, copies):
    """Up to ``copies`` argument overrides, each a copy of ``B`` and its full-weight host descriptor
    (BSDescriptor may be an unread alias of BDescriptor). Fewer when memory runs short; none when
    B's descriptor covers a different span than B."""
    span_bytes = weight.numel() * weight.element_size()
    descriptor = nargs.get("BDescriptor")
    if isinstance(descriptor, TensorDescriptor):
        base = descriptor.base
        # Existing wrappers describe all of B, sometimes through a byte/shape view.
        if (not isinstance(base, torch.Tensor) or not weight.is_contiguous()
                or not base.is_contiguous() or base.data_ptr() != weight.data_ptr()
                or base.numel() * base.element_size() != span_bytes):
            return []
    overrides = []
    with torch.no_grad():
        while len(overrides) < copies and span_bytes <= free_device_bytes(weight.device):
            try:
                clone = torch.empty_strided(weight.shape, weight.stride(), dtype=weight.dtype, device=weight.device)
            except torch.OutOfMemoryError:
                break
            clone.copy_(weight)
            override = {"B": clone}
            if isinstance(descriptor, TensorDescriptor):
                base = clone.view(descriptor.base.dtype).view(descriptor.base.shape)
                override["BDescriptor"] = maybe_descriptor(base, list(descriptor.block_shape))
                if nargs.get("BSDescriptor") is descriptor:
                    override["BSDescriptor"] = override["BDescriptor"]
            overrides.append(override)
    return overrides


def gemm_cold_banks(nargs):
    """``cold_banks`` for the GEMMs: launch-argument sets that together read more than twice L2 of
    distinct weights ``B``, and whether that many fit. A routed call reroutes to other experts of its
    stack (``routed_banks``); copies of ``B`` make up the rest."""
    weight = nargs["B"]
    nbytes = weight.numel() * weight.element_size()
    routing = active_experts(nargs.get("ExpertIds"), nargs.get("ExpertStart"), weight.shape[0]) if weight.ndim == 3 else None
    active_bytes = nbytes if routing is None else nbytes * routing[0].numel() // weight.shape[0]
    count = rotation_count(active_bytes, weight.device)
    banks = [nargs] if routing is None else routed_banks(nargs, *routing, count)
    copies = []
    if len(banks) < count and torch._debug_has_internal_overlap(weight) == 0:
        copies = _weight_copies(nargs, weight, -(-count // len(banks)) - 1)
    banks = [dict(bank, **copy) for copy in [{}, *copies] for bank in banks][:count]
    return banks, len(banks) == count
