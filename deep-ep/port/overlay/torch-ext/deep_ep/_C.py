"""
kernels-community: the `deep_ep._C` API of upstream DeepEP, on top of Torch ops.

Upstream builds `deep_ep._C` as a pybind11 module. In the Hub kernel, its free
functions are Torch ops and its stateful objects are Torch custom classes
(`torch-ext/torch_binding.cpp`). This module gives them back the upstream
names, signatures and attributes, so the rest of the package is upstream code
with only its `_C` imports made relative.
"""

from typing import Any, List, Optional, Sequence, Tuple

import torch

from ._ops import add_op_namespace_prefix, ops

_namespace = add_op_namespace_prefix("")[: -len("::")]
_classes = getattr(torch.classes, _namespace)


def _class_name(value: Any) -> Optional[str]:
    if not isinstance(value, torch.ScriptObject):
        return None
    return value._type().qualified_name().rsplit(".", 1)[-1]


def _wrap(value: Any) -> Any:
    """Convert custom class objects returned by the C++ runtime into upstream's Python objects."""
    name = _class_name(value)
    if name == "Epilogue":
        return lambda: _wrap(value.run())
    if name == "Context":
        return Context(value)
    if isinstance(value, tuple):
        return tuple(_wrap(item) for item in value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _unwrap_context(context: "Context") -> torch.ScriptObject:
    return context._runtime


def _ensure_initialized() -> None:
    # Deferred from import time, see `_runtime.py`
    from ._runtime import ensure_initialized
    ensure_initialized()


# Constants
def get_num_allocation_alignment() -> int:
    return ops.get_num_allocation_alignment()


def get_num_tma_alignment() -> int:
    return ops.get_num_tma_alignment()


def get_num_rdma_alignment() -> int:
    return ops.get_num_rdma_alignment()


topk_idx_t = {32: torch.int32, 64: torch.int64}[ops.get_num_topk_idx_bits()]


# JIT
def init_jit(library_root_path: str, nccl_root_path: str) -> None:
    ops.init_jit(library_root_path, nccl_root_path)


# NCCL versions (kernels-community additions, used by the import-time checks)
def get_nccl_runtime_version() -> int:
    return ops.get_nccl_runtime_version()


def get_nccl_compile_version() -> int:
    return ops.get_nccl_compile_version()


# Events
EventHandle = _classes.EventHandle


# Communication
class Context:
    """A communication context, shared by the buffers that use it."""

    def __init__(self, runtime: torch.ScriptObject):
        self._runtime = runtime

    def get_physical_domain_size(self) -> Tuple[int, int]:
        return self._runtime.get_physical_domain_size()

    def get_logical_domain_size(self) -> Tuple[int, int]:
        return self._runtime.get_logical_domain_size()

    @property
    def num_workspace_bytes(self) -> int:
        return self._runtime.num_workspace_bytes()

    @property
    def num_gpu_buffer_bytes(self) -> int:
        return self._runtime.num_gpu_buffer_bytes()

    @property
    def num_rdma_storage_bytes(self) -> int:
        return self._runtime.num_rdma_storage_bytes()


def barrier(context: Context, with_cpu_sync: bool, sequential: bool, wait_comm_stream: bool) -> None:
    _unwrap_context(context).barrier(with_cpu_sync, sequential, wait_comm_stream)


def get_local_nccl_unique_id() -> torch.Tensor:
    return ops.get_local_nccl_unique_id()


def create_nccl_comm(root_unique_id: torch.Tensor, num_ranks: int, rank_idx: int) -> int:
    return ops.create_nccl_comm(root_unique_id, num_ranks, rank_idx)


def destroy_nccl_comm(nccl_comm: int) -> None:
    ops.destroy_nccl_comm(nccl_comm)


def get_physical_domain_size(nccl_comm: int) -> Tuple[int, int]:
    return tuple(ops.get_physical_domain_size(nccl_comm))


def get_logical_domain_size(nccl_comm: int, allow_hybrid_mode: bool) -> Tuple[int, int]:
    return tuple(ops.get_logical_domain_size(nccl_comm, allow_hybrid_mode))


def get_comm_stream() -> torch.cuda.Stream:
    stream_id, device_index, device_type = ops.get_comm_stream()
    return torch.cuda.Stream(stream_id=stream_id, device_index=device_index, device_type=device_type)


# Buffers
class BufferBase:
    """Common runtime interface of the C++ buffers."""

    def __init__(self, runtime: torch.ScriptObject):
        self._runtime = runtime

    @property
    def main_context(self) -> Optional[Context]:
        return _wrap(self._runtime.main_context())

    def print_memory_usage(self) -> None:
        self._runtime.print_memory_usage()

    def destroy(self) -> None:
        self._runtime.destroy()


class EPBuffer(BufferBase):
    def __init__(self, *args):
        _ensure_initialized()
        super().__init__(_classes.EPBuffer(*args))
        self.context = _wrap(self._runtime.context())

    @property
    def lb_storage(self) -> torch.Tensor:
        return self._runtime.lb_storage()

    def dispatch(self, *args) -> tuple:
        args = list(args)
        # `cached_num_recv_tokens_per_expert_list`
        if args[7] is not None:
            args[7] = list(args[7])
        return _wrap(self._runtime.dispatch(*args))

    def combine(self, *args) -> tuple:
        return _wrap(self._runtime.combine(*args))

    def lb_prefetch_weights(self, redundant_expert_weights: Sequence[torch.Tensor],
                            expert_weights: Sequence[torch.Tensor], redundancy_mapping: torch.Tensor,
                            num_sms: int, previous_event: Optional[EventHandle]) -> Optional[EventHandle]:
        return self._runtime.lb_prefetch_weights(list(redundant_expert_weights), list(expert_weights),
                                                 redundancy_mapping, num_sms, previous_event)

    def lb_reduce_grads(self, redundant_expert_grads: torch.Tensor, expert_grads: torch.Tensor,
                        redundancy_mapping: torch.Tensor, num_sms: int,
                        previous_event: Optional[EventHandle]) -> Optional[EventHandle]:
        return self._runtime.lb_reduce_grads(redundant_expert_grads, expert_grads, redundancy_mapping,
                                             num_sms, previous_event)


def calculate_ep_buffer_size(nccl_comm: int, num_max_tokens_per_rank: int, hidden: int, num_topk: int,
                             use_fp8_dispatch: bool, allow_hybrid_mode: bool, allow_multiple_reduction: bool) -> int:
    return ops.calculate_ep_buffer_size(nccl_comm, num_max_tokens_per_rank, hidden, num_topk,
                                        use_fp8_dispatch, allow_hybrid_mode, allow_multiple_reduction)


def get_ep_buffer_alignment() -> int:
    return ops.get_ep_buffer_alignment()


class BucketBuffer(BufferBase):
    def __init__(self, rank_indices: List[int], num_ranks: List[int], nccl_comms: List[int],
                 num_buffer_bytes: int, sl_idx: Optional[int], num_gpu_timeout_secs: int,
                 nvl_gbs: float, rdma_gbs: float, explicitly_destroy: bool = False):
        _ensure_initialized()
        super().__init__(_classes.BucketBuffer(list(rank_indices), list(num_ranks), list(nccl_comms),
                                               num_buffer_bytes, sl_idx, num_gpu_timeout_secs,
                                               float(nvl_gbs), float(rdma_gbs), explicitly_destroy))
        self.contexts = _wrap(self._runtime.contexts())
        self.storage = self._runtime.storage()

    def reduce_scatter(self, srcs: List[torch.Tensor], context_idx: int, num_sms: int,
                       comm_precision: str = "fp32", scale: float = 1.0) -> tuple:
        return _wrap(self._runtime.reduce_scatter(list(srcs), context_idx, num_sms, comm_precision, float(scale)))

    def all_reduce(self, srcs: List[torch.Tensor], context_idx: int, num_sms: int, scale: float = 1.0) -> tuple:
        return _wrap(self._runtime.all_reduce(list(srcs), context_idx, num_sms, float(scale)))

    def all_gather(self, srcs: List[torch.Tensor], dsts: List[torch.Tensor],
                   context_idx: int = 0, num_sms: int = 0) -> tuple:
        return _wrap(self._runtime.all_gather(list(srcs), list(dsts), context_idx, num_sms))


class EngramBuffer(BufferBase):
    def __init__(self, rank_idx: int, num_ranks: int, nccl_comm: int, shared_comm: List[Tuple[int, int]],
                 num_gpu_buffer_bytes: int, num_rdma_storage_bytes: int,
                 use_cpu_rdma_storage: bool, allow_hybrid_mode: bool,
                 sl_idx: Optional[int], num_allocated_qps: int, qp_depth: int,
                 num_gpu_timeout_secs: int, explicitly_destroy: bool):
        _ensure_initialized()
        shared_comm_flat = [int(value) for pair in shared_comm for value in pair]
        super().__init__(_classes.EngramBuffer(rank_idx, num_ranks, nccl_comm, shared_comm_flat,
                                               num_gpu_buffer_bytes, num_rdma_storage_bytes,
                                               use_cpu_rdma_storage, allow_hybrid_mode,
                                               sl_idx, num_allocated_qps, qp_depth,
                                               num_gpu_timeout_secs, explicitly_destroy))
        self.context = _wrap(self._runtime.context())

    def set_config(self, num_entries_per_layer: List[int], hidden: int, elem_size: int,
                   num_max_tokens: int, num_entries_per_token: int, num_sf_packs: int) -> None:
        self._runtime.set_config(list(num_entries_per_layer), hidden, elem_size,
                                 num_max_tokens, num_entries_per_token, num_sf_packs)

    def write(self, storages: List[torch.Tensor], sfs: Optional[List[torch.Tensor]]) -> None:
        self._runtime.write(list(storages), None if sfs is None else list(sfs))

    def fetch(self, indices: torch.Tensor, num_qps: int, use_tma_aligned_col_major_sf: bool) -> list:
        return _wrap(self._runtime.fetch(indices, num_qps, use_tma_aligned_col_major_sf))

    @staticmethod
    def create_shared_handle(num_rdma_storage_bytes: int, use_cpu_rdma_storage: bool) -> Tuple[int, int]:
        return tuple(ops.engram_create_shared_handle(num_rdma_storage_bytes, use_cpu_rdma_storage))

    @staticmethod
    def calculate_storage_size(nccl_comm: int, num_entries_per_layer: List[int], hidden: int, elem_size: int,
                               num_max_tokens: int, num_entries_per_token: int, num_sf_packs: int) -> Tuple[int, int]:
        return tuple(ops.engram_calculate_storage_size(nccl_comm, list(num_entries_per_layer), hidden, elem_size,
                                                       num_max_tokens, num_entries_per_token, num_sf_packs))

    @staticmethod
    def get_theoretical_config(nccl_comm: int, num_layers: int, num_max_tokens: int, num_entries_per_token: int,
                               nic_read_mpps: float, num_max_read_requests_per_qp: int) -> Tuple[int, int, bool]:
        return tuple(ops.engram_get_theoretical_config(nccl_comm, num_layers, num_max_tokens, num_entries_per_token,
                                                       float(nic_read_mpps), num_max_read_requests_per_qp))


class PPBuffer(BufferBase):
    def __init__(self, rank_idx: int, num_ranks: int, nccl_comm: int,
                 num_max_tensor_bytes: int, num_max_inflight_tensors: int,
                 sl_idx: Optional[int], num_gpu_timeout_secs: int, explicitly_destroy: bool = False):
        _ensure_initialized()
        super().__init__(_classes.PPBuffer(rank_idx, num_ranks, nccl_comm, num_max_tensor_bytes,
                                           num_max_inflight_tensors, sl_idx, num_gpu_timeout_secs,
                                           explicitly_destroy))
        self.storage = self._runtime.storage()

    def send(self, x: torch.Tensor, dst_rank_idx: int, num_sms: int = 0) -> None:
        self._runtime.send(x, dst_rank_idx, num_sms)

    def recv(self, x: torch.Tensor, src_rank_idx: int, num_sms: int = 0) -> None:
        self._runtime.recv(x, src_rank_idx, num_sms)
