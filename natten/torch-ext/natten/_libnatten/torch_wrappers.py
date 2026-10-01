#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
# kernel-builder port of upstream `natten/_libnatten/torch_wrappers.py`.
#
# Upstream registers Python `torch.library.custom_op`s that allocate outputs
# and call into the pybind11 `libnatten` extension. In this port the ops are
# registered in C++ (`torch-ext/torch_binding.cpp`) as out-variant ops under
# the build-time namespace exposed through `.._ops`. The functions here keep
# the exact upstream calling conventions (allocate outputs, handle kv-split
# defaults, varlen zero-init) and call the C++ ops, and each C++ op gets a
# fake (meta) registration so the whole surface stays torch.compile-safe.
#
# Schema conventions of the C++ ops:
#   - `kernel_size`/`stride`/`dilation`/tile shapes are `int[]`.
#   - Multi-dimensional causal masks are passed as `int[]` (0/1) because
#     boolean arrays are less uniformly supported in op schemas.
#   - `scale` is a `float`.

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from .._ops import add_op_namespace_prefix, ops
from ..utils.tuples import ceil_div_tuple, mul_tuple

register_fake = torch.library.register_fake


def maybe_contiguous(x):
    return x.contiguous()


def _ints(v: Sequence) -> list:
    return [int(x) for x in v]


################################################################################
############################ Fake (meta) registration ##########################
################################################################################
# All C++ ops are out-variant: they only mutate output arguments and return
# nothing, so their fake impls are no-ops. Shape inference happens in the
# Python wrappers below, which allocate the outputs.


def _register_noop_fake(op_name: str) -> None:
    def _fake(*args, **kwargs) -> None:
        return None

    register_fake(add_op_namespace_prefix(op_name))(_fake)


for _na_dim in (1, 2, 3):
    for _prefix in ("", "hopper_", "blackwell_", "reference_"):
        _register_noop_fake(f"{_prefix}na{_na_dim}d_forward")
        _register_noop_fake(f"{_prefix}na{_na_dim}d_backward")
    _register_noop_fake(f"token_permute_{_na_dim}d")
    _register_noop_fake(f"token_unpermute_{_na_dim}d")

for _prefix in ("", "hopper_", "blackwell_"):
    _register_noop_fake(f"{_prefix}fmha_forward")
    _register_noop_fake(f"{_prefix}fmha_backward")

_register_noop_fake("compute_delta")


################################################################################
################################### FMHA ops ###################################
################################################################################


def blackwell_fmha_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    run_persistent_kernel: bool,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

    output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros if is_varlen else torch.empty

    output = init_fn(output_shape, device=query.device, dtype=query.dtype)
    logsumexp = init_fn(query.shape[:-1], dtype=torch.float32, device=query.device)

    # Skip kernel launch when all sequences are empty
    if is_varlen and max_seqlen_Q == 0 and max_seqlen_KV == 0:
        return output, logsumexp

    ops.blackwell_fmha_forward(
        output,
        query,
        key,
        value,
        logsumexp,
        bool(is_causal),
        float(scale),
        int(q_tile_size),
        int(kv_tile_size),
        bool(run_persistent_kernel),
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        int(max_seqlen_Q),
        int(max_seqlen_KV),
    )

    return output, logsumexp


def blackwell_fmha_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
    deterministic: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    output, d_output, logsumexp = [
        maybe_contiguous(x) for x in (output, d_output, logsumexp)
    ]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros_like if is_varlen else torch.empty_like

    d_query = init_fn(query)
    d_key = init_fn(key)
    d_value = init_fn(value)

    # Skip kernel launch when all sequences are empty
    if is_varlen and max_seqlen_Q == 0 and max_seqlen_KV == 0:
        return d_query, d_key, d_value

    ops.blackwell_fmha_backward(
        d_query,
        d_key,
        d_value,
        query,
        key,
        value,
        output,
        d_output,
        logsumexp,
        bool(is_causal),
        float(scale),
        int(q_tile_size),
        int(kv_tile_size),
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        int(max_seqlen_Q),
        int(max_seqlen_KV),
        bool(deterministic),
    )

    return d_query, d_key, d_value


def hopper_fmha_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    kernel_schedule_int: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

    output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros if is_varlen else torch.empty

    output = init_fn(output_shape, device=query.device, dtype=query.dtype)
    logsumexp = init_fn(query.shape[:-1], dtype=torch.float32, device=query.device)

    # Skip kernel launch when all sequences are empty
    if is_varlen and max_seqlen_Q == 0 and max_seqlen_KV == 0:
        return output, logsumexp

    ops.hopper_fmha_forward(
        output,
        query,
        key,
        value,
        logsumexp,
        bool(is_causal),
        float(scale),
        int(q_tile_size),
        int(kv_tile_size),
        int(kernel_schedule_int),
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        int(max_seqlen_Q),
        int(max_seqlen_KV),
    )

    return output, logsumexp


def hopper_fmha_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    output, d_output, logsumexp = [
        maybe_contiguous(x) for x in (output, d_output, logsumexp)
    ]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros_like if is_varlen else torch.empty_like

    d_query = init_fn(query)
    d_key = init_fn(key)
    d_value = init_fn(value)

    # Skip kernel launch when all sequences are empty
    if is_varlen and max_seqlen_Q == 0 and max_seqlen_KV == 0:
        return d_query, d_key, d_value

    ops.hopper_fmha_backward(
        d_query,
        d_key,
        d_value,
        query,
        key,
        value,
        output,
        d_output,
        logsumexp,
        bool(is_causal),
        float(scale),
        int(q_tile_size),
        int(kv_tile_size),
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        int(max_seqlen_Q),
        int(max_seqlen_KV),
    )

    return d_query, d_key, d_value


def fmha_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

    output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros if is_varlen else torch.empty

    output = init_fn(output_shape, device=query.device, dtype=query.dtype)
    logsumexp = init_fn(query.shape[:-1], dtype=torch.float32, device=query.device)

    # Skip kernel launch when all sequences are empty
    if is_varlen and max_seqlen_Q == 0 and max_seqlen_KV == 0:
        return output, logsumexp

    ops.fmha_forward(
        output,
        query,
        key,
        value,
        logsumexp,
        bool(is_causal),
        float(scale),
        int(q_tile_size),
        int(kv_tile_size),
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        int(max_seqlen_Q),
        int(max_seqlen_KV),
    )

    return output, logsumexp


def fmha_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    num_kv_splits: Optional[int],
    compute_delta_with_pt: bool,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
    deterministic: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    from ..backends.configs.cutlass.backward_knobs import check_fmha_kv_splits

    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    output, d_output, logsumexp = [
        maybe_contiguous(x) for x in (output, d_output, logsumexp)
    ]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros_like if is_varlen else torch.empty_like

    d_query = init_fn(query)
    d_key = init_fn(key)
    d_value = init_fn(value)

    # Skip kernel launch when all sequences are empty
    if is_varlen and max_seqlen_Q == 0 and max_seqlen_KV == 0:
        return d_query, d_key, d_value

    if deterministic:
        # Torch reduction seems to have slight reproducibility issues, even with determinism on
        compute_delta_with_pt = False
        # TODO: this is the only way to get determinism in this kernel, but it's very slow
        num_kv_splits = 1
    else:
        # Compute default kv_splits if not specified
        # max_seqlen must be at least 2 to satisfy static checks that are just too complicated to
        # relax at this point. Kernel launch will be skipped if max_seqlen is 0 anyway. Prior checks
        # should prevent negative max seqlens.
        max_seqlen = max(2, max_seqlen_KV) if is_varlen else None
        num_kv_splits = check_fmha_kv_splits(
            kv_splits=num_kv_splits,
            input_tensor=key,
            kv_tile_size=kv_tile_size,
            deterministic=deterministic,
            max_seqlen=max_seqlen,
        )

    ops.fmha_backward(
        d_query,
        d_key,
        d_value,
        query,
        key,
        value,
        output,
        d_output,
        logsumexp,
        bool(is_causal),
        float(scale),
        int(q_tile_size),
        int(kv_tile_size),
        int(num_kv_splits),
        bool(compute_delta_with_pt),
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        int(max_seqlen_Q),
        int(max_seqlen_KV),
    )

    return d_query, d_key, d_value


################################################################################
################################### FNA ops  ###################################
################################################################################


def make_blackwell_fna_ops(na_dim):
    fwd_op = getattr(ops, f"blackwell_na{na_dim}d_forward")
    bwd_op = getattr(ops, f"blackwell_na{na_dim}d_backward")

    def blackwell_fna_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale: float,
        q_shape,
        kv_shape,
        qkv_shape,
        q_tile_shape,
        kv_tile_shape,
        run_persistent_kernel: bool,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        fwd_op(
            output,
            query,
            key,
            value,
            logsumexp,
            _ints(kernel_size),
            _ints(stride),
            _ints(dilation),
            _ints(is_causal),
            float(scale),
            _ints(q_shape),
            _ints(kv_shape),
            _ints(qkv_shape),
            _ints(q_tile_shape),
            _ints(kv_tile_shape),
            bool(run_persistent_kernel),
        )

        return output, logsumexp

    def blackwell_fna_backward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale: float,
        q_shape,
        kv_shape,
        qkv_shape,
        q_tile_shape,
        kv_tile_shape,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        bwd_op(
            d_query,
            d_key,
            d_value,
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            _ints(kernel_size),
            _ints(stride),
            _ints(dilation),
            _ints(is_causal),
            float(scale),
            _ints(q_shape),
            _ints(kv_shape),
            _ints(qkv_shape),
            _ints(q_tile_shape),
            _ints(kv_tile_shape),
        )

        return d_query, d_key, d_value

    return blackwell_fna_forward, blackwell_fna_backward


def make_hopper_fna_ops(na_dim):
    fwd_op = getattr(ops, f"hopper_na{na_dim}d_forward")
    bwd_op = getattr(ops, f"hopper_na{na_dim}d_backward")

    def hopper_fna_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale: float,
        q_shape,
        kv_shape,
        qkv_shape,
        q_tile_shape,
        kv_tile_shape,
        kernel_schedule_int: int,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        fwd_op(
            output,
            query,
            key,
            value,
            logsumexp,
            _ints(kernel_size),
            _ints(stride),
            _ints(dilation),
            _ints(is_causal),
            float(scale),
            _ints(q_shape),
            _ints(kv_shape),
            _ints(qkv_shape),
            _ints(q_tile_shape),
            _ints(kv_tile_shape),
            int(kernel_schedule_int),
        )

        return output, logsumexp

    def hopper_fna_backward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale: float,
        q_shape,
        kv_shape,
        qkv_shape,
        q_tile_shape,
        kv_tile_shape,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        bwd_op(
            d_query,
            d_key,
            d_value,
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            _ints(kernel_size),
            _ints(stride),
            _ints(dilation),
            _ints(is_causal),
            float(scale),
            _ints(q_shape),
            _ints(kv_shape),
            _ints(qkv_shape),
            _ints(q_tile_shape),
            _ints(kv_tile_shape),
        )

        return d_query, d_key, d_value

    return hopper_fna_forward, hopper_fna_backward


def make_fna_ops(na_dim):
    fwd_op = getattr(ops, f"na{na_dim}d_forward")
    bwd_op = getattr(ops, f"na{na_dim}d_backward")

    def fna_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale: float,
        q_tile_shape,
        kv_tile_shape,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        fwd_op(
            output,
            query,
            key,
            value,
            logsumexp,
            _ints(kernel_size),
            _ints(stride),
            _ints(dilation),
            _ints(is_causal),
            float(scale),
            _ints(q_tile_shape),
            _ints(kv_tile_shape),
        )

        return output, logsumexp

    def fna_backward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale: float,
        q_tile_shape,
        kv_tile_shape,
        num_kv_splits,
        compute_delta_with_pt: bool,
        deterministic: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        from ..backends.configs.cutlass.backward_knobs import check_fna_kv_splits

        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        if deterministic:
            # Torch reduction seems to have slight reproducibility issues, even with determinism on
            compute_delta_with_pt = False
            # TODO: this is the only way to get determinism in this kernel, but it's very slow
            num_kv_splits = tuple(1 for _ in range(na_dim))
        else:
            # Compute default kv_splits if not specified
            num_kv_splits = check_fna_kv_splits(
                kv_splits=tuple(num_kv_splits) if num_kv_splits is not None else None,
                input_tensor=key,
                kv_tile_shape=tuple(kv_tile_shape),
                deterministic=deterministic,
                dilation=tuple(dilation),
            )

        bwd_op(
            d_query,
            d_key,
            d_value,
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            _ints(kernel_size),
            _ints(stride),
            _ints(dilation),
            _ints(is_causal),
            float(scale),
            _ints(q_tile_shape),
            _ints(kv_tile_shape),
            _ints(num_kv_splits),
            bool(compute_delta_with_pt),
        )

        return d_query, d_key, d_value

    return fna_forward, fna_backward


def make_reference_fna_ops(na_dim):
    fwd_op = getattr(ops, f"reference_na{na_dim}d_forward")
    bwd_op = getattr(ops, f"reference_na{na_dim}d_backward")

    def reference_fna_forward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale: float,
        qkv_shape,
        num_extra_kv: int,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        fwd_op(
            output,
            query,
            key,
            value,
            logsumexp,
            _ints(kernel_size),
            _ints(stride),
            _ints(dilation),
            _ints(is_causal),
            float(scale),
            _ints(qkv_shape),
            int(num_extra_kv),
        )

        return output, logsumexp

    def reference_fna_backward(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale: float,
        qkv_shape,
        num_extra_kv: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        bwd_op(
            d_query,
            d_key,
            d_value,
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            _ints(kernel_size),
            _ints(stride),
            _ints(dilation),
            _ints(is_causal),
            float(scale),
            _ints(qkv_shape),
            int(num_extra_kv),
        )

        return d_query, d_key, d_value

    return reference_fna_forward, reference_fna_backward


################################################################################
################################# TokPerm ops  #################################
################################################################################


def make_token_permute_ops(na_dim):
    permute_op = getattr(ops, f"token_permute_{na_dim}d")
    unpermute_op = getattr(ops, f"token_unpermute_{na_dim}d")

    def token_permute(
        input_tensor: Tensor,
        tile_shape,
        dilation,
        flip_tiled_dims: bool,
    ) -> Tensor:
        input_tensor = maybe_contiguous(input_tensor)

        token_layout = tuple(x for x in input_tensor.shape[1 : na_dim + 1])
        token_layout_padded = mul_tuple(
            mul_tuple(
                ceil_div_tuple(ceil_div_tuple(token_layout, tile_shape), dilation),
                dilation,
            ),
            tile_shape,
        )
        output_shape = [
            input_tensor.shape[0],
            math.prod(token_layout_padded),
            input_tensor.shape[-2],
            input_tensor.shape[-1],
        ]
        output = torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
        permute_op(
            output,
            input_tensor,
            _ints(tile_shape),
            _ints(dilation),
            bool(flip_tiled_dims),
        )

        # Fold dilation in batch dimension so that attention is correct.
        output = output.reshape(
            input_tensor.shape[0] * math.prod(dilation),
            -1,
            input_tensor.shape[-2],
            input_tensor.shape[-1],
        )

        return output

    def token_unpermute(
        input_tensor: Tensor,
        token_layout_shape,
        tile_shape,
        dilation,
        flip_tiled_dims: bool,
    ) -> Tensor:
        input_tensor = maybe_contiguous(input_tensor)

        # Unfold dilation in batch dimension
        num_dilation_groups = math.prod(dilation)
        assert input_tensor.shape[0] % num_dilation_groups == 0
        input_tensor = input_tensor.reshape(
            input_tensor.shape[0] // num_dilation_groups,
            -1,
            input_tensor.shape[-2],
            input_tensor.shape[-1],
        )

        output_shape = [
            input_tensor.shape[0],
            *token_layout_shape,
            input_tensor.shape[-2],
            input_tensor.shape[-1],
        ]
        output = torch.empty(
            output_shape, device=input_tensor.device, dtype=input_tensor.dtype
        )
        unpermute_op(
            output,
            input_tensor,
            _ints(tile_shape),
            _ints(dilation),
            bool(flip_tiled_dims),
        )

        return output

    return token_permute, token_unpermute


(blackwell_na1d_forward, blackwell_na1d_backward) = make_blackwell_fna_ops(1)
(blackwell_na2d_forward, blackwell_na2d_backward) = make_blackwell_fna_ops(2)
(blackwell_na3d_forward, blackwell_na3d_backward) = make_blackwell_fna_ops(3)

(hopper_na1d_forward, hopper_na1d_backward) = make_hopper_fna_ops(1)
(hopper_na2d_forward, hopper_na2d_backward) = make_hopper_fna_ops(2)
(hopper_na3d_forward, hopper_na3d_backward) = make_hopper_fna_ops(3)

(na1d_forward, na1d_backward) = make_fna_ops(1)
(na2d_forward, na2d_backward) = make_fna_ops(2)
(na3d_forward, na3d_backward) = make_fna_ops(3)

(reference_na1d_forward, reference_na1d_backward) = make_reference_fna_ops(1)
(reference_na2d_forward, reference_na2d_backward) = make_reference_fna_ops(2)
(reference_na3d_forward, reference_na3d_backward) = make_reference_fna_ops(3)

(token_permute_1d, token_unpermute_1d) = make_token_permute_ops(1)
(token_permute_2d, token_unpermute_2d) = make_token_permute_ops(2)
(token_permute_3d, token_unpermute_3d) = make_token_permute_ops(3)


# This is only used in unit tests, and not even auto-diffable
def compute_delta(out: Tensor, d_out: Tensor, delta: Tensor) -> None:
    ops.compute_delta(out, d_out, delta)


__all__ = [
    "blackwell_fmha_backward",
    "blackwell_fmha_forward",
    "blackwell_na1d_backward",
    "blackwell_na1d_forward",
    "blackwell_na2d_backward",
    "blackwell_na2d_forward",
    "blackwell_na3d_backward",
    "blackwell_na3d_forward",
    "compute_delta",
    "fmha_backward",
    "fmha_forward",
    "hopper_fmha_backward",
    "hopper_fmha_forward",
    "hopper_na1d_backward",
    "hopper_na1d_forward",
    "hopper_na2d_backward",
    "hopper_na2d_forward",
    "hopper_na3d_backward",
    "hopper_na3d_forward",
    "na1d_backward",
    "na1d_forward",
    "na2d_backward",
    "na2d_forward",
    "na3d_backward",
    "na3d_forward",
    "reference_na1d_backward",
    "reference_na1d_forward",
    "reference_na2d_backward",
    "reference_na2d_forward",
    "reference_na3d_backward",
    "reference_na3d_forward",
    "token_permute_1d",
    "token_permute_2d",
    "token_permute_3d",
    "token_unpermute_1d",
    "token_unpermute_2d",
    "token_unpermute_3d",
]
