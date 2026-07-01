# SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
# SPDX-License-Identifier: MIT

"""Paged decode split-KV scheduling backed by the precompiled Torch op.

The CUDA implementation lives in ``csrc/build_decode_schedule.cu`` and is
built ahead of time by kernel-builder. The op returns the schedule arrays
plus a fixed-order scalar summary, which is reassembled into the schedule
dict here.
"""

from __future__ import annotations

import torch

from ....._ops import ops

# Order of the scalar summary returned by the op; must match
# csrc/build_decode_schedule.cu.
_SCALAR_KEYS = (
    "split_kv",
    "cta_tile_q",
    "num_q_tiles",
    "kv_chunk_size_pages",
    "kv_chunk_size_tokens",
    "work_count",
    "padded_work_count",
    "partial_rows",
    "max_split_count",
    "max_grid_size",
    "active_blocks_per_sm",
    "num_sms",
    "base_cta",
)


def build_decode_schedule(
    seqused_k: torch.Tensor,
    *,
    page_size: int,
    seqlen_q: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    enable_cuda_graph: bool = False,
    max_grid_size: int = 0,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
) -> dict[str, object]:
    """GPU-only schedule build: single CUDA kernel produces all schedule
    index arrays on device.  Only a small summary tensor is D2H'd at the end
    so the wrapper can size O_partial, pick the kernel grid, and choose
    split/non-split compile path.

    ``max_seqlen_k`` is required as the host-side worst-case bound for
    padding the work-tile arrays.
    """

    (
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        block_valid_mask,
        split_counts,
        kv_pages,
        merge_indptr,
        o_indptr,
        scalars,
    ) = ops.build_decode_schedule(
        seqused_k,
        int(page_size),
        int(seqlen_q),
        int(num_qo_heads),
        int(num_kv_heads),
        int(head_dim),
        int(max_seqlen_k),
        bool(enable_cuda_graph),
        int(max_grid_size),
        int(fixed_split_size),
        bool(disable_split_kv),
    )

    raw: dict[str, object] = dict(zip(_SCALAR_KEYS, (int(s) for s in scalars)))
    raw["split_kv"] = bool(raw["split_kv"])
    raw["request_indices"] = request_indices
    raw["qo_tile_indices"] = qo_tile_indices
    raw["kv_tile_indices"] = kv_tile_indices
    raw["block_valid_mask"] = block_valid_mask
    raw["split_counts"] = split_counts
    raw["kv_pages"] = kv_pages
    raw["merge_indptr"] = merge_indptr
    raw["o_indptr"] = o_indptr

    # The CUDA kernel writes into worst-case-padded buffers (size =
    # batch * num_q_tiles * max_pages_global) but only the first
    # ``padded_work_count`` entries are valid.  Downstream consumers
    # (tile_scheduler) take grid size from ``request_indices.shape[0]``
    # so we narrow the views to that count; the underlying allocation
    # is unchanged so this is a view, no copy.
    pad = int(raw["padded_work_count"])
    for key in (
        "request_indices",
        "qo_tile_indices",
        "kv_tile_indices",
        "block_valid_mask",
    ):
        raw[key] = raw[key].narrow(0, 0, pad)
    return raw


__all__ = ["build_decode_schedule"]
