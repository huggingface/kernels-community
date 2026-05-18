"""Sliding-window + attention-sinks CSR builder.

StreamingLLM (Xiao et al. 2023, arXiv:2309.17453) showed that always keeping
the first N tokens ("attention sinks") visible recovers semantic quality
when a non-SW-trained LM is forced into a sliding-window attention regime.
Without sinks, the softmax distribution loses the anchor it implicitly
deposits on the first tokens during pretraining; with sinks, models like
Qwen2.5-Coder / Qwen3 stay coherent at long context under SW=4096.

This builder produces the same ``(row_ptr, col_idx, seq_lens)`` triple as
``build_sliding_window_csr`` but for each Q-block additionally includes the
first ``ceil(sinks / BLOCK_K)`` K-blocks. Deduplicated when the SW range
already covers the sinks. Per-row col_idx is sorted ascending with the
diagonal block as the last entry (kernel convention — see ``csr.py``).
"""
from __future__ import annotations

import math

import torch

from .csr import _broadcast_csr


def build_sw_sinks_csr(
    window: int,
    seq_len: int,
    block_size: int,
    batch_size: int,
    num_heads: int,
    device: torch.device | str,
    sinks: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sliding-window + first-N-sinks causal CSR.

    Same shape contract as ``build_sliding_window_csr``: returns
    ``(row_ptr, col_idx, seq_lens)`` broadcast across (B, H).

    For Q-block ``qb``:
      - Off-diagonal SW range (block ids): ``[k_min_sw, qb)`` where
        ``k_min_sw = max(0, (qb*BS - window + 1)) // BS`` if positive
        else 0.
      - PLUS sink K-blocks: ``[0, sinks_blocks)`` with
        ``sinks_blocks = ceil(sinks / BS)``.
      - PLUS diagonal placeholder ``qb`` (always the last entry).

    The union is deduplicated and sorted ascending; the diagonal is
    appended last per the kernel's row-walker contract.

    Edge cases:
      - sinks <= 0 OR sinks_blocks == 0: degrades to plain SW.
      - sinks_blocks >= num_q_blocks: degrades to dense causal.
      - window covers the sinks (qb*BS < window): sinks already
        included; dedup makes this a no-op.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if sinks < 0:
        raise ValueError(f"sinks must be non-negative, got {sinks}")

    num_q_blocks = math.ceil(seq_len / block_size)
    sinks_blocks = math.ceil(sinks / block_size) if sinks > 0 else 0
    # Clip so we never claim a sink block past the sequence end.
    sinks_blocks = min(sinks_blocks, num_q_blocks)

    row_ptr_row = [0]
    col_idx_row: list[int] = []
    for q_block in range(num_q_blocks):
        # SW left-edge in K-block space (same logic as build_sliding_window_csr).
        left_bound = q_block * block_size - window + 1
        if left_bound <= 0:
            k_min_sw = 0
        else:
            k_min_sw = left_bound // block_size

        sw_range_end = q_block  # exclusive (diagonal excluded)
        sw_lo = min(k_min_sw, sw_range_end)

        # Sink block ids that fall strictly before the diagonal AND strictly
        # before the SW range start (anything inside SW is already covered).
        sink_lo = 0
        sink_hi = min(sinks_blocks, sw_lo)  # exclusive

        if sink_hi > sink_lo:
            col_idx_row.extend(range(sink_lo, sink_hi))
        col_idx_row.extend(range(sw_lo, sw_range_end))
        # Diagonal placeholder must be the last entry per kernel contract.
        col_idx_row.append(q_block)
        row_ptr_row.append(len(col_idx_row))

    return _broadcast_csr(row_ptr_row, col_idx_row, batch_size, num_heads, seq_len, device)
