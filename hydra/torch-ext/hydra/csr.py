"""CSR (Compressed Sparse Row) builders for the Blackwell FA kernel.

Convention
----------
The kernel iterates each Q-block's CSR row as ``ColIdx[ci_lo .. ci_hi)`` where
the **last entry** is the diagonal block (``col_idx[ci_hi - 1] == q_block_id``)
and the preceding entries are off-diagonal lower-triangular K-blocks (all
strictly less than ``q_block_id``, by the causal constraint).

The kernel applies causal masking + hi/lo precision split *within* the
diagonal block and a standard online-softmax merge for the off-diagonal
blocks. Builders MUST place the diagonal as the last entry per Q row;
otherwise the kernel reads the wrong block as "the one needing causal
masking" and produces silently-wrong attention outputs.

All builders return ``(row_ptr, col_idx, seq_lens)``:
    row_ptr:  (B, H, num_q_blocks + 1) int32 — CSR row pointers per (B, H).
    col_idx:  (B, H, total_nnz) int32     — CSR column indices.
    seq_lens: (B,)            int32       — per-batch sequence length.

The (B, H) broadcast is materialized as a contiguous expand so the kernel can
index without per-head pointer arithmetic.
"""
from __future__ import annotations

import math

import torch


def _broadcast_csr(
    row_ptr_row: list[int],
    col_idx_row: list[int],
    batch_size: int,
    num_heads: int,
    seq_len: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lift per-row Python lists to per-(B,H) contiguous int32 tensors."""
    row_ptr = (
        torch.tensor(row_ptr_row, device=device, dtype=torch.int32)
        .view(1, 1, -1)
        .expand(batch_size, num_heads, -1)
        .contiguous()
    )
    col_idx = (
        torch.tensor(col_idx_row, device=device, dtype=torch.int32)
        .view(1, 1, -1)
        .expand(batch_size, num_heads, -1)
        .contiguous()
    )
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    return row_ptr, col_idx, seq_lens


def build_dense_causal_csr(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    block_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense lower-triangular causal pattern.

    For Q-block i: off-diagonal entries enumerate K-blocks 0..i-1, then the
    diagonal placeholder i. nnz scales as O(num_q_blocks^2 / 2).
    """
    num_q_blocks = math.ceil(seq_len / block_size)
    row_ptr_row = [0]
    col_idx_row: list[int] = []
    for q_block in range(num_q_blocks):
        col_idx_row.extend(range(q_block))
        col_idx_row.append(q_block)
        row_ptr_row.append(len(col_idx_row))
    return _broadcast_csr(row_ptr_row, col_idx_row, batch_size, num_heads, seq_len, device)


def build_sliding_window_csr(
    window: int,
    seq_len: int,
    block_size: int,
    batch_size: int,
    num_heads: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Causal sliding-window pattern, widened for per-Q-token masking.

    ``window`` is measured in tokens (transformers / Mistral convention):
    Q-token at position p attends K-tokens [max(0, p - window + 1), p].

    The CSR is built using the **"some Q-token attends"** criterion — a
    K-block is included if ANY Q-token in the Q-block could attend ANY
    K-token in it. The kernel applies a per-Q-token in-window mask inside
    the off-diagonal-leftmost and diagonal blocks, so over-included
    K-tokens at the leftmost edge get masked to -inf and contribute zero.

    Concretely, for Q-block i, the leftmost-attended K-token across all
    Q-tokens in the block is ``max(0, i*BS - window + 1)``. The K-block
    containing that token is its position // BS.

    Edge cases:
        - window >= seq_len: degrades to dense_causal (full lower triangle).
        - window == 1: each Q-token only attends itself; diagonal-only with
          within-block causal mask.

    nnz scales as O(num_q_blocks * ceil(window / BS)) — linear in T for
    fixed window, which is the point of sliding window.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    num_q_blocks = math.ceil(seq_len / block_size)
    row_ptr_row = [0]
    col_idx_row: list[int] = []
    for q_block in range(num_q_blocks):
        # Smallest K-token position attended by any Q-token in this Q-block.
        # The leftmost Q-token (position q_block * BS) has the most reach to
        # the left; its window starts at q_block*BS - window + 1.
        left_bound = q_block * block_size - window + 1
        if left_bound <= 0:
            k_min_block = 0
        else:
            # Smallest n with (n+1)*BS - 1 >= left_bound, i.e., n = left_bound // BS.
            k_min_block = left_bound // block_size
        # Off-diagonal entries: k_min_block .. q_block - 1
        col_idx_row.extend(range(k_min_block, q_block))
        # Diagonal placeholder
        col_idx_row.append(q_block)
        row_ptr_row.append(len(col_idx_row))
    return _broadcast_csr(row_ptr_row, col_idx_row, batch_size, num_heads, seq_len, device)
