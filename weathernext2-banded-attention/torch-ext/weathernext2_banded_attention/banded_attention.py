"""A Triton kernel for WeatherNext 2's block-tridiagonal mesh attention.

The mesh nodes are ordered by reverse Cuthill-McKee, so the k-hop adjacency is banded: a block of
`block_size` consecutive nodes can only reach itself and its two neighbours. The PyTorch path spells
that out by materializing the three neighbouring key/value blocks with `gather_neighbouring_blocks`,
which triples the key/value traffic, and then handing a `[blocks, 1, block, 3 * block]` mask to
`scaled_dot_product_attention`.

This kernel reads the three neighbours straight out of the ungathered key/value tensors instead, so
the 3x copy never happens, and streams the mask a tile at a time. Everything stays in float32 with
float32 accumulation, because upstream casts q, k and v to float32 before attention
(`sparse_transformer.py`, `upcast_attn_to_fp32`) and the released configs all set it.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from .utils import device_context


# Default follows the compiler's float32 dot precision; IEEE requests full float32 multiplication.
PRECISION_DEFAULT = 0
PRECISION_IEEE = 1
_PRECISIONS = {"default": PRECISION_DEFAULT, "ieee": PRECISION_IEEE}


class PreparedMask(NamedTuple):
    packed: torch.Tensor
    tiles: torch.Tensor
    offsets: torch.Tensor
    shape: tuple[int, int, int]


@triton.jit
def _pack_active_mask_kernel(
    mask_ptr,
    tiles_ptr,
    offsets_ptr,
    compact_tiles_ptr,
    out_ptr,
    stride_b,
    stride_m,
    stride_n,
    block_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    block = tl.program_id(1)
    neighbour = tl.program_id(2)
    slot = (block * tl.cdiv(block_size, BLOCK_M) + tile) * 3 + neighbour
    key_tiles = tl.cdiv(block_size, BLOCK_N)
    offset = tl.load(offsets_ptr + slot)
    count = tl.load(offsets_ptr + slot + 1) - offset
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    bits = tl.arange(0, BLOCK_N)
    for index in range(count):
        key_tile = tl.load(tiles_ptr + slot * key_tiles + index)
        tl.store(compact_tiles_ptr + offset + index, key_tile)
        columns = key_tile * BLOCK_N + bits
        keep = tl.load(
            mask_ptr
            + block * stride_b
            + rows[:, None] * stride_m
            + (neighbour * block_size + columns[None, :]) * stride_n,
            mask=(rows[:, None] < block_size) & (columns[None, :] < block_size),
            other=0,
        )
        word = tl.sum(keep.to(tl.uint32) << bits[None, :], axis=1)
        tl.store(out_ptr + (offset + index) * BLOCK_M + tl.arange(0, BLOCK_M), word)


@triton.jit
def _active_tiles_kernel(
    mask_ptr,
    tiles_ptr,
    counts_ptr,
    stride_mb,
    stride_mm,
    stride_mn,
    block_size,
    num_blocks,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    block = tl.program_id(1)
    neighbour = tl.program_id(2)
    key_tiles = tl.cdiv(block_size, BLOCK_N)
    query_tiles = tl.cdiv(block_size, BLOCK_M)
    slot = (block * query_tiles + tile) * 3 + neighbour
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    count = 0
    source = block + neighbour - 1
    if (source >= 0) and (source < num_blocks):
        for index in range(key_tiles):
            columns = index * BLOCK_N + tl.arange(0, BLOCK_N)
            keep = tl.load(
                mask_ptr
                + block * stride_mb
                + rows[:, None] * stride_mm
                + (neighbour * block_size + columns[None, :]) * stride_mn,
                mask=(rows[:, None] < block_size) & (columns[None, :] < block_size),
                other=0,
            )
            if tl.sum(keep.to(tl.int32)) > 0:
                tl.store(tiles_ptr + slot * key_tiles + count, index)
                count += 1
    tl.store(counts_ptr + slot, count)


@triton.jit
def _banded_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    out_ptr,
    tiles_ptr,
    offsets_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    num_blocks,
    block_size,
    scaling,
    HEAD_DIM: tl.constexpr,
    PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One program per (query tile, mesh block, head), looping over the three neighbour blocks."""
    tile = tl.program_id(0)
    flat_block = tl.program_id(1)  # batch * num_blocks + block
    head = tl.program_id(2)

    block_index = flat_block % num_blocks
    rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    row_valid = rows < block_size

    query_base = query_ptr + flat_block * stride_qb + head * stride_qh
    query = tl.load(
        query_base + rows[:, None] * stride_qm + dims[None, :] * stride_qd,
        mask=row_valid[:, None],
        other=0.0,
    )

    # Running softmax, flash style: one pass over the band, rescaling as the maximum moves.
    running_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for neighbour in range(3):
        source = block_index + neighbour - 1
        if (source >= 0) and (source < num_blocks):
            source_flat = flat_block - block_index + source
            key_base = key_ptr + source_flat * stride_kb + head * stride_kh
            value_base = value_ptr + source_flat * stride_vb + head * stride_vh
            slot = (block_index * tl.cdiv(block_size, BLOCK_M) + tile) * 3 + neighbour
            offset = tl.load(offsets_ptr + slot)
            end = tl.load(offsets_ptr + slot + 1)
            for index in range(offset, end):
                start = tl.load(tiles_ptr + index) * BLOCK_N
                columns = start + tl.arange(0, BLOCK_N)
                column_valid = columns < block_size
                word = tl.load(mask_ptr + index * BLOCK_M + tl.arange(0, BLOCK_M))
                keep = ((word[:, None] >> tl.arange(0, BLOCK_N)[None, :]) & 1).to(tl.int1)
                key = tl.load(
                    key_base + columns[:, None] * stride_km + dims[None, :] * stride_kd,
                    mask=column_valid[:, None],
                    other=0.0,
                )
                if PRECISION == 0:
                    logits = tl.dot(query, tl.trans(key))
                else:
                    logits = tl.dot(query, tl.trans(key), input_precision="ieee")
                # Match Faster-WeatherNext's QK dot followed by scaling.
                logits = tl.where(keep, logits * scaling, -1e30)
                tile_max = tl.max(logits, axis=1)
                new_max = tl.maximum(running_max, tile_max)
                weights = tl.where(keep, tl.exp(logits - new_max[:, None]), 0.0)
                rescale = tl.exp(running_max - new_max)
                running_sum = rescale * running_sum + tl.sum(weights, axis=1)
                running_max = new_max
                value = tl.load(
                    value_base + columns[:, None] * stride_vm + dims[None, :] * stride_vd,
                    mask=column_valid[:, None],
                    other=0.0,
                )
                weights = weights.to(value.dtype)
                if PRECISION == 0:
                    contribution = tl.dot(weights, value)
                else:
                    contribution = tl.dot(weights, value, input_precision="ieee")
                accumulator = rescale[:, None] * accumulator + contribution

    accumulator = accumulator / tl.maximum(running_sum, 1e-30)[:, None]
    out_base = out_ptr + flat_block * stride_ob + head * stride_oh
    tl.store(
        out_base + rows[:, None] * stride_om + dims[None, :] * stride_od,
        accumulator,
        mask=row_valid[:, None],
    )


def _prepare_mask(mask):
    blocks, block_size, _ = mask.shape
    query_tiles, key_tiles = triton.cdiv(block_size, 64), triton.cdiv(block_size, 32)
    counts = torch.empty((blocks, query_tiles, 3), dtype=torch.int32, device=mask.device)
    tiles = torch.empty((*counts.shape, key_tiles), dtype=torch.int32, device=mask.device)
    with device_context(mask.device):
        _active_tiles_kernel[(query_tiles, blocks, 3)](
            mask, tiles, counts, *mask.stride(), block_size, blocks, BLOCK_M=64, BLOCK_N=32, num_warps=4
        )
        offsets = torch.cat((counts.new_zeros(1), counts.flatten().cumsum(0, dtype=torch.int32)))
        active_tiles = int(offsets[-1].item())
        packed = torch.empty((active_tiles, 64), dtype=torch.uint32, device=mask.device)
        compact_tiles = torch.empty((active_tiles,), dtype=torch.int32, device=mask.device)
        _pack_active_mask_kernel[(query_tiles, blocks, 3)](
            mask,
            tiles,
            offsets,
            compact_tiles,
            packed,
            *mask.stride(),
            block_size,
            BLOCK_M=64,
            BLOCK_N=32,
            num_warps=4,
        )
    return PreparedMask(packed, compact_tiles, offsets, tuple(mask.shape))


def banded_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | PreparedMask,
    scaling: float,
    precision: str = "default",
) -> torch.Tensor:
    """Attention over the three block-diagonals of the mesh adjacency.

    Args:
        query, key, value: `[batch, num_blocks, heads, block_size, head_dim]`, float32. Note that
            key and value are *not* gathered over neighbours; the kernel walks them itself.
        mask: `[num_blocks, block_size, 3 * block_size]`, bool, or a shared PreparedMask.
        scaling: the usual `head_dim ** -0.5`.
        precision: how `tl.dot` treats the float32 inputs. `"default"` uses Triton's default for the
            active backend; `"ieee"` forces true float32 for strict numerical comparisons.

    Returns:
        `[batch, num_blocks, heads, block_size, head_dim]`.
    """
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(_PRECISIONS)}, got {precision!r}")
    batch, num_blocks, heads, block_size, head_dim = query.shape
    # `HEAD_DIM` is a `tl.constexpr` tile width and the loads along it are not masked, so a value
    # Triton cannot tile reads past the end of every row. That is silent, so reject it up front.
    # `tl.dot` also needs at least 16 along its reduction dimension.
    if head_dim < 16 or (head_dim & (head_dim - 1)) != 0:
        raise ValueError(
            f"head_dim must be a power of two and at least 16, got {head_dim}. WeatherNext 2's "
            "released checkpoints use 128."
        )
    if mask.shape != (num_blocks, block_size, 3 * block_size):
        raise ValueError(f"mask is {tuple(mask.shape)}, expected {(num_blocks, block_size, 3 * block_size)}")

    query, key, value = (t.reshape(batch * num_blocks, heads, block_size, head_dim) for t in (query, key, value))
    out = torch.empty(query.shape, dtype=query.dtype, device=query.device)

    def grid(meta):
        return (triton.cdiv(block_size, meta["BLOCK_M"]), batch * num_blocks, heads)

    # Sharded layers must launch on their tensors' device, not the current device.
    with device_context(query.device):
        prepared = mask if isinstance(mask, PreparedMask) else _prepare_mask(mask)
        _banded_attention_kernel[grid](
            query,
            key,
            value,
            prepared.packed,
            out,
            prepared.tiles,
            prepared.offsets,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            num_blocks,
            block_size,
            scaling,
            HEAD_DIM=head_dim,
            PRECISION=_PRECISIONS[precision],
            BLOCK_M=64,
            BLOCK_N=32,
            num_warps=4,
            num_stages=1,
        )
    return out.reshape(batch, num_blocks, heads, block_size, head_dim)
