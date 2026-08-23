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

import torch
import triton
import triton.language as tl

from .utils import device_context


# Let Triton choose its backend default unless strict IEEE arithmetic was requested. The default is
# TF32 on supported NVIDIA GPUs and IEEE on AMD, so the same source remains portable across both.
PRECISION_DEFAULT = 0
PRECISION_IEEE = 1
_PRECISIONS = {"default": PRECISION_DEFAULT, "ieee": PRECISION_IEEE}


def _is_hip() -> bool:
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


def _configs():
    """Tile shapes to sweep. AMD wants fewer stages, since LDS is smaller than Hopper's SMEM."""
    stages = (1, 2) if _is_hip() else (2, 3)
    return [
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_warps=w, num_stages=s)
        for m in (64, 128)
        for n in (32, 64, 128)
        for w in (4, 8)
        for s in stages
    ]


@triton.autotune(configs=_configs(), key=["block_size", "HEAD_DIM"])
@triton.jit
def _banded_attention_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    mask_ptr,
    out_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_mb,
    stride_mm,
    stride_mn,
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
    query = query * scaling

    # Running softmax, flash style: one pass over the band, rescaling as the maximum moves.
    running_max = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for neighbour in range(3):
        source = block_index + neighbour - 1
        # Blocks off either end have no neighbour there. The PyTorch path zero-pads them and relies
        # on the mask being False; skipping them outright is the same answer without the traffic.
        if (source >= 0) and (source < num_blocks):
            source_flat = flat_block - block_index + source
            key_base = key_ptr + source_flat * stride_qb + head * stride_qh
            value_base = value_ptr + source_flat * stride_qb + head * stride_qh

            for start in range(0, block_size, BLOCK_N):
                columns = start + tl.arange(0, BLOCK_N)
                column_valid = columns < block_size

                # The mask spans three blocks side by side, so the neighbour picks the third. Read it
                # first: the band is sparse, and a tile nothing reaches costs two matmuls to compute
                # and then throw away.
                mask_columns = neighbour * block_size + columns
                keep = tl.load(
                    mask_ptr
                    + block_index * stride_mb
                    + rows[:, None] * stride_mm
                    + mask_columns[None, :] * stride_mn,
                    mask=row_valid[:, None] & column_valid[None, :],
                    other=0,
                ).to(tl.int1)

                if tl.sum(keep.to(tl.int32)) > 0:
                    key = tl.load(
                        key_base + columns[:, None] * stride_qm + dims[None, :] * stride_qd,
                        mask=column_valid[:, None],
                        other=0.0,
                    )
                    if PRECISION == 0:
                        logits = tl.dot(query, tl.trans(key))
                    else:
                        logits = tl.dot(query, tl.trans(key), input_precision="ieee")
                    logits = tl.where(keep, logits, float("-inf"))

                    tile_max = tl.max(logits, axis=1)
                    new_max = tl.maximum(running_max, tile_max)
                    # A row that has seen nothing yet stays at -inf; guard so it contributes zero.
                    safe_max = tl.where(new_max == float("-inf"), 0.0, new_max)
                    weights = tl.exp(logits - safe_max[:, None])
                    weights = tl.where(keep, weights, 0.0)

                    rescale = tl.exp(tl.where(running_max == float("-inf"), 0.0, running_max) - safe_max)
                    rescale = tl.where(running_max == float("-inf"), 0.0, rescale)
                    running_sum = running_sum * rescale + tl.sum(weights, axis=1)
                    accumulator = accumulator * rescale[:, None]

                    value = tl.load(
                        value_base + columns[:, None] * stride_qm + dims[None, :] * stride_qd,
                        mask=column_valid[:, None],
                        other=0.0,
                    )
                    # The accumulator-passing form of `tl.dot` folds the running sum into the matmul.
                    weights = weights.to(value.dtype)
                    if PRECISION == 0:
                        accumulator = tl.dot(weights, value, accumulator)
                    else:
                        accumulator = tl.dot(weights, value, accumulator, input_precision="ieee")
                    running_max = new_max

    accumulator = accumulator / tl.where(running_sum == 0.0, 1.0, running_sum)[:, None]
    out_base = out_ptr + flat_block * stride_qb + head * stride_qh
    tl.store(
        out_base + rows[:, None] * stride_qm + dims[None, :] * stride_qd,
        accumulator,
        mask=row_valid[:, None],
    )


def banded_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
    scaling: float,
    precision: str = "default",
) -> torch.Tensor:
    """Attention over the three block-diagonals of the mesh adjacency.

    Args:
        query, key, value: `[batch, num_blocks, heads, block_size, head_dim]`, float32. Note that
            key and value are *not* gathered over neighbours; the kernel walks them itself.
        mask: `[num_blocks, block_size, 3 * block_size]`, bool.
        scaling: the usual `head_dim ** -0.5`.
        precision: how `tl.dot` treats the float32 inputs. `"default"` uses Triton's default for the
            active backend; `"ieee"` forces true float32 for strict numerical comparisons.

    Returns:
        `[batch, num_blocks, heads, block_size, head_dim]`.
    """
    if precision not in _PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(_PRECISIONS)}, got {precision!r}")
    batch, num_blocks, heads, block_size, head_dim = query.shape
    if mask.shape != (num_blocks, block_size, 3 * block_size):
        raise ValueError(f"mask is {tuple(mask.shape)}, expected {(num_blocks, block_size, 3 * block_size)}")

    query, key, value = (
        t.reshape(batch * num_blocks, heads, block_size, head_dim) for t in (query, key, value)
    )
    query, key, value = (t.contiguous() for t in (query, key, value))
    mask = mask.contiguous()
    out = torch.empty_like(query)

    def grid(meta):
        return (triton.cdiv(block_size, meta["BLOCK_M"]), batch * num_blocks, heads)

    # Triton launches on whichever device is current, not on the one the tensors live on, so a
    # shard placed on cuda:1 by `device_map="auto"` would otherwise be launched against cuda:0.
    with device_context(query.device):
        _banded_attention_kernel[grid](
            query,
            key,
            value,
            mask,
            out,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            num_blocks,
            block_size,
            scaling,
            HEAD_DIM=head_dim,
            PRECISION=_PRECISIONS[precision],
        )
    return out.reshape(batch, num_blocks, heads, block_size, head_dim)
