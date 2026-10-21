import math
from collections.abc import Sequence

import torch
import triton
import triton.language as tl


BLOCK = 256
DOT_TILE = (32, 64)
DOT_MIN_SCALE = 2


@triton.jit
def _upsample_argmax_kernel(
    logits,
    output,
    classes,
    height,
    width,
    out_height,
    out_width,
    height_scale,
    width_scale,
    BLOCK: tl.constexpr,
):
    """Bilinear upsample + argmax, `BLOCK` output pixels per program.

    For each class, every pixel reads its 4 neighbors in the logits and keeps the best class so far."""
    image = tl.program_id(1).to(tl.int64)
    pixels = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = pixels < out_height * out_width
    rows = pixels // out_width
    columns = pixels % out_width
    source_row = tl.maximum((rows.to(tl.float32) + 0.5) * height_scale - 0.5, 0.0)
    source_column = tl.maximum((columns.to(tl.float32) + 0.5) * width_scale - 0.5, 0.0)
    row0 = source_row.to(tl.int32)
    column0 = source_column.to(tl.int32)
    row1 = tl.minimum(row0 + 1, height - 1)
    column1 = tl.minimum(column0 + 1, width - 1)
    row_weight1 = source_row - row0.to(tl.float32)
    column_weight1 = source_column - column0.to(tl.float32)
    row_weight0 = 1.0 - row_weight1
    column_weight0 = 1.0 - column_weight1
    offset00 = row0 * width + column0
    offset01 = row0 * width + column1
    offset10 = row1 * width + column0
    offset11 = row1 * width + column1
    plane = height.to(tl.int64) * width
    channel = logits + image * classes * plane
    best = tl.full([BLOCK], float("-inf"), tl.float32)
    best_class = tl.zeros([BLOCK], tl.int64)
    for class_index in range(classes):
        top = column_weight0 * tl.load(channel + offset00, mask=active).to(tl.float32) + column_weight1 * tl.load(
            channel + offset01, mask=active
        ).to(tl.float32)
        bottom = column_weight0 * tl.load(channel + offset10, mask=active).to(tl.float32) + column_weight1 * tl.load(
            channel + offset11, mask=active
        ).to(tl.float32)
        value = row_weight0 * top + row_weight1 * bottom
        better = value > best
        best = tl.where(better, value, best)
        best_class = tl.where(better, class_index, best_class)
        channel += plane
    tl.store(output + image * out_height * out_width + pixels, best_class, mask=active)


@triton.jit
def _dot_upsample_argmax_kernel(
    logits,
    output,
    classes,
    height,
    width,
    out_height,
    out_width,
    height_scale,
    width_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PATCH_H: tl.constexpr,
    PATCH_W: tl.constexpr,
):
    """Bilinear upsample + argmax with tensor cores.

    Each program computes a `BLOCK_M x BLOCK_N` output tile from a `PATCH_H x PATCH_W` patch of the logits. For each
    class, the tile is `row_matrix @ patch @ column_matrix` in float16, where both matrices hold the bilinear weights.
    """
    image = tl.program_id(2).to(tl.int64)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    source_row = tl.maximum((rows.to(tl.float32) + 0.5) * height_scale - 0.5, 0.0)
    source_column = tl.maximum((columns.to(tl.float32) + 0.5) * width_scale - 0.5, 0.0)
    row0 = source_row.to(tl.int32)
    column0 = source_column.to(tl.int32)
    row1 = tl.minimum(row0 + 1, height - 1)
    column1 = tl.minimum(column0 + 1, width - 1)
    row_weight1 = source_row - row0.to(tl.float32)
    column_weight1 = source_column - column0.to(tl.float32)
    first_row = tl.min(row0, axis=0)
    first_column = tl.min(column0, axis=0)
    patch_rows = tl.arange(0, PATCH_H)
    patch_columns = tl.arange(0, PATCH_W)
    row_matrix = tl.where(patch_rows[None, :] == (row0 - first_row)[:, None], 1.0 - row_weight1[:, None], 0.0)
    row_matrix += tl.where(patch_rows[None, :] == (row1 - first_row)[:, None], row_weight1[:, None], 0.0)
    column_matrix = tl.where(
        patch_columns[:, None] == (column0 - first_column)[None, :], 1.0 - column_weight1[None, :], 0.0
    )
    column_matrix += tl.where(
        patch_columns[:, None] == (column1 - first_column)[None, :], column_weight1[None, :], 0.0
    )
    row_matrix = row_matrix.to(tl.float16)
    column_matrix = column_matrix.to(tl.float16)
    source_rows = first_row + patch_rows
    source_columns = first_column + patch_columns
    patch_offsets = source_rows[:, None] * width + source_columns[None, :]
    patch_mask = (source_rows[:, None] < height) & (source_columns[None, :] < width)
    plane = height.to(tl.int64) * width
    channel = logits + image * classes * plane
    best = tl.full([BLOCK_M, BLOCK_N], float("-inf"), tl.float32)
    best_class = tl.zeros([BLOCK_M, BLOCK_N], tl.int32)
    for class_index in range(classes):
        patch = tl.load(channel + patch_offsets, mask=patch_mask, other=0.0).to(tl.float16)
        horizontal = tl.dot(patch, column_matrix)
        value = tl.dot(row_matrix, horizontal.to(tl.float16))
        better = value > best
        best = tl.where(better, value, best)
        best_class = tl.where(better, class_index, best_class)
        channel += plane
    output_mask = (rows[:, None] < out_height) & (columns[None, :] < out_width)
    output_offsets = image * out_height * out_width + rows[:, None] * out_width + columns[None, :]
    tl.store(output + output_offsets, best_class.to(tl.int64), mask=output_mask)


def _patch_side(block, in_length, out_length):
    return max(16, triton.next_power_of_2(math.ceil(block * in_length / out_length) + 3))


def _launch(logits, out_height, out_width):
    batch, classes, height, width = logits.shape
    output = torch.empty(batch, out_height, out_width, dtype=torch.int64, device=logits.device)
    if out_height >= DOT_MIN_SCALE * height and out_width >= DOT_MIN_SCALE * width:
        block_m, block_n = DOT_TILE
        grid = (triton.cdiv(out_width, block_n), triton.cdiv(out_height, block_m), batch)
        _dot_upsample_argmax_kernel[grid](
            logits,
            output,
            classes,
            height,
            width,
            out_height,
            out_width,
            height / out_height,
            width / out_width,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            PATCH_H=_patch_side(block_m, height, out_height),
            PATCH_W=_patch_side(block_n, width, out_width),
        )
        return output
    grid = (triton.cdiv(out_height * out_width, BLOCK), batch)
    _upsample_argmax_kernel[grid](
        logits,
        output,
        classes,
        height,
        width,
        out_height,
        out_width,
        height / out_height,
        width / out_width,
        BLOCK=BLOCK,
    )
    return output


def upsample_argmax(logits: torch.Tensor, sizes: Sequence[tuple[int, int]]) -> list[torch.Tensor]:
    """Class of every pixel of `logits` `(B, C, h, w)` bilinearly upsampled to `sizes[i]`, without the upsampled logits.

    Matches `F.interpolate(logits[i:i + 1], size=sizes[i], mode="bilinear", align_corners=False)[0].argmax(0)`.
    When both sides are upsampled at least `DOT_MIN_SCALE` times, the interpolation runs on tensor cores in float16
    and a few pixels at near-ties can change class; otherwise it runs in float32. Returns one int64 `(H, W)` tensor
    per image.
    """
    if logits.ndim != 4 or not logits.is_cuda:
        raise ValueError(f"Expected (B, C, h, w) CUDA logits, got shape {tuple(logits.shape)} on {logits.device}.")
    if len(sizes) != len(logits):
        raise ValueError(f"Pass one size per image, got {len(sizes)} sizes for {len(logits)} images.")
    logits = logits.contiguous()
    sizes = [tuple(int(side) for side in size) for size in sizes]
    with torch.cuda.device(logits.device):
        if len(set(sizes)) == 1:
            return list(_launch(logits, *sizes[0]).unbind(0))
        return [_launch(logits[index : index + 1], *size)[0] for index, size in enumerate(sizes)]
