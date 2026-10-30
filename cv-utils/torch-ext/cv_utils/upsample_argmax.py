import math
from collections.abc import Sequence
from functools import lru_cache
from itertools import accumulate

import torch
import triton
import triton.language as tl

from .resize import MAX_PLANS, _as_tensors


BLOCK = 256
DOT_TILE = (32, 64)
DOT_MIN_SCALE = 2


@triton.jit
def _tile_image(
    tile_ends, images, height, width, out_height, out_width, BLOCK_IMAGES: tl.constexpr, UNIFORM: tl.constexpr
):
    """Image of the program's tile, its tile index in that image, and the image layout.

    With `UNIFORM`, every image has the output size `out_height x out_width` and the grid has one row per image.
    Otherwise the grid is flat, the image is the number of images whose last tile is before this one, and its layout
    is read from `images`."""
    if UNIFORM:
        tile = tl.program_id(0)
        image = tl.program_id(1).to(tl.int64)
        out_start = image * out_height * out_width
    else:
        index = tl.sum((tl.load(tile_ends + tl.arange(0, BLOCK_IMAGES)) <= tl.program_id(0)).to(tl.int32), axis=0)
        tile = tl.program_id(0) - tl.load(tile_ends + index - 1, mask=index > 0, other=0)
        out_height = tl.load(images + index * 4).to(tl.int32)
        out_width = tl.load(images + index * 4 + 1).to(tl.int32)
        out_start = tl.load(images + index * 4 + 2)
        image = tl.load(images + index * 4 + 3)
    height_scale = height.to(tl.float32) / out_height.to(tl.float32)
    width_scale = width.to(tl.float32) / out_width.to(tl.float32)
    return tile, image, out_height, out_width, out_start, height_scale, width_scale


@triton.jit(do_not_specialize=["out_height", "out_width"])
def _upsample_argmax_kernel(
    logits,
    output,
    tile_ends,
    images,
    out_height,
    out_width,
    classes,
    height,
    width,
    BLOCK: tl.constexpr,
    BLOCK_IMAGES: tl.constexpr,
    UNIFORM: tl.constexpr,
):
    """Bilinear upsample + argmax, `BLOCK` output pixels per program, one program per tile of every image.

    For each class, every pixel reads its 4 neighbors in the logits and keeps the best class so far."""
    tile, image, out_height, out_width, out_start, height_scale, width_scale = _tile_image(
        tile_ends, images, height, width, out_height, out_width, BLOCK_IMAGES, UNIFORM
    )
    pixels = tile * BLOCK + tl.arange(0, BLOCK)
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
    tl.store(output + out_start + pixels, best_class, mask=active)


@triton.jit(do_not_specialize=["out_height", "out_width"])
def _dot_upsample_argmax_kernel(
    logits,
    output,
    tile_ends,
    images,
    out_height,
    out_width,
    classes,
    height,
    width,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PATCH_H: tl.constexpr,
    PATCH_W: tl.constexpr,
    BLOCK_IMAGES: tl.constexpr,
    UNIFORM: tl.constexpr,
):
    """Bilinear upsample + argmax with tensor cores.

    Each program computes a `BLOCK_M x BLOCK_N` output tile from a `PATCH_H x PATCH_W` patch of the logits. For each
    class, the tile is `row_matrix @ patch @ column_matrix` in float16, where both matrices hold the bilinear weights.
    """
    tile, image, out_height, out_width, out_start, height_scale, width_scale = _tile_image(
        tile_ends, images, height, width, out_height, out_width, BLOCK_IMAGES, UNIFORM
    )
    tiles_per_row = tl.cdiv(out_width, BLOCK_N)
    rows = (tile // tiles_per_row) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = (tile % tiles_per_row) * BLOCK_N + tl.arange(0, BLOCK_N)
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
    row_matrix = tl.where(
        patch_rows[None, :] == (row0 - first_row)[:, None],
        1.0 - row_weight1[:, None],
        0.0,
    )
    row_matrix += tl.where(patch_rows[None, :] == (row1 - first_row)[:, None], row_weight1[:, None], 0.0)
    column_matrix = tl.where(
        patch_columns[:, None] == (column0 - first_column)[None, :],
        1.0 - column_weight1[None, :],
        0.0,
    )
    column_matrix += tl.where(
        patch_columns[:, None] == (column1 - first_column)[None, :],
        column_weight1[None, :],
        0.0,
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
    output_offsets = out_start + rows[:, None] * out_width + columns[None, :]
    tl.store(output + output_offsets, best_class.to(tl.int64), mask=output_mask)


def _patch_side(block, in_length, out_length):
    return max(16, triton.next_power_of_2(math.ceil(block * in_length / out_length) + 3))


def _uses_tensor_cores(height, width, out_height, out_width):
    return out_height >= DOT_MIN_SCALE * height and out_width >= DOT_MIN_SCALE * width


@lru_cache(maxsize=MAX_PLANS)
def _upsample_argmax_plan(device, height, width, sizes):
    """One launch per kernel path for the whole batch. Each launch gets the last tile of every image and the image
    layouts in one transfer, so a program finds its image without a launch per image. A batch with one output size
    needs no metadata on the device: its grid has one row per image and the output size is a kernel argument."""
    uniform = len(set(sizes)) == 1
    out_starts = list(accumulate([0] + [out_height * out_width for out_height, out_width in sizes[:-1]]))
    launches = []
    for tensor_cores in (False, True):
        indices = [
            index for index, size in enumerate(sizes) if _uses_tensor_cores(height, width, *size) == tensor_cores
        ]
        if not indices:
            continue
        if tensor_cores:
            tile_counts = [
                triton.cdiv(sizes[index][0], DOT_TILE[0]) * triton.cdiv(sizes[index][1], DOT_TILE[1])
                for index in indices
            ]
        else:
            tile_counts = [triton.cdiv(sizes[index][0] * sizes[index][1], BLOCK) for index in indices]
        if uniform:
            block_images = 1
            arguments = (None, None, *sizes[0])
            grid = (tile_counts[0], len(sizes))
        else:
            block_images = triton.next_power_of_2(len(indices) + 1)
            tile_ends = list(accumulate(tile_counts)) + [2**62] * (block_images - len(indices))
            images = [value for index in indices for value in (*sizes[index], out_starts[index], index)]
            arguments = (*_as_tensors(device, torch.int64, tile_ends, images), 0, 0)
            grid = (sum(tile_counts),)
        constants = {"BLOCK_IMAGES": block_images, "UNIFORM": uniform}
        if tensor_cores:
            constants.update(
                BLOCK_M=DOT_TILE[0],
                BLOCK_N=DOT_TILE[1],
                PATCH_H=max(_patch_side(DOT_TILE[0], height, sizes[index][0]) for index in indices),
                PATCH_W=max(_patch_side(DOT_TILE[1], width, sizes[index][1]) for index in indices),
            )
        else:
            constants["BLOCK"] = BLOCK
        kernel = _dot_upsample_argmax_kernel if tensor_cores else _upsample_argmax_kernel
        launches.append((kernel, grid, arguments, constants))
    return launches


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
    _, classes, height, width = logits.shape
    sizes = tuple(tuple(int(side) for side in size) for size in sizes)
    output = torch.empty(
        sum(out_height * out_width for out_height, out_width in sizes),
        dtype=torch.int64,
        device=logits.device,
    )
    with torch.cuda.device(logits.device):
        for kernel, grid, arguments, constants in _upsample_argmax_plan(logits.device, height, width, sizes):
            kernel[grid](logits, output, *arguments, classes, height, width, **constants)
    return [
        chunk.view(size)
        for chunk, size in zip(
            output.split([out_height * out_width for out_height, out_width in sizes]),
            sizes,
        )
    ]
