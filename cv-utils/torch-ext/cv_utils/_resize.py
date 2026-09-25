import math
from functools import lru_cache
from itertools import accumulate

import torch
import triton
import triton.language as tl


HORIZONTAL_TILE = (16, 64)
VERTICAL_TILE = (4, 256)
TABLE_BLOCK = 256


@triton.jit
def _tap_count(scale, CUBIC: tl.constexpr, ANTIALIAS: tl.constexpr):
    filter_scale = tl.maximum(scale, 1.0) if ANTIALIAS else 1.0
    return (tl.ceil((2.0 if CUBIC else 1.0) * filter_scale) * 2 + 1).to(tl.int32)


@triton.jit
def _filter_table_kernel(
    weights,
    first_taps,
    in_sizes,
    resize_sizes,
    crop_starts,
    out_sizes,
    table_offsets,
    taps_stride,
    cubic_coeff,
    BLOCK: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
):
    item = tl.program_id(0)
    out_size = tl.load(out_sizes + item)
    if tl.program_id(1) * BLOCK >= out_size:
        return
    positions = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = positions < out_size
    size = tl.load(in_sizes + item)
    scale = size.to(tl.float32) / tl.load(resize_sizes + item).to(tl.float32)
    filter_scale = tl.maximum(scale, 1.0) if ANTIALIAS else 1.0
    center = scale * ((positions + tl.load(crop_starts + item)).to(tl.float32) + 0.5)
    first_tap = tl.floor(center - (2.0 if CUBIC else 1.0) * filter_scale + 0.5)
    taps = _tap_count(scale, CUBIC, ANTIALIAS)
    rows = (tl.load(table_offsets + item) + positions).to(tl.int64)
    weight_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for tap in range(taps):
        tap_position = first_tap + tap
        distance = tl.abs(tap_position - center + 0.5) / filter_scale
        if CUBIC:
            square = distance * distance
            cube = square * distance
            near = (cubic_coeff + 2.0) * cube - (cubic_coeff + 3.0) * square + 1.0
            far = cubic_coeff * (cube - 5.0 * square + 8.0 * distance - 4.0)
            weight = tl.where(distance <= 1.0, near, tl.where(distance < 2.0, far, 0.0))
        else:
            weight = tl.maximum(1.0 - distance, 0.0)
        if ANTIALIAS:
            weight = tl.where((tap_position >= 0.0) & (tap_position < size.to(tl.float32)), weight, 0.0)
        tl.store(weights + rows * taps_stride + tap, weight, mask=active)
        weight_sum += weight
    for tap in range(taps):
        weight = tl.load(weights + rows * taps_stride + tap, mask=active)
        tl.store(weights + rows * taps_stride + tap, weight / weight_sum, mask=active)
    tl.store(first_taps + rows, first_tap.to(tl.int32), mask=active)


@triton.jit
def _round(values, ROUND_TO_UINT8: tl.constexpr):
    if ROUND_TO_UINT8:
        values = tl.minimum(tl.maximum(tl.floor(values + 0.5), 0.0), 255.0)
    return values


@triton.jit
def _horizontal_kernel(
    frame_pointers,
    intermediate,
    intermediate_offsets,
    heights,
    widths,
    resize_widths,
    out_widths,
    weights,
    first_taps,
    table_offsets,
    taps_stride,
    CHANNELS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
):
    item = tl.program_id(0)
    plane_rows = CHANNELS * tl.load(heights + item)
    out_width = tl.load(out_widths + item)
    first_row = tl.program_id(1) * BLOCK_ROWS
    first_column = tl.program_id(2) * BLOCK_COLUMNS
    if first_row >= plane_rows or first_column >= out_width:
        return
    width = tl.load(widths + item)
    rows = (first_row + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    columns = first_column + tl.arange(0, BLOCK_COLUMNS)
    column_active = columns < out_width
    active = (rows < plane_rows)[:, None] & column_active[None, :]
    taps = _tap_count(width.to(tl.float32) / tl.load(resize_widths + item).to(tl.float32), CUBIC, ANTIALIAS)
    table_rows = (tl.load(table_offsets + item) + columns).to(tl.int64)
    first_tap = tl.load(first_taps + table_rows, mask=column_active, other=0)
    source = tl.load(frame_pointers + item).to(tl.pointer_type(tl.uint8)) + rows[:, None] * width
    total = tl.zeros([BLOCK_ROWS, BLOCK_COLUMNS], dtype=tl.float32)
    for tap in range(taps):
        weight = tl.load(weights + table_rows * taps_stride + tap, mask=column_active, other=0)
        index = tl.minimum(tl.maximum(first_tap + tap, 0), width - 1)
        total += weight[None, :] * tl.load(source + index[None, :], mask=active, other=0).to(tl.float32)
    destination = intermediate + tl.load(intermediate_offsets + item) + rows[:, None] * out_width + columns[None, :]
    tl.store(destination, _round(total, ROUND_TO_UINT8).to(intermediate.dtype.element_ty), mask=active)


@triton.jit
def _vertical_kernel(
    intermediate,
    output,
    intermediate_offsets,
    heights,
    resize_heights,
    weights,
    first_taps,
    table_offsets,
    taps_stride,
    means,
    stds,
    out_height,
    out_width,
    CHANNELS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
):
    item = tl.program_id(0)
    height = tl.load(heights + item)
    rows = tl.program_id(1) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    columns = tl.program_id(2) * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
    row_active = rows < out_height
    active = row_active[:, None] & (columns < out_width)[None, :]
    taps = _tap_count(height.to(tl.float32) / tl.load(resize_heights + item).to(tl.float32), CUBIC, ANTIALIAS)
    table_rows = (tl.load(table_offsets + item) + rows).to(tl.int64)
    first_tap = tl.load(first_taps + table_rows, mask=row_active, other=0)
    for channel in tl.static_range(CHANNELS):
        plane_start = tl.load(intermediate_offsets + item) + channel * height.to(tl.int64) * out_width
        source = intermediate + plane_start + columns[None, :]
        total = tl.zeros([BLOCK_ROWS, BLOCK_COLUMNS], dtype=tl.float32)
        for tap in range(taps):
            weight = tl.load(weights + table_rows * taps_stride + tap, mask=row_active, other=0)
            index = tl.minimum(tl.maximum(first_tap + tap, 0), height - 1).to(tl.int64)
            pixels = tl.load(source + index[:, None] * out_width, mask=active, other=0)
            total += weight[:, None] * pixels.to(tl.float32)
        total = (_round(total, ROUND_TO_UINT8) - tl.load(means + channel)) / tl.load(stds + channel)
        plane = (item * CHANNELS + channel).to(tl.int64) * out_height
        tl.store(output + (plane + rows)[:, None] * out_width + columns[None, :], total, mask=active)


def _as_tensors(device, dtype, *lists):
    """Copy several lists to `device` in one transfer, and return one tensor view per list."""
    lists = [list(values) for values in lists]
    packed = torch.tensor([value for values in lists for value in values], dtype=dtype, device=device)
    return packed.split([len(values) for values in lists])


def _tile(tile, out_width):
    """`tile` with its columns capped at the power of two above `out_width`, and the rows grown to keep its size."""
    rows, columns = tile
    capped_columns = min(columns, triton.next_power_of_2(out_width))
    return rows * columns // capped_columns, capped_columns


@lru_cache
def _normalization(mean, std, rescale, device):
    """Mean and std divided by `rescale`, so that `(x - mean') / std' == (x * rescale - mean) / std`."""
    return _as_tensors(device, torch.float32, [value / rescale for value in mean], [value / rescale for value in std])


def _horizontal_pass(
    frames, resize_widths, crop_lefts, out_widths, resize_heights, crop_tops, out_heights, cubic, antialias, round_to_uint8
):
    """Resize the width of every frame into one packed buffer of `(C, H, out_width)` planes, uint8 when rounding.

    Also returns the filter table of the vertical pass, built in the same launch as the horizontal one.
    """
    device = frames[0].device
    channels = frames[0].shape[0]
    frames = [frame.contiguous() for frame in frames]
    heights = [frame.shape[1] for frame in frames]
    widths = [frame.shape[2] for frame in frames]
    in_sizes, resize_sizes, out_sizes = widths + heights, resize_widths + resize_heights, out_widths + out_heights
    taps_stride = max(
        math.ceil((2 if cubic else 1) * (max(in_size / resize_size, 1.0) if antialias else 1.0)) * 2 + 1
        for in_size, resize_size in zip(in_sizes, resize_sizes)
    )
    (
        heights_tensor,
        widths_tensor,
        resize_widths_tensor,
        out_widths_tensor,
        resize_heights_tensor,
        in_sizes_tensor,
        resize_sizes_tensor,
        crop_starts_tensor,
        out_sizes_tensor,
        table_offsets,
    ) = _as_tensors(
        device,
        torch.int32,
        heights,
        widths,
        resize_widths,
        out_widths,
        resize_heights,
        in_sizes,
        resize_sizes,
        crop_lefts + crop_tops,
        out_sizes,
        accumulate([0] + out_sizes[:-1]),
    )
    intermediate_sizes = [channels * height * out_width for height, out_width in zip(heights, out_widths)]
    frame_pointers, intermediate_offsets = _as_tensors(
        device, torch.int64, [frame.data_ptr() for frame in frames], accumulate([0] + intermediate_sizes[:-1])
    )
    weights = torch.empty(sum(out_sizes) * taps_stride, device=device, dtype=torch.float32)
    first_taps = torch.empty(sum(out_sizes), device=device, dtype=torch.int32)
    _filter_table_kernel[(len(in_sizes), triton.cdiv(max(out_sizes), TABLE_BLOCK))](
        weights,
        first_taps,
        in_sizes_tensor,
        resize_sizes_tensor,
        crop_starts_tensor,
        out_sizes_tensor,
        table_offsets,
        taps_stride,
        -0.5 if antialias else -0.75,
        BLOCK=TABLE_BLOCK,
        CUBIC=cubic,
        ANTIALIAS=antialias,
    )
    intermediate_dtype = torch.uint8 if round_to_uint8 else torch.float32
    intermediate = torch.empty(sum(intermediate_sizes), device=device, dtype=intermediate_dtype)
    block_rows, block_columns = _tile(HORIZONTAL_TILE, max(out_widths))
    grid = (len(frames), triton.cdiv(channels * max(heights), block_rows), triton.cdiv(max(out_widths), block_columns))
    _horizontal_kernel[grid](
        frame_pointers,
        intermediate,
        intermediate_offsets,
        heights_tensor,
        widths_tensor,
        resize_widths_tensor,
        out_widths_tensor,
        weights,
        first_taps,
        table_offsets,
        taps_stride,
        CHANNELS=channels,
        BLOCK_ROWS=block_rows,
        BLOCK_COLUMNS=block_columns,
        CUBIC=cubic,
        ANTIALIAS=antialias,
        ROUND_TO_UINT8=round_to_uint8,
    )
    vertical_table = (weights, first_taps, table_offsets[len(frames) :], taps_stride)
    return intermediate, intermediate_offsets, heights_tensor, resize_heights_tensor, vertical_table


def resize_normalize(
    images,
    size,
    image_mean,
    image_std,
    rescale_factor,
    resample,
    antialias,
    crop_size=None,
    resize_mode="square",
    round_to_uint8=False,
):
    """Resize uint8 CHW images, center crop them to `crop_size`, rescale and normalize, into one `(N, C, H, W)` tensor.

    `resize_mode="square"` resizes every image to `size = (height, width)`. `resize_mode="shortest_edge"` resizes the
    short side to `size` and keeps the aspect ratio, and needs a `crop_size`.
    """
    shapes = [image.shape[1:] for image in images]
    if resize_mode == "shortest_edge":
        resize_sizes = [
            (size, int(width * size / height)) if height <= width else (int(height * size / width), size)
            for height, width in shapes
        ]
    else:
        resize_sizes = [tuple(size)] * len(images)
    out_height, out_width = crop_size if crop_size is not None else size
    cubic = resample == "bicubic"
    resize_heights = [height for height, _ in resize_sizes]
    intermediate, intermediate_offsets, heights, resize_heights_tensor, vertical_table = _horizontal_pass(
        images,
        [width for _, width in resize_sizes],
        [(width - out_width) // 2 for _, width in resize_sizes],
        [out_width] * len(images),
        resize_heights,
        [(height - out_height) // 2 for height in resize_heights],
        [out_height] * len(images),
        cubic,
        antialias,
        round_to_uint8,
    )
    device = images[0].device
    channels = images[0].shape[0]
    means, stds = _normalization(tuple(image_mean), tuple(image_std), rescale_factor, device)
    output = torch.empty((len(images), channels, out_height, out_width), device=device, dtype=torch.float32)
    block_rows, block_columns = _tile(VERTICAL_TILE, out_width)
    _vertical_kernel[(len(images), triton.cdiv(out_height, block_rows), triton.cdiv(out_width, block_columns))](
        intermediate,
        output,
        intermediate_offsets,
        heights,
        resize_heights_tensor,
        *vertical_table,
        means,
        stds,
        out_height,
        out_width,
        CHANNELS=channels,
        BLOCK_ROWS=block_rows,
        BLOCK_COLUMNS=block_columns,
        CUBIC=cubic,
        ANTIALIAS=antialias,
        ROUND_TO_UINT8=round_to_uint8,
    )
    return output
