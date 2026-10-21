import math
from collections.abc import Sequence
from functools import lru_cache
from itertools import accumulate

import torch
import triton
import triton.language as tl


HORIZONTAL_TILE = (64, 32)
HORIZONTAL_WARPS = 8
VERTICAL_TILE = (4, 256)
TABLE_BLOCK = 256
ROW_ALIGNMENT = 16
MAX_PLANS = 1024


@triton.jit
def _tap_count(scale, CUBIC: tl.constexpr, ANTIALIAS: tl.constexpr):
    filter_scale = tl.maximum(scale, 1.0) if ANTIALIAS else 1.0
    return (tl.ceil((2.0 if CUBIC else 1.0) * filter_scale) * 2 + 1).to(tl.int32)


@triton.jit
def _filter_table_kernel(
    weight_pointers,
    first_tap_pointers,
    in_sizes,
    resize_sizes,
    crop_starts,
    out_sizes,
    cubic_coeff,
    BLOCK: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
):
    table = tl.program_id(0)
    out_size = tl.load(out_sizes + table)
    if tl.program_id(1) * BLOCK >= out_size:
        return
    positions = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = positions < out_size
    size = tl.load(in_sizes + table)
    scale = size.to(tl.float32) / tl.load(resize_sizes + table).to(tl.float32)
    filter_scale = tl.maximum(scale, 1.0) if ANTIALIAS else 1.0
    center = scale * ((positions + tl.load(crop_starts + table)).to(tl.float32) + 0.5)
    first_tap = tl.floor(center - (2.0 if CUBIC else 1.0) * filter_scale + 0.5)
    taps = _tap_count(scale, CUBIC, ANTIALIAS)
    weights = tl.load(weight_pointers + table).to(tl.pointer_type(tl.float32)) + positions * taps
    first_taps = tl.load(first_tap_pointers + table).to(tl.pointer_type(tl.int32))
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
        tl.store(weights + tap, weight, mask=active)
        weight_sum += weight
    for tap in range(taps):
        weight = tl.load(weights + tap, mask=active)
        tl.store(weights + tap, weight / weight_sum, mask=active)
    tl.store(first_taps + positions, first_tap.to(tl.int32), mask=active)


@triton.jit
def _round(values, ROUND_TO_UINT8: tl.constexpr):
    if ROUND_TO_UINT8:
        values = tl.minimum(tl.maximum(tl.floor(values + 0.5), 0.0), 255.0)
    return values


@triton.jit
def _horizontal_kernel(
    frames,
    intermediate,
    intermediate_offsets,
    row_strides,
    heights,
    widths,
    resize_widths,
    out_widths,
    weight_pointers,
    first_tap_pointers,
    CHANNELS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
    FRAME_POINTERS: tl.constexpr,
):
    frame = tl.program_id(0)
    plane_rows = CHANNELS * tl.load(heights + frame).to(tl.int32)
    out_width = tl.load(out_widths + frame).to(tl.int32)
    first_row = tl.program_id(1) * BLOCK_ROWS
    first_column = tl.program_id(2) * BLOCK_COLUMNS
    if first_row >= plane_rows or first_column >= out_width:
        return
    width = tl.load(widths + frame).to(tl.int32)
    rows = (first_row + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    columns = first_column + tl.arange(0, BLOCK_COLUMNS)
    column_active = columns < out_width
    active = (rows < plane_rows)[:, None] & column_active[None, :]
    taps = _tap_count(width.to(tl.float32) / tl.load(resize_widths + frame).to(tl.float32), CUBIC, ANTIALIAS)
    weights = tl.load(weight_pointers + frame).to(tl.pointer_type(tl.float32)) + columns * taps
    first_taps = tl.load(first_tap_pointers + frame).to(tl.pointer_type(tl.int32))
    first_tap = tl.load(first_taps + columns, mask=column_active, other=0)
    if FRAME_POINTERS:
        source = tl.load(frames + frame).to(tl.pointer_type(tl.uint8))
    else:
        source = frames + frame.to(tl.int64) * plane_rows * width
    source += rows[:, None] * width
    total = tl.zeros([BLOCK_ROWS, BLOCK_COLUMNS], dtype=tl.float32)
    for tap in range(taps):
        weight = tl.load(weights + tap, mask=column_active, other=0)
        index = tl.minimum(tl.maximum(first_tap + tap, 0), width - 1)
        total += weight[None, :] * tl.load(source + index[None, :], mask=active, other=0).to(tl.float32)
    row_stride = tl.load(row_strides + frame)
    destination = intermediate + tl.load(intermediate_offsets + frame) + rows[:, None] * row_stride + columns[None, :]
    tl.store(destination, _round(total, ROUND_TO_UINT8).to(intermediate.dtype.element_ty), mask=active)


@triton.jit
def _vertical_kernel(
    intermediate,
    output,
    intermediate_offsets,
    row_strides,
    heights,
    resize_heights,
    weight_pointers,
    first_tap_pointers,
    means,
    stds,
    out_heights,
    out_widths,
    output_offsets,
    CHANNELS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
    ALIGNED: tl.constexpr,
    OUTPUT_ALIGNED: tl.constexpr,
    ALIGNMENT: tl.constexpr,
):
    frame = tl.program_id(0)
    out_height = tl.load(out_heights + frame)
    out_width = tl.load(out_widths + frame)
    first_row = tl.program_id(1) * BLOCK_ROWS
    first_column = tl.program_id(2) * BLOCK_COLUMNS
    if first_row >= out_height or first_column >= out_width:
        return
    height = tl.load(heights + frame).to(tl.int32)
    row_stride = tl.load(row_strides + frame)
    intermediate_offset = tl.load(intermediate_offsets + frame)
    if ALIGNED:
        row_stride = tl.multiple_of(row_stride, ALIGNMENT)
        intermediate_offset = tl.multiple_of(intermediate_offset, ALIGNMENT)
    output_offset = tl.load(output_offsets + frame)
    if OUTPUT_ALIGNED:
        out_width = tl.multiple_of(out_width, ALIGNMENT)
        output_offset = tl.multiple_of(output_offset, ALIGNMENT)
    rows = first_row + tl.arange(0, BLOCK_ROWS)
    columns = first_column + tl.arange(0, BLOCK_COLUMNS)
    row_active = rows < out_height
    active = row_active[:, None] & (columns < out_width)[None, :]
    taps = _tap_count(height.to(tl.float32) / tl.load(resize_heights + frame).to(tl.float32), CUBIC, ANTIALIAS)
    weights = tl.load(weight_pointers + frame).to(tl.pointer_type(tl.float32)) + rows * taps
    first_taps = tl.load(first_tap_pointers + frame).to(tl.pointer_type(tl.int32))
    first_tap = tl.load(first_taps + rows, mask=row_active, other=0)
    for channel in tl.static_range(CHANNELS):
        source = intermediate + intermediate_offset + channel * height.to(tl.int64) * row_stride + columns[None, :]
        total = tl.zeros([BLOCK_ROWS, BLOCK_COLUMNS], dtype=tl.float32)
        for tap in range(taps):
            weight = tl.load(weights + tap, mask=row_active, other=0)
            index = tl.minimum(tl.maximum(first_tap + tap, 0), height - 1).to(tl.int64)
            pixels = tl.load(source + index[:, None] * row_stride, mask=active, other=0)
            total += weight[:, None] * pixels.to(tl.float32)
        total = (_round(total, ROUND_TO_UINT8) - tl.load(means + channel)) / tl.load(stds + channel)
        plane_rows = (channel * out_height + rows).to(tl.int64)
        destination = output + output_offset + plane_rows[:, None] * out_width + columns[None, :]
        tl.store(destination, total, mask=active)


def _as_tensors(device, dtype, *lists):
    """Copy several lists to `device` in one transfer, and return one tensor view per list."""
    lists = [list(values) for values in lists]
    packed = torch.tensor([value for values in lists for value in values], dtype=dtype, pin_memory=True)
    packed = packed.to(device, non_blocking=True)
    return packed.split([len(values) for values in lists])


def _tile(tile, out_width):
    """`tile` with its columns capped at the power of two above `out_width`, and the rows grown to keep its size."""
    rows, columns = tile
    capped_columns = min(columns, triton.next_power_of_2(out_width))
    return rows * columns // capped_columns, capped_columns


def _normalization(mean, std, rescale, channels, device):
    """Per-channel mean and std tensors for the kernels. A single number is used for every channel."""
    mean = (mean,) * channels if isinstance(mean, (int, float)) else tuple(mean)
    std = (std,) * channels if isinstance(std, (int, float)) else tuple(std)
    if len(mean) != channels or len(std) != channels:
        raise ValueError(f"`image_mean` and `image_std` need one value per channel, the images have {channels}.")
    return _rescaled_normalization(mean, std, rescale, device)


@lru_cache
def _rescaled_normalization(mean, std, rescale, device):
    """Mean and std divided by `rescale`, so that `(x - mean') / std' == (x * rescale - mean) / std`."""
    return _as_tensors(device, torch.float32, [value / rescale for value in mean], [value / rescale for value in std])


def _check_frames(frames, ndim):
    if not frames:
        raise ValueError("Pass at least one input.")
    device = frames[0].device
    channels = frames[0].shape[-3]
    if any(
        frame.dtype != torch.uint8 or frame.ndim != ndim or frame.device != device or frame.shape[-3] != channels
        for frame in frames
    ):
        raise ValueError(
            f"Inputs must be {ndim}-dimensional uint8 tensors on one device with the same number of channels."
        )


def _plan_passes(
    device, channels, frame_sizes, resize_sizes, crop_starts, out_sizes, cubic, antialias, vertical_lists
):
    """Device metadata and filter tables of both resize passes, built with one copy and one launch.

    Each argument list holds one `(height, width)` per frame. The horizontal pass is skipped when no frame changes
    width, and the vertical pass then reads the frames packed in one buffer. Intermediate rows are padded to `ROW_ALIGNMENT` bytes so that the vertical pass can load them aligned.
    `vertical_lists` are int64 lists of the vertical pass, returned as tensors after the pass metadata.
    """
    heights = [height for height, _ in frame_sizes]
    widths = [width for _, width in frame_sizes]
    resize_heights = [height for height, _ in resize_sizes]
    resize_widths = [width for _, width in resize_sizes]
    out_heights = [height for height, _ in out_sizes]
    out_widths = [width for _, width in out_sizes]
    frame_lengths = [channels * height * width for height, width in frame_sizes]
    frame_offsets = list(accumulate([0] + frame_lengths[:-1]))
    skip_horizontal = all(
        width == resize_width == out_width for width, resize_width, out_width in zip(widths, resize_widths, out_widths)
    )
    if skip_horizontal:
        row_strides, intermediate_offsets, intermediate_length = widths, frame_offsets, 0
    else:
        row_strides = [-(-out_width // ROW_ALIGNMENT) * ROW_ALIGNMENT for out_width in out_widths]
        intermediate_lengths = [channels * height * stride for height, stride in zip(heights, row_strides)]
        intermediate_offsets = list(accumulate([0] + intermediate_lengths[:-1]))
        intermediate_length = sum(intermediate_lengths)

    horizontal_keys = (
        [] if skip_horizontal else list(zip(widths, resize_widths, [left for _, left in crop_starts], out_widths))
    )
    vertical_keys = list(zip(heights, resize_heights, [top for top, _ in crop_starts], out_heights))
    tables = list(dict.fromkeys(horizontal_keys + vertical_keys))
    # The float32 scale of the kernels can round up to one more step of 2 taps than this float64 one.
    row_sizes = [
        math.ceil((2 if cubic else 1) * (max(size / resize_size, 1.0) if antialias else 1.0)) * 2 + 3
        for size, resize_size, _, _ in tables
    ]
    weight_lengths = [out_size * row_size for (_, _, _, out_size), row_size in zip(tables, row_sizes)]
    weights = torch.empty(sum(weight_lengths), device=device, dtype=torch.float32)
    first_taps = torch.empty(sum(out_size for *_, out_size in tables), device=device, dtype=torch.int32)
    weight_pointers = {
        key: weights.data_ptr() + 4 * offset for key, offset in zip(tables, accumulate([0] + weight_lengths[:-1]))
    }
    first_tap_pointers = {
        key: first_taps.data_ptr() + 4 * offset
        for key, offset in zip(tables, accumulate([0] + [out_size for *_, out_size in tables[:-1]]))
    }
    (
        intermediate_offsets_tensor,
        row_strides_tensor,
        heights_tensor,
        widths_tensor,
        resize_widths_tensor,
        out_widths_tensor,
        horizontal_weights,
        horizontal_first_taps,
        resize_heights_tensor,
        vertical_weights,
        vertical_first_taps,
        *table_tensors,
    ) = _as_tensors(
        device,
        torch.int64,
        intermediate_offsets,
        row_strides,
        heights,
        widths,
        resize_widths,
        out_widths,
        [weight_pointers[key] for key in horizontal_keys],
        [first_tap_pointers[key] for key in horizontal_keys],
        resize_heights,
        [weight_pointers[key] for key in vertical_keys],
        [first_tap_pointers[key] for key in vertical_keys],
        [weight_pointers[key] for key in tables],
        [first_tap_pointers[key] for key in tables],
        *zip(*tables),
        *vertical_lists,
    )
    table_arguments, vertical_list_tensors = (
        table_tensors[: -len(vertical_lists)],
        table_tensors[-len(vertical_lists) :],
    )
    with torch.cuda.device(device):
        _filter_table_kernel[(len(tables), triton.cdiv(max(out_size for *_, out_size in tables), TABLE_BLOCK))](
            *table_arguments,
            -0.5 if antialias else -0.75,
            BLOCK=TABLE_BLOCK,
            CUBIC=cubic,
            ANTIALIAS=antialias,
        )
    horizontal = None
    if not skip_horizontal:
        block_rows, block_columns = _tile(HORIZONTAL_TILE, max(out_widths))
        horizontal = (
            (
                len(frame_sizes),
                triton.cdiv(channels * max(heights), block_rows),
                triton.cdiv(max(out_widths), block_columns),
            ),
            (block_rows, block_columns),
            (
                intermediate_offsets_tensor,
                row_strides_tensor,
                heights_tensor,
                widths_tensor,
                resize_widths_tensor,
                out_widths_tensor,
                horizontal_weights,
                horizontal_first_taps,
            ),
        )
    block_rows, block_columns = _tile(VERTICAL_TILE, max(out_widths))
    return {
        "tables": (weights, first_taps),
        "horizontal": horizontal,
        "intermediate_length": intermediate_length,
        "vertical_grid": (
            len(frame_sizes),
            triton.cdiv(max(out_heights), block_rows),
            triton.cdiv(max(out_widths), block_columns),
        ),
        "vertical_tile": (block_rows, block_columns),
        "vertical": (
            intermediate_offsets_tensor,
            row_strides_tensor,
            heights_tensor,
            resize_heights_tensor,
            vertical_weights,
            vertical_first_taps,
        ),
        "vertical_lists": vertical_list_tensors,
        "aligned": all(value % ROW_ALIGNMENT == 0 for value in row_strides + intermediate_offsets),
        "uniform_output": len(set(out_sizes)) == 1,
    }


def _horizontal_pass(sources, plan, channels, cubic, antialias, round_to_uint8):
    """Resize the width of the frames of `sources` into the intermediate buffer of `plan`.

    One source is read in place. Several sources are read through a table of frame pointers. When the pass is skipped,
    the frames are returned packed in one buffer for the vertical pass.
    """
    sources = [source.contiguous() for source in sources]
    if plan["horizontal"] is None:
        return sources[0].reshape(-1) if len(sources) == 1 else torch.cat([source.reshape(-1) for source in sources])
    frames = sources[0]
    if len(sources) > 1:
        pointers = []
        for source in sources:
            frame_length = source.shape[-3] * source.shape[-2] * source.shape[-1]
            pointers += range(source.data_ptr(), source.data_ptr() + source.numel(), frame_length)
        frames = _as_tensors(frames.device, torch.int64, pointers)[0]
    grid, (block_rows, block_columns), metadata = plan["horizontal"]
    intermediate = torch.empty(
        plan["intermediate_length"], device=frames.device, dtype=torch.uint8 if round_to_uint8 else torch.float32
    )
    with torch.cuda.device(frames.device):
        _horizontal_kernel[grid](
            frames,
            intermediate,
            *metadata,
            CHANNELS=channels,
            BLOCK_ROWS=block_rows,
            BLOCK_COLUMNS=block_columns,
            CUBIC=cubic,
            ANTIALIAS=antialias,
            ROUND_TO_UINT8=round_to_uint8,
            FRAME_POINTERS=len(sources) > 1,
            num_warps=HORIZONTAL_WARPS,
        )
    return intermediate


@lru_cache(maxsize=MAX_PLANS)
def _resize_plan(device, channels, image_sizes, sizes, crop_size, cubic, antialias):
    """Metadata of one batch layout. Its tensors are written on the stream that is current when it is built."""
    out_sizes = list(sizes) if crop_size is None else [crop_size] * len(sizes)
    output_lengths = [channels * height * width for height, width in out_sizes]
    plan = _plan_passes(
        device,
        channels,
        image_sizes,
        sizes,
        [
            ((height - out_height) // 2, (width - out_width) // 2)
            for (height, width), (out_height, out_width) in zip(sizes, out_sizes)
        ],
        out_sizes,
        cubic,
        antialias,
        [
            [height for height, _ in out_sizes],
            [width for _, width in out_sizes],
            accumulate([0] + output_lengths[:-1]),
        ],
    )
    plan["out_sizes"] = out_sizes
    plan["output_lengths"] = output_lengths
    plan["output_aligned"] = all(width % ROW_ALIGNMENT == 0 for _, width in out_sizes)
    return plan


def resize_normalize(
    images: Sequence[torch.Tensor],
    sizes: Sequence[tuple[int, int]],
    image_mean: float | Sequence[float],
    image_std: float | Sequence[float],
    rescale_factor: float,
    resample: str,
    crop_size: tuple[int, int] | None = None,
    antialias: bool = True,
    round_to_uint8: bool = True,
) -> list[torch.Tensor]:
    """Resize uint8 CHW images, center crop them to `crop_size`, rescale and normalize them.

    `sizes[i]` is the `(height, width)` that `images[i]` is resized to. Returns one float32 `(C, H, W)` tensor per
    image, all views of one buffer. `round_to_uint8=True` rounds after each resize pass like the torchvision uint8 path.
    """
    if len(sizes) != len(images):
        raise ValueError(f"Pass one size per image, got {len(sizes)} sizes for {len(images)} images.")
    if crop_size is not None and any(crop_size[0] > height or crop_size[1] > width for height, width in sizes):
        raise ValueError(f"crop_size {crop_size} is larger than a resized image.")
    _check_frames(images, 3)
    if all(image.shape == images[0].shape for image in images):
        image_sizes = (tuple(images[0].shape[1:]),) * len(images)
        sources = [images[0] if len(images) == 1 else torch.stack(images)]
    else:
        image_sizes = tuple(tuple(image.shape[1:]) for image in images)
        sources = images
    cubic = resample == "bicubic"
    device = images[0].device
    channels = images[0].shape[0]
    means, stds = _normalization(image_mean, image_std, rescale_factor, channels, device)
    plan = _resize_plan(
        device,
        channels,
        image_sizes,
        tuple(map(tuple, sizes)),
        None if crop_size is None else tuple(crop_size),
        cubic,
        antialias,
    )
    intermediate = _horizontal_pass(sources, plan, channels, cubic, antialias, round_to_uint8)
    if plan["uniform_output"]:
        output = torch.empty((len(images), channels, *plan["out_sizes"][0]), device=device, dtype=torch.float32)
    else:
        output = torch.empty(sum(plan["output_lengths"]), device=device, dtype=torch.float32)
    block_rows, block_columns = plan["vertical_tile"]
    with torch.cuda.device(device):
        _vertical_kernel[plan["vertical_grid"]](
            intermediate,
            output,
            *plan["vertical"],
            means,
            stds,
            *plan["vertical_lists"],
            CHANNELS=channels,
            BLOCK_ROWS=block_rows,
            BLOCK_COLUMNS=block_columns,
            CUBIC=cubic,
            ANTIALIAS=antialias,
            ROUND_TO_UINT8=round_to_uint8,
            ALIGNED=plan["aligned"],
            OUTPUT_ALIGNED=plan["output_aligned"],
            ALIGNMENT=ROW_ALIGNMENT,
        )
    if plan["uniform_output"]:
        return list(output.unbind(0))
    return [
        values.view(channels, height, width)
        for values, (height, width) in zip(output.split(plan["output_lengths"]), plan["out_sizes"])
    ]
