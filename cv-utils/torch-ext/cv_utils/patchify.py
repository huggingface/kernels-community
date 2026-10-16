from functools import lru_cache
from itertools import accumulate

import torch
import triton
import triton.language as tl

from .resize import (
    MAX_PLANS,
    _check_frames,
    _horizontal_pass,
    _normalization,
    _plan_passes,
    _round,
    _tap_count,
)


@triton.jit
def _vertical_patchify_kernel(
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
    input_offsets,
    frames_in_input,
    frame_counts,
    CHANNELS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
    ALIGNED: tl.constexpr,
    PATCH: tl.constexpr,
    PATCH_BLOCK: tl.constexpr,
    TEMPORAL: tl.constexpr,
):
    frame = tl.program_id(0)
    out_height = tl.load(out_heights + frame).to(tl.int32)
    out_width = tl.load(out_widths + frame).to(tl.int32)
    first_row = tl.program_id(1) * BLOCK_ROWS
    first_column = tl.program_id(2) * BLOCK_COLUMNS
    if first_row >= out_height or first_column >= out_width:
        return
    height = tl.load(heights + frame).to(tl.int32)
    row_stride = tl.load(row_strides + frame)
    intermediate_offset = tl.load(intermediate_offsets + frame)
    if ALIGNED:
        row_stride = tl.multiple_of(row_stride, 16)
        intermediate_offset = tl.multiple_of(intermediate_offset, 16)
    frame_in_input = tl.load(frames_in_input + frame).to(tl.int32)
    frame_count = tl.load(frame_counts + frame).to(tl.int32)
    group = frame_in_input // TEMPORAL
    temporal_start = frame_in_input % TEMPORAL
    temporal_count = tl.where(frame_in_input == frame_count - 1, TEMPORAL - temporal_start, 1)
    rows = first_row + tl.arange(0, BLOCK_ROWS)
    columns = first_column + tl.arange(0, BLOCK_COLUMNS)
    row_active = rows < out_height
    active = row_active[:, None] & (columns < out_width)[None, :]
    patch_block_rows = out_height // (PATCH * PATCH_BLOCK)
    patch_block_columns = out_width // (PATCH * PATCH_BLOCK)
    grid_rows = rows // PATCH
    grid_columns = columns // PATCH
    row_part = (group * patch_block_rows + grid_rows // PATCH_BLOCK) * patch_block_columns
    patch = (row_part[:, None] + (grid_columns // PATCH_BLOCK)[None, :]) * PATCH_BLOCK
    patch = (patch + (grid_rows % PATCH_BLOCK)[:, None]) * PATCH_BLOCK + (grid_columns % PATCH_BLOCK)[None, :]
    within_patch = ((rows % PATCH) * PATCH)[:, None] + (columns % PATCH)[None, :]
    patch_start = tl.load(input_offsets + frame) + patch.to(tl.int64) * (CHANNELS * TEMPORAL * PATCH * PATCH)
    patch_start += within_patch
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
        for temporal in tl.static_range(TEMPORAL):
            write_index = patch_start + (channel * TEMPORAL + temporal_start + temporal) * PATCH * PATCH
            tl.store(output + write_index, total, mask=active & (temporal < temporal_count))


@lru_cache(maxsize=MAX_PLANS)
def _patchify_plan(device, channels, input_shapes, target_sizes, patch_size, temporal_patch_size, cubic, antialias):
    patch_dim = channels * temporal_patch_size * patch_size * patch_size
    grid_thw = [
        (-(-frame_count // temporal_patch_size), height // patch_size, width // patch_size)
        for (frame_count, _, _), (height, width) in zip(input_shapes, target_sizes)
    ]
    input_lengths = [t * h * w * patch_dim for t, h, w in grid_thw]
    input_offsets = accumulate([0] + input_lengths[:-1])
    frame_sizes, out_sizes, frame_input_offsets, frames_in_input, frame_counts = [], [], [], [], []
    for (frame_count, height, width), target_size, input_offset in zip(input_shapes, target_sizes, input_offsets):
        frame_sizes += [(height, width)] * frame_count
        out_sizes += [target_size] * frame_count
        frame_input_offsets += [input_offset] * frame_count
        frames_in_input += list(range(frame_count))
        frame_counts += [frame_count] * frame_count
    plan = _plan_passes(
        device,
        channels,
        frame_sizes,
        out_sizes,
        [(0, 0)] * len(frame_sizes),
        out_sizes,
        cubic,
        antialias,
        [
            [height for height, _ in out_sizes],
            [width for _, width in out_sizes],
            frame_input_offsets,
            frames_in_input,
            frame_counts,
        ],
    )
    plan["patches"] = (sum(input_lengths) // patch_dim, patch_dim)
    plan["grid_thw"] = grid_thw
    return plan


def resize_normalize_patchify(
    inputs,
    target_sizes,
    image_mean,
    image_std,
    rescale_factor,
    resample,
    patch_size,
    block_size,
    temporal_patch_size,
    antialias=True,
    round_to_uint8=True,
):
    """Resize, normalize and patchify uint8 `(T, C, H, W)` inputs into Qwen2-VL `pixel_values` and an int64 `grid_thw`.

    `target_sizes[i]` is the frame size of `inputs[i]`, with `T=1` for an image. A missing frame of the last
    temporal patch repeats the last frame. Patches are written in `block_size x block_size` blocks, `block_size=1`
    writes them row by row.
    """
    _check_frames(inputs, 4)
    device = inputs[0].device
    channels = inputs[0].shape[1]
    cubic = resample == "bicubic"
    factor = patch_size * block_size
    if any(height % factor or width % factor for height, width in target_sizes):
        raise ValueError(f"Every target size must be a multiple of patch_size * block_size = {factor}.")
    means, stds = _normalization(image_mean, image_std, rescale_factor, channels, device)
    plan = _patchify_plan(
        device,
        channels,
        tuple((frames.shape[0], frames.shape[2], frames.shape[3]) for frames in inputs),
        tuple((height, width) for height, width in target_sizes),
        patch_size,
        temporal_patch_size,
        cubic,
        antialias,
    )
    same_shape = len(inputs) > 1 and all(frames.shape == inputs[0].shape for frames in inputs)
    sources = [torch.cat(inputs)] if same_shape else inputs
    intermediate = _horizontal_pass(sources, plan, channels, cubic, antialias, round_to_uint8)
    output = torch.empty(plan["patches"], device=device, dtype=torch.float32)
    block_rows, block_columns = plan["vertical_tile"]
    _vertical_patchify_kernel[plan["vertical_grid"]](
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
        PATCH=patch_size,
        PATCH_BLOCK=block_size,
        TEMPORAL=temporal_patch_size,
    )
    return output, torch.tensor(plan["grid_thw"], dtype=torch.int64)
