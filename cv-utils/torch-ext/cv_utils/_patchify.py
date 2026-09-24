import torch
import triton
import triton.language as tl

from ._resize import (
    VERTICAL_TILE,
    _as_tensor,
    _filter_table,
    _horizontal_pass,
    _normalization,
    _round,
    _tap_count,
    _tile,
)


@triton.jit
def _vertical_patchify_kernel(
    intermediate,
    output,
    intermediate_offsets,
    heights,
    out_heights,
    out_widths,
    slot_frames,
    slot_temporal_starts,
    slot_temporal_counts,
    slot_groups,
    slot_output_offsets,
    weights,
    first_taps,
    table_offsets,
    taps_stride,
    means,
    stds,
    CHANNELS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLUMNS: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
    PATCH: tl.constexpr,
    MERGE: tl.constexpr,
    TEMPORAL: tl.constexpr,
):
    slot = tl.program_id(0)
    frame = tl.load(slot_frames + slot)
    out_height = tl.load(out_heights + frame)
    out_width = tl.load(out_widths + frame)
    first_row = tl.program_id(1) * BLOCK_ROWS
    first_column = tl.program_id(2) * BLOCK_COLUMNS
    if first_row >= out_height or first_column >= out_width:
        return
    height = tl.load(heights + frame)
    rows = first_row + tl.arange(0, BLOCK_ROWS)
    columns = first_column + tl.arange(0, BLOCK_COLUMNS)
    row_active = rows < out_height
    active = row_active[:, None] & (columns < out_width)[None, :]
    merged_rows = out_height // (PATCH * MERGE)
    merged_columns = out_width // (PATCH * MERGE)
    grid_rows = rows // PATCH
    grid_columns = columns // PATCH
    row_part = (tl.load(slot_groups + slot) * merged_rows + grid_rows // MERGE) * merged_columns
    patch = ((row_part[:, None] + (grid_columns // MERGE)[None, :]) * MERGE + (grid_rows % MERGE)[:, None]) * MERGE
    patch += (grid_columns % MERGE)[None, :]
    within_patch = ((rows % PATCH) * PATCH)[:, None] + (columns % PATCH)[None, :]
    patch_start = tl.load(slot_output_offsets + slot) + patch.to(tl.int64) * (CHANNELS * TEMPORAL * PATCH * PATCH)
    patch_start += within_patch
    temporal_start = tl.load(slot_temporal_starts + slot)
    temporal_count = tl.load(slot_temporal_counts + slot)
    taps = _tap_count(height.to(tl.float32) / out_height.to(tl.float32), CUBIC, ANTIALIAS)
    table_rows = (tl.load(table_offsets + frame) + rows).to(tl.int64)
    first_tap = tl.load(first_taps + table_rows, mask=row_active, other=0)
    for channel in tl.static_range(CHANNELS):
        plane_start = tl.load(intermediate_offsets + frame) + channel * height.to(tl.int64) * out_width
        source = intermediate + plane_start + columns[None, :]
        total = tl.zeros([BLOCK_ROWS, BLOCK_COLUMNS], dtype=tl.float32)
        for tap in range(taps):
            weight = tl.load(weights + table_rows * taps_stride + tap, mask=row_active, other=0)
            index = tl.minimum(tl.maximum(first_tap + tap, 0), height - 1).to(tl.int64)
            pixels = tl.load(source + index[:, None] * out_width, mask=active, other=0)
            total += weight[:, None] * pixels.to(tl.float32)
        total = (_round(total, ROUND_TO_UINT8) - tl.load(means + channel)) / tl.load(stds + channel)
        for temporal in tl.static_range(TEMPORAL):
            write_index = patch_start + (channel * TEMPORAL + temporal_start + temporal) * PATCH * PATCH
            tl.store(output + write_index, total, mask=active & (temporal < temporal_count))


def resize_normalize_patchify(
    frames,
    target_sizes,
    items,
    image_mean,
    image_std,
    rescale_factor,
    resample,
    antialias,
    patch_size,
    merge_size,
    temporal_patch_size,
    round_to_uint8=False,
):
    """Resize, normalize and patchify uint8 CHW frames into Qwen2-VL `pixel_values` and `grid_thw`.

    `target_sizes[j]` is the size of `frames[j]`. `items` lists the frame indices of each output item: `[i]` for an
    image, the frames of a video in order. A missing frame of the last temporal patch repeats the last frame.
    """
    device = frames[0].device
    channels = frames[0].shape[0]
    cubic = resample == "bicubic"
    out_heights = [height for height, _ in target_sizes]
    out_widths = [width for _, width in target_sizes]
    intermediate, intermediate_offsets, heights = _horizontal_pass(
        frames, out_widths, [0] * len(frames), out_widths, cubic, antialias, round_to_uint8
    )

    patch_dim = channels * temporal_patch_size * patch_size * patch_size
    slots, grid_thw, total_patches = [], [], 0
    for item in items:
        padded = list(item) + [item[-1]] * (-len(item) % temporal_patch_size)
        grid_t = len(padded) // temporal_patch_size
        grid_h, grid_w = out_heights[item[0]] // patch_size, out_widths[item[0]] // patch_size
        for group in range(grid_t):
            group_frames = padded[group * temporal_patch_size : (group + 1) * temporal_patch_size]
            position = 0
            while position < temporal_patch_size:
                count = 1
                while (
                    position + count < temporal_patch_size and group_frames[position + count] == group_frames[position]
                ):
                    count += 1
                slots.append((group_frames[position], position, count, group, total_patches * patch_dim))
                position += count
        grid_thw.append((grid_t, grid_h, grid_w))
        total_patches += grid_t * grid_h * grid_w

    weights, first_taps, table_offsets, taps_stride = _filter_table(
        [frame.shape[1] for frame in frames], out_heights, [0] * len(frames), out_heights, cubic, antialias, device
    )
    means, stds = _normalization(image_mean, image_std, rescale_factor, device)
    output = torch.empty((total_patches, patch_dim), device=device, dtype=torch.float32)
    slot_frames, slot_temporal_starts, slot_temporal_counts, slot_groups, slot_output_offsets = zip(*slots)
    block_rows, block_columns = _tile(VERTICAL_TILE, max(out_widths))
    grid = (len(slots), triton.cdiv(max(out_heights), block_rows), triton.cdiv(max(out_widths), block_columns))
    _vertical_patchify_kernel[grid](
        intermediate,
        output,
        intermediate_offsets,
        heights,
        _as_tensor(out_heights, device),
        _as_tensor(out_widths, device),
        _as_tensor(slot_frames, device),
        _as_tensor(slot_temporal_starts, device),
        _as_tensor(slot_temporal_counts, device),
        _as_tensor(slot_groups, device),
        _as_tensor(slot_output_offsets, device, torch.int64),
        weights,
        first_taps,
        table_offsets,
        taps_stride,
        means,
        stds,
        CHANNELS=channels,
        BLOCK_ROWS=block_rows,
        BLOCK_COLUMNS=block_columns,
        CUBIC=cubic,
        ANTIALIAS=antialias,
        ROUND_TO_UINT8=round_to_uint8,
        PATCH=patch_size,
        MERGE=merge_size,
        TEMPORAL=temporal_patch_size,
    )
    return output, grid_thw
