import torch
import triton
import triton.language as tl

from ._resize import _as_tensor, _horizontal_pass, _max_taps, _normalization, _resample


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
    means,
    stds,
    cubic_coeff,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
    PATCH: tl.constexpr,
    MERGE: tl.constexpr,
    TEMPORAL: tl.constexpr,
):
    slot = tl.program_id(0)
    frame = tl.load(slot_frames + slot)
    height = tl.load(heights + frame)
    out_height = tl.load(out_heights + frame)
    out_width = tl.load(out_widths + frame)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = index < out_height * out_width
    row = index // out_width
    column = index % out_width
    grid_row = row // PATCH
    grid_column = column // PATCH
    merged_rows = out_height // (PATCH * MERGE)
    merged_columns = out_width // (PATCH * MERGE)
    patch = (
        ((tl.load(slot_groups + slot) * merged_rows + grid_row // MERGE) * merged_columns + grid_column // MERGE)
        * MERGE
        + grid_row % MERGE
    ) * MERGE + grid_column % MERGE
    patch_start = tl.load(slot_output_offsets + slot) + patch.to(tl.int64) * (CHANNELS * TEMPORAL * PATCH * PATCH)
    patch_start += (row % PATCH) * PATCH + column % PATCH
    temporal_start = tl.load(slot_temporal_starts + slot)
    temporal_count = tl.load(slot_temporal_counts + slot)
    scale = height.to(tl.float32) / out_height.to(tl.float32)
    for channel in tl.static_range(CHANNELS):
        start = tl.load(intermediate_offsets + frame) + channel * height.to(tl.int64) * out_width + column
        value = _resample(
            intermediate,
            start,
            out_width,
            row,
            height,
            scale,
            active,
            cubic_coeff,
            BLOCK,
            CUBIC,
            ANTIALIAS,
            MAX_TAPS,
            ROUND_TO_UINT8,
        )
        value = (value - tl.load(means + channel)) / tl.load(stds + channel)
        for temporal in tl.static_range(TEMPORAL):
            write_index = patch_start + (channel * TEMPORAL + temporal_start + temporal) * PATCH * PATCH
            tl.store(output + write_index, value, mask=active & (temporal < temporal_count))


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
    block=256,
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
        frames, out_widths, [0] * len(frames), out_widths, cubic, antialias, round_to_uint8, block
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

    means, stds = _normalization(image_mean, image_std, rescale_factor, device)
    output = torch.empty((total_patches, patch_dim), device=device, dtype=torch.float32)
    slot_frames, slot_temporal_starts, slot_temporal_counts, slot_groups, slot_output_offsets = zip(*slots)
    grid = (len(slots), triton.cdiv(max(height * width for height, width in target_sizes), block))
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
        means,
        stds,
        -0.5 if antialias else -0.75,
        CHANNELS=channels,
        BLOCK=block,
        CUBIC=cubic,
        ANTIALIAS=antialias,
        MAX_TAPS=_max_taps([frame.shape[1] for frame in frames], out_heights, cubic, antialias),
        ROUND_TO_UINT8=round_to_uint8,
        PATCH=patch_size,
        MERGE=merge_size,
        TEMPORAL=temporal_patch_size,
    )
    return output, grid_thw
