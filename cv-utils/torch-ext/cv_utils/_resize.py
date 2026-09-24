import math
from itertools import accumulate

import torch
import triton
import triton.language as tl


@triton.jit
def _resample(
    source,
    start,
    stride,
    position,
    size,
    scale,
    active,
    cubic_coeff,
    BLOCK: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
):
    """Filter along one axis: output `position` from `source[start + i * stride]`, `i` in `[0, size)`."""
    filter_scale = tl.maximum(scale, 1.0) if ANTIALIAS else 1.0
    center = scale * (position.to(tl.float32) + 0.5)
    first_tap = tl.floor(center - (2.0 if CUBIC else 1.0) * filter_scale + 0.5)
    total = tl.zeros([BLOCK], dtype=tl.float32)
    weight_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for tap in tl.static_range(MAX_TAPS):
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
        index = tl.minimum(tl.maximum(tap_position.to(tl.int32), 0), size - 1).to(tl.int64)
        total += weight * tl.load(source + start + index * stride, mask=active, other=0).to(tl.float32)
        weight_sum += weight
    total = total / weight_sum
    if ROUND_TO_UINT8:
        total = tl.minimum(tl.maximum(tl.floor(total + 0.5), 0.0), 255.0)
    return total


@triton.jit
def _horizontal_kernel(
    pixels,
    intermediate,
    pixel_offsets,
    intermediate_offsets,
    heights,
    widths,
    resize_widths,
    crop_lefts,
    out_widths,
    cubic_coeff,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
):
    item = tl.program_id(0)
    height = tl.load(heights + item)
    width = tl.load(widths + item)
    out_width = tl.load(out_widths + item)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = index < height * out_width
    row = (index // out_width).to(tl.int64)
    column = index % out_width
    scale = width.to(tl.float32) / tl.load(resize_widths + item).to(tl.float32)
    for channel in tl.static_range(CHANNELS):
        plane_row = channel * height + row
        value = _resample(
            pixels,
            tl.load(pixel_offsets + item) + plane_row * width,
            1,
            column + tl.load(crop_lefts + item),
            width,
            scale,
            active,
            cubic_coeff,
            BLOCK,
            CUBIC,
            ANTIALIAS,
            MAX_TAPS,
            ROUND_TO_UINT8,
        )
        tl.store(
            intermediate + tl.load(intermediate_offsets + item) + plane_row * out_width + column, value, mask=active
        )


@triton.jit
def _vertical_kernel(
    intermediate,
    output,
    intermediate_offsets,
    heights,
    resize_heights,
    crop_tops,
    means,
    stds,
    out_height,
    out_width,
    cubic_coeff,
    CHANNELS: tl.constexpr,
    BLOCK: tl.constexpr,
    CUBIC: tl.constexpr,
    ANTIALIAS: tl.constexpr,
    MAX_TAPS: tl.constexpr,
    ROUND_TO_UINT8: tl.constexpr,
):
    item = tl.program_id(0)
    height = tl.load(heights + item)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = index < out_height * out_width
    row = index // out_width + tl.load(crop_tops + item)
    column = (index % out_width).to(tl.int64)
    scale = height.to(tl.float32) / tl.load(resize_heights + item).to(tl.float32)
    for channel in tl.static_range(CHANNELS):
        start = tl.load(intermediate_offsets + item) + channel * height.to(tl.int64) * out_width + column
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
        tl.store(
            output + (item * CHANNELS + channel).to(tl.int64) * out_height * out_width + index, value, mask=active
        )


def _as_tensor(values, device, dtype=torch.int32):
    return torch.tensor(list(values), device=device, dtype=dtype)


def _max_taps(in_sizes, out_sizes, cubic, antialias):
    return max(
        math.ceil((2 if cubic else 1) * (max(in_size / out_size, 1.0) if antialias else 1.0)) * 2 + 1
        for in_size, out_size in zip(in_sizes, out_sizes)
    )


def _normalization(mean, std, rescale, device):
    """Mean and std divided by `rescale`, so that `(x - mean') / std' == (x * rescale - mean) / std`."""
    return _as_tensor([value / rescale for value in mean], device, torch.float32), _as_tensor(
        [value / rescale for value in std], device, torch.float32
    )


def _horizontal_pass(frames, resize_widths, crop_lefts, out_widths, cubic, antialias, round_to_uint8, block):
    """Resize the width of every frame into one packed float buffer of `(C, H, out_width)` planes."""
    device = frames[0].device
    channels = frames[0].shape[0]
    heights = [frame.shape[1] for frame in frames]
    widths = [frame.shape[2] for frame in frames]
    intermediate_sizes = [channels * height * out_width for height, out_width in zip(heights, out_widths)]
    intermediate_offsets = _as_tensor(accumulate([0] + intermediate_sizes[:-1]), device, torch.int64)
    intermediate = torch.empty(sum(intermediate_sizes), device=device, dtype=torch.float32)
    heights_tensor = _as_tensor(heights, device)
    grid = (len(frames), triton.cdiv(max(height * out_width for height, out_width in zip(heights, out_widths)), block))
    _horizontal_kernel[grid](
        torch.cat([frame.reshape(-1) for frame in frames]),
        intermediate,
        _as_tensor(accumulate([0] + [frame.numel() for frame in frames[:-1]]), device, torch.int64),
        intermediate_offsets,
        heights_tensor,
        _as_tensor(widths, device),
        _as_tensor(resize_widths, device),
        _as_tensor(crop_lefts, device),
        _as_tensor(out_widths, device),
        -0.5 if antialias else -0.75,
        CHANNELS=channels,
        BLOCK=block,
        CUBIC=cubic,
        ANTIALIAS=antialias,
        MAX_TAPS=_max_taps(widths, resize_widths, cubic, antialias),
        ROUND_TO_UINT8=round_to_uint8,
    )
    return intermediate, intermediate_offsets, heights_tensor


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
    block=256,
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
    intermediate, intermediate_offsets, heights = _horizontal_pass(
        images,
        [width for _, width in resize_sizes],
        [(width - out_width) // 2 for _, width in resize_sizes],
        [out_width] * len(images),
        cubic,
        antialias,
        round_to_uint8,
        block,
    )
    device = images[0].device
    means, stds = _normalization(image_mean, image_std, rescale_factor, device)
    output = torch.empty((len(images), images[0].shape[0], out_height, out_width), device=device, dtype=torch.float32)
    _vertical_kernel[(len(images), triton.cdiv(out_height * out_width, block))](
        intermediate,
        output,
        intermediate_offsets,
        heights,
        _as_tensor([height for height, _ in resize_sizes], device),
        _as_tensor([(height - out_height) // 2 for height, _ in resize_sizes], device),
        means,
        stds,
        out_height,
        out_width,
        -0.5 if antialias else -0.75,
        CHANNELS=images[0].shape[0],
        BLOCK=block,
        CUBIC=cubic,
        ANTIALIAS=antialias,
        MAX_TAPS=_max_taps([shape[0] for shape in shapes], [height for height, _ in resize_sizes], cubic, antialias),
        ROUND_TO_UINT8=round_to_uint8,
    )
    return output
