from collections.abc import Sequence

import torch
import triton
import triton.language as tl


BLOCK = 256


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


def _launch(logits, out_height, out_width):
    batch, classes, height, width = logits.shape
    output = torch.empty(batch, out_height, out_width, dtype=torch.int64, device=logits.device)
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

    Matches `F.interpolate(logits[i:i + 1], size=sizes[i], mode="bilinear", align_corners=False)[0].argmax(0)`,
    computed in float32. Returns one int64 `(H, W)` tensor per image.
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
