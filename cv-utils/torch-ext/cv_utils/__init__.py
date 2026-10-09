import torch
from typing import List

from ._ops import ops
from .patchify import resize_normalize_patchify
from .resize import resize_normalize

def cc_2d(inputs: torch.Tensor, get_counts: bool) -> List[torch.Tensor]:
    return ops.cc_2d(inputs, get_counts)

def connected_component_areas(mask: torch.Tensor) -> torch.Tensor:
    """Area of the 8-connected component of every foreground pixel of a `(N, 1, H, W)` mask of any height and width.

    Non-zero values are foreground and background pixels get 0. Unlike `cc_2d`, the height and width can be odd.
    """
    height, width = mask.shape[-2:]
    padded = torch.nn.functional.pad(mask.to(torch.uint8), (0, width % 2, 0, height % 2))
    _, areas = ops.cc_2d(padded.contiguous(), True)
    return areas[..., :height, :width]

def generic_nms(dets: torch.Tensor, scores: torch.Tensor, iou_threshold: float, use_iou_matrix: bool) -> torch.Tensor:
    return ops.generic_nms(dets, scores, iou_threshold, use_iou_matrix)

__all__ = ["cc_2d", "connected_component_areas", "generic_nms", "resize_normalize", "resize_normalize_patchify"]