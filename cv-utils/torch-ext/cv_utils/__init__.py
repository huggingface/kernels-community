import torch
from typing import List

from ._ops import ops

def cc_2d(inputs: torch.Tensor, get_counts: bool) -> List[torch.Tensor]:
    return ops.cc_2d(inputs, get_counts)

def generic_nms(dets: torch.Tensor, scores: torch.Tensor, iou_threshold: float, use_iou_matrix: bool) -> torch.Tensor:
    return ops.generic_nms(dets, scores, iou_threshold, use_iou_matrix)

__all__ = ["cc_2d", "generic_nms"]