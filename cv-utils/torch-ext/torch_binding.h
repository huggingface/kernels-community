#pragma once

#include <vector>

#include <torch/csrc/stable/tensor.h>

std::vector<torch::stable::Tensor> connected_components_labeling_2d(
    const torch::stable::Tensor &inputs, bool get_counts);
torch::stable::Tensor generic_nms(const torch::stable::Tensor &dets,
                                  const torch::stable::Tensor &scores,
                                  double iou_threshold, bool use_iou_matrix);
