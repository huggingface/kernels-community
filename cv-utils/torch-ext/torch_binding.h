#pragma once

#include <torch/torch.h>

std::vector<torch::Tensor> connected_components_labeling_2d(const torch::Tensor &inputs, bool get_counts);
torch::Tensor generic_nms(const torch::Tensor &dets, const torch::Tensor &scores, double iou_threshold, bool use_iou_matrix);