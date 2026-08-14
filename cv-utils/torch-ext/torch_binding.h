#pragma once

#include <torch/csrc/stable/tensor.h>

#include <tuple>

using cv_utils_tensor = torch::stable::Tensor;

std::tuple<cv_utils_tensor, cv_utils_tensor>
connected_components_labeling_2d(const cv_utils_tensor &inputs, bool get_counts);

cv_utils_tensor generic_nms(const cv_utils_tensor &dets,
                            const cv_utils_tensor &scores,
                            double iou_threshold,
                            bool use_iou_matrix);
