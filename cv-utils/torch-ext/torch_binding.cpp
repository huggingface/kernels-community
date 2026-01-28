#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("cc_2d(Tensor inputs, bool get_counts) -> Tensor[]");
  ops.impl("cc_2d", torch::kCUDA, &connected_components_labeling_2d);

  ops.def("generic_nms(Tensor dets, Tensor scores, float iou_threshold, bool use_iou_matrix) -> Tensor");
  ops.impl("generic_nms", torch::kCUDA, &generic_nms);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)