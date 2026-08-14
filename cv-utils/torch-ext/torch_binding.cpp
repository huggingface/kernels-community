#include <torch/csrc/stable/library.h>

#include "registration.h"
#include "torch_binding.h"

STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("cc_2d(Tensor inputs, bool get_counts) -> Tensor[]");
  ops.def("generic_nms(Tensor dets, Tensor scores, float iou_threshold, bool use_iou_matrix) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, ops) {
  ops.impl("cc_2d", TORCH_BOX(&connected_components_labeling_2d));
  ops.impl("generic_nms", TORCH_BOX(&generic_nms));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
