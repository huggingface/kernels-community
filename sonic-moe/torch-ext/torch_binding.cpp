#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "count_cumsum(Tensor x, Tensor(a!) count_output, "
      "Tensor(b!) cumsum_output, bool do_cumsum) -> ()");
  ops.impl("count_cumsum", torch::kCUDA, &count_cumsum_cuda);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
