/* Metal implementation of the entry point.
 *
 * No Metal header is included here: all Metal contact sits behind the extern "C" boundary in
 * dispatch.mm, and every output comes from torch's allocator so the op composes with the rest of a
 * model's memory.
 *
 * A torch MPS tensor's storage is a whole MTLBuffer that the tensor may be a view into, so each
 * buffer crosses the boundary as a (buffer, byte offset) pair.
 */

#include <torch/torch.h>

#include <vector>

#include "common.h"
#include "torch_binding.h"

namespace {

void *mtl_buffer(const at::Tensor &t) { return const_cast<void *>(t.storage().data()); }

size_t byte_offset(const at::Tensor &t) {
  return static_cast<size_t>(t.storage_offset()) * t.element_size();
}

}  // namespace

std::vector<at::Tensor> top_k(const at::Tensor &logits, int64_t k, bool softmax) {
  TORCH_CHECK(logits.is_mps() && logits.dim() == 2, "logits must be a 2D mps tensor");
  const at::Tensor x =
      logits.scalar_type() == at::kFloat ? logits.contiguous() : logits.to(at::kFloat).contiguous();
  const int64_t rows = x.size(0), n = x.size(1);
  TORCH_CHECK(k > 0 && k <= n, "k must be within the row");
  at::Tensor indices = at::empty({rows, k}, x.options().dtype(at::kInt));
  at::Tensor values = at::empty({rows, k}, x.options());
  const int status = topk_metal_top_k(mtl_buffer(x), byte_offset(x), mtl_buffer(indices),
                                      byte_offset(indices), mtl_buffer(values), byte_offset(values),
                                      rows, n, k, softmax ? 1 : 0);
  TORCH_CHECK(status == 0, "top_k: no kernel for this build (", status, ")");
  return {values, indices};
}
