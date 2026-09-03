/* Metal implementation of the entry points.
 *
 * No ggml header is included here: all ggml and Metal contact sits behind the extern "C" boundary in
 * ggml_dispatch.mm, and every output comes from torch's allocator so the ops compose with the rest of
 * a model's memory.
 *
 * A torch MPS tensor's storage is a whole MTLBuffer that the tensor may be a view into, so each
 * buffer crosses the boundary as a (buffer, byte offset) pair.
 */

#include <torch/torch.h>

#include "common.h"
#include "torch_binding.h"

namespace {

void *mtl_buffer(const at::Tensor &t) { return const_cast<void *>(t.storage().data()); }

size_t byte_offset(const at::Tensor &t) {
  return static_cast<size_t>(t.storage_offset()) * t.element_size();
}

}  // namespace

at::Tensor rms_norm(const at::Tensor &x, const at::Tensor &weight, double eps) {
  TORCH_CHECK(x.is_mps() && weight.is_mps(), "rms_norm expects mps tensors");
  TORCH_CHECK(x.scalar_type() == at::kFloat && weight.scalar_type() == at::kFloat,
              "rms_norm is f32 only");
  const at::Tensor xc = x.contiguous(), wc = weight.contiguous();
  const int64_t cols = xc.size(-1);
  TORCH_CHECK(wc.numel() == cols, "rms_norm: weight must be one row of ", cols);
  at::Tensor out = at::empty_like(xc);
  const int status = ggml_norm_metal_rms_norm(
      mtl_buffer(xc), byte_offset(xc), mtl_buffer(wc), byte_offset(wc), mtl_buffer(out),
      byte_offset(out), xc.numel() / cols, cols, (float)eps);
  TORCH_CHECK(status == 0, "rms_norm: no kernel for this build (", status, ")");
  return out;
}
