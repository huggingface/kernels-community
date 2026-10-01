/* Metal implementation of the entry points.
 *
 * No ggml header is included here: all ggml and Metal contact sits behind the extern "C" boundary in
 * ggml_dispatch.mm, and every output comes from torch's allocator so the op composes with the rest of
 * a model's memory.
 *
 * A torch MPS tensor's storage is a whole MTLBuffer that the tensor may be a view into, so each
 * buffer crosses the boundary as a (buffer, byte offset) pair.
 */

#include <torch/torch.h>

#include <optional>

#include "common.h"
#include "torch_binding.h"

namespace {

void *mtl_buffer(const at::Tensor &t) { return const_cast<void *>(t.storage().data()); }

size_t byte_offset(const at::Tensor &t) {
  return static_cast<size_t>(t.storage_offset()) * t.element_size();
}

at::Tensor as_f32(const at::Tensor &t) {
  return t.scalar_type() == at::kFloat ? t.contiguous() : t.to(at::kFloat).contiguous();
}

}  // namespace

bool supports_flash_attn(int64_t n_q, int64_t head_dim_k, int64_t head_dim_v) {
  return ggml_attn_metal_supports_flash_attn(n_q, head_dim_k, head_dim_v) != 0;
}

at::Tensor flash_attn(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                      const std::optional<at::Tensor> &mask, double scale) {
  TORCH_CHECK(q.is_mps(), "flash_attn: expected mps tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "flash_attn: q, k and v must be (n_seqs, n_heads, n_positions, head_dim)");

  const int64_t n_seqs = q.size(0), n_heads = q.size(1), n_q = q.size(2), head_dim_k = q.size(3);
  const int64_t n_heads_kv = k.size(1), n_kv = k.size(2), head_dim_v = v.size(3);
  TORCH_CHECK(k.size(3) == head_dim_k, "flash_attn: k's head_dim must match q's");
  TORCH_CHECK(v.size(2) == n_kv && v.size(1) == n_heads_kv, "flash_attn: k and v must agree");
  TORCH_CHECK(n_heads_kv > 0 && n_heads % n_heads_kv == 0,
              "flash_attn: n_heads must be a multiple of n_heads_kv, got ", n_heads, " and ",
              n_heads_kv);

  const auto qc = as_f32(q), kc = as_f32(k), vc = as_f32(v);

  // ggml's kernel reads an f16 mask. A caller's additive f32 mask is small (one row per query), so
  // converting here costs little and keeps the caller's side ordinary.
  const bool has_mask = mask.has_value() && mask->defined();
  at::Tensor mc;
  if (has_mask) {
    TORCH_CHECK(mask->dim() == 4, "flash_attn: mask must be (n_seqs, 1, n_q, n_kv)");
    TORCH_CHECK(mask->size(2) >= n_q && mask->size(3) == n_kv,
                "flash_attn: mask must be at least n_q rows and exactly n_kv wide");
    mc = mask->to(at::kHalf).contiguous();
  } else {
    mc = qc;  // a buffer still has to be bound; the kernel will not read it
  }

  int64_t pad_floats = 0, tmp_floats = 0, blk_floats = 0;
  ggml_attn_metal_flash_attn_scratch(n_seqs, n_heads, n_heads_kv, n_q, n_kv, head_dim_k, head_dim_v,
                                     has_mask, &pad_floats, &tmp_floats, &blk_floats);
  auto pad = at::empty({pad_floats}, qc.options().dtype(at::kFloat));
  auto tmp = at::empty({tmp_floats}, qc.options().dtype(at::kFloat));
  // The tiled path's block map. Zeroed rather than `empty`: the blk kernel only writes it when there
  // is a mask, and the attention kernel is bound to it either way.
  auto blk = at::zeros({blk_floats}, qc.options().dtype(at::kFloat));

  // Tokens before heads: that is the layout ggml writes, and it is what a caller wants -- SDPA's own
  // `.transpose(1, 2)` produces the same thing.
  auto dst = at::empty({n_seqs, n_q, n_heads, head_dim_v}, qc.options().dtype(at::kFloat));

  const int rc = ggml_attn_metal_flash_attn(
      mtl_buffer(qc), byte_offset(qc), mtl_buffer(kc), byte_offset(kc), mtl_buffer(vc),
      byte_offset(vc), mtl_buffer(mc), byte_offset(mc), mtl_buffer(pad), byte_offset(pad),
      mtl_buffer(tmp), byte_offset(tmp), mtl_buffer(blk), byte_offset(blk), mtl_buffer(dst),
      byte_offset(dst), n_seqs, n_heads,
      n_heads_kv, n_q, n_kv, head_dim_k, head_dim_v, static_cast<float>(scale), has_mask);
  TORCH_CHECK(rc == 0, "ggml-attn: no kernel for n_q ", n_q, ", head_dim ", head_dim_k, "/",
              head_dim_v, " (rc ", rc, ") -- ask `supports_flash_attn` first");
  return dst;
}
