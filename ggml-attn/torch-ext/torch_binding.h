#pragma once

#include <torch/torch.h>

#include <optional>

// Flash attention through ggml's `flash_attn_ext`, vector path -- the one upstream picks for decode.
//
// Shapes, torch order (SDPA's own):
//   q     (n_seqs, n_heads,    n_q,  head_dim)  f32
//   k, v  (n_seqs, n_heads_kv, n_kv, head_dim)  f32   -- GQA is native, do not expand them
//   mask  (n_seqs, 1, n_q, n_kv)                additive, cast to f16 internally; may be undefined
//
// Returns `(n_seqs, n_q, n_heads, head_dim)` f32 -- tokens before heads, which is what SDPA gives
// after its own transpose, so a caller usually wants exactly this.
//
// ggml takes no `is_causal` argument: the mask *is* the causality, and no mask means attend to
// everything. For `n_q > 1` a causal caller must pass one.
at::Tensor flash_attn(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                      const std::optional<at::Tensor> &mask, double scale);

// Whether this build has a flash-attention kernel for these shapes. Only the decode-shaped vector
// path is ported: ask, and fall back to torch's attention when it says no.
bool supports_flash_attn(int64_t n_q, int64_t head_dim_k, int64_t head_dim_v);
