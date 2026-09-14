#pragma once

#include <cstddef>
#include <cstdint>

/* The ggml-facing boundary.
 *
 * Everything that touches ggml or Metal lives behind these functions, so torch headers and ggml
 * headers never meet in one translation unit. Buffers arrive as (MTLBuffer, byte offset) pairs
 * because a torch tensor's storage is a whole MTLBuffer that the tensor may only be a view into.
 *
 * Each returns 0 on success, or non-zero when this build has no kernel for the shape it was asked
 * for, which the caller reports rather than faulting.
 */

extern "C" {

// Flash attention, ggml's GGML_OP_FLASH_ATTN_EXT, vector path only (see `supports` below).
//
// Shapes in torch order, which is also SDPA's:
//   q     (n_seqs, n_heads,    n_q,  head_dim_k)  f32
//   k     (n_seqs, n_heads_kv, n_kv, head_dim_k)  f32
//   v     (n_seqs, n_heads_kv, n_kv, head_dim_v)  f32
//   mask  (n_seqs, 1, n_q, n_kv)                  f16, additive; may be null
//   dst   (n_seqs, n_q, n_heads, head_dim_v)      f32   <- note: heads and tokens swapped
//
// Grouped-query attention is native: `n_heads_kv` may be smaller than `n_heads` and no expansion of
// k or v is needed. `dst` comes out with tokens before heads, which is the layout a caller wants
// anyway -- it is what SDPA's own `.transpose(1, 2)` produces.
//
// A null mask means "attend to everything". ggml has no `is_causal` argument -- in ggml the mask *is*
// the causality -- so a caller wanting causal attention with `n_q > 1` must supply one.
//
// `pad`, `tmp` and `blk` are scratch the kernels need; ask for their sizes first. Each may be a single
// element when the path in use does not read it -- `pad` when `n_kv` is already a multiple of the
// cache-values-per-simdgroup, `tmp` on the tiled path, `blk` on the vector path or without a mask.
int ggml_attn_metal_flash_attn(void *q, size_t q_off, void *k, size_t k_off, void *v, size_t v_off,
                          void *mask, size_t mask_off, void *pad, size_t pad_off, void *tmp,
                          size_t tmp_off, void *blk, size_t blk_off, void *dst, size_t dst_off, int64_t n_seqs, int64_t n_heads,
                          int64_t n_heads_kv, int64_t n_q, int64_t n_kv, int64_t head_dim_k,
                          int64_t head_dim_v, float scale, int has_mask);

// Scratch sizes in floats for the shapes above, so the caller allocates them from torch.
void ggml_attn_metal_flash_attn_scratch(int64_t n_seqs, int64_t n_heads, int64_t n_heads_kv, int64_t n_q,
                                   int64_t n_kv, int64_t head_dim_k, int64_t head_dim_v,
                                   int has_mask, int64_t *pad_floats, int64_t *tmp_floats,
                                   int64_t *blk_floats);

// True when this build has a flash-attention kernel for these shapes. Only the vector path is
// ported, which is the one upstream itself picks for decode (`n_q < 20`); anything wider should use
// torch's own attention rather than this.
int ggml_attn_metal_supports_flash_attn(int64_t n_q, int64_t head_dim_k, int64_t head_dim_v);
}
