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

// One step (or `n_tokens` steps) of the gated delta rule, ggml's GGML_OP_GATED_DELTA_NET.
//
// Shapes follow ggml's declaration in ggml.h, given here in torch order (last dim fastest):
//   q, k   (n_seqs, n_tokens, n_heads, head_dim)   f32
//   v      (n_seqs, n_tokens, n_heads, head_dim)   f32
//   g      (n_seqs, n_tokens, n_heads)             f32   scalar gate per head
//   beta   (n_seqs, n_tokens, n_heads)             f32
//   state  (n_seqs, n_heads, head_dim, head_dim)   f32   initial recurrent state, indexed
//          [value_index][key_index] -- transposed relative to the k-outer-v product. Upstream
//          stores it that way so a thread's row is contiguous, and the op's own output is in the
//          same layout, so a caller that keeps its cache as this returns never transposes: a
//          fresh state is zeros, which is symmetric.
//
// `dst` is one allocation holding the outputs followed by the final state, which is how upstream's
// kernel writes them:
//   dst[0 .. n_seqs*n_tokens*n_heads*head_dim)            the outputs
//   dst[that .. + n_seqs*n_heads*head_dim*head_dim)       the final recurrent state
// The caller slices the two views out of it rather than getting two buffers, because the kernel
// takes a single destination pointer and offsets within it.
//
// `q` and `k` must already carry one head per value head: upstream maps a value head to a key head
// with `i21 % ne01`, which is the tiled convention, and passing them pre-expanded makes that the
// identity so the caller's own head order is the one that applies.
//
// The kernel scales the output by 1/sqrt(head_dim) itself.
int ggml_gdn_metal_gated_delta_net(void *q, size_t q_off, void *k, size_t k_off, void *v,
                                    size_t v_off, void *g, size_t g_off, void *beta,
                                    size_t beta_off, void *state, size_t state_off, void *dst,
                                    size_t dst_off, int64_t n_seqs, int64_t n_tokens,
                                    int64_t n_heads, int64_t head_dim);

// True when this build can run the gated delta rule for `head_dim`. Upstream's kernel covers a
// state row with 32 threads times `nsg = head_dim/32` values each, so head_dim must be a multiple
// of 32, and the templates instantiated upstream stop at nsg = 4.
int ggml_gdn_metal_supports_gated_delta_net(int64_t head_dim);

// Flash attention, ggml's GGML_OP_FLASH_ATTN_EXT, decode path only (see `supports` below).
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
// `pad` and `tmp` are scratch the kernels need; ask for their sizes first. `pad` may be a single
// element when `n_kv` is already a multiple of the cache-values-per-simdgroup.
int ggml_gdn_metal_flash_attn(void *q, size_t q_off, void *k, size_t k_off, void *v, size_t v_off,
                               void *mask, size_t mask_off, void *pad, size_t pad_off, void *tmp,
                               size_t tmp_off, void *dst, size_t dst_off, int64_t n_seqs,
                               int64_t n_heads, int64_t n_heads_kv, int64_t n_q, int64_t n_kv,
                               int64_t head_dim_k, int64_t head_dim_v, float scale, int has_mask);

// Scratch sizes in floats for the shapes above, so the caller allocates them from torch.
void ggml_gdn_metal_flash_attn_scratch(int64_t n_seqs, int64_t n_heads, int64_t n_heads_kv,
                                        int64_t n_q, int64_t n_kv, int64_t head_dim_k,
                                        int64_t head_dim_v, int has_mask, int64_t *pad_floats,
                                        int64_t *tmp_floats);

// The two scalar gates of a gated-delta-net step, in one dispatch:
//   beta = sigmoid(b)                                  (n_heads values)
//   g    = -exp(a_log) * softplus(a + dt_bias)         (n_heads values)
// All six buffers are f32 and `n_heads` long. Worth a kernel not for the arithmetic but for the launches:
// six torch ops over 32 floats cost 59 us a layer, more than the recurrence they feed.
int ggml_gdn_metal_delta_gates(void *b, size_t b_off, void *a, size_t a_off, void *a_log, size_t a_log_off,
                                void *dt_bias, size_t dt_bias_off, void *beta, size_t beta_off, void *g,
                                size_t g_off, int64_t n_heads);

// RMS-normalise each row of `n_rows` rows of `n_cols` f32 values and scale by `1 + weight`.
//
// The `1 +` is folded in because models that store the norm weight zero-centred would otherwise need it
// materialised: either a launch per norm, or a cached tensor that is wrong the moment the weight changes.
int ggml_gdn_metal_rms_norm_gain(void *x, size_t x_off, void *weight, size_t weight_off, void *out,
                                  size_t out_off, int64_t n_rows, int64_t n_cols, float eps);

// `rms_norm(x) * weight * silu(gate)`, the tail of a gated-delta-net layer, in one dispatch.
int ggml_gdn_metal_rms_norm_gate(void *x, size_t x_off, void *weight, size_t weight_off, void *gate,
                                  size_t gate_off, void *out, size_t out_off, int64_t n_rows,
                                  int64_t n_cols, float eps);

// L2-normalise each row of `n_rows` rows of `n_cols` f32 values: `x / max(|x|, eps)`.
// That is ggml's own epsilon placement, and `F.normalize`'s; a caller writing `x * rsqrt(sum + eps)`
// differs from it by ~3e-8, including for all-zero and denormal rows.
int ggml_gdn_metal_l2_norm(void *x, size_t x_off, void *out, size_t out_off, int64_t n_rows,
                            int64_t n_cols, float eps);

// SwiGLU over two separate tensors: `out = silu(gate) * up`, one dispatch instead of two.
// `n_rows` rows of `n_cols` f32 values each, all three tensors the same shape.
int ggml_gdn_metal_swiglu(void *gate, size_t gate_off, void *up, size_t up_off, void *out,
                           size_t out_off, int64_t n_rows, int64_t n_cols);

// True when this build has a flash-attention kernel for these shapes. Only the vector path is
// ported, which is the one upstream itself picks for decode (`n_q < 20`); anything wider should use
// torch's own attention rather than this.
int ggml_gdn_metal_supports_flash_attn(int64_t n_q, int64_t head_dim_k, int64_t head_dim_v);

// `causal_conv1d_update` in one dispatch: reads the cached state and this token, rolls the cache
// forward, and applies the bias and the activation. `state` is `(channels, swidth)` and is updated
// in place; `out` is `(channels,)`.
int ggml_gdn_metal_causal_conv_update(void *state, size_t state_off, void *x, size_t x_off,
                                      void *weight, size_t weight_off, void *bias, size_t bias_off,
                                      void *out, size_t out_off, int64_t channels, int64_t swidth,
                                      int64_t k, int has_bias, int silu);
}
