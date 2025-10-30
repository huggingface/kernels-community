#include <torch/all.h>

#include "src/prefill.hpp"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_xpu(), #x " must be on XPU")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

inline int round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

std::vector<at::Tensor>
mha_fwd(
        at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {

    auto device_idx = q.device().index();
    COMPAT::select_device(device_idx);

    // check inputs
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

    // XPU requires head_size to be a multiple of 32
    const int head_size_padded = round_multiple(head_size_og, 32);

    at::Tensor q_padded = q;
    at::Tensor k_padded = k;
    at::Tensor v_padded = v;

    // Apply padding if needed
    if (head_size_og != head_size_padded) {
        const int pad_size = head_size_padded - head_size_og;
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, pad_size}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, pad_size}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, pad_size}));
    }

    at::Tensor out_padded;
    if (out_.has_value()) {
        auto out_val = out_.value();
        if (head_size_og != head_size_padded) {
            const int pad_size = head_size_padded - head_size_og;
            out_padded = torch::nn::functional::pad(out_val, torch::nn::functional::PadFuncOptions({0, pad_size}));
        } else {
            out_padded = out_val;
        }
    } else {
        out_padded = torch::zeros_like(q_padded);
    }

    cutlass_prefill_fixed_impl(q_padded, k_padded, v_padded, out_padded, softmax_scale, is_causal);

    // Remove padding from output
    at::Tensor out = out_padded;
    if (head_size_og != head_size_padded) {
        out = out_padded.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)});
    }
    out = out.contiguous();

    // TODO: current do not support store softmax_lse out
    // hard code to return empty tensor for softmax_lse, S_dmask, rng_state
    at::Tensor softmax_lse;
    at::Tensor S_dmask;
    at::Tensor rng_state;
    return {out, softmax_lse, S_dmask, rng_state};
  }

std::vector<at::Tensor>
mha_varlen_fwd(
              at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
              const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
              const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
              std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const at::Tensor &cu_seqlens_q,  // b+1
              const at::Tensor &cu_seqlens_k,  // b+1
              std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
              std::optional<const at::Tensor> &leftpad_k_, // batch_size
              std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
              std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
              int max_seqlen_q,
              const int max_seqlen_k,
              const float p_dropout,
              const float softmax_scale,
              const bool zero_tensors,
              bool is_causal,
              int window_size_left,
              int window_size_right,
              const float softcap,
              const bool return_softmax,
              std::optional<at::Generator> gen_) {

    auto device_idx = q.device().index();
    COMPAT::select_device(device_idx);

    // check inputs
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);
    TORCH_CHECK(cu_seqlens_q.dim() == 1, "cu_seqlens_q must be 1-dimensional, but got ", cu_seqlens_q.dim(), " dimensions");
    TORCH_CHECK(cu_seqlens_k.dim() == 1, "cu_seqlens_k must be 1-dimensional, but got ", cu_seqlens_k.dim(), " dimensions");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    // Extract dimensions
    const auto sizes = q.sizes();
    const int total_q = sizes[0];
    const int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    const int batch_size = cu_seqlens_q.numel() - 1;

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    // XPU requires head_size to be a multiple of 32
    const int head_size_padded = round_multiple(head_size_og, 32);

    at::Tensor q_padded = q;
    at::Tensor k_padded = k;
    at::Tensor v_padded = v;

    // Apply padding if needed
    if (head_size_og != head_size_padded) {
        const int pad_size = head_size_padded - head_size_og;
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, pad_size}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, pad_size}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, pad_size}));
    }

    at::Tensor out_padded;
    if (out_.has_value()) {
        auto out_val = out_.value();
        if (head_size_og != head_size_padded) {
            const int pad_size = head_size_padded - head_size_og;
            out_padded = torch::nn::functional::pad(out_val, torch::nn::functional::PadFuncOptions({0, pad_size}));
        } else {
            out_padded = out_val;
        }
    } else {
        out_padded = torch::zeros_like(q_padded);
    }

    cutlass_prefill_varlen_impl(q_padded, k_padded, v_padded, out_padded,
                              cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k,
                              softmax_scale, is_causal);

    // Remove padding from output
    at::Tensor out = out_padded;
    if (head_size_og != head_size_padded) {
        out = out_padded.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                torch::indexing::Slice(0, head_size_og)});
    }
    out = out.contiguous();

    // TODO: current do not support store softmax_lse out
    // hard code to return empty tensor for softmax_lse, S_dmask, rng_state
    at::Tensor softmax_lse;
    at::Tensor S_dmask;
    at::Tensor rng_state;
    return {out, softmax_lse, S_dmask, rng_state};
  }
}  // namespace FLASH_NAMESPACE

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
std::vector<torch::Tensor>
mha_fwd(
    torch::Tensor &q, 
    const torch::Tensor &k, 
    const torch::Tensor &v,
    c10::optional<torch::Tensor> out_,
    c10::optional<torch::Tensor> alibi_slopes_,
    const double p_dropout, 
    const double softmax_scale, 
    bool is_causal,
    const int64_t window_size_left, 
    const int64_t window_size_right,
    const double softcap, 
    const bool return_softmax,
    c10::optional<at::Generator> gen_) {
    return FLASH_NAMESPACE::mha_fwd(
      q, 
      k, 
      v, 
      out_, 
      alibi_slopes_, 
      static_cast<float>(p_dropout),
      static_cast<float>(softmax_scale), 
      is_causal,
      static_cast<int>(window_size_left), 
      static_cast<int>(window_size_right),
      static_cast<float>(softcap), 
      return_softmax,
      gen_
    );
}

std::vector<torch::Tensor>
mha_varlen_fwd(
    torch::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const torch::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const torch::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    std::optional<torch::Tensor> out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const torch::Tensor &cu_seqlens_q,  // b+1
    const torch::Tensor &cu_seqlens_k,  // b+1
    std::optional<torch::Tensor> seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<torch::Tensor> leftpad_k_, // batch_size
    std::optional<torch::Tensor> block_table_, // batch_size x max_num_blocks_per_seq
    std::optional<torch::Tensor> alibi_slopes_, // num_heads or b x num_heads
    int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const double p_dropout,
    const double softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    const double softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_) {    
    return FLASH_NAMESPACE::mha_varlen_fwd(
        const_cast<at::Tensor &>(q), 
        k, 
        v, 
        out_, 
        cu_seqlens_q, 
        cu_seqlens_k,
        seqused_k,
        reinterpret_cast<std::optional<const at::Tensor>&>(leftpad_k_),
        block_table_,
        alibi_slopes_,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        zero_tensors,
        is_causal,
        static_cast<int>(window_size_left), 
        static_cast<int>(window_size_right),
        static_cast<float>(softcap),
        return_softmax,
        gen_
    );
}
