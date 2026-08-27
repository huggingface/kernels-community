#include <torch/all.h>
#include "src/fmha_fwd.hpp"


namespace FLASH_NAMESPACE {

inline at::Tensor ensure_contiguous(const at::Tensor& tensor) {
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cpu(), #x " must be on CPU")


std::vector<at::Tensor>
mha_varlen_fwd(
              at::Tensor &q,  // [total_q, num_heads, head_size]
              const at::Tensor &k,  // [total_k, num_heads_k, head_size]
              const at::Tensor &v,  // [total_k, num_heads_k, head_size]
              std::optional<at::Tensor> &out_, // [total_q, num_heads, head_size]
              const at::Tensor &cu_seqlens_q,  // [batch_size + 1]
              const at::Tensor &cu_seqlens_k,  // [batch_size + 1]
              std::optional<at::Tensor> &seqused_k, // [batch_size] (unused in CPU impl)
              std::optional<const at::Tensor> &leftpad_k_, // [batch_size] (unused in CPU impl)
              std::optional<at::Tensor> &block_table_, // (unused, paged cache not supported)
              std::optional<at::Tensor> &alibi_slopes_, // (unused in CPU impl)
              int max_seqlen_q,
              const int max_seqlen_k,
              const float p_dropout,  // (unused in CPU impl)
              const float softmax_scale,
              const bool zero_tensors,  // (unused in CPU impl)
              bool is_causal,
              int window_size_left,  // must be -1
              int window_size_right,  // must be -1
              const float softcap,  // (unused in CPU impl)
              const bool return_softmax,  // (unused in CPU impl)
              std::optional<at::Generator> gen_) {  // (unused in CPU impl)

    // check inputs
    q = ensure_contiguous(q);
    const auto sizes = q.sizes();
    const int head_size = sizes[2];

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(k.size(-1) == head_size, "Key head dimension must match Query head dimension");
    TORCH_CHECK(v.size(-1) == head_size, "Value head dimension must match Query head dimension");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

    at::Tensor k_contig = ensure_contiguous(k);
    at::Tensor v_contig = ensure_contiguous(v);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
    } else {
        out = torch::zeros_like(q);
    }

    bool is_paged = block_table_.has_value() && block_table_->defined();

    // Current implementation limitations
    TORCH_CHECK(!is_paged, "Paged cache is not supported in CPU implementation");
    TORCH_CHECK(window_size_left == -1, "Sliding window attention (window_size_left != -1) is not supported in CPU implementation");
    TORCH_CHECK(window_size_right == -1, "Sliding window attention (window_size_right != -1) is not supported in CPU implementation");

    flash_attn_cpu::fmha_fwd_varlen_impl(
        q, k_contig, v_contig, out,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale, is_causal);

    out = ensure_contiguous(out);

    // TODO: current do not support store softmax_lse out
    // hard code to return empty tensor for softmax_lse, S_dmask, rng_state
    at::Tensor softmax_lse;
    at::Tensor S_dmask;
    at::Tensor rng_state;
    return {out, softmax_lse, S_dmask, rng_state};
  }
}  // namespace FLASH_NAMESPACE


std::vector<torch::Tensor>
mha_varlen_fwd(
    torch::Tensor &q,  // [total_q, num_heads, head_size]
    const torch::Tensor &k,  // [total_k, num_heads_k, head_size]
    const torch::Tensor &v,  // [total_k, num_heads_k, head_size]
    std::optional<torch::Tensor> out_, // [total_q, num_heads, head_size]
    const torch::Tensor &cu_seqlens_q,  // [batch_size + 1]
    const torch::Tensor &cu_seqlens_k,  // [batch_size + 1]
    std::optional<torch::Tensor> seqused_k, // (unused in CPU impl)
    std::optional<torch::Tensor> leftpad_k_, // (unused in CPU impl)
    std::optional<torch::Tensor> block_table_, // (unused, paged cache not supported)
    std::optional<torch::Tensor> alibi_slopes_, // (unused in CPU impl)
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
