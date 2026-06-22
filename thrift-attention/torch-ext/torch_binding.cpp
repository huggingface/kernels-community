#include <torch/all.h>
#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace {

void check_hi_qkv(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "Q/K/V must be 4D [batch, heads, seq, head_dim]");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0),
                "Q/K/V batch dimensions must match");
    TORCH_CHECK(k.size(1) == v.size(1), "K and V head dimensions must match");
    TORCH_CHECK(k.size(2) == v.size(2), "K and V sequence lengths must match");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3),
                "Q/K/V head dimensions must match");
}

void check_packed_qkv(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t) {
    TORCH_CHECK(q_packed.is_cuda() && k_packed.is_cuda() && v_packed_t.is_cuda(),
                "packed Q/K/V must be CUDA tensors");
    TORCH_CHECK(q_scale.is_cuda() && k_scale.is_cuda() && v_scale_t.is_cuda(),
                "Q/K/V scales must be CUDA tensors");
    TORCH_CHECK(q_packed.dim() == 4 && k_packed.dim() == 4 && v_packed_t.dim() == 4,
                "packed Q/K/V must be 4D tensors");
    TORCH_CHECK(q_packed.size(0) == k_packed.size(0), "packed Q/K batch dimensions must match");
    TORCH_CHECK(q_packed.size(1) % k_packed.size(1) == 0,
                "Q heads must be divisible by KV heads");
    TORCH_CHECK(q_packed.size(3) == k_packed.size(3), "packed Q/K head dimensions must match");
}

int single_query_mixed_target_split_ctas(int total_kv_blocks) {
    if (const char* env = std::getenv("SAGE_MIXED_SPLIT_CTAS")) {
        const int value = std::atoi(env);
        if (value > 0) {
            return value;
        }
    }
    return (total_kv_blocks <= 512) ? 384
         : (total_kv_blocks <= 1024) ? 384
         : 896;
}

}  // namespace

// Implemented in csrc/cuda/sm120/nvfp4/fp4_attention.cu.
void fp4_attention_causal_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void fp4_attention_noncausal_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/mxfp4/fp4_attention.cu.
void fp4_attention_causal_mxfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void fp4_attention_noncausal_mxfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/nvfp4/single_query_fp4_attention.cu.
void fp4_attention_single_query_nvfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* workspace,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/mxfp4/single_query_fp4_attention.cu.
void fp4_attention_single_query_mxfp4(
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* workspace,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);

// Implemented in csrc/cuda/sm120/nvfp4/thrift_attention.cu.
void thrift_attention_causal_nvfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const void* selected_blocks, int topk_count,
    const void* topk_mask, int topk_word_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* rowmax_state, void* rowsum_state,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void thrift_attention_noncausal_nvfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const void* selected_blocks, int topk_count,
    const void* topk_mask, int topk_word_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* rowmax_state, void* rowsum_state,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/mxfp4/thrift_attention.cu.
void thrift_attention_causal_mxfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const void* selected_blocks, int topk_count,
    const void* topk_mask, int topk_word_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* rowmax_state, void* rowsum_state,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
void thrift_attention_noncausal_mxfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const void* selected_blocks, int topk_count,
    const void* topk_mask, int topk_word_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* rowmax_state, void* rowsum_state,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/nvfp4/single_query_attention.cu.
void thrift_attention_single_query_nvfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const int32_t* selected_blocks, int topk_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* workspace,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);
// Implemented in csrc/cuda/sm120/mxfp4/single_query_attention.cu.
void thrift_attention_single_query_mxfp4(
    const void* Q_fp16, const void* K_fp16, const void* V_fp16,
    const int32_t* selected_blocks, int topk_count,
    const void* Q, const void* K, const void* V,
    const void* S_Q, const void* S_K, const void* S_V,
    void* O, void* workspace,
    int bs, int q_len, int kv_len, int kv_capacity,
    int num_q_heads, int num_kv_heads, int head_dim, bool is_bf16);

// Implemented in csrc/cuda/sm120/nvfp4/quantization.cu.
void nvfp4_quantise(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);
void nvfp4_quantise_permute_seq(
    void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool inverse, bool is_bf16);
void nvfp4_quantise_transpose(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);
void nvfp4_quantise_transpose_permute_seq(
    void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool inverse, bool is_bf16);
// Implemented in csrc/cuda/sm120/mxfp4/quantization.cu.
void mxfp4_quantise(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);
void mxfp4_quantise_permute_seq(
    void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool inverse, bool is_bf16);
void mxfp4_quantise_transpose(void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool is_bf16);
void mxfp4_quantise_transpose_permute_seq(
    void* X, void* X_fp4, void* X_scale, int bs, int seq_len, int head_dim, bool inverse, bool is_bf16);

// Implemented in csrc/cuda/sm120/shared/block_selection.cu.
void block_mean_topk(
    const void* q_mean,
    const void* k_mean,
    void* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int head_dim,
    int topk_count,
    bool causal,
    bool is_bf16);
void quest_block_topk(
    const void* q_mean,
    const void* k_min,
    const void* k_max,
    void* topk_out,
    int flat_heads,
    int num_q_blocks,
    int num_kv_blocks,
    int head_dim,
    int topk_count,
    bool causal,
    bool is_bf16);
void single_query_key_mean_topk(
    const void* q_grouped,
    const void* k_mean,
    void* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int head_dim,
    int topk_count,
    bool is_bf16);
void single_query_quest_topk(
    const void* q_grouped,
    const void* k_min,
    const void* k_max,
    void* topk_out,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_stat_capacity_blocks,
    int head_dim,
    int topk_count,
    bool is_bf16);
void single_query_key_mean_topk_chunked(
    const void* q_grouped,
    const void* k_mean,
    void* topk_out,
    void* local_scores,
    void* local_indices,
    void* done_counts,
    int flat_heads,
    int groups,
    int num_kv_blocks,
    int k_mean_capacity_blocks,
    int head_dim,
    int topk_count,
    int chunk_count,
    int local_count,
    bool is_bf16);
void pack_topk_mask(
    const int32_t* topk,
    void* topk_mask,
    int row_count,
    int topk_count,
    int total_units,
    int word_count);

static at::Tensor fp4_attention_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool causal,
    bool is_bf16) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));

    if (causal) {
        fp4_attention_causal_nvfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else {
        fp4_attention_noncausal_nvfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    }

    return out;
}

static at::Tensor fp4_attention_causal_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return fp4_attention_nvfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, true, is_bf16);
}

static at::Tensor fp4_attention_noncausal_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return fp4_attention_nvfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, is_bf16);
}

static at::Tensor fp4_attention_mxfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool causal,
    bool is_bf16) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));

    if (causal) {
        fp4_attention_causal_mxfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else {
        fp4_attention_noncausal_mxfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    }

    return out;
}

static at::Tensor fp4_attention_causal_mxfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return fp4_attention_mxfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, true, is_bf16);
}

static at::Tensor fp4_attention_noncausal_mxfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return fp4_attention_mxfp4_packed(
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, is_bf16);
}

static at::Tensor fp4_attention_single_query_nvfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(q_len >= 1 && q_len <= 16,
                "single-query FP4 attention expects grouped query length in [1, 16], got ",
                q_len);

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));

    constexpr int block_kv = 64;
    const int total_kv_blocks = (kv_len + block_kv - 1) / block_kv;
    const bool use_split = total_kv_blocks >= 128;

    auto dispatch = [&](void* workspace) {
        fp4_attention_single_query_nvfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), workspace,
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    };

    if (!use_split) {
        dispatch(nullptr);
        return out;
    }

    int num_kv_splits = std::max(1, std::min(total_kv_blocks, (256 + flat_q_heads - 1) / flat_q_heads));
    num_kv_splits = std::min(num_kv_splits, total_kv_blocks);
    const int64_t workspace_elems =
        static_cast<int64_t>(flat_q_heads) * num_kv_splits * 16 * (head_dim + 2) + flat_q_heads;
    at::Tensor workspace = at::empty({workspace_elems},
        at::TensorOptions().dtype(at::kFloat).device(q_packed.device()));
    dispatch(workspace.data_ptr());
    return out;
}

static at::Tensor fp4_attention_single_query_mxfp4_packed(
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(q_len >= 1 && q_len <= 16,
                "single-query FP4 attention expects grouped query length in [1, 16], got ",
                q_len);

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));

    constexpr int block_kv = 64;
    const int total_kv_blocks = (kv_len + block_kv - 1) / block_kv;
    const bool use_split = total_kv_blocks >= 128;

    auto dispatch = [&](void* workspace) {
        fp4_attention_single_query_mxfp4(
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), workspace,
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    };

    if (!use_split) {
        dispatch(nullptr);
        return out;
    }

    int num_kv_splits = std::max(1, std::min(total_kv_blocks, (256 + flat_q_heads - 1) / flat_q_heads));
    num_kv_splits = std::min(num_kv_splits, total_kv_blocks);
    const int64_t workspace_elems =
        static_cast<int64_t>(flat_q_heads) * num_kv_splits * 16 * (head_dim + 2) + flat_q_heads;
    at::Tensor workspace = at::empty({workspace_elems},
        at::TensorOptions().dtype(at::kFloat).device(q_packed.device()));
    dispatch(workspace.data_ptr());
    return out;
}

static at::Tensor thrift_attention_nvfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool causal,
    bool is_bf16) {
    check_hi_qkv(q_hi, k_hi, v_hi);
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);
    TORCH_CHECK(selected_blocks.is_cuda(), "selected_blocks must be a CUDA tensor");
    TORCH_CHECK(selected_blocks.dtype() == at::kInt, "selected_blocks must be int32");
    TORCH_CHECK(selected_blocks.is_contiguous(), "selected_blocks must be contiguous");
    TORCH_CHECK(selected_blocks.dim() == 3,
                "selected_blocks must be [batch * heads, q_blocks, top_k]");

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    constexpr int block_q = 64;
    const int num_q_blocks = (q_len + block_q - 1) / block_q;
    const int topk_count = selected_blocks.size(2);
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(selected_blocks.size(0) == flat_q_heads,
                "selected_blocks first dimension must equal batch * Q heads");
    TORCH_CHECK(selected_blocks.size(1) == num_q_blocks,
                "selected_blocks second dimension must equal number of Q blocks");

    if (topk_count == 0) {
        return fp4_attention_nvfp4_packed(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, causal, is_bf16);
    }

    constexpr int topk_unit_tokens = 64;
    const int total_topk_units = (kv_len + topk_unit_tokens - 1) / topk_unit_tokens;
    const int topk_word_count = (total_topk_units + 63) / 64;
    TORCH_CHECK(topk_word_count <= 32,
                "ThriftAttention supports at most 2048 64-token KV blocks, got ",
                total_topk_units);

    auto mask_opts = at::TensorOptions().dtype(at::kLong).device(selected_blocks.device());
    at::Tensor topk_mask = at::empty({flat_q_heads, num_q_blocks, topk_word_count}, mask_opts);
    pack_topk_mask(
        static_cast<const int32_t*>(selected_blocks.data_ptr()),
        topk_mask.data_ptr(),
        flat_q_heads * num_q_blocks, topk_count, total_topk_units, topk_word_count);

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));
    auto state_opts = at::TensorOptions().dtype(at::kFloat).device(q_packed.device());
    at::Tensor rowmax_state = at::empty({batch, num_q_heads, q_len}, state_opts);
    at::Tensor rowsum_state = at::empty({batch, num_q_heads, q_len}, state_opts);

    if (causal) {
        thrift_attention_causal_nvfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            selected_blocks.data_ptr(), topk_count,
            topk_mask.data_ptr(), topk_word_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), rowmax_state.data_ptr(), rowsum_state.data_ptr(),
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else {
        thrift_attention_noncausal_nvfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            selected_blocks.data_ptr(), topk_count,
            topk_mask.data_ptr(), topk_word_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), rowmax_state.data_ptr(), rowsum_state.data_ptr(),
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    }

    return out;
}

static at::Tensor thrift_attention_causal_nvfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return thrift_attention_nvfp4_packed(
        q_hi, k_hi, v_hi, selected_blocks,
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, true, is_bf16);
}

static at::Tensor thrift_attention_noncausal_nvfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return thrift_attention_nvfp4_packed(
        q_hi, k_hi, v_hi, selected_blocks,
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, is_bf16);
}

static at::Tensor thrift_attention_mxfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool causal,
    bool is_bf16) {
    check_hi_qkv(q_hi, k_hi, v_hi);
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);
    TORCH_CHECK(selected_blocks.is_cuda(), "selected_blocks must be a CUDA tensor");
    TORCH_CHECK(selected_blocks.dtype() == at::kInt, "selected_blocks must be int32");
    TORCH_CHECK(selected_blocks.is_contiguous(), "selected_blocks must be contiguous");
    TORCH_CHECK(selected_blocks.dim() == 3,
                "selected_blocks must be [batch * heads, q_blocks, top_k]");

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    constexpr int block_q = 64;
    const int num_q_blocks = (q_len + block_q - 1) / block_q;
    const int topk_count = selected_blocks.size(2);
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(selected_blocks.size(0) == flat_q_heads,
                "selected_blocks first dimension must equal batch * Q heads");
    TORCH_CHECK(selected_blocks.size(1) == num_q_blocks,
                "selected_blocks second dimension must equal number of Q blocks");

    if (topk_count == 0) {
        return fp4_attention_mxfp4_packed(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, causal, is_bf16);
    }

    constexpr int topk_unit_tokens = 64;
    const int total_topk_units = (kv_len + topk_unit_tokens - 1) / topk_unit_tokens;
    const int topk_word_count = (total_topk_units + 63) / 64;
    TORCH_CHECK(topk_word_count <= 32,
                "ThriftAttention supports at most 2048 64-token KV blocks, got ",
                total_topk_units);

    auto mask_opts = at::TensorOptions().dtype(at::kLong).device(selected_blocks.device());
    at::Tensor topk_mask = at::empty({flat_q_heads, num_q_blocks, topk_word_count}, mask_opts);
    pack_topk_mask(
        static_cast<const int32_t*>(selected_blocks.data_ptr()),
        topk_mask.data_ptr(),
        flat_q_heads * num_q_blocks, topk_count, total_topk_units, topk_word_count);

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));
    auto state_opts = at::TensorOptions().dtype(at::kFloat).device(q_packed.device());
    at::Tensor rowmax_state = at::empty({batch, num_q_heads, q_len}, state_opts);
    at::Tensor rowsum_state = at::empty({batch, num_q_heads, q_len}, state_opts);

    if (causal) {
        thrift_attention_causal_mxfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            selected_blocks.data_ptr(), topk_count,
            topk_mask.data_ptr(), topk_word_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), rowmax_state.data_ptr(), rowsum_state.data_ptr(),
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    } else {
        thrift_attention_noncausal_mxfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            selected_blocks.data_ptr(), topk_count,
            topk_mask.data_ptr(), topk_word_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), rowmax_state.data_ptr(), rowsum_state.data_ptr(),
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    }

    return out;
}

static at::Tensor thrift_attention_causal_mxfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return thrift_attention_mxfp4_packed(
        q_hi, k_hi, v_hi, selected_blocks,
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, true, is_bf16);
}

static at::Tensor thrift_attention_noncausal_mxfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    return thrift_attention_mxfp4_packed(
        q_hi, k_hi, v_hi, selected_blocks,
        q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, false, is_bf16);
}

static at::Tensor thrift_attention_single_query_nvfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    check_hi_qkv(q_hi, k_hi, v_hi);
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);
    TORCH_CHECK(selected_blocks.is_cuda(), "selected_blocks must be a CUDA tensor");
    TORCH_CHECK(selected_blocks.dtype() == at::kInt, "selected_blocks must be int32");
    TORCH_CHECK(selected_blocks.is_contiguous(), "selected_blocks must be contiguous");
    TORCH_CHECK(selected_blocks.dim() == 2,
                "selected_blocks must be [batch * grouped_heads, top_k]");

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int topk_count = selected_blocks.size(1);
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(q_len >= 1 && q_len <= 16,
                "single-query ThriftAttention expects grouped query length in [1, 16], got ",
                q_len);
    TORCH_CHECK(q_hi.size(0) == batch && q_hi.size(1) == num_q_heads &&
                q_hi.size(2) == q_len && q_hi.size(3) == head_dim,
                "q_hi shape must match packed Q");
    TORCH_CHECK(selected_blocks.size(0) == flat_q_heads,
                "selected_blocks first dimension must equal batch * grouped_heads");

    constexpr int block_kv = 64;
    const int total_kv_blocks = (kv_len + block_kv - 1) / block_kv;
    TORCH_CHECK(topk_count >= 0 && topk_count <= total_kv_blocks,
                "selected block count must be in [0, number of KV blocks]");

    if (topk_count == 0) {
        return fp4_attention_single_query_nvfp4_packed(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, is_bf16);
    }

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));
    const bool use_split = total_kv_blocks >= 64;

    auto dispatch = [&](void* workspace) {
        thrift_attention_single_query_nvfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            static_cast<const int32_t*>(selected_blocks.data_ptr()), topk_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), workspace,
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    };

    if (!use_split) {
        dispatch(nullptr);
        return out;
    }

    const int target_split_ctas = single_query_mixed_target_split_ctas(total_kv_blocks);
    int num_kv_splits =
        std::max(1, std::min(total_kv_blocks, (target_split_ctas + flat_q_heads - 1) / flat_q_heads));
    num_kv_splits = std::min(num_kv_splits, total_kv_blocks);
    const int64_t workspace_elems =
        static_cast<int64_t>(flat_q_heads) * num_kv_splits * q_len * (head_dim + 2) + flat_q_heads;
    at::Tensor workspace = at::empty({workspace_elems},
        at::TensorOptions().dtype(at::kFloat).device(q_packed.device()));
    dispatch(workspace.data_ptr());
    return out;
}

static at::Tensor thrift_attention_single_query_mxfp4_packed(
    const at::Tensor& q_hi,
    const at::Tensor& k_hi,
    const at::Tensor& v_hi,
    const at::Tensor& selected_blocks,
    const at::Tensor& q_packed,
    const at::Tensor& k_packed,
    const at::Tensor& v_packed_t,
    const at::Tensor& q_scale,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale_t,
    bool is_bf16 = false) {
    check_hi_qkv(q_hi, k_hi, v_hi);
    check_packed_qkv(q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t);
    TORCH_CHECK(selected_blocks.is_cuda(), "selected_blocks must be a CUDA tensor");
    TORCH_CHECK(selected_blocks.dtype() == at::kInt, "selected_blocks must be int32");
    TORCH_CHECK(selected_blocks.is_contiguous(), "selected_blocks must be contiguous");
    TORCH_CHECK(selected_blocks.dim() == 2,
                "selected_blocks must be [batch * grouped_heads, top_k]");

    const int batch = q_packed.size(0);
    const int num_q_heads = q_packed.size(1);
    const int num_kv_heads = k_packed.size(1);
    const int flat_q_heads = batch * num_q_heads;
    const int q_len = q_packed.size(2);
    const int kv_len = k_packed.size(2);
    const int head_dim = q_packed.size(3) * 2;
    const int topk_count = selected_blocks.size(1);
    const int kv_capacity = static_cast<int>(k_packed.stride(1)) / (head_dim / 2);
    const at::ScalarType out_dtype = is_bf16 ? at::kBFloat16 : at::kHalf;
    TORCH_CHECK(q_len >= 1 && q_len <= 16,
                "single-query ThriftAttention expects grouped query length in [1, 16], got ",
                q_len);
    TORCH_CHECK(q_hi.size(0) == batch && q_hi.size(1) == num_q_heads &&
                q_hi.size(2) == q_len && q_hi.size(3) == head_dim,
                "q_hi shape must match packed Q");
    TORCH_CHECK(selected_blocks.size(0) == flat_q_heads,
                "selected_blocks first dimension must equal batch * grouped_heads");

    constexpr int block_kv = 64;
    const int total_kv_blocks = (kv_len + block_kv - 1) / block_kv;
    TORCH_CHECK(topk_count >= 0 && topk_count <= total_kv_blocks,
                "selected block count must be in [0, number of KV blocks]");

    if (topk_count == 0) {
        return fp4_attention_single_query_mxfp4_packed(
            q_packed, k_packed, v_packed_t, q_scale, k_scale, v_scale_t, is_bf16);
    }

    at::Tensor out = at::empty({batch, num_q_heads, q_len, head_dim},
        at::TensorOptions().dtype(out_dtype).device(q_packed.device()));
    const bool use_split = total_kv_blocks >= 64;

    auto dispatch = [&](void* workspace) {
        thrift_attention_single_query_mxfp4(
            q_hi.data_ptr(), k_hi.data_ptr(), v_hi.data_ptr(),
            static_cast<const int32_t*>(selected_blocks.data_ptr()), topk_count,
            q_packed.data_ptr(), k_packed.data_ptr(), v_packed_t.data_ptr(),
            q_scale.data_ptr(), k_scale.data_ptr(), v_scale_t.data_ptr(),
            out.data_ptr(), workspace,
            flat_q_heads, q_len, kv_len, kv_capacity,
            num_q_heads, num_kv_heads, head_dim, is_bf16);
    };

    if (!use_split) {
        dispatch(nullptr);
        return out;
    }

    const int target_split_ctas = single_query_mixed_target_split_ctas(total_kv_blocks);
    int num_kv_splits =
        std::max(1, std::min(total_kv_blocks, (target_split_ctas + flat_q_heads - 1) / flat_q_heads));
    num_kv_splits = std::min(num_kv_splits, total_kv_blocks);
    const int64_t workspace_elems =
        static_cast<int64_t>(flat_q_heads) * num_kv_splits * q_len * (head_dim + 2) + flat_q_heads;
    at::Tensor workspace = at::empty({workspace_elems},
        at::TensorOptions().dtype(at::kFloat).device(q_packed.device()));
    dispatch(workspace.data_ptr());
    return out;
}

static std::vector<at::Tensor> nvfp4_quantize(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 16}, opts_f8);

    nvfp4_quantise(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> nvfp4_quantize_permuted(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 16}, opts_f8);

    nvfp4_quantise_permute_seq(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, false, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> nvfp4_quantize_transposed(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);

    constexpr int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;
    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, head_dim, padded_seq / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, head_dim, padded_seq / 16}, opts_f8);

    nvfp4_quantise_transpose(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> nvfp4_quantize_transposed_permuted(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);

    constexpr int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;
    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_f8 = at::TensorOptions().dtype(at::kFloat8_e4m3fn).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, head_dim, padded_seq / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, head_dim, padded_seq / 16}, opts_f8);

    nvfp4_quantise_transpose_permute_seq(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, false, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> mxfp4_quantize(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_e8m0 = at::TensorOptions().dtype(at::kFloat8_e8m0fnu).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 32}, opts_e8m0);

    mxfp4_quantise(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> mxfp4_quantize_permuted(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);

    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_e8m0 = at::TensorOptions().dtype(at::kFloat8_e8m0fnu).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, seq_len, head_dim / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, seq_len, head_dim / 32}, opts_e8m0);

    mxfp4_quantise_permute_seq(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, false, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> mxfp4_quantize_transposed(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);

    constexpr int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;
    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_e8m0 = at::TensorOptions().dtype(at::kFloat8_e8m0fnu).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, head_dim, padded_seq / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, head_dim, padded_seq / 32}, opts_e8m0);

    mxfp4_quantise_transpose(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, is_bf16);

    return {x_packed, x_scale};
}

static std::vector<at::Tensor> mxfp4_quantize_transposed_permuted(const at::Tensor& x, bool is_bf16 = false) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [batch, heads, seq, head_dim]");

    const int batch = x.size(0);
    const int heads = x.size(1);
    const int seq_len = x.size(2);
    const int head_dim = x.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);

    constexpr int seq_block = 128;
    const int padded_seq = ((seq_len + seq_block - 1) / seq_block) * seq_block;
    auto opts_u8 = at::TensorOptions().dtype(at::kByte).device(x.device());
    auto opts_e8m0 = at::TensorOptions().dtype(at::kFloat8_e8m0fnu).device(x.device());
    at::Tensor x_packed = at::empty({batch, heads, head_dim, padded_seq / 2}, opts_u8);
    at::Tensor x_scale = at::empty({batch, heads, head_dim, padded_seq / 32}, opts_e8m0);

    mxfp4_quantise_transpose_permute_seq(
        x.data_ptr(), x_packed.data_ptr(), x_scale.data_ptr(),
        batch * heads, seq_len, head_dim, false, is_bf16);

    return {x_packed, x_scale};
}

static at::Tensor block_mean_topk_impl(
    const at::Tensor& q_mean,
    const at::Tensor& k_mean,
    int topk_count,
    bool causal = true,
    bool is_bf16 = false) {
    TORCH_CHECK(q_mean.is_cuda() && k_mean.is_cuda(),
                "q_mean and k_mean must be CUDA tensors");
    TORCH_CHECK(q_mean.is_contiguous() && k_mean.is_contiguous(),
                "q_mean and k_mean must be contiguous");
    TORCH_CHECK(q_mean.dim() == 4 && k_mean.dim() == 4,
                "q_mean and k_mean must be 4D tensors");
    TORCH_CHECK(q_mean.size(0) == k_mean.size(0),
                "q_mean and k_mean batch dimensions must match");
    TORCH_CHECK(q_mean.size(1) == k_mean.size(1),
                "q_mean and k_mean head dimensions must match");
    TORCH_CHECK(q_mean.size(3) == k_mean.size(3),
                "q_mean and k_mean head_dim must match");

    const int batch = q_mean.size(0);
    const int heads = q_mean.size(1);
    const int flat_heads = batch * heads;
    const int num_q_blocks = q_mean.size(2);
    const int num_kv_blocks = k_mean.size(2);
    const int head_dim = q_mean.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);
    TORCH_CHECK(num_kv_blocks <= 2048,
                "block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    auto opts = at::TensorOptions().dtype(at::kInt).device(q_mean.device());
    at::Tensor topk = at::empty({flat_heads, num_q_blocks, topk_count}, opts);
    if (topk_count == 0) {
        return topk;
    }

    block_mean_topk(
        q_mean.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
        flat_heads, num_q_blocks, num_kv_blocks, head_dim, topk_count, causal, is_bf16);

    return topk;
}

static at::Tensor quest_block_topk_impl(
    const at::Tensor& q_mean,
    const at::Tensor& k_min,
    const at::Tensor& k_max,
    int topk_count,
    bool causal = true,
    bool is_bf16 = false) {
    TORCH_CHECK(q_mean.is_cuda() && k_min.is_cuda() && k_max.is_cuda(),
                "q_mean, k_min, and k_max must be CUDA tensors");
    TORCH_CHECK(q_mean.is_contiguous() && k_min.is_contiguous() && k_max.is_contiguous(),
                "q_mean, k_min, and k_max must be contiguous");
    TORCH_CHECK(q_mean.dim() == 4 && k_min.dim() == 4 && k_max.dim() == 4,
                "q_mean, k_min, and k_max must be 4D tensors");
    TORCH_CHECK(k_min.sizes() == k_max.sizes(),
                "k_min and k_max shapes must match");
    TORCH_CHECK(q_mean.size(0) == k_min.size(0),
                "q_mean and key stat batch dimensions must match");
    TORCH_CHECK(q_mean.size(1) == k_min.size(1),
                "q_mean and key stat head dimensions must match");
    TORCH_CHECK(q_mean.size(3) == k_min.size(3),
                "q_mean and key stat head_dim must match");

    const int batch = q_mean.size(0);
    const int heads = q_mean.size(1);
    const int flat_heads = batch * heads;
    const int num_q_blocks = q_mean.size(2);
    const int num_kv_blocks = k_min.size(2);
    const int head_dim = q_mean.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);
    TORCH_CHECK(num_kv_blocks <= 2048,
                "QUEST block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    auto opts = at::TensorOptions().dtype(at::kInt).device(q_mean.device());
    at::Tensor topk = at::empty({flat_heads, num_q_blocks, topk_count}, opts);
    if (topk_count == 0) {
        return topk;
    }

    quest_block_topk(
        q_mean.data_ptr(), k_min.data_ptr(), k_max.data_ptr(), topk.data_ptr(),
        flat_heads, num_q_blocks, num_kv_blocks, head_dim, topk_count, causal, is_bf16);

    return topk;
}

static at::Tensor single_query_key_mean_topk_impl(
    const at::Tensor& q_grouped,
    const at::Tensor& k_mean,
    int topk_count,
    int num_kv_blocks,
    bool is_bf16 = false) {
    TORCH_CHECK(q_grouped.is_cuda() && k_mean.is_cuda(),
                "q_grouped and k_mean must be CUDA tensors");
    TORCH_CHECK(q_grouped.is_contiguous() && k_mean.is_contiguous(),
                "q_grouped and k_mean must be contiguous");
    TORCH_CHECK(q_grouped.dim() == 4 && k_mean.dim() == 4,
                "q_grouped and k_mean must be 4D tensors");
    TORCH_CHECK(q_grouped.size(0) == k_mean.size(0),
                "q_grouped and k_mean batch dimensions must match");
    TORCH_CHECK(q_grouped.size(1) == k_mean.size(1),
                "q_grouped and k_mean KV-head dimensions must match");
    TORCH_CHECK(q_grouped.size(3) == k_mean.size(3),
                "q_grouped and k_mean head_dim must match");

    const int batch = q_grouped.size(0);
    const int kv_heads = q_grouped.size(1);
    const int flat_heads = batch * kv_heads;
    const int groups = q_grouped.size(2);
    const int head_dim = q_grouped.size(3);
    const int k_mean_capacity_blocks = k_mean.size(2);
    TORCH_CHECK(groups >= 1 && groups <= 16,
                "grouped single-query selector expects groups in [1, 16], got ", groups);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);
    TORCH_CHECK(num_kv_blocks >= 0 && num_kv_blocks <= k_mean_capacity_blocks,
                "num_kv_blocks must be in [0, k_mean capacity]");
    TORCH_CHECK(num_kv_blocks <= 2048,
                "single-query block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    auto opts = at::TensorOptions().dtype(at::kInt).device(q_grouped.device());
    at::Tensor topk = at::empty({flat_heads, topk_count}, opts);
    if (topk_count == 0) {
        return topk;
    }

    if (num_kv_blocks <= 128) {
        single_query_key_mean_topk(
            q_grouped.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
            flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
            head_dim, topk_count, is_bf16);
        return topk;
    }

    constexpr int chunk_blocks = 64;
    const int chunk_count = (num_kv_blocks + chunk_blocks - 1) / chunk_blocks;
    const int local_count = std::min(topk_count, chunk_blocks);
    at::Tensor local_scores = at::empty(
        {flat_heads, chunk_count, local_count},
        at::TensorOptions().dtype(at::kFloat).device(q_grouped.device()));
    at::Tensor local_indices = at::empty({flat_heads, chunk_count, local_count}, opts);
    at::Tensor done_counts = at::empty({flat_heads}, opts);
    done_counts.zero_();
    single_query_key_mean_topk_chunked(
        q_grouped.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
        local_scores.data_ptr(), local_indices.data_ptr(), done_counts.data_ptr(),
        flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
        head_dim, topk_count, chunk_count, local_count, is_bf16);

    return topk;
}

static at::Tensor single_query_quest_topk_impl(
    const at::Tensor& q_grouped,
    const at::Tensor& k_min,
    const at::Tensor& k_max,
    int topk_count,
    int num_kv_blocks,
    bool is_bf16 = false) {
    TORCH_CHECK(q_grouped.is_cuda() && k_min.is_cuda() && k_max.is_cuda(),
                "q_grouped, k_min, and k_max must be CUDA tensors");
    TORCH_CHECK(q_grouped.is_contiguous() && k_min.is_contiguous() && k_max.is_contiguous(),
                "q_grouped, k_min, and k_max must be contiguous");
    TORCH_CHECK(q_grouped.dim() == 4 && k_min.dim() == 4 && k_max.dim() == 4,
                "q_grouped, k_min, and k_max must be 4D tensors");
    TORCH_CHECK(k_min.sizes() == k_max.sizes(),
                "k_min and k_max shapes must match");
    TORCH_CHECK(q_grouped.size(0) == k_min.size(0),
                "q_grouped and key stat batch dimensions must match");
    TORCH_CHECK(q_grouped.size(1) == k_min.size(1),
                "q_grouped and key stat KV-head dimensions must match");
    TORCH_CHECK(q_grouped.size(3) == k_min.size(3),
                "q_grouped and key stat head_dim must match");

    const int batch = q_grouped.size(0);
    const int kv_heads = q_grouped.size(1);
    const int flat_heads = batch * kv_heads;
    const int groups = q_grouped.size(2);
    const int head_dim = q_grouped.size(3);
    const int k_stat_capacity_blocks = k_min.size(2);
    TORCH_CHECK(groups >= 1 && groups <= 16,
                "grouped single-query selector expects groups in [1, 16], got ", groups);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);
    TORCH_CHECK(num_kv_blocks >= 0 && num_kv_blocks <= k_stat_capacity_blocks,
                "num_kv_blocks must be in [0, key stat capacity]");
    TORCH_CHECK(num_kv_blocks <= 2048,
                "QUEST single-query block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");

    auto opts = at::TensorOptions().dtype(at::kInt).device(q_grouped.device());
    at::Tensor topk = at::empty({flat_heads, topk_count}, opts);
    if (topk_count == 0) {
        return topk;
    }

    single_query_quest_topk(
        q_grouped.data_ptr(), k_min.data_ptr(), k_max.data_ptr(), topk.data_ptr(),
        flat_heads, groups, num_kv_blocks, k_stat_capacity_blocks,
        head_dim, topk_count, is_bf16);

    return topk;
}

static at::Tensor single_query_key_mean_topk_into_impl(
    const at::Tensor& q_grouped,
    const at::Tensor& k_mean,
    at::Tensor& topk,
    at::Tensor& local_scores,
    at::Tensor& local_indices,
    at::Tensor& done_counts,
    int num_kv_blocks,
    bool is_bf16 = false) {
    TORCH_CHECK(q_grouped.is_cuda() && k_mean.is_cuda(),
                "q_grouped and k_mean must be CUDA tensors");
    TORCH_CHECK(topk.is_cuda() && local_scores.is_cuda() &&
                local_indices.is_cuda() && done_counts.is_cuda(),
                "top-k workspace tensors must be CUDA tensors");
    TORCH_CHECK(q_grouped.is_contiguous() && k_mean.is_contiguous() &&
                topk.is_contiguous() && local_scores.is_contiguous() &&
                local_indices.is_contiguous() && done_counts.is_contiguous(),
                "q_grouped, k_mean, and top-k workspace tensors must be contiguous");
    TORCH_CHECK(topk.dtype() == at::kInt && local_indices.dtype() == at::kInt &&
                done_counts.dtype() == at::kInt,
                "topk, local_indices, and done_counts must be int32");
    TORCH_CHECK(local_scores.dtype() == at::kFloat,
                "local_scores must be float32");
    TORCH_CHECK(q_grouped.dim() == 4 && k_mean.dim() == 4,
                "q_grouped and k_mean must be 4D tensors");
    TORCH_CHECK(q_grouped.size(0) == k_mean.size(0),
                "q_grouped and k_mean batch dimensions must match");
    TORCH_CHECK(q_grouped.size(1) == k_mean.size(1),
                "q_grouped and k_mean KV-head dimensions must match");
    TORCH_CHECK(q_grouped.size(3) == k_mean.size(3),
                "q_grouped and k_mean head_dim must match");

    const int batch = q_grouped.size(0);
    const int kv_heads = q_grouped.size(1);
    const int flat_heads = batch * kv_heads;
    const int groups = q_grouped.size(2);
    const int head_dim = q_grouped.size(3);
    const int k_mean_capacity_blocks = k_mean.size(2);
    const int topk_count = topk.size(1);
    TORCH_CHECK(topk.dim() == 2 && topk.size(0) == flat_heads,
                "topk must be shaped [batch * kv_heads, topk_count]");
    TORCH_CHECK(done_counts.dim() == 1 && done_counts.size(0) == flat_heads,
                "done_counts must be shaped [batch * kv_heads]");
    TORCH_CHECK(groups >= 1 && groups <= 16,
                "grouped single-query selector expects groups in [1, 16], got ", groups);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
                "head_dim must be 64 or 128, got ", head_dim);
    TORCH_CHECK(num_kv_blocks >= 0 && num_kv_blocks <= k_mean_capacity_blocks,
                "num_kv_blocks must be in [0, k_mean capacity]");
    TORCH_CHECK(num_kv_blocks <= 2048,
                "single-query block selector supports <= 2048 KV blocks, got ", num_kv_blocks);
    TORCH_CHECK(topk_count >= 0 && topk_count <= num_kv_blocks,
                "topk_count must be in [0, num_kv_blocks]");
    if (topk_count == 0) {
        return topk;
    }
    done_counts.zero_();

    if (num_kv_blocks <= 128) {
        single_query_key_mean_topk(
            q_grouped.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
            flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
            head_dim, topk_count, is_bf16);
        return topk;
    }

    constexpr int chunk_blocks = 64;
    const int chunk_count = (num_kv_blocks + chunk_blocks - 1) / chunk_blocks;
    const int local_count = std::min(topk_count, chunk_blocks);
    TORCH_CHECK(local_scores.dim() == 3 && local_scores.size(0) == flat_heads &&
                local_scores.size(1) >= chunk_count && local_scores.size(2) >= local_count,
                "local_scores workspace is too small");
    TORCH_CHECK(local_indices.dim() == 3 && local_indices.size(0) == flat_heads &&
                local_indices.size(1) >= chunk_count && local_indices.size(2) >= local_count,
                "local_indices workspace is too small");
    single_query_key_mean_topk_chunked(
        q_grouped.data_ptr(), k_mean.data_ptr(), topk.data_ptr(),
        local_scores.data_ptr(), local_indices.data_ptr(), done_counts.data_ptr(),
        flat_heads, groups, num_kv_blocks, k_mean_capacity_blocks,
        head_dim, topk_count, chunk_count, local_count, is_bf16);

    return topk;
}

static at::Tensor block_mean_topk_dispatch(
    const at::Tensor& q_mean,
    const at::Tensor& k_mean,
    int64_t topk_count,
    bool causal,
    bool is_bf16) {
    return block_mean_topk_impl(
        q_mean,
        k_mean,
        static_cast<int>(topk_count),
        causal,
        is_bf16);
}

static at::Tensor single_query_key_mean_topk_dispatch(
    const at::Tensor& q_grouped,
    const at::Tensor& k_mean,
    int64_t topk_count,
    int64_t num_kv_blocks,
    bool is_bf16) {
    return single_query_key_mean_topk_impl(
        q_grouped,
        k_mean,
        static_cast<int>(topk_count),
        static_cast<int>(num_kv_blocks),
        is_bf16);
}

}  // namespace

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("nvfp4_quantize(Tensor x, bool is_bf16) -> Tensor[]");
    ops.impl("nvfp4_quantize", torch::kCUDA, &nvfp4_quantize);

    ops.def("nvfp4_quantize_permuted(Tensor x, bool is_bf16) -> Tensor[]");
    ops.impl("nvfp4_quantize_permuted", torch::kCUDA, &nvfp4_quantize_permuted);

    ops.def("nvfp4_quantize_transposed(Tensor x, bool is_bf16) -> Tensor[]");
    ops.impl("nvfp4_quantize_transposed", torch::kCUDA, &nvfp4_quantize_transposed);

    ops.def("block_mean_topk(Tensor q_mean, Tensor k_mean, int topk_count, bool causal, bool is_bf16) -> Tensor");
    ops.impl("block_mean_topk", torch::kCUDA, &block_mean_topk_dispatch);

    ops.def("single_query_key_mean_topk(Tensor q_grouped, Tensor k_mean, int topk_count, int num_kv_blocks, bool is_bf16) -> Tensor");
    ops.impl("single_query_key_mean_topk", torch::kCUDA, &single_query_key_mean_topk_dispatch);

    ops.def("thrift_attention_causal_nvfp4_packed(Tensor q_hi, Tensor k_hi, Tensor v_hi, Tensor selected_blocks, Tensor q_packed, Tensor k_packed, Tensor v_packed_t, Tensor q_scale, Tensor k_scale, Tensor v_scale_t, bool is_bf16) -> Tensor");
    ops.impl("thrift_attention_causal_nvfp4_packed", torch::kCUDA, &thrift_attention_causal_nvfp4_packed);

    ops.def("thrift_attention_noncausal_nvfp4_packed(Tensor q_hi, Tensor k_hi, Tensor v_hi, Tensor selected_blocks, Tensor q_packed, Tensor k_packed, Tensor v_packed_t, Tensor q_scale, Tensor k_scale, Tensor v_scale_t, bool is_bf16) -> Tensor");
    ops.impl("thrift_attention_noncausal_nvfp4_packed", torch::kCUDA, &thrift_attention_noncausal_nvfp4_packed);

    ops.def("thrift_attention_single_query_nvfp4_packed(Tensor q_hi, Tensor k_hi, Tensor v_hi, Tensor selected_blocks, Tensor q_packed, Tensor k_packed, Tensor v_packed_t, Tensor q_scale, Tensor k_scale, Tensor v_scale_t, bool is_bf16) -> Tensor");
    ops.impl("thrift_attention_single_query_nvfp4_packed", torch::kCUDA, &thrift_attention_single_query_nvfp4_packed);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
