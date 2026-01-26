#include <torch/all.h>
#include <c10/xpu/XPUStream.h>
#include <cute/util/compat/device.hpp>

#include "src/fmha_fwd.hpp"
#include "src/fmha_bwd.hpp"

namespace FLASH_NAMESPACE {

inline int round_multiple(int x, int m) {
    int pad_res = (x + m - 1) / m * m;
    if (pad_res == 224) {
        pad_res = 256;
    }
    return pad_res;
}

inline at::Tensor ensure_contiguous(const at::Tensor& tensor) {
    return tensor.is_contiguous() ? tensor : tensor.contiguous();
}

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_xpu(), #x " must be on XPU")

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
    compat::select_device(device_idx);

    // check inputs
    q = ensure_contiguous(q);
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(k.size(-1) == head_size_og, "Key head dimension must match Query head dimension");
    TORCH_CHECK(v.size(-1) == head_size_og, "Value head dimension must match Query head dimension");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

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

    // Allocate softmax_lse output tensor: (batch_size, num_heads, seqlen_q)
    auto opts = q.options().dtype(torch::kFloat32);
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts);

    bool is_local = (window_size_left != -1) | (window_size_right != -1);

    q_padded = ensure_contiguous(q_padded);
    k_padded = ensure_contiguous(k_padded);
    v_padded = ensure_contiguous(v_padded);

    auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();

    cutlass_fmha_fwd_fix_impl(
        queue,
        q_padded, k_padded, v_padded, out_padded,
        softmax_lse,
        softmax_scale,
        window_size_left, window_size_right,
        is_causal, is_local);

    // Remove padding from output
    at::Tensor out = out_padded;
    if (head_size_og != head_size_padded) {
        out = out_padded.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)});
    }
    out = ensure_contiguous(out);

    at::Tensor S_dmask;
    at::Tensor rng_state;
    return {out, softmax_lse, S_dmask, rng_state};
  }


std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &q,     // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,     // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,     // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,  // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,
        std::optional<at::Tensor> &dk_,
        std::optional<at::Tensor> &dv_,
        std::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        std::optional<at::Generator> gen_,
        std::optional<at::Tensor> &rng_state) {

    auto device_idx = q.device().index();
    compat::select_device(device_idx);

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention backward only supports fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    CHECK_DEVICE(out);
    CHECK_DEVICE(dout);

    // Ensure inputs are contiguous (k, v may come from kv-packed slices)
    at::Tensor k_contig = ensure_contiguous(k);
    at::Tensor v_contig = ensure_contiguous(v);
    at::Tensor q_contig = ensure_contiguous(q);
    at::Tensor out_contig = ensure_contiguous(out);
    at::Tensor dout_contig = ensure_contiguous(dout);

    TORCH_CHECK(q_contig.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k_contig.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v_contig.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out_contig.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout_contig.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q_contig.sizes();
    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k_contig.size(1);
    const int num_heads_k = k_contig.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // XPU requires head_size to be a multiple of 32
    const int head_size_padded = round_multiple(head_size_og, 32);
    const bool needs_padding = (head_size_og != head_size_padded);

    at::Tensor q_padded, k_padded, v_padded, out_padded, dout_padded;

    // Apply padding only if needed
    if (needs_padding) {
        const int pad_size = head_size_padded - head_size_og;
        q_padded = torch::nn::functional::pad(q_contig, torch::nn::functional::PadFuncOptions({0, pad_size}));
        k_padded = torch::nn::functional::pad(k_contig, torch::nn::functional::PadFuncOptions({0, pad_size}));
        v_padded = torch::nn::functional::pad(v_contig, torch::nn::functional::PadFuncOptions({0, pad_size}));
        out_padded = torch::nn::functional::pad(out_contig, torch::nn::functional::PadFuncOptions({0, pad_size}));
        dout_padded = torch::nn::functional::pad(dout_contig, torch::nn::functional::PadFuncOptions({0, pad_size}));
    } else {
        // No padding needed, reuse contiguous tensors directly
        q_padded = q_contig;
        k_padded = k_contig;
        v_padded = v_contig;
        out_padded = out_contig;
        dout_padded = dout_contig;
    }

    auto opts = q_contig.options();

    // Allocate output gradients
    // When no padding needed and user provides contiguous tensors, reuse them directly
    at::Tensor dq, dk, dv;
    at::Tensor dq_work, dk_work, dv_work;  // Working tensors for kernel
    bool dq_needs_copy = false, dk_needs_copy = false, dv_needs_copy = false;

    if (!needs_padding && num_heads_k == num_heads) {
        // Optimal path: no padding, no MQA/GQA - can write directly to output
        if (dq_.has_value() && dq_.value().is_contiguous()) {
            dq_work = dq_.value();
        } else {
            dq_work = torch::empty({batch_size, seqlen_q, num_heads, head_size_og}, opts);
            if (dq_.has_value()) dq_needs_copy = true;
        }
        if (dk_.has_value() && dk_.value().is_contiguous()) {
            dk_work = dk_.value();
        } else {
            dk_work = torch::empty({batch_size, seqlen_k, num_heads_k, head_size_og}, opts);
            if (dk_.has_value()) dk_needs_copy = true;
        }
        if (dv_.has_value() && dv_.value().is_contiguous()) {
            dv_work = dv_.value();
        } else {
            dv_work = torch::empty({batch_size, seqlen_k, num_heads_k, head_size_og}, opts);
            if (dv_.has_value()) dv_needs_copy = true;
        }
    } else {
        // Need padding or MQA/GQA - allocate with padded size
        dq_work = torch::empty({batch_size, seqlen_q, num_heads, head_size_padded}, opts);
        dk_work = torch::empty({batch_size, seqlen_k, num_heads_k, head_size_padded}, opts);
        dv_work = torch::empty({batch_size, seqlen_k, num_heads_k, head_size_padded}, opts);
    }

    // Allocate intermediate buffers
    at::Tensor softmax_d = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    // Handle MQA/GQA
    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size_padded}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size_padded}, opts);
    } else {
        dk_expanded = dk_work;
        dv_expanded = dv_work;
    }

    bool is_local = (window_size_left != -1) | (window_size_right != -1);

    auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();

    // Call the cutlass backward implementation
    cutlass_fmha_bwd_fix_impl(
        queue,
        dout_padded, q_padded, k_padded, v_padded, out_padded, softmax_lse,
        dq_work, dk_expanded, dv_expanded, softmax_d,
        softmax_scale, window_size_left, window_size_right, is_causal, is_local);

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk_work, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size_padded}), {3});
        at::sum_out(dv_work, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size_padded}), {3});
    }

    // Remove padding from output gradients if needed
    if (needs_padding) {
        dq = dq_work.index({torch::indexing::Slice(), torch::indexing::Slice(),
                           torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)}).contiguous();
        dk = dk_work.index({torch::indexing::Slice(), torch::indexing::Slice(),
                           torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)}).contiguous();
        dv = dv_work.index({torch::indexing::Slice(), torch::indexing::Slice(),
                           torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)}).contiguous();
        // Copy to user tensors if provided
        if (dq_.has_value()) dq_.value().copy_(dq);
        if (dk_.has_value()) dk_.value().copy_(dk);
        if (dv_.has_value()) dv_.value().copy_(dv);
    } else if (num_heads_k != num_heads) {
        // MQA/GQA without padding: need to slice and copy
        dq = dq_work.index({torch::indexing::Slice(), torch::indexing::Slice(),
                           torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)}).contiguous();
        dk = dk_work.index({torch::indexing::Slice(), torch::indexing::Slice(),
                           torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)}).contiguous();
        dv = dv_work.index({torch::indexing::Slice(), torch::indexing::Slice(),
                           torch::indexing::Slice(), torch::indexing::Slice(0, head_size_og)}).contiguous();
        if (dq_.has_value()) dq_.value().copy_(dq);
        if (dk_.has_value()) dk_.value().copy_(dk);
        if (dv_.has_value()) dv_.value().copy_(dv);
    } else {
        // Optimal path: no padding, no MQA/GQA - results already in place
        dq = dq_work;
        dk = dk_work;
        dv = dv_work;
        // Copy to non-contiguous user tensors if needed
        if (dq_needs_copy) dq_.value().copy_(dq);
        if (dk_needs_copy) dk_.value().copy_(dk);
        if (dv_needs_copy) dv_.value().copy_(dv);
    }

    return {dq, dk, dv, softmax_d};
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
    compat::select_device(device_idx);

    // check inputs
    q = ensure_contiguous(q);
    const auto sizes = q.sizes();
    const int total_q = sizes[0];
    const int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    const int batch_size = cu_seqlens_q.numel() - 1;

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(k.size(-1) == head_size_og, "Key head dimension must match Query head dimension");
    TORCH_CHECK(v.size(-1) == head_size_og, "Value head dimension must match Query head dimension");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);

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

    bool is_local = (window_size_left != -1) | (window_size_right != -1);
    bool is_paged = block_table_.has_value() && block_table_->defined();

    q_padded = ensure_contiguous(q_padded);
    k_padded = ensure_contiguous(k_padded);
    v_padded = ensure_contiguous(v_padded);

    // Allocate softmax_lse output tensor: (batch_size, num_heads, max_seqlen_q)
    // For varlen, we use max_seqlen_q as the third dimension
    auto opts = q.options().dtype(torch::kFloat32);
    at::Tensor softmax_lse = torch::empty({batch_size, num_heads, max_seqlen_q}, opts);

    auto queue = c10::xpu::getCurrentXPUStream(device_idx).queue();

    cutlass_fmha_fwd_varlen_impl(
        queue,
        q_padded, k_padded, v_padded, out_padded,
        softmax_lse,
        block_table_,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        softmax_scale,
        window_size_left, window_size_right,
        true, is_paged, is_causal, is_local);

    // Remove padding from output
    at::Tensor out = out_padded;
    if (head_size_og != head_size_padded) {
        out = out_padded.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                torch::indexing::Slice(0, head_size_og)});
    }
    out = ensure_contiguous(out);

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
std::vector<torch::Tensor>
mha_bwd(const torch::Tensor &dout,
        const torch::Tensor &q,
        const torch::Tensor &k,
        const torch::Tensor &v,
        const torch::Tensor &out,
        const torch::Tensor &softmax_lse,
        const c10::optional<torch::Tensor> &dq_,
        const c10::optional<torch::Tensor> &dk_,
        const c10::optional<torch::Tensor> &dv_,
        const c10::optional<torch::Tensor> &alibi_slopes_,
        const double p_dropout,
        const double softmax_scale,
        const bool is_causal,
        const int64_t window_size_left,
        const int64_t window_size_right,
        const double softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        const c10::optional<torch::Tensor> &rng_state) {
    // Convert optional types
    std::optional<at::Tensor> dq_opt = dq_.has_value() ? std::optional<at::Tensor>(dq_.value()) : std::nullopt;
    std::optional<at::Tensor> dk_opt = dk_.has_value() ? std::optional<at::Tensor>(dk_.value()) : std::nullopt;
    std::optional<at::Tensor> dv_opt = dv_.has_value() ? std::optional<at::Tensor>(dv_.value()) : std::nullopt;
    std::optional<at::Tensor> alibi_opt = alibi_slopes_.has_value() ? std::optional<at::Tensor>(alibi_slopes_.value()) : std::nullopt;
    std::optional<at::Tensor> rng_opt = rng_state.has_value() ? std::optional<at::Tensor>(rng_state.value()) : std::nullopt;
    
    return FLASH_NAMESPACE::mha_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq_opt,
        dk_opt,
        dv_opt,
        alibi_opt,
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        is_causal,
        static_cast<int>(window_size_left),
        static_cast<int>(window_size_right),
        static_cast<float>(softcap),
        deterministic,
        gen_,
        rng_opt
    );
}