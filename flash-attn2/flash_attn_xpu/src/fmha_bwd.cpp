#include "fmha_bwd.hpp"
#include "torch/all.h"
#include <sycl/sycl.hpp>

// Helper to convert tensor dtype to BwdCutlassType
inline BwdCutlassType aten_to_Bwd_Cutlass_dtype(const at::Tensor& tensor) {
    if (tensor.scalar_type() == at::ScalarType::Half) {
        return BwdCutlassType::half;
    } else if (tensor.scalar_type() == at::ScalarType::BFloat16) {
        return BwdCutlassType::bfloat16;
    } else {
        throw std::runtime_error("Unsupported dtype for backward pass. Expected half or bfloat16.");
    }
}

// Helper function to round up to multiple
inline int round_multiple(int x, int m) {
    return (x + m - 1) / m * m;
}

void cutlass_fmha_bwd_fix_impl(
    sycl::queue& queue,
    const at::Tensor& dout,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& out,
    const at::Tensor& softmax_lse,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    at::Tensor& softmax_d,
    float sm_scale,
    int window_size_left,
    int window_size_right,
    bool is_causal,
    bool is_local) {

    // Get dimensions from tensors - assuming BSHD layout (batch, seq, head, dim)
    int batch_size = q.size(0);
    int seqlen_q = q.size(1);
    int num_heads_q = q.size(2);
    int head_size = q.size(3);

    int seqlen_k = k.size(1);
    int num_heads_k = k.size(2);

    // Round up sequence lengths for internal buffers
    int seqlen_q_rounded = round_multiple(seqlen_q, 64);
    int seqlen_k_rounded = round_multiple(seqlen_k, 64);

    // Allocate dq_accum buffer (float)
    auto dq_accum = at::zeros({batch_size, seqlen_q_rounded, num_heads_q, head_size}, 
                               q.options().dtype(at::kFloat));

    // Build args structure
    fmha_bwd_args_t args = {
        dout.data_ptr(),
        out.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        softmax_lse.data_ptr(),
        dq.data_ptr(),
        dk.data_ptr(),
        dv.data_ptr(),
        softmax_d.data_ptr(),
        dq_accum.data_ptr(),
        batch_size,
        num_heads_q,
        num_heads_k,
        seqlen_q,
        seqlen_k,
        head_size,
        seqlen_q_rounded,
        seqlen_k_rounded,
        sm_scale,
        is_causal,
        is_local,
        q.scalar_type() == at::ScalarType::BFloat16,
        false,  // deterministic
        window_size_left,
        window_size_right
    };

    BwdCutlassType cuType = aten_to_Bwd_Cutlass_dtype(q);
    const int h = args.head_size;

    // Dispatch based on head size, causal mode, and local mode
    if (h <= 32) {
        if (is_causal && is_local) {
            bwd_policy_dispatch<bwd_policy_head32, 1, 1>(queue, cuType, args);
        } else if (is_causal) {
            bwd_policy_dispatch<bwd_policy_head32, 1, 0>(queue, cuType, args);
        } else if (is_local) {
            bwd_policy_dispatch<bwd_policy_head32, 0, 1>(queue, cuType, args);
        } else {
            bwd_policy_dispatch<bwd_policy_head32, 0, 0>(queue, cuType, args);
        }
    }
    // else if (h <= 64) {
    //     if (is_causal && is_local) {
    //         bwd_policy_dispatch<bwd_policy_head64, 1, 1>(queue, cuType, args);
    //     } else if (is_causal) {
    //         bwd_policy_dispatch<bwd_policy_head64, 1, 0>(queue, cuType, args);
    //     } else if (is_local) {
    //         bwd_policy_dispatch<bwd_policy_head64, 0, 1>(queue, cuType, args);
    //     } else {
    //         bwd_policy_dispatch<bwd_policy_head64, 0, 0>(queue, cuType, args);
    //     }
    // }
    // else if (h <= 96) {
    //     if (is_causal && is_local) {
    //         bwd_policy_dispatch<bwd_policy_head96, 1, 1>(queue, cuType, args);
    //     } else if (is_causal) {
    //         bwd_policy_dispatch<bwd_policy_head96, 1, 0>(queue, cuType, args);
    //     } else if (is_local) {
    //         bwd_policy_dispatch<bwd_policy_head96, 0, 1>(queue, cuType, args);
    //     } else {
    //         bwd_policy_dispatch<bwd_policy_head96, 0, 0>(queue, cuType, args);
    //     }
    // }
    // else if (h <= 128) {
    //     if (is_causal && is_local) {
    //         bwd_policy_dispatch<bwd_policy_head128, 1, 1>(queue, cuType, args);
    //     } else if (is_causal) {
    //         bwd_policy_dispatch<bwd_policy_head128, 1, 0>(queue, cuType, args);
    //     } else if (is_local) {
    //         bwd_policy_dispatch<bwd_policy_head128, 0, 1>(queue, cuType, args);
    //     } else {
    //         bwd_policy_dispatch<bwd_policy_head128, 0, 0>(queue, cuType, args);
    //     }
    // }
    // else if (h <= 160) {
    //     if (is_causal && is_local) {
    //         bwd_policy_dispatch<bwd_policy_head160, 1, 1>(queue, cuType, args);
    //     } else if (is_causal) {
    //         bwd_policy_dispatch<bwd_policy_head160, 1, 0>(queue, cuType, args);
    //     } else if (is_local) {
    //         bwd_policy_dispatch<bwd_policy_head160, 0, 1>(queue, cuType, args);
    //     } else {
    //         bwd_policy_dispatch<bwd_policy_head160, 0, 0>(queue, cuType, args);
    //     }
    // }
    // if (h <= 192) {
    //     if (is_causal && is_local) {
    //         bwd_policy_dispatch<bwd_policy_head192, 1, 1>(queue, cuType, args);
    //     } else if (is_causal) {
    //         bwd_policy_dispatch<bwd_policy_head192, 1, 0>(queue, cuType, args);
    //     } else if (is_local) {
    //         bwd_policy_dispatch<bwd_policy_head192, 0, 1>(queue, cuType, args);
    //     } else {
    //         bwd_policy_dispatch<bwd_policy_head192, 0, 0>(queue, cuType, args);
    //     }
    // }
    // else if (h <= 256) {
    //     if (is_causal && is_local) {
    //         bwd_policy_dispatch<bwd_policy_head256, 1, 1>(queue, cuType, args);
    //     } else if (is_causal) {
    //         bwd_policy_dispatch<bwd_policy_head256, 1, 0>(queue, cuType, args);
    //     } else if (is_local) {
    //         bwd_policy_dispatch<bwd_policy_head256, 0, 1>(queue, cuType, args);
    //     } else {
    //         bwd_policy_dispatch<bwd_policy_head256, 0, 0>(queue, cuType, args);
    //     }
    // }
    else {
        throw std::runtime_error("Unsupported head_size: " + std::to_string(h) + ". Max supported head_size is 256");
    }
}
