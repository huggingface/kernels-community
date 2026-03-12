#include "fmha_bwd.hpp"
#include "torch/all.h"
#include <sycl/sycl.hpp>

namespace {

/// Map PyTorch scalar type to BwdCutlassType enum.
[[nodiscard]] inline BwdCutlassType aten_to_Bwd_Cutlass_dtype(const at::Tensor& tensor) {
    if (tensor.scalar_type() == at::ScalarType::Half) {
        return BwdCutlassType::half;
    }
    if (tensor.scalar_type() == at::ScalarType::BFloat16) {
        return BwdCutlassType::bfloat16;
    }
    throw std::runtime_error("Unsupported dtype for backward pass. Expected half or bfloat16.");
}

/// Round `x` up to the nearest multiple of `m`.
constexpr int round_up(int x, int m) {
    return (x + m - 1) / m * m;
}

/// Dispatch backward kernel by head_size with pre-resolved causal/local flags.
template <typename Policy>
void dispatch_bwd_causal_local(sycl::queue& queue, BwdCutlassType cuType,
                               const fmha_bwd_args_t& args,
                               bool is_causal, bool is_local) {
    if (is_causal && is_local) {
        bwd_policy_dispatch<Policy, 1, 1>(queue, cuType, args);
    } else if (is_causal) {
        bwd_policy_dispatch<Policy, 1, 0>(queue, cuType, args);
    } else if (is_local) {
        bwd_policy_dispatch<Policy, 0, 1>(queue, cuType, args);
    } else {
        bwd_policy_dispatch<Policy, 0, 0>(queue, cuType, args);
    }
}

/// Dispatch backward kernel by head dimension.
void dispatch_bwd_by_head(sycl::queue& queue, BwdCutlassType cuType,
                          const fmha_bwd_args_t& args, int head_size,
                          bool is_causal, bool is_local) {
    if      (head_size <=  32) dispatch_bwd_causal_local<bwd_policy_head32> (queue, cuType, args, is_causal, is_local);
    else if (head_size <=  64) dispatch_bwd_causal_local<bwd_policy_head64> (queue, cuType, args, is_causal, is_local);
    else if (head_size <=  96) dispatch_bwd_causal_local<bwd_policy_head96> (queue, cuType, args, is_causal, is_local);
    else if (head_size <= 128) dispatch_bwd_causal_local<bwd_policy_head128>(queue, cuType, args, is_causal, is_local);
    else if (head_size <= 160) dispatch_bwd_causal_local<bwd_policy_head160>(queue, cuType, args, is_causal, is_local);
    else if (head_size <= 192) dispatch_bwd_causal_local<bwd_policy_head192>(queue, cuType, args, is_causal, is_local);
    else if (head_size <= 256) dispatch_bwd_causal_local<bwd_policy_head256>(queue, cuType, args, is_causal, is_local);
    else throw std::runtime_error("Unsupported head_size: " + std::to_string(head_size) + ". Max supported is 256");
}

}  // anonymous namespace

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
    bool is_local,
    float p_dropout,
    uint64_t philox_seed,
    uint64_t philox_offset,
    bool deterministic) {

    // Get dimensions from tensors — assuming BSHD layout (batch, seq, head, dim)
    const int batch_size   = q.size(0);
    const int seqlen_q     = q.size(1);
    const int num_heads_q  = q.size(2);
    const int head_size    = q.size(3);

    const int seqlen_k     = k.size(1);
    const int num_heads_k  = k.size(2);

    // Round up sequence lengths for internal buffers
    const int seqlen_q_rounded = round_up(seqlen_q, 64);
    const int seqlen_k_rounded = round_up(seqlen_k, 64);

    // Allocate dq_accum buffer (float)
    // For deterministic mode, allocate separate splits to avoid atomicAdd races
    int nsplits = 1;
    int dq_accum_split_stride = 0;
    at::Tensor dq_accum;
    if (!deterministic) {
        dq_accum = at::zeros({batch_size, seqlen_q_rounded, num_heads_q, head_size}, 
                              q.options().dtype(at::kFloat));
    } else {
        // Each work-group gets its own split to write dQ accumulator
        // Use a reasonable number of splits based on batch/head parallelism
        // Similar to CUDA: nsplits = ceil(num_compute_units / (batch * heads))
        // Query actual XPU compute units from the device
        const int num_compute_units = static_cast<int>(queue.get_device().get_info<sycl::info::device::max_compute_units>());
        nsplits = std::max((num_compute_units + batch_size * num_heads_q - 1) / (batch_size * num_heads_q), 1);
        // Cap nsplits by max possible N_BLOCK to avoid excessive memory allocation.
        // The minimum kBlockN across all head size policies is 32.
        const int max_n_blocks = std::max((seqlen_k + 31) / 32, 1);
        nsplits = std::min(nsplits, max_n_blocks);
        dq_accum = at::zeros({nsplits, batch_size, seqlen_q_rounded, num_heads_q, head_size}, 
                              q.options().dtype(at::kFloat));
        dq_accum_split_stride = batch_size * seqlen_q_rounded * num_heads_q * head_size;
    }

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
        deterministic,
        nsplits,
        dq_accum_split_stride,
        window_size_left,
        window_size_right,
        p_dropout,
        philox_seed,
        philox_offset
    };

    const BwdCutlassType cuType = aten_to_Bwd_Cutlass_dtype(q);
    dispatch_bwd_by_head(queue, cuType, args, args.head_size, is_causal, is_local);
}
