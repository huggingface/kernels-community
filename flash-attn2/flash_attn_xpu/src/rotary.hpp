#pragma once

#include <torch/all.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

/// Apply rotary embedding to Q or K tensor (interleaved mode)
template <typename scalar_t>
struct ApplyRotaryInterleavedKernel {
    scalar_t* x;
    const scalar_t* cos;
    const scalar_t* sin;
    const int* seqlen_offsets;
    int batch_size, seqlen, num_heads, head_dim, rotary_dim, cos_sin_stride;

    void operator()(sycl::nd_item<1> item) const {
        int idx = item.get_global_id(0);
        int total_elements = batch_size * seqlen * num_heads * (rotary_dim / 2);
        if (idx >= total_elements) return;
        int half_rotary = rotary_dim / 2;
        int pair_idx = idx % half_rotary;
        int temp = idx / half_rotary;
        int head_idx = temp % num_heads;
        temp = temp / num_heads;
        int seq_idx = temp % seqlen;
        int batch_idx = temp / seqlen;
        int pos = (seqlen_offsets != nullptr) ? seqlen_offsets[batch_idx] + seq_idx : seq_idx;
        float c = static_cast<float>(cos[pos * cos_sin_stride + pair_idx]);
        float s = static_cast<float>(sin[pos * cos_sin_stride + pair_idx]);
        int base_offset = ((batch_idx * seqlen + seq_idx) * num_heads + head_idx) * head_dim;
        int x0_idx = base_offset + pair_idx * 2;
        int x1_idx = base_offset + pair_idx * 2 + 1;
        float x0 = static_cast<float>(x[x0_idx]);
        float x1 = static_cast<float>(x[x1_idx]);
        x[x0_idx] = static_cast<scalar_t>(x0 * c - x1 * s);
        x[x1_idx] = static_cast<scalar_t>(x0 * s + x1 * c);
    }
};

/// Apply rotary embedding (non-interleaved / GPT-NeoX style)
template <typename scalar_t>
struct ApplyRotaryContiguousKernel {
    scalar_t* x;
    const scalar_t* cos;
    const scalar_t* sin;
    const int* seqlen_offsets;
    int batch_size, seqlen, num_heads, head_dim, rotary_dim, cos_sin_stride;

    void operator()(sycl::nd_item<1> item) const {
        int idx = item.get_global_id(0);
        int total_elements = batch_size * seqlen * num_heads * (rotary_dim / 2);
        if (idx >= total_elements) return;
        int half_rotary = rotary_dim / 2;
        int pair_idx = idx % half_rotary;
        int temp = idx / half_rotary;
        int head_idx = temp % num_heads;
        temp = temp / num_heads;
        int seq_idx = temp % seqlen;
        int batch_idx = temp / seqlen;
        int pos = (seqlen_offsets != nullptr) ? seqlen_offsets[batch_idx] + seq_idx : seq_idx;
        float c = static_cast<float>(cos[pos * cos_sin_stride + pair_idx]);
        float s = static_cast<float>(sin[pos * cos_sin_stride + pair_idx]);
        int base_offset = ((batch_idx * seqlen + seq_idx) * num_heads + head_idx) * head_dim;
        int x0_idx = base_offset + pair_idx;
        int x1_idx = base_offset + pair_idx + half_rotary;
        float x0 = static_cast<float>(x[x0_idx]);
        float x1 = static_cast<float>(x[x1_idx]);
        x[x0_idx] = static_cast<scalar_t>(x0 * c - x1 * s);
        x[x1_idx] = static_cast<scalar_t>(x0 * s + x1 * c);
    }
};

inline void apply_rotary_emb_inplace(
    at::Tensor& x,
    const at::Tensor& cos,
    const at::Tensor& sin,
    const std::optional<at::Tensor>& seqlen_offsets,
    bool interleaved) {
    auto batch_size = x.size(0);
    auto seqlen = x.size(1);
    auto num_heads = x.size(2);
    auto head_dim = x.size(3);
    auto rotary_dim = cos.size(1) * 2;
    TORCH_CHECK(rotary_dim <= head_dim, "rotary_dim must be <= head_dim");
    auto queue = c10::xpu::getCurrentXPUStream().queue();
    int total_pairs = batch_size * seqlen * num_heads * (rotary_dim / 2);
    int wg_size = 256;
    int num_groups = (total_pairs + wg_size - 1) / wg_size;
    if (interleaved) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf, x.scalar_type(), "apply_rotary_interleaved", [&] {
                const int* offset_ptr = seqlen_offsets.has_value()
                    ? seqlen_offsets->data_ptr<int>() : nullptr;
                ApplyRotaryInterleavedKernel<scalar_t> kernel{
                    x.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(),
                    sin.data_ptr<scalar_t>(), offset_ptr,
                    (int)batch_size, (int)seqlen, (int)num_heads,
                    (int)head_dim, (int)rotary_dim, (int)cos.size(1)};
                queue.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size), kernel);
                });
            });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::kBFloat16, at::kHalf, x.scalar_type(), "apply_rotary_contiguous", [&] {
                const int* offset_ptr = seqlen_offsets.has_value()
                    ? seqlen_offsets->data_ptr<int>() : nullptr;
                ApplyRotaryContiguousKernel<scalar_t> kernel{
                    x.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(),
                    sin.data_ptr<scalar_t>(), offset_ptr,
                    (int)batch_size, (int)seqlen, (int)num_heads,
                    (int)head_dim, (int)rotary_dim, (int)cos.size(1)};
                queue.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::nd_range<1>(num_groups * wg_size, wg_size), kernel);
                });
            });
    }
}
