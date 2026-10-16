// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

#include <optional>

#include "aiter_stream.h"
#include "gfx1201_sage_attention.h"
#include "gfx1201_sage_prepare.h"

namespace {

aiter_tensor_t view(const at::Tensor& t)
{
    aiter_tensor_t v{};
    v.ptr  = t.data_ptr();
    v.ndim = static_cast<int>(t.dim());
    for(int i = 0; i < v.ndim; ++i)
        v.shape[i] = t.size(i);
    return v;
}

std::optional<aiter_tensor_t> view(const std::optional<at::Tensor>& t)
{
    return t.has_value() ? std::optional<aiter_tensor_t>(view(*t)) : std::nullopt;
}

// The vendored kernels launch on aiter's thread-local stream; point it at Torch's.
void use_torch_stream(const at::Tensor& t)
{
    aiter::setCurrentHIPStream(at::hip::getCurrentHIPStream(t.get_device()));
}

} // namespace

void gfx1201_sage_prepare(const at::Tensor& query,
                          const at::Tensor& key,
                          const at::Tensor& value,
                          const std::optional<at::Tensor>& query_weight,
                          const std::optional<at::Tensor>& key_weight,
                          const std::optional<at::Tensor>& cosine,
                          const std::optional<at::Tensor>& sine,
                          at::Tensor& maximum,
                          at::Tensor& query_out,
                          at::Tensor& key_out,
                          at::Tensor& value_out,
                          at::Tensor& query_scale,
                          at::Tensor& key_scale,
                          at::Tensor& value_scale,
                          int64_t rope_dim,
                          double eps,
                          double sm_scale)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{query.device()};
    use_torch_stream(query);
    aiter_tensor_t maximum_v = view(maximum), query_out_v = view(query_out),
                   key_out_v = view(key_out), value_out_v = view(value_out),
                   query_scale_v = view(query_scale), key_scale_v = view(key_scale),
                   value_scale_v = view(value_scale);
    gfx1201_sage_prepare_hip(view(query),
                             view(key),
                             view(value),
                             view(query_weight),
                             view(key_weight),
                             view(cosine),
                             view(sine),
                             maximum_v,
                             query_out_v,
                             key_out_v,
                             value_out_v,
                             query_scale_v,
                             key_scale_v,
                             value_scale_v,
                             rope_dim,
                             eps,
                             sm_scale);
}

void gfx1201_sage_attention_fwd(const at::Tensor& q_int8,
                                const at::Tensor& q_scale,
                                const at::Tensor& k_int8,
                                const at::Tensor& k_scale,
                                const at::Tensor& v_fp8,
                                const at::Tensor& v_scale,
                                at::Tensor& out,
                                int64_t seq_len)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{q_int8.device()};
    use_torch_stream(q_int8);
    aiter_tensor_t out_v = view(out);
    gfx1201_sage_attention_fwd_hip(view(q_int8),
                                   view(q_scale),
                                   view(k_int8),
                                   view(k_scale),
                                   view(v_fp8),
                                   view(v_scale),
                                   out_v,
                                   seq_len);
}
