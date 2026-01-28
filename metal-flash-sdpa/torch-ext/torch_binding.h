#pragma once

#include <torch/torch.h>

void flash_attention_varlen(
    torch::Tensor &out,
    torch::Tensor &query,
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &cu_seqlens_q,
    torch::Tensor &cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    bool do_causal,
    double scale,
    double softcapping);
