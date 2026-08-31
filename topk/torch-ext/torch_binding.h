#pragma once

#include <torch/torch.h>

#include <vector>

// Top-k over each row: returns `{values, indices}`, both `rows x k`, with `indices` as int32.
//
// For a MoE router, where torch's MPS top-k is a full sort -- 71us for k=8 of 256, against 26us
// here -- and a router runs once per layer per token. With `softmax` set, `values` comes back
// softmaxed over the k that were selected, which is what a router wants and saves a dispatch.
std::vector<at::Tensor> top_k(const at::Tensor &logits, int64_t k, bool softmax);
