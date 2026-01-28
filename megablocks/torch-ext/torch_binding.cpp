#include <torch/library.h>
#include <optional>
#include <vector>

#if defined(CUDA_KERNEL)
#include "registration.h"
#include "torch_binding.h"

#include "cumsum.h"
#include "histogram.h"
#include "indices.h"
#include "replicate.h"
#include "sort.h"

#include "grouped_gemm/grouped_gemm.h"
#elif defined(XPU_KERNEL)
#include "../csrc_xpu/core/registration.h"
#include "../csrc_xpu/moe/moe_ops.h"
#include "../csrc_xpu/activation.h"
#include "../csrc_xpu/grouped_gemm/grouped_gemm_interface.h"
#elif defined(CPU_KERNEL)
#include "../csrc_cpu/registration.h"
#include "../csrc_cpu/moe_ops.h"
#include "../csrc_cpu/moe_dispatcher.h"
#endif

#if defined(CUDA_KERNEL)
// ======================== CUDA Implementation ========================

// void exclusive_cumsum(torch::Tensor x, int dim, torch::Tensor out) {
torch::Tensor exclusive_cumsum_wrapper(torch::Tensor x, int64_t dim, torch::Tensor out) {
  megablocks::exclusive_cumsum(x, dim, out);
  return out;
}

// void inclusive_cumsum(torch::Tensor x, int dim, torch::Tensor out) {
torch::Tensor inclusive_cumsum_wrapper(torch::Tensor x, int64_t dim, torch::Tensor out) {
  megablocks::inclusive_cumsum(x, dim, out);
  return out;
}

// torch::Tensor histogram(torch::Tensor x, int num_bins);
torch::Tensor histogram_wrapper(torch::Tensor x, int64_t num_bins) {
  return megablocks::histogram(x, num_bins);
}

// void indices(torch::Tensor padded_bins,
//   int block_size,
//   int output_block_rows,
//   int output_block_columns,
//   torch::Tensor out);
torch::Tensor indices_wrapper(torch::Tensor padded_bins,
                               int64_t block_size,
                               int64_t output_block_rows,
                               int64_t output_block_columns,
                               torch::Tensor out) {
  megablocks::indices(padded_bins, block_size, output_block_rows, output_block_columns, out);
  return out;
}



// Forward pass: replicate values from x according to bin sizes
// void replicate_forward(torch::Tensor x,
//   torch::Tensor bins,
//   torch::Tensor out);
torch::Tensor replicate_forward_wrapper(torch::Tensor x, torch::Tensor bins, torch::Tensor out) {
  megablocks::replicate_forward(x, bins, out);
  return out;
}

// // Backward pass: reduce gradients back to bins using segmented reduction
// void replicate_backward(torch::Tensor grad,
//    torch::Tensor bins,
//    torch::Tensor out);
torch::Tensor replicate_backward_wrapper(torch::Tensor grad, torch::Tensor bins, torch::Tensor out) {
  megablocks::replicate_backward(grad, bins, out);
  return out;
}

// // Public interface function for radix sorting with indices
// void sort(torch::Tensor x,
//   int end_bit,
//   torch::Tensor x_out,
//   torch::Tensor iota_out);
torch::Tensor sort_wrapper(torch::Tensor x, int64_t end_bit, torch::Tensor x_out, torch::Tensor iota_out) {
  megablocks::sort(x, end_bit, x_out, iota_out);
  return x_out;
}

// GroupedGemm operation
torch::Tensor gmm(torch::Tensor a, torch::Tensor b, torch::Tensor c, torch::Tensor batch_sizes, bool trans_a, bool trans_b) {
  grouped_gemm::GroupedGemm(a, b, c, batch_sizes, trans_a, trans_b);
  return c;
}

// Reference implementation:
//
// m.def("exclusive_cumsum", &exclusive_cumsum, "batched exclusive cumsum.");
// m.def("histogram", &histogram, "even width histogram.");
// m.def("inclusive_cumsum", &inclusive_cumsum, "batched inclusive cumsum");
// m.def("indices", &indices, "indices construction for sparse matrix.");
// m.def("replicate_forward", &replicate_forward, "(fwd) replicate a vector dynamically.");
// m.def("replicate_backward", &replicate_backward, "(bwd) replicate a vector dynamically.");
// m.def("sort", &sort, "key/value sort.");

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("exclusive_cumsum(Tensor x, int dim, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("exclusive_cumsum", torch::kCUDA, &exclusive_cumsum_wrapper);

  ops.def("inclusive_cumsum(Tensor x, int dim, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("inclusive_cumsum", torch::kCUDA, &inclusive_cumsum_wrapper);

  ops.def("histogram(Tensor x, int num_bins) -> Tensor");
  ops.impl("histogram", torch::kCUDA, &histogram_wrapper);

  ops.def("indices(Tensor padded_bins, int block_size, int output_block_rows, int output_block_columns, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("indices", torch::kCUDA, &indices_wrapper);

  ops.def("replicate_forward(Tensor x, Tensor bins, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("replicate_forward", torch::kCUDA, &replicate_forward_wrapper);

  ops.def("replicate_backward(Tensor grad, Tensor bins, Tensor(a!) out) -> Tensor(a!)");
  ops.impl("replicate_backward", torch::kCUDA, &replicate_backward_wrapper);
  
  ops.def("sort(Tensor x, int end_bit, Tensor x_out, Tensor iota_out) -> Tensor(x_out)");
  ops.impl("sort", torch::kCUDA, &sort_wrapper);

  // Register the gmm GroupedGemm operation
  ops.def("gmm(Tensor (a!) a, Tensor (b!) b, Tensor(c!) c, Tensor batch_sizes, bool trans_a, bool trans_b) -> Tensor(c!)");
  ops.impl("gmm", torch::kCUDA, &gmm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

#elif defined(XPU_KERNEL)
// ======================== XPU Implementation ========================

// All XPU operations in a single library
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // ==================== MOE Operations ====================
  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  ops.def("moe_sum(Tensor input, Tensor! output) -> ()");
  ops.impl("moe_sum", torch::kXPU, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  ops.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  ops.impl("moe_align_block_size", torch::kXPU, &moe_align_block_size);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size, but for the batched case.
  ops.def(
      "batched_moe_align_block_size(int max_tokens_per_batch,"
      "                     int block_size, Tensor expert_num_tokens,"
      "                     Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  ops.impl(
      "batched_moe_align_block_size",
      torch::kXPU,
      &batched_moe_align_block_size);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  ops.def(
      "moe_lora_align_block_size(Tensor topk_ids,"
      "                     Tensor token_lora_mapping,"
      "                     int num_experts,"
      "                     int block_size, int max_loras, "
      "                     int max_num_tokens_padded, "
      "                     int max_num_m_blocks, "
      "                     Tensor !sorted_token_ids,"
      "                     Tensor !experts_ids,"
      "                     Tensor !num_tokens_post_pad,"
      "                     Tensor !adapter_enabled,"
      "                     Tensor !lora_ids) -> () ");
  ops.impl("moe_lora_align_block_size", torch::kXPU, &moe_lora_align_block_size);

  // Apply grouped topk routing to select experts.
  ops.def(
      "grouped_topk(Tensor scores, Tensor scores_with_bias, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor) -> (Tensor, Tensor)");
  ops.impl("grouped_topk", torch::kXPU, &grouped_topk);

  // Fused Grouped TopK
  ops.def(
      "fused_grouped_topk(Tensor hidden_states, Tensor gating_output, int "
      "n_topk, "
      "bool renormalize, int n_expert_group, int n_topk_group, str "
      "scoring_func, float routed_scaling_factor, Tensor? bias=None) -> "
      "(Tensor, Tensor)");
  ops.impl("fused_grouped_topk", torch::kXPU, &fused_grouped_topk);

  // Apply topk softmax to the gating outputs.
  ops.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize) -> ()");
  ops.impl("topk_softmax", torch::kXPU, &topk_softmax);

  // Gather MOE outputs
  ops.def(
      "moe_gather(Tensor! output, Tensor moe_output, Tensor topk_weights, "
      "Tensor "
      "unpermuted_row_to_permuted_row, int num_experts) -> ()");
  ops.impl("moe_gather", torch::kXPU, &moe_gather);

  // Fused MOE prologue
  ops.def(
      "fused_moe_prologue(Tensor input, Tensor token_selected_experts, "
      "Tensor "
      "token_final_scales, Tensor workspace, int hidden_size, int inter_size, "
      "int num_experts_on_rank) -> "
      "()");
  ops.impl("fused_moe_prologue", torch::kXPU, &fused_moe_prologue);

  // ==================== Activation Operations ====================
  ops.def("silu_and_mul(Tensor! out, Tensor! input) -> ()");
  ops.impl("silu_and_mul", torch::kXPU, &silu_and_mul);

  ops.def("mul_and_silu(Tensor! out, Tensor! input) -> ()");
  ops.impl("mul_and_silu", torch::kXPU, &mul_and_silu);

  ops.def("gelu_and_mul(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_and_mul", torch::kXPU, &gelu_and_mul);

  ops.def("gelu_tanh_and_mul(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_tanh_and_mul", torch::kXPU, &gelu_tanh_and_mul);

  ops.def("gelu_fast(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_fast", torch::kXPU, &gelu_fast);

  ops.def("gelu_new(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_new", torch::kXPU, &gelu_new);

  ops.def("gelu_quick(Tensor! out, Tensor! input) -> ()");
  ops.impl("gelu_quick", torch::kXPU, &gelu_quick);

  ops.def(
      "swigluoai_and_mul(Tensor! out, Tensor input, float alpha=1.702, float "
      "limit=7.0) "
      "-> ()");
  ops.impl("swigluoai_and_mul", torch::kXPU, &swigluoai_and_mul);

  // ==================== Grouped GEMM Operations ====================
  ops.def(
      "cutlass_grouped_gemm_interface(Tensor ptr_A, Tensor ptr_B, "
      "Tensor? ptr_scales, Tensor? ptr_bias, Tensor! ptr_D, "
      "Tensor expert_first_token_offset, int N, int K, int num_experts, "
      "bool is_B_int4, bool is_B_mxfp4) -> Tensor");
  ops.impl("cutlass_grouped_gemm_interface", torch::kXPU, &cutlass_grouped_gemm_interface);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

#elif defined(CPU_KERNEL)
// ======================== CPU Implementation ========================

torch::Tensor convert_weight_packed_wrapper(torch::Tensor weight) {
    // Make a copy since the API requires reference but we receive by value
    auto weight_copy = weight;
    return megablocks::cpu::dispatch::convert_weight_packed(weight_copy);
}

torch::Tensor convert_scale_packed_wrapper(torch::Tensor scale) {
    // Make a copy since the API requires reference but we receive by value
    auto scale_copy = scale;
    return megablocks::cpu::dispatch::convert_scale_packed(scale_copy);
}

torch::Tensor fused_experts_wrapper(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    bool use_mxfp4,
    const c10::optional<torch::Tensor>& w1_scale,
    const c10::optional<torch::Tensor>& w2_scale,
    const c10::optional<std::vector<int64_t>>& block_size,
    const c10::optional<torch::Tensor>& a1_scale,
    const c10::optional<torch::Tensor>& a2_scale,
    const c10::optional<torch::Tensor>& w1_bias,
    const c10::optional<torch::Tensor>& w2_bias,
    const c10::optional<double>& alpha,
    const c10::optional<double>& limit,
    bool is_vnni
) {
    // Create copies for mutable references
    auto hs = hidden_states;
    auto w1_copy = w1;
    auto w2_copy = w2;
    auto tw = topk_weights;
    auto ti = topk_ids;
    return megablocks::cpu::dispatch::fused_experts(
        hs, w1_copy, w2_copy, tw, ti,
        inplace, use_int8_w8a8, use_fp8_w8a16, use_mxfp4,
        w1_scale, w2_scale, block_size, a1_scale, a2_scale,
        w1_bias, w2_bias, alpha, limit, is_vnni
    );
}

torch::Tensor shared_expert_wrapper(
    torch::Tensor hidden_states,
    torch::Tensor w1,
    torch::Tensor w2,
    torch::Tensor fused_experts_out,
    double routed_scaling_factor,
    bool inplace,
    bool use_int8_w8a8,
    bool use_fp8_w8a16,
    const c10::optional<torch::Tensor>& w1_scale,
    const c10::optional<torch::Tensor>& w2_scale,
    const c10::optional<std::vector<int64_t>>& block_size,
    const c10::optional<torch::Tensor>& a1_scale,
    const c10::optional<torch::Tensor>& a2_scale,
    bool is_vnni
) {
    // Create copies for mutable references
    auto hs = hidden_states;
    auto w1_copy = w1;
    auto w2_copy = w2;
    auto feo = fused_experts_out;
    return megablocks::cpu::dispatch::shared_expert(
        hs, w1_copy, w2_copy, feo,
        routed_scaling_factor, inplace, use_int8_w8a8, use_fp8_w8a16,
        w1_scale, w2_scale, block_size, a1_scale, a2_scale, is_vnni
    );
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // Convert weight to VNNI packed format for brgemm
    ops.def("convert_weight_packed(Tensor weight) -> Tensor");
    ops.impl("convert_weight_packed", torch::kCPU, &convert_weight_packed_wrapper);

    // Convert scale to packed format for MXFP4 quantization
    ops.def("convert_scale_packed(Tensor scale) -> Tensor");
    ops.impl("convert_scale_packed", torch::kCPU, &convert_scale_packed_wrapper);

    // Fused experts kernel (sglang compatible)
    ops.def(
        "fused_experts("
        "    Tensor hidden_states,"
        "    Tensor w1,"
        "    Tensor w2,"
        "    Tensor topk_weights,"
        "    Tensor topk_ids,"
        "    bool inplace=False,"
        "    bool use_int8_w8a8=False,"
        "    bool use_fp8_w8a16=False,"
        "    bool use_mxfp4=False,"
        "    Tensor? w1_scale=None,"
        "    Tensor? w2_scale=None,"
        "    int[]? block_size=None,"
        "    Tensor? a1_scale=None,"
        "    Tensor? a2_scale=None,"
        "    Tensor? w1_bias=None,"
        "    Tensor? w2_bias=None,"
        "    float? alpha=None,"
        "    float? limit=None,"
        "    bool is_vnni=False"
        ") -> Tensor"
    );
    ops.impl("fused_experts", torch::kCPU, &fused_experts_wrapper);

    // Shared expert kernel
    ops.def(
        "shared_expert("
        "    Tensor hidden_states,"
        "    Tensor w1,"
        "    Tensor w2,"
        "    Tensor fused_experts_out,"
        "    float routed_scaling_factor,"
        "    bool inplace=False,"
        "    bool use_int8_w8a8=False,"
        "    bool use_fp8_w8a16=False,"
        "    Tensor? w1_scale=None,"
        "    Tensor? w2_scale=None,"
        "    int[]? block_size=None,"
        "    Tensor? a1_scale=None,"
        "    Tensor? a2_scale=None,"
        "    bool is_vnni=False"
        ") -> Tensor"
    );
    ops.impl("shared_expert", torch::kCPU, &shared_expert_wrapper);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)

#endif  // CUDA_KERNEL / XPU_KERNEL / CPU_KERNEL