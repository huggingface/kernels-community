#include <torch/library.h>

#include "registration.h"

#include "torch_binding.h"

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // Attention ops
    // Compute the attention between an input query and the cached
    // keys/values using PagedAttention.
    ops.def(
        "paged_attention_v1("
        "    Tensor! out, Tensor query, Tensor key_cache,"
        "    Tensor value_cache, int num_kv_heads, float scale,"
        "    Tensor block_tables, Tensor seq_lens, int block_size,"
        "    int max_seq_len, Tensor? alibi_slopes,"
        "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
        "    int tp_rank, int blocksparse_local_blocks,"
        "    int blocksparse_vert_stride, int blocksparse_block_size,"
        "    int blocksparse_head_sliding_step) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("paged_attention_v1", torch::kCUDA, &paged_attention_v1);
#elif defined(METAL_KERNEL)
    ops.impl("paged_attention_v1", torch::kMPS, paged_attention_v1);
#endif

    // PagedAttention V2.
    ops.def(
        "paged_attention_v2("
        "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"
        "    Tensor! tmp_out, Tensor query, Tensor key_cache,"
        "    Tensor value_cache, int num_kv_heads, float scale,"
        "    Tensor block_tables, Tensor seq_lens, int block_size,"
        "    int max_seq_len, Tensor? alibi_slopes,"
        "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"
        "    int tp_rank, int blocksparse_local_blocks,"
        "    int blocksparse_vert_stride, int blocksparse_block_size,"
        "    int blocksparse_head_sliding_step) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("paged_attention_v2", torch::kCUDA, &paged_attention_v2);
#elif defined(METAL_KERNEL)
    ops.impl("paged_attention_v2", torch::kMPS, paged_attention_v2);
#endif

    // Swap in (out) the cache blocks from src to dst.
    ops.def(
        "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("swap_blocks", torch::kCUDA, &swap_blocks);
#elif defined(METAL_KERNEL)
    ops.impl("swap_blocks", torch::kMPS, swap_blocks);
#endif

    // Copy the cache blocks from src to dst.
    ops.def(
        "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, "
        "Tensor block_mapping) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("copy_blocks", torch::kCUDA, &copy_blocks);
#elif defined(METAL_KERNEL)
    ops.impl("copy_blocks", torch::kMPS, copy_blocks);
#endif

    // Reshape the key and value tensors and cache them.
    ops.def(
        "reshape_and_cache(Tensor key, Tensor value,"
        "                  Tensor! key_cache, Tensor! value_cache,"
        "                  Tensor slot_mapping,"
        "                  str kv_cache_dtype,"
        "                  Tensor k_scale, Tensor v_scale) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("reshape_and_cache", torch::kCUDA, &reshape_and_cache);
#elif defined(METAL_KERNEL)
    ops.impl("reshape_and_cache", torch::kMPS, reshape_and_cache);
#endif

    // Reshape the key and value tensors and cache them.
    ops.def(
        "reshape_and_cache_flash(Tensor key, Tensor value,"
        "                        Tensor! key_cache,"
        "                        Tensor! value_cache,"
        "                        Tensor slot_mapping,"
        "                        str kv_cache_dtype,"
        "                        Tensor k_scale, Tensor v_scale) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("reshape_and_cache_flash", torch::kCUDA, &reshape_and_cache_flash);
#elif defined(METAL_KERNEL)
    ops.impl("reshape_and_cache_flash", torch::kMPS, reshape_and_cache_flash);
#endif

    // Gets the specified device attribute.
    ops.def("get_device_attribute(int attribute, int device_id) -> int");
    ops.impl("get_device_attribute", &get_device_attribute);

    // Gets the maximum shared memory per block device attribute.
    ops.def(
        "get_max_shared_memory_per_block_device_attribute(int device_id) -> int");
    ops.impl("get_max_shared_memory_per_block_device_attribute",
             &get_max_shared_memory_per_block_device_attribute);

    // Convert the key and value cache to fp8 data type.
    ops.def(
        "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
        "str kv_cache_dtype) -> ()");
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("convert_fp8", torch::kCUDA, &convert_fp8);
#elif defined(METAL_KERNEL)
    ops.impl("convert_fp8", torch::kMPS, convert_fp8);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
