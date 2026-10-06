#ifdef TORCH_TARGET_VERSION
#include <torch/csrc/stable/library.h>
#else
#include <torch/library.h>
#endif

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

#define PAGED_ATTENTION_V1_SCHEMA                                       \
  "paged_attention_v1("                                                 \
  "    Tensor! out, Tensor query, Tensor key_cache,"                    \
  "    Tensor value_cache, int num_kv_heads, float scale,"              \
  "    Tensor block_tables, Tensor seq_lens, int block_size,"           \
  "    int max_seq_len, Tensor? alibi_slopes,"                          \
  "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"             \
  "    int tp_rank, int blocksparse_local_blocks,"                      \
  "    int blocksparse_vert_stride, int blocksparse_block_size,"        \
  "    int blocksparse_head_sliding_step) -> ()"

#define PAGED_ATTENTION_V2_SCHEMA                                       \
  "paged_attention_v2("                                                 \
  "    Tensor! out, Tensor! exp_sums, Tensor! max_logits,"              \
  "    Tensor! tmp_out, Tensor query, Tensor key_cache,"                \
  "    Tensor value_cache, int num_kv_heads, float scale,"              \
  "    Tensor block_tables, Tensor seq_lens, int block_size,"           \
  "    int max_seq_len, Tensor? alibi_slopes,"                          \
  "    str kv_cache_dtype, Tensor k_scale, Tensor v_scale,"             \
  "    int tp_rank, int blocksparse_local_blocks,"                      \
  "    int blocksparse_vert_stride, int blocksparse_block_size,"        \
  "    int blocksparse_head_sliding_step) -> ()"

// Swap in (out) the cache blocks from src to dst.
#define SWAP_BLOCKS_SCHEMA \
  "swap_blocks(Tensor src, Tensor! dst, Tensor block_mapping) -> ()"

// Copy the cache blocks from src to dst.
#define COPY_BLOCKS_SCHEMA                                    \
  "copy_blocks(Tensor(a!)[] key_caches, Tensor[](b!) value_caches, " \
  "Tensor block_mapping) -> ()"

// Reshape the key and value tensors and cache them.
#define RESHAPE_AND_CACHE_SCHEMA                         \
  "reshape_and_cache(Tensor key, Tensor value,"          \
  "                  Tensor! key_cache, Tensor! value_cache," \
  "                  Tensor slot_mapping,"               \
  "                  str kv_cache_dtype,"                \
  "                  Tensor k_scale, Tensor v_scale) -> ()"

// Reshape the key and value tensors and cache them.
#define RESHAPE_AND_CACHE_FLASH_SCHEMA                         \
  "reshape_and_cache_flash(Tensor key, Tensor value,"          \
  "                        Tensor! key_cache,"                 \
  "                        Tensor! value_cache,"               \
  "                        Tensor slot_mapping,"               \
  "                        str kv_cache_dtype,"                \
  "                        Tensor k_scale, Tensor v_scale) -> ()"

// Gets the specified device attribute.
#define GET_DEVICE_ATTRIBUTE_SCHEMA \
  "get_device_attribute(int attribute, int device_id) -> int"

// Gets the maximum shared memory per block device attribute.
#define GET_MAX_SHARED_MEMORY_SCHEMA                   \
  "get_max_shared_memory_per_block_device_attribute(" \
  "int device_id) -> int"

// Convert the key and value cache to fp8 data type.
#define CONVERT_FP8_SCHEMA                                        \
  "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, " \
  "str kv_cache_dtype) -> ()"

#ifdef TORCH_TARGET_VERSION

// Dispatch key of the backend that is being built.
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
#define PAGED_ATTENTION_DISPATCH_KEY CUDA
#elif defined(METAL_KERNEL)
#define PAGED_ATTENTION_DISPATCH_KEY MPS
#else
#error "Unsupported backend for the stable ABI"
#endif

// Stable-ABI registration
STABLE_TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(PAGED_ATTENTION_V1_SCHEMA);
  ops.def(PAGED_ATTENTION_V2_SCHEMA);
  ops.def(SWAP_BLOCKS_SCHEMA);
  ops.def(COPY_BLOCKS_SCHEMA);
  ops.def(RESHAPE_AND_CACHE_SCHEMA);
  ops.def(RESHAPE_AND_CACHE_FLASH_SCHEMA);
  ops.def(GET_DEVICE_ATTRIBUTE_SCHEMA);
  ops.def(GET_MAX_SHARED_MEMORY_SCHEMA);
  ops.def(CONVERT_FP8_SCHEMA);
}

STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME,
                                 PAGED_ATTENTION_DISPATCH_KEY, ops) {
  ops.impl("paged_attention_v1", TORCH_BOX(&paged_attention_v1));
  ops.impl("paged_attention_v2", TORCH_BOX(&paged_attention_v2));
  ops.impl("swap_blocks", TORCH_BOX(&swap_blocks));
  ops.impl("copy_blocks", TORCH_BOX(&copy_blocks));
  ops.impl("reshape_and_cache", TORCH_BOX(&reshape_and_cache));
  ops.impl("reshape_and_cache_flash", TORCH_BOX(&reshape_and_cache_flash));
  ops.impl("convert_fp8", TORCH_BOX(&convert_fp8));
}

// These ops have no tensor arguments, so they cannot be dispatched on a
// backend key.
STABLE_TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME,
                                 CompositeExplicitAutograd, ops) {
  ops.impl("get_device_attribute", TORCH_BOX(&get_device_attribute));
  ops.impl("get_max_shared_memory_per_block_device_attribute",
           TORCH_BOX(&get_max_shared_memory_per_block_device_attribute));
}

#else

// Non-stable registration (currently Metal)
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(PAGED_ATTENTION_V1_SCHEMA);
  ops.impl("paged_attention_v1", torch::kMPS, paged_attention_v1);

  ops.def(PAGED_ATTENTION_V2_SCHEMA);
  ops.impl("paged_attention_v2", torch::kMPS, paged_attention_v2);

  ops.def(SWAP_BLOCKS_SCHEMA);
  ops.impl("swap_blocks", torch::kMPS, swap_blocks);

  ops.def(COPY_BLOCKS_SCHEMA);
  ops.impl("copy_blocks", torch::kMPS, copy_blocks);

  ops.def(RESHAPE_AND_CACHE_SCHEMA);
  ops.impl("reshape_and_cache", torch::kMPS, reshape_and_cache);

  ops.def(RESHAPE_AND_CACHE_FLASH_SCHEMA);
  ops.impl("reshape_and_cache_flash", torch::kMPS, reshape_and_cache_flash);

  ops.def(GET_DEVICE_ATTRIBUTE_SCHEMA);
  ops.impl("get_device_attribute", &get_device_attribute);

  ops.def(GET_MAX_SHARED_MEMORY_SCHEMA);
  ops.impl("get_max_shared_memory_per_block_device_attribute",
           &get_max_shared_memory_per_block_device_attribute);

  ops.def(CONVERT_FP8_SCHEMA);
  ops.impl("convert_fp8", torch::kMPS, convert_fp8);
}

#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
