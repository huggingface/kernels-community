#!/bin/bash

# This script generates explicit template instantiation files for different head dimensions
# to reduce compilation time by compiling each instantiation separately.
#
# Files are split by dtype (fp16/bf16) to reduce per-TU memory usage from ~40 GB to ~20 GB,
# allowing higher build parallelism.
#
# File naming convention:
# - flash_fwd_hdim{N}_varlen_fp16.cpp: Variable length fp16 mode
# - flash_fwd_hdim{N}_varlen_bf16.cpp: Variable length bf16 mode
# - flash_fwd_hdim{N}_fix_fp16.cpp: Fixed length fp16 mode
# - flash_fwd_hdim{N}_fix_bf16.cpp: Fixed length bf16 mode

HDIMS=(32 64 96 128 160 192 256 512)

# Create varlen instantiation files split by dtype
echo "Creating varlen instantiation files (split by dtype)..."
for hdim in "${HDIMS[@]}"; do
  for dtype in fp16 bf16; do
    cat > flash_fwd_hdim${hdim}_varlen_${dtype}.cpp << ENDFILE
#include "fmha_fwd_impl.hpp"

// Varlen mode: IsVarLen=1, dtype=${dtype}

// Varlen prefill + non-paged
template void policy_dispatch_${dtype}<
    prefill_policy_head${hdim},
    PipelineStages_Prefill,
    1, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Varlen prefill + paged
template void policy_dispatch_${dtype}<
    prefill_policy_head${hdim},
    PipelineStages_Prefill,
    1, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Varlen decode + non-paged
template void policy_dispatch_${dtype}<
    decode_policy_head${hdim},
    PipelineStages_Decode,
    1, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Varlen decode + paged
template void policy_dispatch_${dtype}<
    decode_paged_policy_head${hdim},
    PipelineStages_Decode,
    1, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);
ENDFILE
    echo "  Created flash_fwd_hdim${hdim}_varlen_${dtype}.cpp"
  done
done

# Create fixed mode instantiation files split by dtype
echo "Creating fixed mode instantiation files (split by dtype)..."
for hdim in "${HDIMS[@]}"; do
  for dtype in fp16 bf16; do
    cat > flash_fwd_hdim${hdim}_fix_${dtype}.cpp << ENDFILE
#include "fmha_fwd_impl.hpp"

// Fixed mode: IsVarLen=0, IsPaged=0, dtype=${dtype}

// Decode fixed mode
template void policy_dispatch_${dtype}<
    decode_policy_head${hdim},
    PipelineStages_Decode,
    0, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Prefill fixed mode
template void policy_dispatch_${dtype}<
    prefill_policy_head${hdim},
    PipelineStages_Prefill,
    0, 0>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);
ENDFILE
    echo "  Created flash_fwd_hdim${hdim}_fix_${dtype}.cpp"
  done
done

echo ""
echo "Creating kvcache-paged instantiation files (split by dtype)..."
for hdim in "${HDIMS[@]}"; do
  for dtype in fp16 bf16; do
    cat > flash_fwd_hdim${hdim}_kvcache_paged_${dtype}.cpp << ENDFILE
#include "fmha_fwd_impl.hpp"

// Non-varlen + paged: IsVarLen=0, IsPaged=1, dtype=${dtype}
// Used by mha_fwd_kvcache when block_table is provided.

// Prefill paged
template void policy_dispatch_${dtype}<
    prefill_policy_head${hdim},
    PipelineStages_Prefill,
    0, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);

// Decode paged (smaller K-tile to fit page boundaries)
template void policy_dispatch_${dtype}<
    decode_paged_policy_head${hdim},
    PipelineStages_Decode,
    0, 1>(
    sycl::queue& queue,
    const fmha_fwd_args_t& args);
ENDFILE
    echo "  Created flash_fwd_hdim${hdim}_kvcache_paged_${dtype}.cpp"
  done
done

echo ""
echo "✓ All instantiation files created successfully!"
echo "  - $((${#HDIMS[@]} * 2)) varlen files (split by dtype)"
echo "  - $((${#HDIMS[@]} * 2)) fixed files (split by dtype)"
echo "  - $((${#HDIMS[@]} * 2)) kvcache_paged files (split by dtype)"
echo "  Total: $((${#HDIMS[@]} * 6)) files"
