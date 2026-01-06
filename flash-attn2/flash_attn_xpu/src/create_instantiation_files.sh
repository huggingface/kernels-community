#!/bin/bash

# This script generates explicit template instantiation files for different head dimensions
# to reduce compilation time by compiling each instantiation separately.
#
# File naming convention:
# - flash_fwd_hdim{N}_varlen.cpp: Variable length mode (IsVarLen=1) with paged/non-paged variants
# - flash_fwd_hdim{N}_fix.cpp: Fixed length mode (IsVarLen=0, IsPaged=0) for both decode and prefill

HDIMS=(32 64 96 128 160 192 256)

# Create varlen instantiation files (IsVarLen=1, with paged/non-paged)
echo "Creating varlen instantiation files..."
for hdim in "${HDIMS[@]}"; do
  cat > flash_fwd_hdim${hdim}_varlen.cpp << ENDFILE
#include "fmha_fwd_impl.hpp"

// Varlen mode: IsVarLen=1, handles both paged and non-paged cases
// No need for dynamic dispatch since we know it's varlen

// Varlen + non-paged
template void policy_dispatch<
    prefill_policy_head${hdim}, 
    PipelineStages_Prefill, 
    1, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);

// Varlen + paged
template void policy_dispatch<
    prefill_policy_head${hdim}, 
    PipelineStages_Prefill, 
    1, 1>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
ENDFILE
  echo "  Created flash_fwd_hdim${hdim}_varlen.cpp"
done

# Create fixed mode instantiation files (decode + prefill in one file)
echo "Creating fixed mode instantiation files..."
for hdim in "${HDIMS[@]}"; do
  cat > flash_fwd_hdim${hdim}_fix.cpp << ENDFILE
#include "fmha_fwd_impl.hpp"

// Fixed mode: non-varlen (IsVarLen=0), non-paged (IsPaged=0)
// Includes both decode and prefill policies

// Decode fixed mode
template void policy_dispatch<
    decode_policy_head${hdim}, 
    PipelineStages_Decode, 
    0, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);

// Prefill fixed mode
template void policy_dispatch<
    prefill_policy_head${hdim}, 
    PipelineStages_Prefill, 
    0, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
ENDFILE
  echo "  Created flash_fwd_hdim${hdim}_fix.cpp"
done

echo ""
echo "✓ All instantiation files created successfully!"
echo "  - ${#HDIMS[@]} varlen files (IsVarLen=1, paged + non-paged)"
echo "  - ${#HDIMS[@]} fixed files (IsVarLen=0, decode + prefill)"
echo "  Total: $((${#HDIMS[@]} * 2)) files"
