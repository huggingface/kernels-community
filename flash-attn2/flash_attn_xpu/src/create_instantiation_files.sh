#!/bin/bash

# Create prefill instantiation files
for hdim in 32 64 96 128 160 192 256; do
  cat > flash_fwd_hdim${hdim}.cpp << ENDFILE
#include "fmha_fwd_impl.hpp"

template void policy_dispatch_dynamic<
    prefill_policy_head${hdim}, 
    PipelineStages_Prefill>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
ENDFILE
done

# Create decode instantiation files
for hdim in 32 64 96 128 160 192 256; do
  cat > flash_fwd_hdim${hdim}_decode.cpp << ENDFILE
#include "fmha_fwd_impl.hpp"

template void policy_dispatch<
    decode_policy_head${hdim}, 
    PipelineStages_Decode, 
    0, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
ENDFILE
done

# Create prefill fix mode instantiation files
for hdim in 32 64 96 128 160 192 256; do
  cat > flash_fwd_hdim${hdim}_prefill_fix.cpp << ENDFILE
#include "fmha_fwd_impl.hpp"

template void policy_dispatch<
    prefill_policy_head${hdim}, 
    PipelineStages_Prefill, 
    0, 0>(
    sycl::queue& queue, 
    CutlassType cuType, 
    const fmha_fwd_args_t& args);
ENDFILE
done

echo "Created all instantiation files"
