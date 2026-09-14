/* Metal side: load ggml's metallib, specialise its kernels, encode them on torch's own stream.
 *
 * The three things this has that a `torch.mps.compile_shader` caller does not, and which are the
 * reason the backend exists at all:
 *
 *   - MTLFunctionConstantValues, so a kernel is specialised the way ggml specialises it instead of
 *     the source being rewritten before compiling.
 *   - setThreadgroupMemoryLength, without which mul_mm cannot run at all -- and mul_mm is what
 *     keeps prefill from unpacking every weight to dense.
 *   - encoding into torch's current command buffer, so a layer's dispatches join the work already
 *     queued rather than each becoming its own submission.
 */

#import <Metal/Metal.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include <string>
#include <unordered_map>

#include "common.h"
#include "ggml-metal-impl.h"

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#endif

namespace {

// ggml type ids, matching ggml.h
enum { GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6, GGML_Q5_1 = 7, GGML_Q8_0 = 8,
       GGML_Q2_K = 10, GGML_Q3_K = 11, GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14,
       GGML_IQ2_XXS = 16, GGML_IQ2_XS = 17, GGML_IQ3_XXS = 18, GGML_IQ1_S = 19,
       GGML_IQ4_NL = 20, GGML_IQ3_S = 21, GGML_IQ2_S = 22, GGML_IQ4_XS = 23,
       GGML_IQ1_M = 29, GGML_MXFP4 = 39 };

struct TypeInfo {
  const char *name;   // the infix in kernel_mul_mv_<name>_f32
  int block_elems;
  int block_bytes;
  int mv_nsg;         // N_SG_*  from ggml-metal-impl.h
  int mv_nr0;         // N_R0_*
  // Two gemv grids upstream, chosen by type in ggml-metal-ops.cpp's mul_mv dispatch: the dense
  // types and q8_0 take (ne01 + nr0 - 1)/nr0, everything else divides by nsg as well. q8_0 splits
  // the K reduction across simdgroups (r0 = tgpig.x*NR0) and combines them through threadgroup
  // memory in helper_mv_reduce_and_write, so dividing its grid by nsg would drop rows; the others
  // give each simdgroup its own rows (first_row = (tgpig.x*NSG + sgitg)*nr0) and share nothing.
  // Among quantized types q8_0 is the only one on the first grid. Getting it wrong returns garbage
  // rather than failing.
  bool mv_reduce_across_sgs;
  // ggml_metal_library_get_pipeline_mul_mv's `smem`, per type. Mostly 0; the IQ types cache their
  // lookup grid in threadgroup memory and q8_0 needs a reduction buffer.
  size_t mv_smem;
};

// Transcribed from ggml_metal_library_get_pipeline_mul_mv (ggml-metal-device.cpp): every type it
// implements, minus the dense ones (a GGUF weight this reaches is always quantized) and nvfp4,
// which upstream's Metal backend has no kernel for. Block sizes are ggml's GGML_QUANT_SIZES.
const std::unordered_map<int, TypeInfo> &type_table() {
  static const std::unordered_map<int, TypeInfo> table = {
      // legacy quants
      {GGML_Q4_0, {"q4_0", 32, 18, N_SG_Q4_0, N_R0_Q4_0, false, 0}},
      {GGML_Q4_1, {"q4_1", 32, 20, N_SG_Q4_1, N_R0_Q4_1, false, 0}},
      {GGML_Q5_0, {"q5_0", 32, 22, N_SG_Q5_0, N_R0_Q5_0, false, 0}},
      {GGML_Q5_1, {"q5_1", 32, 24, N_SG_Q5_1, N_R0_Q5_1, false, 0}},
      {GGML_Q8_0, {"q8_0", 32, 34, N_SG_Q8_0, N_R0_Q8_0, true, 32 * sizeof(float) * N_R0_Q8_0}},
      // K quants
      {GGML_Q2_K, {"q2_K", 256, 84, N_SG_Q2_K, N_R0_Q2_K, false, 0}},
      {GGML_Q3_K, {"q3_K", 256, 110, N_SG_Q3_K, N_R0_Q3_K, false, 0}},
      {GGML_Q4_K, {"q4_K", 256, 144, N_SG_Q4_K, N_R0_Q4_K, false, 0}},
      {GGML_Q5_K, {"q5_K", 256, 176, N_SG_Q5_K, N_R0_Q5_K, false, 0}},
      {GGML_Q6_K, {"q6_K", 256, 210, N_SG_Q6_K, N_R0_Q6_K, false, 0}},
      // IQ quants
      {GGML_IQ2_XXS, {"iq2_xxs", 256, 66, N_SG_IQ2_XXS, N_R0_IQ2_XXS, false, 256 * 8 + 128}},
      {GGML_IQ2_XS, {"iq2_xs", 256, 74, N_SG_IQ2_XS, N_R0_IQ2_XS, false, 512 * 8 + 128}},
      {GGML_IQ3_XXS, {"iq3_xxs", 256, 98, N_SG_IQ3_XXS, N_R0_IQ3_XXS, false, 256 * 4 + 128}},
      {GGML_IQ1_S, {"iq1_s", 256, 50, N_SG_IQ1_S, N_R0_IQ1_S, false, 0}},
      {GGML_IQ4_NL, {"iq4_nl", 32, 18, N_SG_IQ4_NL, N_R0_IQ4_NL, false, 32 * sizeof(float)}},
      {GGML_IQ3_S, {"iq3_s", 256, 110, N_SG_IQ3_S, N_R0_IQ3_S, false, 512 * 4}},
      {GGML_IQ2_S, {"iq2_s", 256, 82, N_SG_IQ2_S, N_R0_IQ2_S, false, 0}},
      {GGML_IQ4_XS, {"iq4_xs", 256, 136, N_SG_IQ4_XS, N_R0_IQ4_XS, false, 32 * sizeof(float)}},
      {GGML_IQ1_M, {"iq1_m", 256, 56, N_SG_IQ1_M, N_R0_IQ1_M, false, 0}},
      {GGML_MXFP4, {"mxfp4", 32, 17, N_SG_MXFP4, N_R0_MXFP4, false, 32 * sizeof(float)}},
  };
  return table;
}

const TypeInfo *lookup(int ggml_type) {
  auto it = type_table().find(ggml_type);
  return it == type_table().end() ? nullptr : &it->second;
}

id<MTLDevice> device() { return at::mps::MPSDevice::getInstance()->device(); }

id<MTLLibrary> library() {
  static id<MTLLibrary> lib = nil;
  if (lib != nil) {
    return lib;
  }
  NSError *error = nil;
#ifdef EMBEDDED_METALLIB_HEADER
  lib = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device(), &error);
#else
  // Local builds point at a metallib on disk; the packaged build embeds it instead.
  const char *path = getenv("GGUF_METAL_METALLIB");
  if (path == nullptr) {
    NSLog(@"ggml-quantization: GGUF_METAL_METALLIB is unset and no metallib is embedded");
    return nil;
  }
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  lib = [device() newLibraryWithURL:url error:&error];
#endif
  if (lib == nil) {
    NSLog(@"ggml-quantization: could not load the metallib: %@", error);
  } else {
    [lib retain];
  }
  return lib;
}

// Pipelines are cached under the same key ggml uses: the kernel name plus the constants it was
// specialised with, since a different specialisation is a different pipeline.
id<MTLComputePipelineState> pipeline(const std::string &key, const char *fn_name,
                                     MTLFunctionConstantValues *constants) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
  auto it = cache.find(key);
  if (it != cache.end()) {
    return it->second;
  }
  id<MTLLibrary> lib = library();
  if (lib == nil) {
    return nil;
  }
  NSError *error = nil;
  NSString *name = [NSString stringWithUTF8String:fn_name];
  id<MTLFunction> fn = constants == nil
                           ? [lib newFunctionWithName:name]
                           : [lib newFunctionWithName:name constantValues:constants error:&error];
  if (fn == nil) {
    NSLog(@"ggml-quantization: no function %s in the metallib: %@", fn_name, error);
    return nil;
  }
  id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&error];
  [fn release];
  if (pso == nil) {
    NSLog(@"ggml-quantization: could not build a pipeline for %s: %@", fn_name, error);
    return nil;
  }
  cache[key] = pso;
  return pso;
}

void set_short(MTLFunctionConstantValues *cv, int16_t value, NSUInteger index) {
  [cv setConstantValue:&value type:MTLDataTypeShort atIndex:index];
}

void set_bool(MTLFunctionConstantValues *cv, bool value, NSUInteger index) {
  [cv setConstantValue:&value type:MTLDataTypeBool atIndex:index];
}

}  // namespace

extern "C" int gguf_metal_supports(int ggml_type) { return lookup(ggml_type) != nullptr; }

extern "C" int gguf_metal_gemv_types(int *out, int max) {
  int n = 0;
  for (const auto &entry : type_table()) {
    if (n < max) {
      out[n] = entry.first;
    }
    ++n;
  }
  return n;
}

extern "C" int gguf_metal_mul_mat(void *blocks, size_t blocks_off, void *x, size_t x_off, void *out,
                                  size_t out_off, int ggml_type, int64_t K, int64_t N, int64_t M) {
  const TypeInfo *info = lookup(ggml_type);
  if (info == nullptr) {
    return 1;
  }
  const uint64_t nb01 = (uint64_t)(K / info->block_elems) * info->block_bytes;  // bytes per row
  // One batch, no broadcast: this backend is only ever handed a 2-D weight and a 2-D activation.
  const int16_t r2 = 1, r3 = 1;
  const int32_t ne12 = 1;

  // torch keeps one encoder open on its command buffer and coalesces into it; opening a second is
  // an error, so its encoder is reused -- which also puts these dispatches in the same submission
  // as torch's own. That buffer is owned by the stream's serial queue though, so asking for the
  // encoder from the calling thread races with torch committing it, and encoding into a committed
  // buffer aborts (`_status < MTLCommandBufferStatusCommitted`). Everything below therefore runs on
  // that queue, which is also what torch's own extension guidance prescribes.
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  __block int rc = 0;
  dispatch_sync(stream->queue(), ^{
  id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

  id<MTLBuffer> b_blocks = (__bridge id<MTLBuffer>)blocks;
  id<MTLBuffer> b_x = (__bridge id<MTLBuffer>)x;
  id<MTLBuffer> b_out = (__bridge id<MTLBuffer>)out;

  if (M <= 8) {
    // gemv: one token (or a handful), the decode path
    const int nsg = info->mv_nsg, nr0 = info->mv_nr0;
    char fn[128];
    snprintf(fn, sizeof(fn), "kernel_mul_mv_%s_f32", info->name);
    std::string key = std::string(fn) + "_nsg=" + std::to_string(nsg);

    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    set_short(cv, (int16_t)nsg, FC_MUL_MV + 0);
    set_short(cv, (int16_t)ne12, FC_MUL_MV + 2);
    set_short(cv, r2, FC_MUL_MV + 3);
    set_short(cv, r3, FC_MUL_MV + 4);
    id<MTLComputePipelineState> pso = pipeline(key, fn, cv);
    [cv release];
    if (pso == nil) {
      rc = 2;
    } else {
      ggml_metal_kargs_mul_mv args = {
          /*.ne00 =*/ (int32_t)K,   /*.ne01 =*/ (int32_t)N,  /*.ne02 =*/ 1,
          /*.nb00 =*/ (uint64_t)info->block_bytes,
          /*.nb01 =*/ nb01,         /*.nb02 =*/ nb01 * N,    /*.nb03 =*/ nb01 * N,
          /*.ne10 =*/ (int32_t)K,   /*.ne11 =*/ (int32_t)M,  /*.ne12 =*/ 1,
          /*.nb10 =*/ 4,            /*.nb11 =*/ 4 * (uint64_t)K,
          /*.nb12 =*/ 4 * (uint64_t)K * M, /*.nb13 =*/ 4 * (uint64_t)K * M,
          /*.ne0  =*/ (int32_t)N,   /*.ne1  =*/ (int32_t)M,  /*.nr0 =*/ nr0,
          /*.r2   =*/ r2,           /*.r3   =*/ r3,
      };
      [enc setComputePipelineState:pso];
      [enc setBytes:&args length:sizeof(args) atIndex:0];
      [enc setBuffer:b_blocks offset:blocks_off atIndex:1];
      [enc setBuffer:b_x offset:x_off atIndex:2];
      [enc setBuffer:b_out offset:out_off atIndex:3];
      if (info->mv_smem) {
        [enc setThreadgroupMemoryLength:info->mv_smem atIndex:0];
      }
      const NSUInteger gx = info->mv_reduce_across_sgs ? (N + nr0 - 1) / nr0
                                                       : (N + nr0 * nsg - 1) / (nr0 * nsg);
      [enc dispatchThreadgroups:MTLSizeMake(gx, M, 1)
          threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    }
  } else {
    // gemm: prefill. This is the branch that needs threadgroup memory, and the reason a caller
    // without it has to unpack the weight to dense instead.
    const bool bc_inp = (K % 32) != 0;
    const bool bc_out = false;  // the tensor API path, which this backend does not use
    const int nsg = N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;
    const int nr0 = 64, nr1 = 32;
    const size_t smem = bc_out ? 8192 : (4096 + 2048);

    char fn[128];
    snprintf(fn, sizeof(fn), "kernel_mul_mm_%s_f32", info->name);
    std::string key = std::string(fn) + "_bci=" + std::to_string(bc_inp) +
                      "_bco=" + std::to_string(bc_out);

    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    // All six must be set: an unspecified function constant is not defaulted, the pipeline just
    // reads whatever is there.
    set_bool(cv, bc_inp, FC_MUL_MM + 0);
    set_bool(cv, bc_out, FC_MUL_MM + 1);
    set_short(cv, (int16_t)ne12, FC_MUL_MM + 2);
    set_short(cv, 1, FC_MUL_MM + 3);   // ne13
    set_short(cv, r2, FC_MUL_MM + 4);
    set_short(cv, r3, FC_MUL_MM + 5);
    id<MTLComputePipelineState> pso = pipeline(key, fn, cv);
    [cv release];
    if (pso == nil) {
      rc = 2;
    } else {
      ggml_metal_kargs_mul_mm args = {
          /*.ne00 =*/ (int32_t)K,  /*.ne02 =*/ 1,
          /*.nb01 =*/ nb01,        /*.nb02 =*/ nb01 * N,  /*.nb03 =*/ nb01 * N,
          /*.ne12 =*/ 1,
          /*.nb10 =*/ 4,           /*.nb11 =*/ 4 * (uint64_t)K,
          /*.nb12 =*/ 4 * (uint64_t)K * M, /*.nb13 =*/ 4 * (uint64_t)K * M,
          /*.ne0  =*/ (int32_t)N,  /*.ne1  =*/ (int32_t)M,
          /*.r2   =*/ r2,          /*.r3   =*/ r3,
      };
      [enc setComputePipelineState:pso];
      [enc setBytes:&args length:sizeof(args) atIndex:0];
      [enc setBuffer:b_blocks offset:blocks_off atIndex:1];
      [enc setBuffer:b_x offset:x_off atIndex:2];
      [enc setBuffer:b_out offset:out_off atIndex:3];
      [enc setThreadgroupMemoryLength:smem atIndex:0];
      [enc dispatchThreadgroups:MTLSizeMake((M + nr1 - 1) / nr1, (N + nr0 - 1) / nr0, 1)
          threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
    }
  }
  });

  return rc;
}

extern "C" int gguf_metal_mul_mat_id(void *blocks, size_t blocks_off, void *x, size_t x_off,
                                     void *ids, size_t ids_off, void *out, size_t out_off,
                                     int ggml_type, int64_t K, int64_t N, int64_t E, int64_t T,
                                     int64_t U) {
  const TypeInfo *info = lookup(ggml_type);
  if (info == nullptr) {
    return 1;
  }
  const uint64_t nb01 = (uint64_t)(K / info->block_elems) * info->block_bytes;  // bytes per row
  const int nsg = info->mv_nsg, nr0 = info->mv_nr0;

  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  __block int rc = 0;
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

    char fn[128];
    snprintf(fn, sizeof(fn), "kernel_mul_mv_id_%s_f32", info->name);
    std::string key = std::string(fn) + "_nsg=" + std::to_string(nsg);

    // The same constants `mul_mv` takes: `mul_mv_id` resolves the expert and then calls straight
    // into the very same implementation, with a batch of one.
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    set_short(cv, (int16_t)nsg, FC_MUL_MV + 0);
    set_short(cv, (int16_t)1, FC_MUL_MV + 2);
    set_short(cv, (int16_t)1, FC_MUL_MV + 3);
    set_short(cv, (int16_t)1, FC_MUL_MV + 4);
    id<MTLComputePipelineState> pso = pipeline(key, fn, cv);
    [cv release];
    if (pso == nil) {
      rc = 2;
      return;
    }

    // `ne11 = 1` so every slot of a token reads that token's activation: the kernel takes
    // `i11 = idx % ne11`, and only `i12 = token` should move the source pointer.
    ggml_metal_kargs_mul_mv_id args = {
        /*.nei0 =*/ (int32_t)U,   /*.nei1 =*/ (int32_t)T,  /*.nbi1 =*/ (uint64_t)(U * 4),
        /*.ne00 =*/ (int32_t)K,   /*.ne01 =*/ (int32_t)N,  /*.ne02 =*/ (int32_t)E,
        /*.nb00 =*/ (uint64_t)info->block_bytes,
        /*.nb01 =*/ nb01,         /*.nb02 =*/ nb01 * (uint64_t)N,
        /*.ne10 =*/ (int32_t)K,   /*.ne11 =*/ 1,           /*.ne12 =*/ (int32_t)T,
        /*.ne13 =*/ 1,
        /*.nb10 =*/ 4,            /*.nb11 =*/ 4 * (uint64_t)K,
        /*.nb12 =*/ 4 * (uint64_t)K,
        /*.ne0  =*/ (int32_t)N,   /*.ne1  =*/ (int32_t)U,  /*.nb1 =*/ 4 * (uint64_t)N,
        /*.nr0  =*/ nr0,
    };

    [enc setComputePipelineState:pso];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)blocks offset:blocks_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:x_off atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:out_off atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)ids offset:ids_off atIndex:4];
    if (info->mv_smem) {
      [enc setThreadgroupMemoryLength:info->mv_smem atIndex:0];
    }
    // One threadgroup column per (token, slot): the kernel reads its expert out of `ids` by that.
    const NSUInteger gx = info->mv_reduce_across_sgs ? (N + nr0 - 1) / nr0
                                                     : (N + nr0 * nsg - 1) / (nr0 * nsg);
    [enc dispatchThreadgroups:MTLSizeMake(gx, 1, (NSUInteger)(U * T))
        threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
  });
  return rc;
}


extern "C" int gguf_metal_get_rows(void *blocks, size_t blocks_off, void *indices,
                                   size_t indices_off, void *out, size_t out_off, int ggml_type,
                                   int64_t rows, int64_t cols, int out_dtype) {
  const TypeInfo *info = lookup(ggml_type);
  if (info == nullptr) {
    return 1;
  }
  (void)out_dtype;  // ggml's get_rows writes f32; the caller casts

  char fn[128];
  snprintf(fn, sizeof(fn), "kernel_get_rows_%s", info->name);
  id<MTLComputePipelineState> pso = pipeline(fn, fn, nil);
  if (pso == nil) {
    return 2;
  }
  const uint64_t nb01 = (uint64_t)(cols / info->block_elems) * info->block_bytes;

  // ggml walks a quantized row 16 values at a time, so the thread count is derived from ne00/16
  // rather than the element count.
  const int32_t ne00t = (int32_t)(cols / 16);

  ggml_metal_kargs_get_rows args = {
      /*.ne00t =*/ ne00t,               /*.ne00 =*/ (int32_t)cols,
      /*.nb01  =*/ nb01,                /*.nb02 =*/ nb01 * rows, /*.nb03 =*/ nb01 * rows,
      /*.ne10  =*/ (int32_t)rows,
      /*.nb10  =*/ 4,                   /*.nb11 =*/ 4,           /*.nb12 =*/ 4 * (uint64_t)rows,
      /*.nb1   =*/ 4 * (uint64_t)cols,  /*.nb2  =*/ 4 * (uint64_t)cols * rows,
      /*.nb3   =*/ 4 * (uint64_t)cols * rows,
  };

  const NSUInteger max_threads = pso.maxTotalThreadsPerThreadgroup;
  const NSUInteger nth = MIN((NSUInteger)ne00t, max_threads);
  const NSUInteger nw0 = (ne00t + nth - 1) / nth;

  // Same as the matmul: encode on the stream's own queue so this cannot race torch's commits.
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
  id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
  [enc setComputePipelineState:pso];
  [enc setBytes:&args length:sizeof(args) atIndex:0];
  [enc setBuffer:(__bridge id<MTLBuffer>)blocks offset:blocks_off atIndex:1];
  [enc setBuffer:(__bridge id<MTLBuffer>)indices offset:indices_off atIndex:2];
  [enc setBuffer:(__bridge id<MTLBuffer>)out offset:out_off atIndex:3];
  [enc dispatchThreadgroups:MTLSizeMake(nw0 * rows, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
  });
  return 0;
}
