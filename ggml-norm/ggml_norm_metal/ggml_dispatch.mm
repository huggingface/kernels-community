/* Metal side: load ggml's metallib, encode its normalisation kernels on torch's own stream.
 *
 * Upstream ships `norm.metal` as its own file, and this package is the matching unit: one operation,
 * one metallib. What it has that a `torch.mps.compile_shader` caller does not is encoding into
 * torch's *current* command buffer -- so these dispatches join the work already queued rather than
 * each becoming its own submission -- and doing that on the stream's own serial queue, which is what
 * keeps it legal (see below).
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
  const char *path = getenv("GGML_NORM_METALLIB");
  if (path == nullptr) {
    NSLog(@"ggml-norm: GGML_NORM_METALLIB is unset and no metallib is embedded");
    return nil;
  }
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  lib = [device() newLibraryWithURL:url error:&error];
#endif
  if (lib == nil) {
    NSLog(@"ggml-norm: could not load the metallib: %@", error);
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
    NSLog(@"ggml-norm: no function %s in the metallib: %@", fn_name, error);
    return nil;
  }
  id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&error];
  [fn release];
  if (pso == nil) {
    NSLog(@"ggml-norm: could not build a pipeline for %s: %@", fn_name, error);
    return nil;
  }
  cache[key] = pso;
  return pso;
}

}  // namespace

// RMS norm fused with its weight multiply: upstream's `kernel_rms_norm_mul_f32`, which is
// `ggml_metal_op_norm` at n_fuse == 2. Eager torch spells the same thing as five dispatches
// (square, mean, add, rsqrt, and two multiplies), and a model normalises at least twice per layer,
// so at decode this is launch overhead rather than arithmetic.
extern "C" int ggml_norm_metal_rms_norm(void *x, size_t x_off, void *w, size_t w_off, void *out,
                                        size_t out_off, int32_t rows, int32_t cols, float eps) {
  // `_4` reads the row as float4, which is what upstream picks whenever the row divides by 4.
  const bool vec4 = (cols % 4) == 0;
  const int32_t ne00_t = vec4 ? cols / 4 : cols;
  const char *fn = vec4 ? "kernel_rms_norm_mul_f32_4" : "kernel_rms_norm_mul_f32";
  id<MTLComputePipelineState> pso = pipeline(fn, fn, nil);
  if (!pso) return -1;

  // The weight is one row broadcast over every token: extent 1 on axes 1..3. Only the multiply
  // source is used at n_fuse == 2, so the add source's strides are left at the weight's.
  ggml_metal_kargs_norm args = {
      /*.ne00   =*/ cols,
      /*.ne00_t =*/ ne00_t,
      /*.nb1    =*/ 4 * (uint64_t)cols,
      /*.nb2    =*/ 4 * (uint64_t)cols * rows,
      /*.nb3    =*/ 4 * (uint64_t)cols * rows,
      /*.eps    =*/ eps,
      /*.nef1   =*/ { rows, 1, 1 },
      /*.nef2   =*/ { 1, 1, 1 },
      /*.nef3   =*/ { 1, 1, 1 },
      /*.nbf1   =*/ { 4 * (uint64_t)cols, 0, 0 },
      /*.nbf2   =*/ { 4 * (uint64_t)cols * rows, 0, 0 },
      /*.nbf3   =*/ { 4 * (uint64_t)cols * rows, 0, 0 },
  };

  // Upstream's thread count: a SIMD width, doubled until it covers the row.
  NSUInteger nth = 32;
  const NSUInteger max_threads = pso.maxTotalThreadsPerThreadgroup;
  while (nth < (NSUInteger)ne00_t && nth < max_threads) nth *= 2;
  nth = MIN(nth, max_threads);
  nth = MIN(nth, (NSUInteger)((ne00_t + 31) / 32 * 32));

  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pso];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:x_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)w offset:w_off atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)w offset:w_off atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:out_off atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
  });
  return 0;
}
