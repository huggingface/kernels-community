/* Metal side: load the metallib and encode the kernel on torch's own stream.
 *
 * What this has that a `torch.mps.compile_shader` caller does not is encoding into torch's *current*
 * command buffer -- so the dispatch joins the work already queued rather than becoming its own
 * submission -- and doing that on the stream's own serial queue, which is what keeps it legal.
 */

#import <Metal/Metal.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include <string>
#include <unordered_map>

#include "common.h"

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
  const char *path = getenv("TOPK_METALLIB");
  if (path == nullptr) {
    NSLog(@"topk-metal: TOPK_METALLIB is unset and no metallib is embedded");
    return nil;
  }
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  lib = [device() newLibraryWithURL:url error:&error];
#endif
  if (lib == nil) {
    NSLog(@"topk-metal: could not load the metallib: %@", error);
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
    NSLog(@"topk-metal: no function %s in the metallib: %@", fn_name, error);
    return nil;
  }
  id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&error];
  [fn release];
  if (pso == nil) {
    NSLog(@"topk-metal: could not build a pipeline for %s: %@", fn_name, error);
    return nil;
  }
  cache[key] = pso;
  return pso;
}

}  // namespace

extern "C" int topk_metal_top_k(void *logits, size_t logits_off, void *indices, size_t indices_off,
                                void *values, size_t values_off, int64_t rows, int64_t n, int64_t k,
                                int softmax) {
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  __block int rc = 0;
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    id<MTLComputePipelineState> pso = pipeline("kernel_top_k_f32", "kernel_top_k_f32", nil);
    if (pso == nil) {
      rc = 2;
      return;
    }
    const int32_t n32 = (int32_t)n, k32 = (int32_t)k, sm32 = softmax;
    // One threadgroup per row; a power of two so the reduction below can halve cleanly.
    NSUInteger nth = 32;
    while (nth * 2 <= (NSUInteger)n && nth < 256) {
      nth *= 2;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)logits offset:logits_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)indices offset:indices_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)values offset:values_off atIndex:2];
    [enc setBytes:&n32 length:sizeof(n32) atIndex:3];
    [enc setBytes:&k32 length:sizeof(k32) atIndex:4];
    [enc setBytes:&sm32 length:sizeof(sm32) atIndex:5];
    [enc setThreadgroupMemoryLength:nth * (sizeof(float) + sizeof(int)) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
  });
  return rc;
}
