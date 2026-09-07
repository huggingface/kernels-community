/* Metal side: load ggml's metallib, specialise its kernels, encode them on torch's own stream.
 *
 * The three things this has that a `torch.mps.compile_shader` caller does not:
 *
 *   - MTLFunctionConstantValues, so a kernel is specialised the way ggml specialises it instead of
 *     the source being rewritten before compiling.
 *   - encoding into torch's *current* command buffer, so these dispatches join the work already
 *     queued rather than each becoming its own submission.
 *   - doing that on the stream's own serial queue, which is what keeps it legal (see below).
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
  const char *path = getenv("GGML_ATTN_METALLIB");
  if (path == nullptr) {
    NSLog(@"ggml-gated-delta-net: GGML_ATTN_METALLIB is unset and no metallib is embedded");
    return nil;
  }
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  lib = [device() newLibraryWithURL:url error:&error];
#endif
  if (lib == nil) {
    NSLog(@"ggml-gated-delta-net: could not load the metallib: %@", error);
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
    NSLog(@"ggml-gated-delta-net: no function %s in the metallib: %@", fn_name, error);
    return nil;
  }
  id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&error];
  [fn release];
  if (pso == nil) {
    NSLog(@"ggml-gated-delta-net: could not build a pipeline for %s: %@", fn_name, error);
    return nil;
  }
  cache[key] = pso;
  return pso;
}

void set_short(MTLFunctionConstantValues *cv, int16_t value, NSUInteger index) {
  [cv setConstantValue:&value type:MTLDataTypeShort atIndex:index];
}

void set_int(MTLFunctionConstantValues *cv, int32_t value, NSUInteger index) {
  [cv setConstantValue:&value type:MTLDataTypeInt atIndex:index];
}

void set_bool(MTLFunctionConstantValues *cv, bool value, NSUInteger index) {
  [cv setConstantValue:&value type:MTLDataTypeBool atIndex:index];
}

int64_t pad_to(int64_t x, int64_t n) { return ((x + n - 1) / n) * n; }

// Upstream's flash-attention vector path, from ggml-metal-impl.h and the dispatch in
// ggml-metal-ops.cpp. Only the vector variant is ported here.
constexpr int FA_VEC_NQPSG = 1;   // queries per threadgroup
constexpr int FA_VEC_NCPSG = 32;  // cache values per simdgroup
constexpr int FA_VEC_NHPTG = 1;   // heads per threadgroup
constexpr int FA_VEC_NWG = 32;    // workgroups, upstream's non-disabled branch

// nsg grows until a workgroup's slice covers the cache, capped at 4 -- upstream's loop verbatim.
int fa_vec_nsg(int64_t n_kv) {
  int nsg = 1;
  while (2 * FA_VEC_NWG * nsg * FA_VEC_NCPSG < n_kv && nsg < 4) {
    nsg *= 2;
  }
  return nsg;
}

size_t fa_vec_smem(int64_t head_dim_k, int64_t head_dim_v, int nsg) {
  const int64_t inner =
      (pad_to(head_dim_k, 128) + 4 * FA_VEC_NCPSG + 2 * pad_to(head_dim_v, 128)) * nsg;
  return (size_t)pad_to(inner * (int64_t)(sizeof(float) / 2), 16);
}

// Upstream's threadgroup layout: 32 threads cover a state row `nsg` values at a time.
int nsg_for(int64_t head_dim) { return (int)(head_dim / 32); }

}  // namespace

extern "C" int ggml_gdn_metal_supports_gated_delta_net(int64_t head_dim) {
  if (head_dim % 32 != 0) {
    return 0;
  }
  const int nsg = nsg_for(head_dim);
  // upstream instantiates kernel_gated_delta_net_f32_{1,2,4}
  return nsg == 1 || nsg == 2 || nsg == 4;
}

namespace {
// Mirrors rms_norm_gain.metal; upstream's `ggml_metal_kargs_norm` has the same layout but is not exported
// through the vendored headers this file includes.
struct ggml_attn_kargs_norm {
  int32_t ne00;
  int32_t ne00_t;
  uint64_t nb1;
  uint64_t nb2;
  uint64_t nb3;
  float eps;
  int32_t nef1[3];
  int32_t nef2[3];
  int32_t nef3[3];
  uint64_t nbf1[3];
  uint64_t nbf2[3];
  uint64_t nbf3[3];
};
}  // namespace

extern "C" int ggml_gdn_metal_rms_norm_gate(void *x, size_t x_off, void *weight, size_t weight_off,
                                            void *gate, size_t gate_off, void *out, size_t out_off,
                                            int64_t n_rows, int64_t n_cols, float eps) {
  id<MTLComputePipelineState> pso = pipeline("kernel_rms_norm_gate_f32", "kernel_rms_norm_gate_f32", nil);
  if (pso == nil) {
    return 2;
  }
  const uint64_t row = sizeof(float) * n_cols;
  ggml_attn_kargs_norm args = {
      (int32_t)n_cols, (int32_t)n_cols, row, row * n_rows, row * n_rows, eps,
      {(int32_t)n_rows, 1, 1}, {1, 1, 1}, {1, 1, 1}, {row, 0, 0}, {row * n_rows, 0, 0}, {row * n_rows, 0, 0},
  };
  NSUInteger nth = 32;
  while (nth < (NSUInteger)n_cols && nth < pso.maxTotalThreadsPerThreadgroup) {
    nth *= 2;
  }
  nth = MIN(nth, pso.maxTotalThreadsPerThreadgroup);
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pso];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:x_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:weight_off atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)gate offset:gate_off atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:out_off atIndex:4];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
  });
  return 0;
}

extern "C" int ggml_gdn_metal_delta_gates(void *b, size_t b_off, void *a, size_t a_off, void *a_log,
                                           size_t a_log_off, void *dt_bias, size_t dt_bias_off, void *beta,
                                           size_t beta_off, void *g, size_t g_off, int64_t n_heads) {
  id<MTLComputePipelineState> pso = pipeline("kernel_delta_gates_f32", "kernel_delta_gates_f32", nil);
  if (pso == nil) {
    return 2;
  }
  struct {
    int32_t n_heads;
  } args = {(int32_t)n_heads};

  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pso];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)b offset:b_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)a offset:a_off atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)a_log offset:a_log_off atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)dt_bias offset:dt_bias_off atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)beta offset:beta_off atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)g offset:g_off atIndex:6];
    const NSUInteger nth = MIN((NSUInteger)n_heads, pso.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
  });
  return 0;
}

extern "C" int ggml_gdn_metal_l2_norm(void *x, size_t x_off, void *out, size_t out_off,
                                      int64_t n_rows, int64_t n_cols, float eps) {
  // Upstream takes a 4-wide variant when the row length allows it; rows are flattened into ne01 here
  // because the op only ever normalises the last dimension.
  const bool c4 = (n_cols % 4) == 0;
  const char *fn = c4 ? "kernel_l2_norm_f32_f32_4" : "kernel_l2_norm_f32_f32";
  id<MTLComputePipelineState> pso = pipeline(fn, fn, nil);
  if (pso == nil) {
    return 2;
  }
  const uint64_t row = sizeof(float) * n_cols;
  ggml_metal_kargs_l2_norm args = {
      /*.ne00 =*/(int32_t)(c4 ? n_cols / 4 : n_cols),
      /*.ne01 =*/(int32_t)n_rows,
      /*.ne02 =*/1,
      /*.ne03 =*/1,
      /*.nb00 =*/sizeof(float),
      /*.nb01 =*/row,
      /*.nb02 =*/row * n_rows,
      /*.nb03 =*/row * n_rows,
      /*.ne0 =*/(int32_t)(c4 ? n_cols / 4 : n_cols),
      /*.ne1 =*/(int32_t)n_rows,
      /*.ne2 =*/1,
      /*.ne3 =*/1,
      /*.nb0 =*/sizeof(float),
      /*.nb1 =*/row,
      /*.nb2 =*/row * n_rows,
      /*.nb3 =*/row * n_rows,
      /*.eps =*/eps,
  };
  // one simdgroup, doubling until the row is covered
  NSUInteger nth = 32;
  while (nth < (NSUInteger)args.ne00 && nth < pso.maxTotalThreadsPerThreadgroup) {
    nth *= 2;
  }
  nth = MIN(nth, pso.maxTotalThreadsPerThreadgroup);

  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pso];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:x_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:out_off atIndex:2];
    [enc setThreadgroupMemoryLength:32 * sizeof(float) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(n_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
  });
  return 0;
}

extern "C" int ggml_gdn_metal_gated_delta_net(void *q, size_t q_off, void *k, size_t k_off,
                                               void *v, size_t v_off, void *g, size_t g_off,
                                               void *beta, size_t beta_off, void *state,
                                               size_t state_off, void *dst, size_t dst_off,
                                               int64_t n_seqs, int64_t n_tokens, int64_t n_heads,
                                               int64_t head_dim) {
  if (!ggml_gdn_metal_supports_gated_delta_net(head_dim)) {
    return 1;
  }
  const int nsg = nsg_for(head_dim);
  const int16_t ne20 = (int16_t)head_dim;  // S_v
  const int16_t ne30 = 1;                  // scalar gate, one value per (seq, token, head)
  const int16_t K = 1;                     // keep the final state only

  char fn[128];
  snprintf(fn, sizeof(fn), "kernel_gated_delta_net_f32_%d", nsg);
  std::string key = std::string(fn) + "_ne20=" + std::to_string(ne20) +
                    "_ne30=" + std::to_string(ne30) + "_K=" + std::to_string(K);

  MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
  set_short(cv, ne20, FC_GATED_DELTA_NET + 0);
  set_short(cv, ne30, FC_GATED_DELTA_NET + 1);
  set_short(cv, K, FC_GATED_DELTA_NET + 2);
  id<MTLComputePipelineState> pso = pipeline(key, fn, cv);
  [cv release];
  if (pso == nil) {
    return 2;
  }

  // ggml strides, in bytes, for contiguous tensors in the layouts common.h documents. ggml's ne
  // order is fastest-first, which is the reverse of torch's, so ne0 is head_dim and ne3 is n_seqs.
  const uint64_t f32 = sizeof(float);
  const uint64_t nb_q0 = f32, nb_q1 = f32 * head_dim, nb_q2 = nb_q1 * n_heads,
                 nb_q3 = nb_q2 * n_tokens;

  ggml_metal_kargs_gated_delta_net args = {
      /*.ne00 =*/(int32_t)head_dim,  /*.ne01 =*/(int32_t)n_heads,
      /*.ne02 =*/(int32_t)n_tokens,  /*.ne03 =*/(int32_t)n_seqs,
      /*.nb00 =*/nb_q0,              /*.nb01 =*/nb_q1,
      /*.nb02 =*/nb_q2,              /*.nb03 =*/nb_q3,
      /*.ne10 =*/(int32_t)head_dim,  /*.ne11 =*/(int32_t)n_heads,
      /*.ne12 =*/(int32_t)n_tokens,  /*.ne13 =*/(int32_t)n_seqs,
      /*.nb10 =*/nb_q0,              /*.nb11 =*/nb_q1,
      /*.nb12 =*/nb_q2,              /*.nb13 =*/nb_q3,
      /*.ne20 =*/(int32_t)head_dim,  /*.ne21 =*/(int32_t)n_heads,
      /*.ne22 =*/(int32_t)n_tokens,  /*.ne23 =*/(int32_t)n_seqs,
      /*.nb20 =*/nb_q0,              /*.nb21 =*/nb_q1,
      /*.nb22 =*/nb_q2,              /*.nb23 =*/nb_q3,
      // the kernel walks tokens by adding these to a float pointer, so they are element counts
      /*.ns02 =*/(int32_t)(nb_q2 / f32),
      /*.ns12 =*/(int32_t)(nb_q2 / f32),
      /*.ns22 =*/(int32_t)(nb_q2 / f32),
      /*.ne0  =*/(int32_t)(head_dim * n_heads), /*.ne1 =*/(int32_t)n_tokens,
      /*.ne2  =*/(int32_t)n_seqs,               /*.ne3 =*/1,
      /*.nb0  =*/f32,
      /*.nb1  =*/f32 * head_dim * n_heads,
      /*.nb2  =*/f32 * head_dim * n_heads * n_tokens,
      /*.nb3  =*/f32 * head_dim * n_heads * n_tokens * n_seqs,
  };

  // torch keeps one encoder open on its command buffer and coalesces into it; opening a second is
  // an error, so its encoder is reused -- which also puts these dispatches in the same submission
  // as torch's own. That buffer is owned by the stream's serial queue though, so asking for the
  // encoder from the calling thread races with torch committing it, and encoding into a committed
  // buffer aborts (`_status < MTLCommandBufferStatusCommitted`). Everything below therefore runs on
  // that queue, which is also what torch's own extension guidance prescribes. Skipping this is what
  // makes a kernel die with "A command encoder is already encoding to this command buffer".
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pso];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)q offset:q_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)k offset:k_off atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)v offset:v_off atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)g offset:g_off atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)beta offset:beta_off atIndex:5];
    [enc setBuffer:(__bridge id<MTLBuffer>)state offset:state_off atIndex:6];
    // one destination: the outputs, then the final state
    [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:dst_off atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(head_dim / nsg, n_heads, n_seqs)
        threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
  });
  return 0;
}

// The whole `causal_conv1d_update` in one dispatch: the window read, the cache roll, the bias and
// the activation. See causal_conv.metal for why ggml's `ssm_conv` does not serve here.
extern "C" int ggml_gdn_metal_causal_conv_update(void *state, size_t state_off, void *x,
                                                 size_t x_off, void *weight, size_t weight_off,
                                                 void *bias, size_t bias_off, void *out,
                                                 size_t out_off, int64_t channels, int64_t swidth,
                                                 int64_t k, int has_bias, int silu) {
  id<MTLComputePipelineState> pso =
      pipeline("kernel_causal_conv_update_f32", "kernel_causal_conv_update_f32", nil);
  if (!pso) return -1;

  const int32_t c32 = (int32_t)channels, s32 = (int32_t)swidth, k32 = (int32_t)k;
  const int32_t b32 = has_bias, a32 = silu;

  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();
    [enc setComputePipelineState:pso];
    [enc setBuffer:(__bridge id<MTLBuffer>)state offset:state_off atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)x offset:x_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:weight_off atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)(bias ? bias : state) offset:bias ? bias_off : 0 atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)out offset:out_off atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)state offset:state_off atIndex:5];
    [enc setBytes:&c32 length:sizeof(c32) atIndex:6];
    [enc setBytes:&s32 length:sizeof(s32) atIndex:7];
    [enc setBytes:&k32 length:sizeof(k32) atIndex:8];
    [enc setBytes:&b32 length:sizeof(b32) atIndex:9];
    [enc setBytes:&a32 length:sizeof(a32) atIndex:10];
    const NSUInteger nth = MIN((NSUInteger)pso.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)channels, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
  });
  return 0;
}
