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
    NSLog(@"ggml-attn: GGML_ATTN_METALLIB is unset and no metallib is embedded");
    return nil;
  }
  NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
  lib = [device() newLibraryWithURL:url error:&error];
#endif
  if (lib == nil) {
    NSLog(@"ggml-attn: could not load the metallib: %@", error);
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
    NSLog(@"ggml-attn: no function %s in the metallib: %@", fn_name, error);
    return nil;
  }
  id<MTLComputePipelineState> pso = [device() newComputePipelineStateWithFunction:fn error:&error];
  [fn release];
  if (pso == nil) {
    NSLog(@"ggml-attn: could not build a pipeline for %s: %@", fn_name, error);
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

// Upstream's two flash-attention paths, from ggml-metal-impl.h and the dispatch in
// ggml-metal-ops.cpp. Both are ported: the vector one for few queries, the tiled one above that.
constexpr int FA_VEC_NQPSG = 1;   // queries per threadgroup
constexpr int FA_VEC_NCPSG = 32;  // cache values per simdgroup
constexpr int FA_VEC_NHPTG = 1;   // heads per threadgroup
constexpr int FA_VEC_NWG = 32;    // workgroups, upstream's non-disabled branch

// OP_FLASH_ATTN_EXT_NQPSG / _NCPSG. The tiled kernel does 8 queries per threadgroup against 64 cache
// values per simdgroup, using simdgroup matrix multiplies where the vector one uses reductions --
// which is why upstream switches at 20 queries: one query row cannot fill an 8x8 matrix unit.
constexpr int FA_NQPSG = 8;
constexpr int FA_NCPSG = 64;

// `ggml_metal_op_flash_attn_ext_use_vec`, verbatim.
bool fa_use_vec(int64_t n_q, int64_t head_dim_k) {
  return n_q < 20 && head_dim_k % 32 == 0;
}

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

// FATTN_SMEM from ggml-metal-ops.cpp. `is_q` is 0 here -- K is f32, never quantized -- so the
// `16*32*nsg` term for dequantizing into shared memory drops out.
size_t fa_smem(int64_t head_dim_k, int64_t head_dim_v) {
  const int64_t inner =
      FA_NQPSG * (head_dim_k + 2 * pad_to(head_dim_v, 64) + 2 * (2 * FA_NCPSG));
  return (size_t)pad_to(inner * (int64_t)(sizeof(float) / 2), 16);
}

// The dk/dv pairs upstream instantiates. Hardcoded, the way llama.cpp hardcodes the same list in
// `ggml_metal_device_supports_op` ("for new head sizes, add checks here").
//
// Re-derive these after every `vendor.py` bump -- the set moves (it went from 15 tiled pairs to 8
// between llama.cpp 432d7ffe and 50f068ff):
//
//   grep -ohE 'kernel_flash_attn_ext(_vec)?_f32_dk[0-9]+_dv[0-9]+' \
//       vendor/src/ggml-metal/kernels/fa.metal | sort -u
//
// Getting it wrong is loud but late: a pair with no instantiation finds no function and the
// dispatch reports it, except `supports_flash_attn` will already have promised the shape.
//
// The vector kernel covers a narrower set than the tiled one, so the two are asked separately.
constexpr int FA_TILED[][2] = {{64, 64}, {80, 80}, {96, 96}, {112, 112}, {128, 128},
                               {192, 128}, {192, 192}, {256, 256}};
constexpr int FA_VEC[][2] = {{32, 32}, {64, 64}, {96, 96}, {128, 128}, {192, 128},
                             {192, 192}, {256, 256}, {320, 256}, {512, 512}, {576, 512}};

bool fa_listed(const int (*table)[2], size_t count, int64_t dk, int64_t dv) {
  for (size_t i = 0; i < count; ++i) {
    if (table[i][0] == dk && table[i][1] == dv) {
      return true;
    }
  }
  return false;
}

bool fa_has_template(int64_t dk, int64_t dv) {
  return fa_listed(FA_TILED, sizeof(FA_TILED) / sizeof(FA_TILED[0]), dk, dv);
}

}  // namespace

extern "C" int ggml_attn_metal_supports_flash_attn(int64_t n_q, int64_t head_dim_k,
                                                   int64_t head_dim_v) {
  // "for simplicity assume K is larger or equal than V", as upstream asserts
  if (head_dim_k < head_dim_v) {
    return 0;
  }
  if (fa_use_vec(n_q, head_dim_k)) {
    // The vector kernel is instantiated only for equal widths, and needs dk divisible by 4 for its
    // float4 loads on top of the % 32 that selected it. Those two rules are necessary but not
    // sufficient: dk 160 and 224 satisfy both and have no instantiation, so check the list as
    // well rather than claiming a shape whose pipeline would then fail to load.
    return head_dim_k % 4 == 0 && head_dim_k == head_dim_v &&
           fa_listed(FA_VEC, sizeof(FA_VEC) / sizeof(FA_VEC[0]), head_dim_k, head_dim_v);
  }
  // Otherwise the tiled kernel, which covers the wider set of head-dim pairs upstream instantiates.
  return fa_has_template(head_dim_k, head_dim_v) ? 1 : 0;
}

extern "C" void ggml_attn_metal_flash_attn_scratch(int64_t n_seqs, int64_t n_heads,
                                                   int64_t n_heads_kv, int64_t n_q, int64_t n_kv,
                                                   int64_t head_dim_k, int64_t head_dim_v,
                                                   int has_mask, int64_t *pad_floats,
                                                   int64_t *tmp_floats,
                                                   int64_t *blk_floats) {
  const bool is_vec = fa_use_vec(n_q, head_dim_k);
  const int ncpsg = is_vec ? FA_VEC_NCPSG : FA_NCPSG;
  const int nqptg = is_vec ? FA_VEC_NQPSG : FA_NQPSG;

  // The pad kernel writes `ncpsg` extra cache rows for k, v and the mask, so the kernel can read a
  // whole simdgroup's worth past the end of an unaligned cache.
  const bool has_kvpad = (n_kv % ncpsg) != 0;
  int64_t pad = 1;  // a buffer still has to be bound when it is not read
  if (has_kvpad) {
    const int64_t k_row = head_dim_k * n_heads_kv * n_seqs;
    const int64_t v_row = head_dim_v * n_heads_kv * n_seqs;
    // the mask's padding is f16, so half a float per element
    const int64_t m_row = has_mask ? (n_q * n_seqs + 1) / 2 : 0;
    pad = ncpsg * (k_row + v_row + m_row);
  }
  *pad_floats = pad;

  // Each workgroup writes a partial head vector plus its running S and M. Only the vector path uses
  // it -- the tiled kernel writes dst directly -- but upstream reserves it either way.
  const int64_t n_q_max = n_q < 32 ? n_q : 32;
  *tmp_floats = n_q_max * n_heads * n_seqs * FA_VEC_NWG * (head_dim_v + 2);

  // The block map: one i8 per (query block, cache block) telling the tiled kernel whether a block is
  // fully masked out and can be skipped. Only built when there is a mask, and only read by the tiled
  // path, but a buffer is bound either way.
  int64_t blk = 1;
  if (has_mask && !is_vec) {
    const int64_t nblk1 = (n_q + nqptg - 1) / nqptg;
    const int64_t nblk0 = (n_kv + ncpsg - 1) / ncpsg;
    const int64_t bytes = pad_to(nblk0 * nblk1 * 1 * n_seqs, 32);  // ne32 == 1
    blk = (bytes + 3) / 4;
  }
  *blk_floats = blk;
}

// The tiled path: `kernel_flash_attn_ext`, upstream's choice above 20 queries. Three kernels rather
// than the vector path's three-with-a-reduce -- pad, blk, then the attention itself, which writes dst
// directly because 8 queries per threadgroup is enough parallelism without splitting the cache.
static int fa_dispatch_tiled(void *q, size_t q_off, void *k, size_t k_off, void *v, size_t v_off,
                             void *mask, size_t mask_off, void *pad, size_t pad_off, void *blk,
                             size_t blk_off, void *dst, size_t dst_off, int64_t n_seqs,
                             int64_t n_heads, int64_t n_heads_kv, int64_t n_q, int64_t n_kv,
                             int64_t head_dim_k, int64_t head_dim_v, float scale, int has_mask) {
  const bool has_kvpad = (n_kv % FA_NCPSG) != 0;
  // "do bounds checks for the mask?" -- upstream sets it when the query count does not fill whole
  // threadgroups, so the last one would read past the mask.
  const bool bc_mask = has_mask && (n_q % FA_NQPSG != 0);
  const int32_t nsg = head_dim_k >= 512 ? 8 : 4;
  const size_t smem = fa_smem(head_dim_k, head_dim_v);

  const uint64_t f32 = sizeof(float), f16 = 2;
  const uint64_t nb01 = f32 * head_dim_k, nb02 = nb01 * n_q, nb03 = nb02 * n_heads;
  const uint64_t nb11 = f32 * head_dim_k, nb12 = nb11 * n_kv, nb13 = nb12 * n_heads_kv;
  const uint64_t nb21 = f32 * head_dim_v, nb22 = nb21 * n_kv, nb23 = nb22 * n_heads_kv;
  const uint64_t nb31 = f16 * n_kv, nb32 = nb31 * n_q, nb33 = nb32;
  const int32_t ne31 = (int32_t)n_q, ne32 = 1, ne33 = (int32_t)n_seqs;

  const float max_bias = 0.0f, logit_softcap = 0.0f;
  const int32_t n_head_log2 = 1 << (int)floorf(log2f((float)n_heads));

  ggml_metal_kargs_flash_attn_ext args = {
      /*.ne01 =*/(int32_t)n_q,
      /*.ne02 =*/(int32_t)n_heads,
      /*.ne03 =*/(int32_t)n_seqs,
      /*.nb01 =*/nb01,
      /*.nb02 =*/nb02,
      /*.nb03 =*/nb03,
      /*.ne11 =*/(int32_t)n_kv,
      /*.ne_12_2 =*/(int32_t)n_heads_kv,
      /*.ne_12_3 =*/(int32_t)n_seqs,
      /*.ns10 =*/(int32_t)(nb11 / f32),
      /*.nb11 =*/nb11,
      /*.nb12 =*/nb12,
      /*.nb13 =*/nb13,
      /*.ns20 =*/(int32_t)(nb21 / f32),
      /*.nb21 =*/nb21,
      /*.nb22 =*/nb22,
      /*.nb23 =*/nb23,
      /*.ne31 =*/ne31,
      /*.ne32 =*/ne32,
      /*.ne33 =*/ne33,
      /*.nb31 =*/nb31,
      /*.nb32 =*/nb32,
      /*.nb33 =*/nb33,
      /*.ne1 =*/(int32_t)n_heads,
      /*.ne2 =*/(int32_t)n_q,
      /*.ne3 =*/(int32_t)n_seqs,
      /*.scale =*/scale,
      /*.max_bias =*/max_bias,
      /*.m0 =*/1.0f,
      /*.m1 =*/1.0f,
      /*.n_head_log2 =*/n_head_log2,
      /*.logit_softcap =*/logit_softcap,
  };

  char fn[160];
  snprintf(fn, sizeof(fn), "kernel_flash_attn_ext_f32_dk%lld_dv%lld", (long long)head_dim_k,
           (long long)head_dim_v);
  std::string key = std::string(fn) + "_mask=" + std::to_string(has_mask != 0) +
                    "_kvpad=" + std::to_string(has_kvpad) + "_bcm=" + std::to_string(bc_mask) +
                    "_ns10=" + std::to_string(args.ns10) + "_ns20=" + std::to_string(args.ns20) +
                    "_nsg=" + std::to_string(nsg);
  MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
  set_bool(cv, has_mask != 0, FC_FLASH_ATTN_EXT + 0);
  set_bool(cv, false, FC_FLASH_ATTN_EXT + 1);  // sinks
  set_bool(cv, false, FC_FLASH_ATTN_EXT + 2);  // ALiBi
  set_bool(cv, false, FC_FLASH_ATTN_EXT + 3);  // logit softcap
  set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);
  set_bool(cv, bc_mask, FC_FLASH_ATTN_EXT + 10);
  set_int(cv, args.ns10, FC_FLASH_ATTN_EXT + 20);
  set_int(cv, args.ns20, FC_FLASH_ATTN_EXT + 21);
  set_int(cv, nsg, FC_FLASH_ATTN_EXT + 22);
  id<MTLComputePipelineState> pso = pipeline(key, fn, cv);
  [cv release];
  if (pso == nil) {
    return 2;
  }
  if ((NSUInteger)(nsg * 32) > pso.maxTotalThreadsPerThreadgroup) {
    return 3;
  }

  // The block map, built from the mask: one byte per (query block, cache block) saying whether the
  // block is entirely masked out, so the attention kernel can skip it. The vector path has no such
  // step -- with one query row there is nothing to block over.
  id<MTLComputePipelineState> blk_pso = nil;
  ggml_metal_kargs_flash_attn_ext_blk blk_args = {};
  if (has_mask) {
    blk_args = ggml_metal_kargs_flash_attn_ext_blk{
        /*.ne01 =*/(int32_t)n_q,
        /*.ne30 =*/(int32_t)n_kv,
        /*.ne31 =*/ne31,
        /*.ne32 =*/ne32,
        /*.ne33 =*/ne33,
        /*.nb31 =*/nb31,
        /*.nb32 =*/nb32,
        /*.nb33 =*/nb33,
    };
    std::string blk_key = "kernel_flash_attn_ext_blk_nqptg=" + std::to_string(FA_NQPSG) +
                          "_ncpsg=" + std::to_string(FA_NCPSG);
    MTLFunctionConstantValues *cvb = [MTLFunctionConstantValues new];
    set_int(cvb, FA_NQPSG, FC_FLASH_ATTN_EXT_BLK + 24);
    set_int(cvb, FA_NCPSG, FC_FLASH_ATTN_EXT_BLK + 25);
    blk_pso = pipeline(blk_key, "kernel_flash_attn_ext_blk", cvb);
    [cvb release];
    if (blk_pso == nil) {
      return 2;
    }
  }

  id<MTLComputePipelineState> pad_pso = nil;
  ggml_metal_kargs_flash_attn_ext_pad pad_args = {};
  if (has_kvpad) {
    pad_args = ggml_metal_kargs_flash_attn_ext_pad{
        /*.ne11 =*/(int32_t)n_kv,
        /*.ne_12_2 =*/(int32_t)n_heads_kv,
        /*.ne_12_3 =*/(int32_t)n_seqs,
        /*.nb11 =*/nb11,
        /*.nb12 =*/nb12,
        /*.nb13 =*/nb13,
        /*.nb21 =*/nb21,
        /*.nb22 =*/nb22,
        /*.nb23 =*/nb23,
        /*.ne31 =*/ne31,
        /*.ne32 =*/ne32,
        /*.ne33 =*/ne33,
        /*.nb31 =*/nb31,
        /*.nb32 =*/nb32,
        /*.nb33 =*/nb33,
    };
    std::string pad_key = "kernel_flash_attn_ext_pad_mask=" + std::to_string(has_mask != 0) +
                          "_ncpsg=" + std::to_string(FA_NCPSG);
    MTLFunctionConstantValues *cvp = [MTLFunctionConstantValues new];
    set_bool(cvp, has_mask != 0, FC_FLASH_ATTN_EXT_PAD + 0);
    set_int(cvp, FA_NCPSG, FC_FLASH_ATTN_EXT_PAD + 25);
    pad_pso = pipeline(pad_key, "kernel_flash_attn_ext_pad", cvp);
    [cvp release];
    if (pad_pso == nil) {
      return 2;
    }
  }

  void *mask_buf = has_mask ? mask : q;
  const size_t mask_buf_off = has_mask ? mask_off : q_off;

  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

    if (pad_pso != nil) {
      [enc setComputePipelineState:pad_pso];
      [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
      [enc setBuffer:(__bridge id<MTLBuffer>)k offset:k_off atIndex:1];
      [enc setBuffer:(__bridge id<MTLBuffer>)v offset:v_off atIndex:2];
      [enc setBuffer:(__bridge id<MTLBuffer>)mask_buf offset:mask_buf_off atIndex:3];
      [enc setBuffer:(__bridge id<MTLBuffer>)pad offset:pad_off atIndex:4];
      const NSUInteger gy = (NSUInteger)(n_heads_kv > ne32 ? n_heads_kv : ne32);
      const NSUInteger gz = (NSUInteger)(n_seqs > ne33 ? n_seqs : ne33);
      [enc dispatchThreadgroups:MTLSizeMake(FA_NCPSG, gy, gz)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    if (blk_pso != nil) {
      [enc setComputePipelineState:blk_pso];
      [enc setBytes:&blk_args length:sizeof(blk_args) atIndex:0];
      [enc setBuffer:(__bridge id<MTLBuffer>)mask_buf offset:mask_buf_off atIndex:1];
      [enc setBuffer:(__bridge id<MTLBuffer>)blk offset:blk_off atIndex:2];
      const NSUInteger nblk1 = (NSUInteger)((n_q + FA_NQPSG - 1) / FA_NQPSG);
      const NSUInteger nblk0 = (NSUInteger)((n_kv + FA_NCPSG - 1) / FA_NCPSG);
      [enc dispatchThreadgroups:MTLSizeMake(nblk0, nblk1, (NSUInteger)(ne32 * ne33))
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    [enc setComputePipelineState:pso];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)q offset:q_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)k offset:k_off atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)v offset:v_off atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)mask_buf offset:mask_buf_off atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)q offset:q_off atIndex:5];  // sinks, unused
    [enc setBuffer:(__bridge id<MTLBuffer>)pad offset:pad_off atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)blk offset:blk_off atIndex:7];
    [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:dst_off atIndex:8];
    [enc setThreadgroupMemoryLength:smem atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((n_q + FA_NQPSG - 1) / FA_NQPSG),
                                          (NSUInteger)n_heads, (NSUInteger)n_seqs)
        threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];
  });
  return 0;
}

extern "C" int ggml_attn_metal_flash_attn(void *q, size_t q_off, void *k, size_t k_off, void *v,
                                          size_t v_off, void *mask, size_t mask_off, void *pad,
                                          size_t pad_off, void *tmp, size_t tmp_off, void *blk,
                                          size_t blk_off, void *dst, size_t dst_off, int64_t n_seqs,
                                          int64_t n_heads, int64_t n_heads_kv, int64_t n_q,
                                          int64_t n_kv, int64_t head_dim_k, int64_t head_dim_v,
                                          float scale, int has_mask) {
  if (!ggml_attn_metal_supports_flash_attn(n_q, head_dim_k, head_dim_v)) {
    return 1;
  }
  if (!fa_use_vec(n_q, head_dim_k)) {
    return fa_dispatch_tiled(q, q_off, k, k_off, v, v_off, mask, mask_off, pad, pad_off, blk,
                             blk_off, dst, dst_off, n_seqs, n_heads, n_heads_kv, n_q, n_kv,
                             head_dim_k, head_dim_v, scale, has_mask);
  }
  const bool has_kvpad = (n_kv % FA_VEC_NCPSG) != 0;
  const int nsg = fa_vec_nsg(n_kv);
  const int nwg = FA_VEC_NWG;
  const size_t smem = fa_vec_smem(head_dim_k, head_dim_v, nsg);

  // ggml strides in bytes for the contiguous layouts common.h documents, ne0 fastest.
  const uint64_t f32 = sizeof(float), f16 = 2;
  const uint64_t nb01 = f32 * head_dim_k, nb02 = nb01 * n_q, nb03 = nb02 * n_heads;
  const uint64_t nb11 = f32 * head_dim_k, nb12 = nb11 * n_kv, nb13 = nb12 * n_heads_kv;
  const uint64_t nb21 = f32 * head_dim_v, nb22 = nb21 * n_kv, nb23 = nb22 * n_heads_kv;
  const uint64_t nb31 = f16 * n_kv, nb32 = nb31 * n_q, nb33 = nb32;  // mask ne32 == 1
  const int32_t ne31 = (int32_t)n_q, ne32 = 1, ne33 = (int32_t)n_seqs;

  // No ALiBi and no logit softcap: both are off for the models this serves, and leaving them out
  // keeps the specialisation (and the argument block) honest about what has been tested.
  const float max_bias = 0.0f, logit_softcap = 0.0f;
  const int32_t n_head_log2 = 1 << (int)floorf(log2f((float)n_heads));

  ggml_metal_kargs_flash_attn_ext_vec args = {
      /*.ne01 =*/(int32_t)n_q,
      /*.ne02 =*/(int32_t)n_heads,
      /*.ne03 =*/(int32_t)n_seqs,
      /*.nb01 =*/nb01,
      /*.nb02 =*/nb02,
      /*.nb03 =*/nb03,
      /*.ne11 =*/(int32_t)n_kv,
      /*.ne_12_2 =*/(int32_t)n_heads_kv,
      /*.ne_12_3 =*/(int32_t)n_seqs,
      /*.ns10 =*/(int32_t)(nb11 / f32),
      /*.nb11 =*/nb11,
      /*.nb12 =*/nb12,
      /*.nb13 =*/nb13,
      /*.ns20 =*/(int32_t)(nb21 / f32),
      /*.nb21 =*/nb21,
      /*.nb22 =*/nb22,
      /*.nb23 =*/nb23,
      /*.ne31 =*/ne31,
      /*.ne32 =*/ne32,
      /*.ne33 =*/ne33,
      /*.nb31 =*/nb31,
      /*.nb32 =*/nb32,
      /*.nb33 =*/nb33,
      /*.ne1 =*/(int32_t)n_heads,
      /*.ne2 =*/(int32_t)n_q,
      /*.ne3 =*/(int32_t)n_seqs,
      /*.scale =*/scale,
      /*.max_bias =*/max_bias,
      /*.m0 =*/1.0f,
      /*.m1 =*/1.0f,
      /*.n_head_log2 =*/n_head_log2,
      /*.logit_softcap =*/logit_softcap,
  };

  char vec_fn[160];
  snprintf(vec_fn, sizeof(vec_fn), "kernel_flash_attn_ext_vec_f32_dk%lld_dv%lld",
           (long long)head_dim_k, (long long)head_dim_v);
  std::string vec_key = std::string(vec_fn) + "_mask=" + std::to_string(has_mask != 0) +
                        "_kvpad=" + std::to_string(has_kvpad) +
                        "_ns10=" + std::to_string(args.ns10) +
                        "_ns20=" + std::to_string(args.ns20) + "_nsg=" + std::to_string(nsg) +
                        "_nwg=" + std::to_string(nwg);

  MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
  set_bool(cv, has_mask != 0, FC_FLASH_ATTN_EXT_VEC + 0);
  set_bool(cv, false, FC_FLASH_ATTN_EXT_VEC + 1);  // sinks
  set_bool(cv, false, FC_FLASH_ATTN_EXT_VEC + 2);  // ALiBi
  set_bool(cv, false, FC_FLASH_ATTN_EXT_VEC + 3);  // logit softcap
  set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT_VEC + 4);
  set_int(cv, args.ns10, FC_FLASH_ATTN_EXT_VEC + 20);
  set_int(cv, args.ns20, FC_FLASH_ATTN_EXT_VEC + 21);
  set_int(cv, nsg, FC_FLASH_ATTN_EXT_VEC + 22);
  set_int(cv, nwg, FC_FLASH_ATTN_EXT_VEC + 23);
  id<MTLComputePipelineState> vec_pso = pipeline(vec_key, vec_fn, cv);
  [cv release];
  if (vec_pso == nil) {
    return 2;
  }
  if ((NSUInteger)(nsg * 32) > vec_pso.maxTotalThreadsPerThreadgroup) {
    return 3;
  }

  // The reduce pass combines the `nwg` partial results into dst.
  const int32_t nrows = (int32_t)(n_heads * n_q * n_seqs);
  std::string red_key = "kernel_flash_attn_ext_vec_reduce_dv=" + std::to_string(head_dim_v) +
                        "_nwg=" + std::to_string(nwg);
  MTLFunctionConstantValues *cvr = [MTLFunctionConstantValues new];
  set_int(cvr, (int32_t)head_dim_v, FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
  set_int(cvr, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);
  id<MTLComputePipelineState> red_pso =
      pipeline(red_key, "kernel_flash_attn_ext_vec_reduce", cvr);
  [cvr release];
  if (red_pso == nil) {
    return 2;
  }

  // The padding pass, only when the cache does not end on a simdgroup boundary.
  id<MTLComputePipelineState> pad_pso = nil;
  ggml_metal_kargs_flash_attn_ext_pad pad_args = {};
  if (has_kvpad) {
    pad_args = ggml_metal_kargs_flash_attn_ext_pad{
        /*.ne11 =*/(int32_t)n_kv,
        /*.ne_12_2 =*/(int32_t)n_heads_kv,
        /*.ne_12_3 =*/(int32_t)n_seqs,
        /*.nb11 =*/nb11,
        /*.nb12 =*/nb12,
        /*.nb13 =*/nb13,
        /*.nb21 =*/nb21,
        /*.nb22 =*/nb22,
        /*.nb23 =*/nb23,
        /*.ne31 =*/ne31,
        /*.ne32 =*/ne32,
        /*.ne33 =*/ne33,
        /*.nb31 =*/nb31,
        /*.nb32 =*/nb32,
        /*.nb33 =*/nb33,
    };
    std::string pad_key = "kernel_flash_attn_ext_pad_mask=" + std::to_string(has_mask != 0) +
                          "_ncpsg=" + std::to_string(FA_VEC_NCPSG);
    MTLFunctionConstantValues *cvp = [MTLFunctionConstantValues new];
    set_bool(cvp, has_mask != 0, FC_FLASH_ATTN_EXT_PAD + 0);
    // +25, not +20: the pad kernel reads its ncpsg from FC_FLASH_ATTN_EXT_PAD + 25
    set_int(cvp, FA_VEC_NCPSG, FC_FLASH_ATTN_EXT_PAD + 25);
    pad_pso = pipeline(pad_key, "kernel_flash_attn_ext_pad", cvp);
    [cvp release];
    if (pad_pso == nil) {
      return 2;
    }
  }

  // A null mask still has to be bound to something; upstream binds q.
  void *mask_buf = has_mask ? mask : q;
  const size_t mask_buf_off = has_mask ? mask_off : q_off;

  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = stream->commandEncoder();

    if (pad_pso != nil) {
      [enc setComputePipelineState:pad_pso];
      [enc setBytes:&pad_args length:sizeof(pad_args) atIndex:0];
      [enc setBuffer:(__bridge id<MTLBuffer>)k offset:k_off atIndex:1];
      [enc setBuffer:(__bridge id<MTLBuffer>)v offset:v_off atIndex:2];
      [enc setBuffer:(__bridge id<MTLBuffer>)mask_buf offset:mask_buf_off atIndex:3];
      [enc setBuffer:(__bridge id<MTLBuffer>)pad offset:pad_off atIndex:4];
      const NSUInteger gy = (NSUInteger)(n_heads_kv > ne32 ? n_heads_kv : ne32);
      const NSUInteger gz = (NSUInteger)(n_seqs > ne33 ? n_seqs : ne33);
      [enc dispatchThreadgroups:MTLSizeMake(FA_VEC_NCPSG, gy, gz)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
      // Upstream resets its concurrency tracking here; the equivalent on a shared encoder is a
      // barrier, because torch may have opened it for concurrent dispatch.
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }

    [enc setComputePipelineState:vec_pso];
    [enc setBytes:&args length:sizeof(args) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)q offset:q_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)k offset:k_off atIndex:2];
    [enc setBuffer:(__bridge id<MTLBuffer>)v offset:v_off atIndex:3];
    [enc setBuffer:(__bridge id<MTLBuffer>)mask_buf offset:mask_buf_off atIndex:4];
    [enc setBuffer:(__bridge id<MTLBuffer>)q offset:q_off atIndex:5];  // sinks, unused
    [enc setBuffer:(__bridge id<MTLBuffer>)pad offset:pad_off atIndex:6];
    [enc setBuffer:(__bridge id<MTLBuffer>)tmp offset:tmp_off atIndex:7];
    [enc setThreadgroupMemoryLength:smem atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((n_q + FA_VEC_NQPSG - 1) / FA_VEC_NQPSG,
                                          (n_heads + FA_VEC_NHPTG - 1) / FA_VEC_NHPTG,
                                          n_seqs * nwg)
        threadsPerThreadgroup:MTLSizeMake(32, nsg, 1)];

    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    ggml_metal_kargs_flash_attn_ext_vec_reduce red_args = {nrows};
    [enc setComputePipelineState:red_pso];
    [enc setBytes:&red_args length:sizeof(red_args) atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)tmp offset:tmp_off atIndex:1];
    [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:dst_off atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * nwg, 1, 1)];
  });
  return 0;
}
