#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <torch/torch.h>

// Include the auto-generated header with embedded metallib
#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Helper function to get dtype string
static std::string getDtypeString(torch::ScalarType dtype) {
  switch (dtype) {
  case torch::kFloat:
    return "float32";
  case torch::kHalf:
    return "float16";
  case torch::kBFloat16:
    return "bfloat16";
  default:
    TORCH_CHECK(false, "Unsupported dtype for SDPA: ", dtype);
  }
}

// Helper function to get dtype string for kernel names
static std::string getKernelDtypeString(torch::ScalarType dtype) {
  switch (dtype) {
  case torch::kFloat:
    return "float32";  // Match the instantiation names
  case torch::kHalf:
    return "float16";
  case torch::kBFloat16:
    return "bfloat16";
  default:
    TORCH_CHECK(false, "Unsupported dtype for SDPA: ", dtype);
  }
}


// Parameters structure matching Flash Attention's AttnParams
struct AttnParams {
  int32_t B;              // batch size
  int32_t H;              // number of heads
  int32_t D;              // head dimension
  int32_t qL;             // query sequence length (per sequence)
  int32_t kL;             // key sequence length (per sequence)
  int32_t gqa_factor;     // grouped query attention factor
  float scale;            // attention scale
  float softcapping;      // softcapping value (1.0 for no softcapping)
  int32_t NQ;             // number of query blocks
  int32_t NK;             // number of key blocks
  int32_t NQ_aligned;     // aligned query blocks
  int32_t NK_aligned;     // aligned key blocks
  int32_t qL_rem;         // remainder query length
  int32_t kL_rem;         // remainder key length
  int32_t qL_off;         // query offset
  int64_t Q_strides[3];   // query tensor strides
  int64_t K_strides[3];   // key tensor strides
  int64_t V_strides[3];   // value tensor strides
  int64_t O_strides[3];   // output tensor strides
  
  // Flash Attention variable-length support
  int32_t total_q_tokens; // Total number of query tokens
  int32_t total_k_tokens; // Total number of key/value tokens
  int32_t max_seqlen_q;   // Maximum query sequence length
  int32_t max_seqlen_k;   // Maximum key/value sequence length
};

// Forward declarations for kernel implementations
void call_flash_attention_varlen(
    id<MTLDevice> device,
    id<MTLCommandBuffer> cmdBuf,
    id<MTLLibrary> lib,
    torch::Tensor &out,
    torch::Tensor &query,
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &cu_seqlens_q,
    torch::Tensor &cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    bool do_causal,
    double scale,
    double softcapping);


void flash_attention_varlen(
    torch::Tensor &out,           // [total_q_tokens, num_heads, head_size]
    torch::Tensor &query,         // [total_q_tokens, num_heads, head_size]
    torch::Tensor &key,           // [total_k_tokens, num_heads_kv, head_size]
    torch::Tensor &value,         // [total_k_tokens, num_heads_kv, head_size]
    torch::Tensor &cu_seqlens_q,  // [batch_size + 1]
    torch::Tensor &cu_seqlens_k,  // [batch_size + 1]
    int64_t max_seqlen_q,         // Maximum query sequence length
    int64_t max_seqlen_k,         // Maximum key sequence length
    bool do_causal,               // Whether to use causal mask
    double scale,                 // Attention scale
    double softcapping) {         // Softcapping value

  try {
    // Get device and stream
    id<MTLDevice> device = at::mps::MPSDevice::getInstance()->device();
    at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
    TORCH_CHECK(stream, "Failed to get current MPS stream");

  // Get dimensions from Flash Attention format
  int64_t total_q_tokens = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t head_dim = query.size(2);
  int64_t num_heads_kv = key.size(1);
  int64_t batch_size = cu_seqlens_q.size(0) - 1;  // cu_seqlens has batch_size + 1 elements
  
  // Check if we support this head dimension
  std::vector<int> supported_head_dims = {32, 64, 72, 80, 96, 128, 256};
  bool supported_head_dim = std::find(supported_head_dims.begin(), 
                                      supported_head_dims.end(), 
                                      head_dim) != supported_head_dims.end();
  
  TORCH_CHECK(supported_head_dim, "Head dimension ", head_dim, " is not supported");
  TORCH_CHECK(cu_seqlens_q.size(0) == cu_seqlens_k.size(0), 
              "cu_seqlens_q and cu_seqlens_k must have the same size");

  // Load Metal library
  static id<MTLLibrary> lib = nil;
  if (!lib) {
    NSError *error = nil;
    lib = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(lib,
                "Failed to create Metal library from embedded data: ",
                error.localizedDescription.UTF8String);
  }

  // Get command buffer
  id<MTLCommandBuffer> cmdBuf = stream->commandBuffer();
  TORCH_CHECK(cmdBuf, "Failed to get MPS command buffer");

  // For variable-length Flash Attention, always use the full attention kernel
  
  // Call the Flash Attention kernel
  call_flash_attention_varlen(device, cmdBuf, lib, out, query, key, value, 
                              cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                              do_causal, scale, softcapping);
  } catch (const std::exception& e) {
    throw;
  } catch (...) {
    throw;
  }
}

// Implementation of Flash Attention variable-length kernel
void call_flash_attention_varlen(
    id<MTLDevice> device,
    id<MTLCommandBuffer> cmdBuf,
    id<MTLLibrary> lib,
    torch::Tensor &out,
    torch::Tensor &query,
    torch::Tensor &key,
    torch::Tensor &value,
    torch::Tensor &cu_seqlens_q,
    torch::Tensor &cu_seqlens_k,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    bool do_causal,
    double scale,
    double softcapping) {

  // Get dimensions
  int64_t total_q_tokens = query.size(0);
  int64_t num_heads = query.size(1);
  int64_t head_dim = query.size(2);
  int64_t num_heads_kv = key.size(1);
  int64_t batch_size = cu_seqlens_q.size(0) - 1;

  // Grouped Query Attention factor
  int32_t gqa_factor = num_heads / num_heads_kv;

  // Block sizes based on head dimension
  const int BQ = (head_dim == 256) ? 16 : 32;  // Use BQ=16 for head_dim=256
  const int bk = (head_dim == 256) ? 8 : ((head_dim >= 128) ? 16 : 32);  // Use bk=8 for head_dim=256
  const int WM = (head_dim == 256) ? 2 : 4;  // Use WM=2 for head_dim=256
  const int WN = 1;

  // Setup parameters
  AttnParams params = {}; // Zero-initialize all fields
  params.B = batch_size;
  params.H = num_heads;
  params.D = head_dim;
  params.gqa_factor = gqa_factor;
  params.scale = static_cast<float>(scale);
  params.softcapping = static_cast<float>(softcapping);
  params.total_q_tokens = total_q_tokens;
  params.total_k_tokens = key.size(0);
  params.max_seqlen_q = max_seqlen_q;
  params.max_seqlen_k = max_seqlen_k;
  
  // Initialize fields that might be checked but aren't used in Flash Attention
  params.qL = 0;  // Not used in variable-length attention
  params.kL = 0;  // Not used in variable-length attention
  params.NQ = 0;  // Not used
  params.NK = 0;  // Not used
  params.NQ_aligned = 0;
  params.NK_aligned = 0;
  params.qL_rem = 0;
  params.kL_rem = 0;
  params.qL_off = 0;
  
  // Strides are not used for packed tensors (contiguous)
  params.Q_strides[0] = 0;
  params.Q_strides[1] = 0;
  params.Q_strides[2] = 0;
  params.K_strides[0] = 0;
  params.K_strides[1] = 0;
  params.K_strides[2] = 0;
  params.V_strides[0] = 0;
  params.V_strides[1] = 0;
  params.V_strides[2] = 0;
  params.O_strides[0] = 0;
  params.O_strides[1] = 0;
  params.O_strides[2] = 0;
  
  // For variable-length attention, we'll process each sequence separately
  // The kernel will handle the cu_seqlens internally

  bool has_mask = false;  // Masks are not supported in Flash Attention

  // Setup function constants
  MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
  [constants setConstantValue:&has_mask type:MTLDataTypeBool atIndex:300];
  [constants setConstantValue:&do_causal type:MTLDataTypeBool atIndex:301];

  // Construct kernel name based on data type and head dimension
  std::string kernel_name = "steel_attention_";
  kernel_name += getKernelDtypeString(query.scalar_type());
  kernel_name += "_bq" + std::to_string(BQ);
  kernel_name += "_bk" + std::to_string(bk);
  kernel_name += "_bd" + std::to_string(head_dim);
  kernel_name += "_wm" + std::to_string(WM) + "_wn" + std::to_string(WN);
  kernel_name += "_maskbool_";  // Always use bool for mask type (no masks supported)

  // Get kernel function
  NSError *error = nil;
  id<MTLFunction> function = [lib newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]
                                      constantValues:constants
                                              error:&error];
  TORCH_CHECK(function, "Failed to get Metal function: ", kernel_name,
              " Error: ", error ? error.localizedDescription.UTF8String : "unknown");

  // Create compute pipeline
  id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
  TORCH_CHECK(pipeline, "Failed to create compute pipeline: ",
              error ? error.localizedDescription.UTF8String : "unknown");

  // Setup command encoder with dispatch sync
  at::mps::MPSStream *stream = at::mps::getCurrentMPSStream();
  dispatch_queue_t q = stream->queue();
  dispatch_sync(q, ^{
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    TORCH_CHECK(encoder, "Failed to create compute encoder");

    [encoder setComputePipelineState:pipeline];
    
    // Set buffers
    int buffer_idx = 0;
    
    // Query buffer - index 0
    [encoder setBuffer:getMTLBufferStorage(query) 
              offset:query.storage_offset() * query.element_size() 
              atIndex:buffer_idx++];
    
    // Key buffer - index 1
    [encoder setBuffer:getMTLBufferStorage(key) 
              offset:key.storage_offset() * key.element_size() 
              atIndex:buffer_idx++];
    
    // Value buffer - index 2
    [encoder setBuffer:getMTLBufferStorage(value) 
              offset:value.storage_offset() * value.element_size() 
              atIndex:buffer_idx++];
    
    // Output buffer - index 3
    [encoder setBuffer:getMTLBufferStorage(out) 
              offset:out.storage_offset() * out.element_size() 
              atIndex:buffer_idx++];
    
    // Parameters - index 4
    [encoder setBytes:&params length:sizeof(AttnParams) atIndex:buffer_idx++];
    
    // Skip mask parameters - indices 5 and 6 (masks not supported)
    buffer_idx += 2;
    
    // Set cu_seqlens buffers - indices 7 and 8
    [encoder setBuffer:getMTLBufferStorage(cu_seqlens_q) 
              offset:cu_seqlens_q.storage_offset() * cu_seqlens_q.element_size() 
              atIndex:7];
    [encoder setBuffer:getMTLBufferStorage(cu_seqlens_k) 
              offset:cu_seqlens_k.storage_offset() * cu_seqlens_k.element_size() 
              atIndex:8];

    // Calculate grid dimensions
    // We need to process each sequence independently
    int64_t max_blocks_q = (max_seqlen_q + BQ - 1) / BQ;
    
    MTLSize gridSize = MTLSizeMake(max_blocks_q, num_heads, batch_size);
    MTLSize threadgroupSize = MTLSizeMake(32, WM, WN);
    
    [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    stream->synchronize(at::mps::SyncType::COMMIT);
  });
}