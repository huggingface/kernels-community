// Objective-C++ binding for sdpa_int4 Metal kernel
// Bridges PyTorch MPS tensors to the Metal compute pipeline

#include <torch/torch.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <unordered_map>

// Metal library embedded by kernel-builder
#include EMBEDDED_METALLIB_HEADER

// Forward declarations
torch::Tensor sdpa_int4_splitk(
    torch::Tensor queries, torch::Tensor k_quant, torch::Tensor k_scales,
    torch::Tensor k_biases, torch::Tensor v_quant, torch::Tensor v_scales,
    torch::Tensor v_biases, int64_t gqa_factor, int64_t N, double scale,
    int64_t sliding_window, int64_t num_splits);

// Cache pipeline states to avoid redundant Metal library/pipeline creation
static std::unordered_map<int, id<MTLComputePipelineState>> pso_cache;

static id<MTLComputePipelineState> getPipelineState(
    id<MTLDevice> device,
    int head_dim
) {
    auto it = pso_cache.find(head_dim);
    if (it != pso_cache.end()) return it->second;

    NSError* error = nil;
    id<MTLLibrary> library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
    TORCH_CHECK(library, "Failed to create Metal library: ",
                error ? [[error localizedDescription] UTF8String] : "unknown");

    NSString* kernelName;
    switch (head_dim) {
        case 128: kernelName = @"sdpa_int4_128"; break;
        case 256: kernelName = @"sdpa_int4_256"; break;
        case 512: kernelName = @"sdpa_int4_512"; break;
        default:
            TORCH_CHECK(false, "Unsupported head_dim: ", head_dim,
                       ". Must be 128, 256, or 512.");
    }

    id<MTLFunction> function = [library newFunctionWithName:kernelName];
    TORCH_CHECK(function, "Failed to find kernel: ", [kernelName UTF8String]);

    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:function error:&error];
    TORCH_CHECK(pso, "Failed to create pipeline state: ",
                error ? [[error localizedDescription] UTF8String] : "unknown");

    pso_cache[head_dim] = pso;
    return pso;
}

torch::Tensor sdpa_int4(
    torch::Tensor queries,      // (num_heads, D) float32
    torch::Tensor k_quant,      // (num_kv_heads, N, D/8) uint32
    torch::Tensor k_scales,     // (num_kv_heads, N, D/64) float32
    torch::Tensor k_biases,     // (num_kv_heads, N, D/64) float32
    torch::Tensor v_quant,      // (num_kv_heads, N, D/8) uint32
    torch::Tensor v_scales,     // (num_kv_heads, N, D/64) float32
    torch::Tensor v_biases,     // (num_kv_heads, N, D/64) float32
    int64_t gqa_factor,
    int64_t N,
    double scale,
    int64_t sliding_window
) {
    TORCH_CHECK(queries.device().is_mps(), "queries must be on MPS device");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D (num_heads, D)");

    int total_heads = queries.size(0);  // May be batch * num_heads for batched decode
    int D = queries.size(1);
    int n = static_cast<int>(N);
    int num_kv_heads = k_quant.size(0);
    int num_heads = num_kv_heads * static_cast<int>(gqa_factor);  // Per-batch num_heads

    // Adaptive split-K: use when we need more GPU parallelism
    // Split-K doubles threadgroups but only helps when heads < GPU cores.
    int sw = static_cast<int>(sliding_window);
    int effective_n = (sw > 0 && sw < n) ? sw : n;
    if (effective_n >= 2048 && total_heads <= 16) {
        return sdpa_int4_splitk(queries, k_quant, k_scales, k_biases,
                                v_quant, v_scales, v_biases,
                                gqa_factor, N, scale, sliding_window, 2);
    }

    // Regular single-dispatch path for smaller N
    auto output = torch::empty({total_heads, D}, queries.options());

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto device = stream->device();

        auto pso = getPipelineState(device, D);

        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        [encoder setComputePipelineState:pso];

        using at::native::mps::getMTLBufferStorage;
        [encoder setBuffer:getMTLBufferStorage(queries)  offset:queries.storage_offset() * sizeof(float)     atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(k_quant)  offset:k_quant.storage_offset() * sizeof(uint32_t)  atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(k_scales) offset:k_scales.storage_offset() * sizeof(float)    atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(k_biases) offset:k_biases.storage_offset() * sizeof(float)    atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(v_quant)  offset:v_quant.storage_offset() * sizeof(uint32_t)  atIndex:4];
        [encoder setBuffer:getMTLBufferStorage(v_scales) offset:v_scales.storage_offset() * sizeof(float)    atIndex:5];
        [encoder setBuffer:getMTLBufferStorage(v_biases) offset:v_biases.storage_offset() * sizeof(float)    atIndex:6];
        [encoder setBuffer:getMTLBufferStorage(output)   offset:0                                            atIndex:7];

        int gqa = static_cast<int>(gqa_factor);
        float s = static_cast<float>(scale);
        int sw = static_cast<int>(sliding_window);
        int nh = num_heads;
        [encoder setBytes:&gqa length:sizeof(int)   atIndex:8];
        [encoder setBytes:&n   length:sizeof(int)   atIndex:9];
        [encoder setBytes:&s   length:sizeof(float) atIndex:10];
        [encoder setBytes:&sw  length:sizeof(int)   atIndex:11];
        [encoder setBytes:&nh  length:sizeof(int)   atIndex:12];

        int BN = 16;
        MTLSize gridSize = MTLSizeMake(total_heads, 1, 1);
        MTLSize groupSize = MTLSizeMake(BN * 32, 1, 1);
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:groupSize];

        stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
    }

    return output;
}

// Split-K variant for improved GPU utilization
torch::Tensor sdpa_int4_splitk(
    torch::Tensor queries, torch::Tensor k_quant, torch::Tensor k_scales,
    torch::Tensor k_biases, torch::Tensor v_quant, torch::Tensor v_scales,
    torch::Tensor v_biases, int64_t gqa_factor, int64_t N, double scale,
    int64_t sliding_window, int64_t num_splits
) {
    TORCH_CHECK(queries.device().is_mps(), "queries must be on MPS device");
    TORCH_CHECK(queries.dim() == 2, "queries must be 2D (num_heads, D)");

    int total_heads = queries.size(0);
    int D = queries.size(1);
    int ns = static_cast<int>(num_splits);
    int num_kv_heads = k_quant.size(0);
    int num_heads = num_kv_heads * static_cast<int>(gqa_factor);

    // Persistent scratch buffers
    static torch::Tensor s_partial_out, s_partial_max, s_partial_sum;
    static int s_nh = 0, s_ns = 0, s_D = 0;

    if (total_heads != s_nh || ns != s_ns || D != s_D) {
        s_partial_out = torch::empty({total_heads, ns, D}, queries.options());
        s_partial_max = torch::empty({total_heads, ns}, queries.options());
        s_partial_sum = torch::empty({total_heads, ns}, queries.options());
        s_nh = total_heads; s_ns = ns; s_D = D;
    }

    // No initialization needed — kernel writes all elements
    auto& partial_out = s_partial_out;
    auto& partial_max = s_partial_max;
    auto& partial_sum = s_partial_sum;
    auto output = torch::empty({total_heads, D}, queries.options());

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        auto device = stream->device();

        // Get cached split-K pipeline states
        static std::unordered_map<int, std::pair<id<MTLComputePipelineState>, id<MTLComputePipelineState>>> splitk_cache;
        auto sk_it = splitk_cache.find(D);
        id<MTLComputePipelineState> splitk_pso, reduce_pso;
        if (sk_it != splitk_cache.end()) {
            splitk_pso = sk_it->second.first;
            reduce_pso = sk_it->second.second;
        } else {
            NSError* error = nil;
            id<MTLLibrary> library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(device, &error);
            TORCH_CHECK(library, "Failed to create Metal library");

            NSString* splitk_name;
            NSString* reduce_name;
            switch (D) {
                case 128: splitk_name = @"sdpa_int4_splitk_128"; reduce_name = @"sdpa_int4_reduce_128"; break;
                case 256: splitk_name = @"sdpa_int4_splitk_256"; reduce_name = @"sdpa_int4_reduce_256"; break;
                case 512: splitk_name = @"sdpa_int4_splitk_512"; reduce_name = @"sdpa_int4_reduce_512"; break;
                default:
                    TORCH_CHECK(false, "Unsupported head_dim for split-K: ", D);
            }

            id<MTLFunction> splitk_fn = [library newFunctionWithName:splitk_name];
            TORCH_CHECK(splitk_fn, "Failed to find splitk kernel");
            splitk_pso = [device newComputePipelineStateWithFunction:splitk_fn error:&error];
            TORCH_CHECK(splitk_pso, "Failed to create splitk pipeline");

            id<MTLFunction> reduce_fn = [library newFunctionWithName:reduce_name];
            TORCH_CHECK(reduce_fn, "Failed to find reduce kernel");
            reduce_pso = [device newComputePipelineStateWithFunction:reduce_fn error:&error];
            TORCH_CHECK(reduce_pso, "Failed to create reduce pipeline");

            splitk_cache[D] = {splitk_pso, reduce_pso};
        }

        // Use stream's shared encoder — supports multiple dispatches
        // Both split-K and reduction kernels encode on the same encoder
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        // --- Kernel 1: Split-K main computation ---
        [encoder setComputePipelineState:splitk_pso];

        using at::native::mps::getMTLBufferStorage;
        [encoder setBuffer:getMTLBufferStorage(queries)      offset:queries.storage_offset() * sizeof(float)     atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(k_quant)      offset:k_quant.storage_offset() * sizeof(uint32_t)  atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(k_scales)     offset:k_scales.storage_offset() * sizeof(float)    atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(k_biases)     offset:k_biases.storage_offset() * sizeof(float)    atIndex:3];
        [encoder setBuffer:getMTLBufferStorage(v_quant)      offset:v_quant.storage_offset() * sizeof(uint32_t)  atIndex:4];
        [encoder setBuffer:getMTLBufferStorage(v_scales)     offset:v_scales.storage_offset() * sizeof(float)    atIndex:5];
        [encoder setBuffer:getMTLBufferStorage(v_biases)     offset:v_biases.storage_offset() * sizeof(float)    atIndex:6];
        [encoder setBuffer:getMTLBufferStorage(partial_out)  offset:0 atIndex:7];
        [encoder setBuffer:getMTLBufferStorage(partial_max)  offset:0 atIndex:8];
        [encoder setBuffer:getMTLBufferStorage(partial_sum)  offset:0 atIndex:9];

        int gqa = static_cast<int>(gqa_factor);
        int n = static_cast<int>(N);
        float s = static_cast<float>(scale);
        int sw = static_cast<int>(sliding_window);
        int nh = num_heads;
        [encoder setBytes:&gqa length:sizeof(int)   atIndex:10];
        [encoder setBytes:&n   length:sizeof(int)   atIndex:11];
        [encoder setBytes:&s   length:sizeof(float) atIndex:12];
        [encoder setBytes:&sw  length:sizeof(int)   atIndex:13];
        [encoder setBytes:&ns  length:sizeof(int)   atIndex:14];
        [encoder setBytes:&nh  length:sizeof(int)   atIndex:15];

        int BN = 16;
        MTLSize splitk_grid = MTLSizeMake(total_heads, ns, 1);
        MTLSize splitk_group = MTLSizeMake(BN * 32, 1, 1);
        [encoder dispatchThreadgroups:splitk_grid threadsPerThreadgroup:splitk_group];

        // --- Kernel 2: Reduction ---
        [encoder setComputePipelineState:reduce_pso];
        [encoder setBuffer:getMTLBufferStorage(partial_out) offset:0 atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(partial_max) offset:0 atIndex:1];
        [encoder setBuffer:getMTLBufferStorage(partial_sum) offset:0 atIndex:2];
        [encoder setBuffer:getMTLBufferStorage(output)      offset:0 atIndex:3];
        [encoder setBytes:&ns length:sizeof(int) atIndex:4];

        MTLSize reduce_grid = MTLSizeMake(total_heads, 1, 1);
        MTLSize reduce_group = MTLSizeMake(32, 1, 1);  // Single simdgroup per head
        [encoder dispatchThreadgroups:reduce_grid threadsPerThreadgroup:reduce_group];

        // End kernel coalescing to ensure split-K and reduction execute sequentially
        stream->endKernelCoalescing();
        stream->synchronize(at::mps::SyncType::COMMIT_AND_CONTINUE);
    }

    return output;
}

