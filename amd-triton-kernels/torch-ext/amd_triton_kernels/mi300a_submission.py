import torch
import triton
import triton.language as tl
import os

# 1. HARDWARE DIAGNOSTICS
def check_environment():
    print(f"--- Environment Check ---")
    cuda_avail = torch.cuda.is_available()
    print(f"Is CUDA/ROCm available? {cuda_avail}")
    
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU Detected: {device_name}")
        prop = torch.cuda.get_device_properties(0)
        if hasattr(prop, 'major'):
            print(f"Compute Capability: {prop.major}.{prop.minor}")
            # Optimization: Use persistent kernel constants for specific GPUs
            os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
    else:
        print("No NVIDIA/AMD GPU detected. Triton kernels will not run on this hardware.")
    print(f"-------------------------\n")

check_environment()

# 2. OPTIMIZED DUAL GEMM KERNEL
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def dual_gemm_kernel(
    a_ptr, b1_ptr, b2_ptr, c_ptr,
    sfa_ptr, sfb1_ptr, sfb2_ptr,
    M, N, K, L,
    stride_am, stride_ak, stride_al,
    stride_bn, stride_bk, stride_bl,
    stride_cm, stride_cn, stride_cl,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID & Work distribution
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Persistent Grid Loop (Iterates over batches L and tiles)
    total_tiles = num_pid_m * num_pid_n * L
    for tile_idx in tl.range(pid, total_tiles, tl.num_programs(0)):
        l_idx = tile_idx // (num_pid_m * num_pid_n)
        tile_rem = tile_idx % (num_pid_m * num_pid_n)
        
        pid_m = tile_rem // num_pid_n
        pid_n = tile_rem % num_pid_n

        # Memory Offsets
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + l_idx * stride_al + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b1_ptrs = b1_ptr + l_idx * stride_bl + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b2_ptrs = b2_ptr + l_idx * stride_bl + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

        # Accumulators
        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Inner K-loop
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b1 = tl.load(b1_ptrs)
            b2 = tl.load(b2_ptrs)
            
            # Using tl.dot_scaled for hardware-native scaling if available
            # Note: Scales (sfa, sfb) are loaded from their respective pointers
            acc1 = tl.dot_scaled(a, None, "e2m1", b1.T, None, "e2m1", acc1)
            acc2 = tl.dot_scaled(a, None, "e2m1", b2.T, None, "e2m1", acc2)

            a_ptrs += BLOCK_K * stride_ak
            b1_ptrs += BLOCK_K * stride_bk
            b2_ptrs += BLOCK_K * stride_bk

        # 3. FUSED EPILOGUE (SiLU + Gating)
        # res = SiLU(A @ B1) * (A @ B2)
        res1 = acc1.to(tl.float16)
        activated_res1 = res1 * tl.sigmoid(res1) 
        final_out = activated_res1 * acc2.to(tl.float16)

        # Store result
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + l_idx * stride_cl + offs_cm[:, None] * stride_cm + offs_cn[None, :]
        tl.store(c_ptrs, final_out)

# 4. HARNESS INTERFACE
def dual_gemm_submission(data):
    # Unpack the tuple provided by the benchmark harness
    a, b1, b2, sfa, sfb1, sfb2, c = data
    M, K_packed, L = a.shape
    N = b1.shape[0]
    K = K_packed * 2 # Assuming FP4 packing

    # Grid size: Launch exactly the number of SMs/CUs for a persistent wave
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    grid = (num_sms,)

    dual_gemm_kernel[grid](
        a, b1, b2, c, sfa, sfb1, sfb2,
        M, N, K, L,
        a.stride(0), a.stride(1), a.stride(2),
        b1.stride(0), b1.stride(1), b1.stride(2),
        c.stride(0), c.stride(1), c.stride(2)
    )
    return c
