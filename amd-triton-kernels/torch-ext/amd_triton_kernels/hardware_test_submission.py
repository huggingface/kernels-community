import torch
import triton
import triton.language as tl
import os

# 1. HARDWARE DIAGNOSTICS & OS PREP
def check_environment():
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        # Optimization: Force kernel arguments to device to save PCIe latency
        os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
        # Disable compiler cache for benchmarking clean runs
        os.environ["TRITON_CACHE_DIR"] = ""

check_environment()

# 2. THE SOL KERNEL
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def dual_gemm_hardware_kernel(
    a_ptr, b1_ptr, b2_ptr, c_ptr,
    sfa_ptr, sfb1_ptr, sfb2_ptr,
    M, N, K, L,
    stride_am, stride_ak, stride_al,
    stride_bn, stride_bk, stride_bl,
    stride_cm, stride_cn, stride_cl,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Persistent Grid logic
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n * L
    
    for tile_idx in tl.range(pid, total_tiles, tl.num_programs(0)):
        l_idx = tile_idx // (num_pid_m * num_pid_n)
        tile_rem = tile_idx % (num_pid_m * num_pid_n)
        
        # Swizzle for L2 Locality
        pid_m, pid_n = tl.swizzle2d(tile_rem, num_pid_m, num_pid_n, GROUP_SIZE_M)

        # Base offsets
        rm = pid_m * BLOCK_M
        rn = pid_n * BLOCK_N
        
        # Ranges
        offs_m = rm + tl.arange(0, BLOCK_M)
        offs_n = rn + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Memory Pointers
        a_ptrs = a_ptr + l_idx * stride_al + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b1_ptrs = b1_ptr + l_idx * stride_bl + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        b2_ptrs = b2_ptr + l_idx * stride_bl + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

        # Accumulators
        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Scale Factor Load Offsets (Assuming OCP 1 scale per 16 elements)
        # sfa: (M, K/16, L), sfb: (N, K/16, L)
        sfa_base = sfa_ptr + l_idx * (M * (K // 16)) + (offs_m[:, None] * (K // 16))
        sfb1_base = sfb1_ptr + l_idx * (N * (K // 16)) + (offs_n[None, :] * (K // 16))
        sfb2_base = sfb2_ptr + l_idx * (N * (K // 16)) + (offs_n[None, :] * (K // 16))

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            # 1. Load Data
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K), other=0.0)
            b1 = tl.load(b1_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_K), other=0.0)
            b2 = tl.load(b2_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K - k * BLOCK_K), other=0.0)

            # 2. Load Scales for current K-block
            # Blackwell uses a 32x4 atom, but for pointers, we load the K-slice
            curr_sfa = tl.load(sfa_base + (k * (BLOCK_K // 16)), mask=(offs_m[:, None] < M), other=1.0)
            curr_sfb1 = tl.load(sfb1_base + (k * (BLOCK_K // 16)), mask=(offs_n[None, :] < N), other=1.0)
            curr_sfb2 = tl.load(sfb2_base + (k * (BLOCK_K // 16)), mask=(offs_n[None, :] < N), other=1.0)

            # 3. Hardware DOT Scaled
            acc1 = tl.dot_scaled(a, curr_sfa, "e2m1", b1, curr_sfb1, "e2m1", acc1)
            acc2 = tl.dot_scaled(a, curr_sfa, "e2m1", b2, curr_sfb2, "e2m1", acc2)

            # Advance K
            a_ptrs += BLOCK_K * stride_ak
            b1_ptrs += BLOCK_K * stride_bk
            b2_ptrs += BLOCK_K * stride_bk

        # 4. Epilogue (Fused SiLU + Gating)
        res1 = acc1.to(tl.float16)
        activated = res1 * tl.sigmoid(res1)
        final_out = activated * acc2.to(tl.float16)

        # 5. Masked Store
        c_ptrs = c_ptr + l_idx * stride_cl + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, final_out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 3. SUBMISSION INTERFACE
def dual_gemm_submission(data):
    a, b1, b2, sfa, sfb1, sfb2, c = data
    M, K_packed, L = a.shape
    N = b1.shape[0]
    K = K_packed * 2 # FP4 2nd element expansion

    # Saturate Device (148 for B200, 304 for MI300X)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    grid = (num_sms,)

    dual_gemm_hardware_kernel[grid](
        a, b1, b2, c, sfa, sfb1, sfb2,
        M, N, K, L,
        a.stride(0), a.stride(1), a.stride(2),
        b1.stride(0), b1.stride(1), b1.stride(2),
        c.stride(0), c.stride(1), c.stride(2)
    )
    return c
