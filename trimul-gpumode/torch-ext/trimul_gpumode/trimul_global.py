# from utils import make_match_reference, DisableCuDNNTF32
from .task import input_t, output_t

import torch
from torch import nn, einsum
import math
import os
import requests

import triton
import triton.language as tl

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

# Set allocator for TMA descriptors (required for on-device TMA)
def alloc_fn(size: int, alignment: int, stream=None):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)

# os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
# os.environ['MLIR_ENABLE_DIAGNOSTICS'] = 'warnings,remarks'

# Reference code in PyTorch
class TriMul(nn.Module):
    # Based on https://github.com/lucidrains/triangle-multiplicative-module/blob/main/triangle_multiplicative_module/triangle_multiplicative_module.py
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]

        Returns:
            output: [bs, seq_len, seq_len, dim]
        """
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        # This einsum is the same as the following:
        # out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)
        
        # # Compute using nested loops
        # for b in range(batch_size):
        #     for i in range(seq_len):
        #         for j in range(seq_len):
        #             # Compute each output element
        #             for k in range(seq_len):
        #                 out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

@triton.jit
def triton_sigmoid(x):
    """
    Compute sigmoid function: 1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + tl.exp(-x))

def two_mm_kernel_configs_wrapper():
    if torch.cuda.get_device_capability() == (12, 0):
        def two_mm_kernel_configs():
            configs = []
            for BLOCK_M in [16, 32]:
                for BLOCK_N in [16, 32, 64]:
                    for BLOCK_K in [16, 32, 64]:
                        for num_stages in [2, 3]:
                            configs.append(triton.Config({
                                'BLOCK_M': BLOCK_M,
                                'BLOCK_N': BLOCK_N,
                                'BLOCK_K': BLOCK_K,
                                'GROUP_SIZE_M': 8
                            }, num_stages=num_stages, num_warps=8))
            return configs
            
    elif torch.cuda.get_device_capability()[0] == 9:
        def get_optimal_two_mm_config_h100(B, seq_len, dim):
            configs = {
                (1, 128, 128): (128, 64, 128, 2, 8),
                (1, 128, 256): (128, 64, 128, 2, 8),
                (1, 128, 384): (128, 64, 64, 3, 8),
                (1, 128, 512): (128, 64, 64, 3, 8),
                (1, 128, 768): (128, 64, 64, 3, 8),
                (1, 128, 1024): (128, 64, 64, 3, 8),
                (1, 256, 128): (128, 64, 128, 2, 8),
                (1, 256, 256): (128, 64, 128, 2, 8),
                (1, 256, 384): (128, 64, 64, 3, 8),
                (1, 256, 512): (128, 64, 64, 3, 8),
                (1, 256, 768): (128, 64, 64, 3, 8),
                (1, 256, 1024): (128, 64, 64, 3, 8),
                (1, 512, 128): (128, 64, 128, 2, 8),
                (1, 512, 256): (128, 64, 128, 2, 8),
                (1, 512, 384): (128, 64, 128, 2, 8),
                (1, 512, 512): (128, 64, 128, 2, 8),
                (1, 512, 768): (128, 64, 64, 3, 8),
                (1, 512, 1024): (128, 64, 64, 3, 8),
                (1, 1024, 128): (128, 64, 128, 2, 8),
                (1, 1024, 256): (128, 64, 64, 2, 8),
                (1, 1024, 384): (128, 64, 128, 2, 8),
                (1, 1024, 512): (128, 64, 128, 2, 8),
                (1, 1024, 768): (128, 64, 128, 2, 8),
                (1, 1024, 1024): (128, 64, 128, 2, 8),
                (2, 128, 128): (128, 64, 128, 2, 8),
                (2, 128, 256): (128, 64, 128, 2, 8),
                (2, 128, 384): (128, 64, 64, 3, 8),
                (2, 128, 512): (128, 64, 64, 3, 8),
                (2, 128, 768): (128, 64, 64, 3, 8),
                (2, 128, 1024): (128, 64, 64, 3, 8),
                (2, 256, 128): (128, 64, 128, 2, 8),
                (2, 256, 256): (128, 64, 128, 2, 8),
                (2, 256, 384): (128, 64, 128, 2, 8),
                (2, 256, 512): (128, 64, 128, 2, 8),
                (2, 256, 768): (128, 64, 64, 3, 8),
                (2, 256, 1024): (128, 64, 64, 3, 8),
                (2, 512, 128): (128, 64, 128, 2, 8),
                (2, 512, 256): (128, 64, 128, 2, 8),
                (2, 512, 384): (128, 64, 128, 2, 8),
                (2, 512, 512): (128, 64, 128, 2, 8),
                (2, 512, 768): (128, 64, 128, 2, 8),
                (2, 512, 1024): (128, 64, 128, 2, 8),
                (2, 1024, 128): (128, 64, 128, 2, 8),
                (2, 1024, 256): (128, 64, 128, 2, 8),
                (2, 1024, 384): (128, 64, 128, 2, 8),
                (2, 1024, 512): (128, 64, 128, 2, 8),
                (2, 1024, 768): (128, 64, 128, 2, 8),
                (2, 1024, 1024): (128, 64, 128, 2, 8),
            }
            return configs.get((B, seq_len, dim), (64, 64, 32, 2, 8))  # default fallback

        def two_mm_kernel_configs():
            # This function is kept for compatibility but will be overridden for H100
            return [
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
            ]
        
    elif torch.cuda.get_device_capability()[0] == 10 and False:
        def get_optimal_two_mm_config(B, seq_len, dim):
            configs = {
                (1, 128, 128): (64, 128, 64, 2, 8),
                (1, 128, 256): (128, 64, 128, 2, 8),
                (1, 128, 384): (128, 64, 128, 2, 8),
                (1, 128, 512): (128, 64, 128, 2, 8),
                (1, 128, 768): (128, 64, 64, 3, 8),
                (1, 128, 1024): (128, 64, 64, 3, 8),
                (1, 256, 128): (128, 64, 128, 2, 8),
                (1, 256, 256): (128, 64, 128, 2, 8),
                (1, 256, 384): (128, 64, 128, 2, 8),
                (1, 256, 512): (128, 64, 64, 3, 8),
                (1, 256, 768): (128, 64, 64, 3, 8),
                (1, 256, 1024): (128, 64, 64, 3, 8),
                (1, 512, 128): (128, 64, 128, 2, 8),
                (1, 512, 256): (128, 64, 128, 2, 8),
                (1, 512, 384): (128, 64, 128, 2, 8),
                (1, 512, 512): (128, 64, 128, 2, 8),
                (1, 512, 768): (128, 64, 64, 3, 8),
                (1, 512, 1024): (128, 64, 64, 3, 8),
                (1, 1024, 128): (128, 64, 128, 2, 8),
                (1, 1024, 256): (128, 64, 128, 2, 8),
                (1, 1024, 384): (128, 64, 128, 2, 8),
                (1, 1024, 512): (128, 64, 128, 2, 8),
                (1, 1024, 768): (128, 64, 64, 3, 8),
                (1, 1024, 1024): (128, 64, 64, 3, 8),
                (2, 128, 128): (128, 64, 128, 2, 8),
                (2, 128, 256): (128, 64, 128, 2, 8),
                (2, 128, 384): (128, 64, 128, 2, 8),
                (2, 128, 512): (128, 64, 64, 3, 8),
                (2, 128, 768): (128, 64, 64, 3, 8),
                (2, 128, 1024): (128, 64, 64, 3, 8),
                (2, 256, 128): (128, 64, 128, 2, 8),
                (2, 256, 256): (128, 64, 128, 2, 8),
                (2, 256, 384): (128, 64, 128, 2, 8),
                (2, 256, 512): (128, 64, 64, 3, 8),
                (2, 256, 768): (128, 64, 64, 3, 8),
                (2, 256, 1024): (128, 64, 64, 3, 8),
                (2, 512, 128): (128, 64, 128, 2, 8),
                (2, 512, 256): (128, 64, 128, 2, 8),
                (2, 512, 384): (128, 64, 128, 2, 8),
                (2, 512, 512): (128, 64, 128, 2, 8),
                (2, 512, 768): (128, 64, 64, 3, 8),
                (2, 512, 1024): (128, 64, 64, 3, 8),
                (2, 1024, 128): (128, 64, 128, 2, 8),
                (2, 1024, 256): (128, 64, 128, 2, 8),
                (2, 1024, 384): (128, 64, 128, 2, 8),
                (2, 1024, 512): (128, 64, 128, 2, 8),
                (2, 1024, 768): (128, 64, 64, 3, 8),
                (2, 1024, 1024): (128, 64, 64, 3, 8),
            }
            return configs.get((B, seq_len, dim), (64, 64, 32, 2, 8))  # default fallback

        def two_mm_kernel_configs():
            # This function is kept for compatibility but will be overridden
            return [
                triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
                triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            ]
    elif torch.cuda.get_device_capability()[0] == 8:
        # A100
        def two_mm_kernel_configs():
            configs = []
            for BLOCK_M in [64]:
                for BLOCK_N in [64, 128]:
                    for BLOCK_K in [16]:
                        for num_stages in [3, 4]:
                            for num_warps in [4, 8]:
                                configs.append(triton.Config({
                                    'BLOCK_M': BLOCK_M,
                                    'BLOCK_N': BLOCK_N,
                                    'BLOCK_K': BLOCK_K,
                                    'GROUP_SIZE_M': 8
                                }, num_stages=num_stages, num_warps=num_warps))
            return configs
    else:
        def two_mm_kernel_configs():
            configs = []
            for BLOCK_M in [64, 128]:
                for BLOCK_N in [64, 128]:
                    for BLOCK_K in [64, 128]:
                        for num_stages in [2, 3]:
                            configs.append(triton.Config({
                                'BLOCK_M': BLOCK_M,
                                'BLOCK_N': BLOCK_N,
                                'BLOCK_K': BLOCK_K,
                                'GROUP_SIZE_M': 8
                            }, num_stages=num_stages, num_warps=8))
            return configs 

    return two_mm_kernel_configs

def two_mm_kernel_wrapper():
    if torch.cuda.get_device_capability()[0] == 8:
        @triton.jit
        def two_mm_kernel(a_ptr, b1_ptr, b2_ptr, b3_ptr, b4_ptr, b5_ptr, c1_ptr, c2_ptr, d_ptr, mask_ptr, M, N, K, stride_a0, stride_a1, stride_a2, stride_a3, stride_bk, stride_bn, stride_c0, stride_c1, stride_c2, stride_c3, seq_len, stride_d0, stride_d1, stride_d2, stride_d3, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr):
            # Persistent kernel using standard tl.load operations
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            k_tiles = tl.cdiv(K, BLOCK_K)
            num_tiles = num_pid_m * num_pid_n

            # tile_id_c is used in the epilogue to break the dependency between
            # the prologue and the epilogue
            tile_id_c = start_pid - NUM_SMS
            num_pid_in_group = GROUP_SIZE_M * num_pid_n

            # Persistent loop over tiles
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
                # Calculate PID for this tile using improved swizzling
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (tile_id % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                # Calculate block offsets
                offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                # Initialize accumulators for all outputs
                accumulator1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                accumulator2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                accumulator3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                accumulator4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                accumulator_d = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                # Main computation loop over K dimension
                for ki in range(k_tiles):
                    k_start = ki * BLOCK_K
                    k_offsets = k_start + offs_k

                    # Create pointers for A matrix (2D flattened view)
                    a_ptrs = a_ptr + offs_am[:, None] * stride_a2 + k_offsets[None, :] * stride_a3
                    a_mask = (offs_am[:, None] < M) & (k_offsets[None, :] < K)

                    # Create pointers for B matrices [N, K] layout
                    b1_ptrs = b1_ptr + offs_bn[:, None] * stride_bn + k_offsets[None, :] * stride_bk
                    b2_ptrs = b2_ptr + offs_bn[:, None] * stride_bn + k_offsets[None, :] * stride_bk
                    b3_ptrs = b3_ptr + offs_bn[:, None] * stride_bn + k_offsets[None, :] * stride_bk
                    b4_ptrs = b4_ptr + offs_bn[:, None] * stride_bn + k_offsets[None, :] * stride_bk
                    b5_ptrs = b5_ptr + offs_bn[:, None] * stride_bn + k_offsets[None, :] * stride_bk
                    b_mask = (offs_bn[:, None] < N) & (k_offsets[None, :] < K)

                    # Load blocks from A and all weight matrices using standard tl.load
                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                    b1 = tl.load(b1_ptrs, mask=b_mask, other=0.0)
                    b2 = tl.load(b2_ptrs, mask=b_mask, other=0.0)
                    b3 = tl.load(b3_ptrs, mask=b_mask, other=0.0)
                    b4 = tl.load(b4_ptrs, mask=b_mask, other=0.0)
                    b5 = tl.load(b5_ptrs, mask=b_mask, other=0.0)

                    # Perform matrix multiplications using TF32
                    accumulator1 = tl.dot(a, b1.T, accumulator1, allow_tf32=True)  # A @ B1.T
                    accumulator2 = tl.dot(a, b2.T, accumulator2, allow_tf32=True)  # A @ B2.T
                    accumulator3 = tl.dot(a, b3.T, accumulator3, allow_tf32=True)  # A @ B3.T
                    accumulator4 = tl.dot(a, b4.T, accumulator4, allow_tf32=True)  # A @ B4.T
                    accumulator_d = tl.dot(a, b5.T, accumulator_d, allow_tf32=True)  # A @ B5.T

                # Store results using separate tile_id_c for epilogue
                tile_id_c += NUM_SMS
                group_id = tile_id_c // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (tile_id_c % group_size_m)
                pid_n = (tile_id_c % num_pid_in_group) // group_size_m

                # Calculate output offsets and pointers
                offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

                # Create masks for bounds checking
                d_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

                # Calculate pointer addresses using 4D strides
                stride_cm = stride_c2  # Stride to next element in flattened M dimension
                stride_cn = stride_c3  # N is the innermost dimension

                # For D tensor: use separate D strides
                stride_dm = stride_d2  # Stride to next element in flattened M dimension
                stride_dn = stride_d3  # N is the innermost dimension

                off_c_batch = offs_cm // (seq_len * seq_len)
                off_c_sl1 = (offs_cm // seq_len) % seq_len
                off_c_sl2 = offs_cm % seq_len
                off_c_dim = offs_cn

                c_offsets = (off_c_batch * stride_c0 + off_c_sl1 * stride_c1 + off_c_sl2 * stride_c2)[:, None] + off_c_dim[None, :] * stride_c3
                c_mask = d_mask

                c1_ptrs = c1_ptr + c_offsets
                c2_ptrs = c2_ptr + c_offsets
                d_ptrs = d_ptr + stride_dm * offs_cm[:, None] + stride_dn * offs_cn[None, :]

                mask = tl.load(mask_ptr + offs_cm, mask=(offs_cm < M))

                # Broadcast mask to match accumulator dimensions [BLOCK_M, BLOCK_N]
                mask_2d = mask[:, None]  # Convert to [BLOCK_M, 1] then broadcast
                # Apply masking only to left_proj and right_proj results (C1, C2)
                accumulator1 = tl.where(mask_2d, accumulator1, 0)
                accumulator2 = tl.where(mask_2d, accumulator2, 0)

                # Apply sigmoid to gate values
                left_gate_sigmoid = triton_sigmoid(accumulator3)
                right_gate_sigmoid = triton_sigmoid(accumulator4)
                accumulator_d = triton_sigmoid(accumulator_d)

                # Apply elementwise multiplication with gated values
                # C1 = left * left_gate, C2 = right * right_gate
                accumulator1 = accumulator1 * left_gate_sigmoid  # left * left_gate
                accumulator2 = accumulator2 * right_gate_sigmoid  # right * right_gate

                # Convert to appropriate output dtype and store with normal tl.store
                c1 = accumulator1.to(c1_ptr.dtype.element_ty)
                c2 = accumulator2.to(c2_ptr.dtype.element_ty)
                d = accumulator_d.to(d_ptr.dtype.element_ty)

                tl.store(c1_ptrs, c1, mask=c_mask)
                tl.store(c2_ptrs, c2, mask=c_mask)
                tl.store(d_ptrs, d, mask=d_mask)
    else:
        @triton.jit
        def two_mm_kernel(a_ptr, b1_ptr, b2_ptr, b3_ptr, b4_ptr, b5_ptr, c1_ptr, c2_ptr, d_ptr, mask_ptr, M, N, K, stride_a0, stride_a1, stride_a2, stride_a3, stride_bk, stride_bn, stride_c0, stride_c1, stride_c2, stride_c3, seq_len, stride_d0, stride_d1, stride_d2, stride_d3, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr):
            # Persistent kernel using on-device TMA descriptors
            start_pid = tl.program_id(axis=0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            num_pid_n = tl.cdiv(N, BLOCK_N)
            k_tiles = tl.cdiv(K, BLOCK_K)
            num_tiles = num_pid_m * num_pid_n

            # Create on-device TMA descriptors
            a_desc = tl._experimental_make_tensor_descriptor(
                a_ptr,
                shape=[M, K],
                strides=[stride_a2, stride_a3],
                block_shape=[BLOCK_M, BLOCK_K],
            )
            b1_desc = tl._experimental_make_tensor_descriptor(
                b1_ptr,
                shape=[N, K],
                strides=[stride_bn, stride_bk],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            b2_desc = tl._experimental_make_tensor_descriptor(
                b2_ptr,
                shape=[N, K],
                strides=[stride_bn, stride_bk],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            b3_desc = tl._experimental_make_tensor_descriptor(
                b3_ptr,
                shape=[N, K],
                strides=[stride_bn, stride_bk],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            b4_desc = tl._experimental_make_tensor_descriptor(
                b4_ptr,
                shape=[N, K],
                strides=[stride_bn, stride_bk],
                block_shape=[BLOCK_N, BLOCK_K],
            )
            b5_desc = tl._experimental_make_tensor_descriptor(
                b5_ptr,
                shape=[N, K],
                strides=[stride_bn, stride_bk],
                block_shape=[BLOCK_N, BLOCK_K],
            )

            # tile_id_c is used in the epilogue to break the dependency between
            # the prologue and the epilogue
            tile_id_c = start_pid - NUM_SMS
            num_pid_in_group = GROUP_SIZE_M * num_pid_n

            # Persistent loop over tiles
            for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=False):
                # Calculate PID for this tile using improved swizzling
                group_id = tile_id // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (tile_id % group_size_m)
                pid_n = (tile_id % num_pid_in_group) // group_size_m

                # Calculate block offsets
                offs_am = pid_m * BLOCK_M
                offs_bn = pid_n * BLOCK_N

                # Initialize accumulators for all outputs
                accumulator1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                accumulator2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                accumulator3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                accumulator4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                accumulator_d = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                # Main computation loop over K dimension
                for ki in range(k_tiles):
                    offs_k = ki * BLOCK_K
                    # Load blocks from A and all weight matrices using on-device TMA
                    a = a_desc.load([offs_am, offs_k])
                    b1 = b1_desc.load([offs_bn, offs_k])
                    b2 = b2_desc.load([offs_bn, offs_k])
                    b3 = b3_desc.load([offs_bn, offs_k])
                    b4 = b4_desc.load([offs_bn, offs_k])
                    b5 = b5_desc.load([offs_bn, offs_k])

                    # Perform matrix multiplications using TF32
                    accumulator1 = tl.dot(a, b1.T, accumulator1, allow_tf32=True)  # A @ B1.T
                    accumulator2 = tl.dot(a, b2.T, accumulator2, allow_tf32=True)  # A @ B2.T
                    accumulator3 = tl.dot(a, b3.T, accumulator3, allow_tf32=True)  # A @ B3.T
                    accumulator4 = tl.dot(a, b4.T, accumulator4, allow_tf32=True)  # A @ B4.T
                    accumulator_d = tl.dot(a, b5.T, accumulator_d, allow_tf32=True)  # A @ B5.T

                # Store results using separate tile_id_c for epilogue
                tile_id_c += NUM_SMS
                group_id = tile_id_c // num_pid_in_group
                first_pid_m = group_id * GROUP_SIZE_M
                group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
                pid_m = first_pid_m + (tile_id_c % group_size_m)
                pid_n = (tile_id_c % num_pid_in_group) // group_size_m

                # Calculate output offsets and pointers
                offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

                # Create masks for bounds checking
                d_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

                # Calculate pointer addresses using 4D strides
                # For C tensors: compute effective 2D strides from 4D strides
                # Output tensor is [B, I, J, N], flattened to [M, N] where M = B*I*J
                stride_cm = stride_c2  # Stride to next element in flattened M dimension
                stride_cn = stride_c3  # N is the innermost dimension

                # For D tensor: use separate D strides
                stride_dm = stride_d2  # Stride to next element in flattened M dimension
                stride_dn = stride_d3  # N is the innermost dimension

                off_c_batch = offs_cm // (seq_len * seq_len)
                off_c_sl1 = (offs_cm // seq_len) % seq_len
                off_c_sl2 = offs_cm % seq_len
                off_c_dim = offs_cn

                # TODO update the mask_c so we don't IMA
                c_offsets = (off_c_batch * stride_c0 + off_c_sl1 * stride_c1 + off_c_sl2 * stride_c2)[:, None] + off_c_dim[None, :] * stride_c3
                # c_offsets = offs_cm[:, None] * stride_c2 + offs_cn[None, :] * stride_c3
                c_mask = d_mask

                c1_ptrs = c1_ptr + c_offsets
                c2_ptrs = c2_ptr + c_offsets
                # c1_ptrs = c1_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                # c2_ptrs = c2_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                d_ptrs = d_ptr + stride_dm * offs_cm[:, None] + stride_dn * offs_cn[None, :]

                mask = tl.load(mask_ptr + offs_cm, mask=(offs_cm < M))

                # Broadcast mask to match accumulator dimensions [BLOCK_M, BLOCK_N]
                mask_2d = mask[:, None]  # Convert to [BLOCK_M, 1] then broadcast
                # Apply masking only to left_proj and right_proj results (C1, C2)
                accumulator1 = tl.where(mask_2d, accumulator1, 0)
                accumulator2 = tl.where(mask_2d, accumulator2, 0)

                # Apply sigmoid to gate values
                left_gate_sigmoid = triton_sigmoid(accumulator3)
                right_gate_sigmoid = triton_sigmoid(accumulator4)
                accumulator_d = triton_sigmoid(accumulator_d)

                # Apply elementwise multiplication with gated values
                # C1 = left * left_gate, C2 = right * right_gate
                accumulator1 = accumulator1 * left_gate_sigmoid  # left * left_gate
                accumulator2 = accumulator2 * right_gate_sigmoid  # right * right_gate

                # Convert to appropriate output dtype and store with normal tl.store
                c1 = accumulator1.to(c1_ptr.dtype.element_ty)
                c2 = accumulator2.to(c2_ptr.dtype.element_ty)
                d = accumulator_d.to(d_ptr.dtype.element_ty)

                tl.store(c1_ptrs, c1, mask=c_mask)
                tl.store(c2_ptrs, c2, mask=c_mask)
                tl.store(d_ptrs, d, mask=d_mask)


    if torch.cuda.get_device_capability()[0] not in [9, 10.2]:
        two_mm_kernel = triton.autotune(
            (two_mm_kernel_configs_wrapper())(), key=["M", "N", "K"]
        )(two_mm_kernel)

    return two_mm_kernel


def two_mm(A, left_proj, right_proj, left_gate, right_gate, out_gate, mask):
    """
    Persistent matrix multiplication for all weight matrices using on-device TMA descriptors.

    Args:
        A: [..., K] tensor (arbitrary leading dimensions)
        left_proj: [N, K] matrix (will be transposed)
        right_proj: [N, K] matrix (will be transposed)
        left_gate: [N, K] left gate weight matrix
        right_gate: [N, K] right gate weight matrix
        out_gate: [N, K] output gate weight matrix
        mask: mask tensor

    Returns:
        (C1, C2, D): Tuple of result tensors [..., N] with same leading dims as A
            C1 = (A @ left_proj.T) * sigmoid(A @ left_gate.T) (masked)
            C2 = (A @ right_proj.T) * sigmoid(A @ right_gate.T) (masked)
            D = sigmoid(A @ out_gate.T) (unmasked)
    """
    # Check constraints
    assert A.shape[-1] == left_proj.shape[1] == right_proj.shape[1], "Incompatible K dimensions"
    assert A.dtype == left_proj.dtype == right_proj.dtype, "Incompatible dtypes"

    # Assert that all weight matrices have the same strides (same [N, K] shape)
    assert left_proj.stride() == right_proj.stride() == left_gate.stride() == right_gate.stride() == out_gate.stride(), \
        "All weight matrices must have identical strides"

    # Get dimensions
    original_shape = A.shape[:-1]  # All dimensions except the last
    K = A.shape[-1]
    N = left_proj.shape[0]
    B, seq_len, _, _ = A.shape
    dtype = A.dtype

    # Flatten A to 2D for kernel processing
    A_2d = A.view(-1, K)  # [M, K] where M is product of all leading dims
    M = A_2d.shape[0]

    # Get number of streaming multiprocessors
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    # Launch persistent kernel with limited number of blocks
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])),)

    # Get original 4D strides for A and output tensors
    A_strides = A.stride()  # (stride_0, stride_1, stride_2, stride_3)

    # Create output tensors with proper 4D shape to get correct strides
    output_shape = original_shape + (N,)
    # C1 = torch.empty(output_shape, device=A.device, dtype=dtype)
    # C2 = torch.empty(output_shape, device=A.device, dtype=dtype)
    C1 = torch.empty(B, N, seq_len, seq_len, device=A.device, dtype=torch.float16).permute(0, 2, 3, 1)
    C2 = torch.empty(B, N, seq_len, seq_len, device=A.device, dtype=torch.float16).permute(0, 2, 3, 1)
    D = torch.empty(output_shape, device=A.device, dtype=torch.float16)

    C_strides = C1.stride()  # (stride_0, stride_1, stride_2, stride_3)
    D_strides = D.stride()   # (stride_0, stride_1, stride_2, stride_3)

    # Use optimal configuration for B200/H100 or fallback to autotuning for other GPUs
    if torch.cuda.get_device_capability()[0] == 10:
        # Get optimal configuration for B200
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps = (two_mm_kernel_configs_wrapper())(B, seq_len, K)
        grid_size = min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))

        two_mm_kernel_wrapper()[(grid_size,)](
            A_2d, left_proj, right_proj, left_gate, right_gate, out_gate,
            C1, C2, D, mask,
            M, N, K,
            *A_strides,  # 4D strides for A
            left_proj.stride(1), left_proj.stride(0),  # B matrices [N, K] shape strides
            *C_strides,  # 4D strides for C
            seq_len,
            *D_strides,  # 4D strides for D
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=8, NUM_SMS=NUM_SMS,
            num_stages=num_stages, num_warps=num_warps
        )
    elif torch.cuda.get_device_capability()[0] == 9:
        # Get optimal configuration for H100
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps = (two_mm_kernel_configs_wrapper())(B, seq_len, K)
        grid_size = min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N))

        two_mm_kernel_wrapper()[(grid_size,)](
            A_2d, left_proj, right_proj, left_gate, right_gate, out_gate,
            C1, C2, D, mask,
            M, N, K,
            *A_strides,  # 4D strides for A
            left_proj.stride(1), left_proj.stride(0),  # B matrices [N, K] shape strides
            *C_strides,  # 4D strides for C
            seq_len,
            *D_strides,  # 4D strides for D
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_SIZE_M=8, NUM_SMS=NUM_SMS,
            num_stages=num_stages, num_warps=num_warps
        )
    else:
        # Use autotuning for other GPUs
        two_mm_kernel_wrapper()[grid](
            A_2d, left_proj, right_proj, left_gate, right_gate, out_gate,
            C1, C2, D, mask,
            M, N, K,
            *A_strides,  # 4D strides for A
            left_proj.stride(1), left_proj.stride(0),  # B matrices [N, K] shape strides
            *C_strides,  # 4D strides for C
            seq_len,
            *D_strides,  # 4D strides for D
            NUM_SMS=NUM_SMS
        )

    return C1, C2, D


def second_layernorm_mul(inp, hidden_dim, weight, bias, mul_operand):
    ln = torch.nn.functional.layer_norm(inp, (hidden_dim,), eps=1e-5, weight=weight.to(inp.dtype), bias=bias.to(inp.dtype))
    out = ln * mul_operand
    return out

'''
@triton.autotune(
    [triton.Config({"ROW_BLOCK_SIZE": 16}, num_warps=4, num_stages=3)],
    key=["R", "C"]
)
'''
@triton.jit
def layernorm_kernel_first(
    X,
    Y,
    Weight,
    Bias,
    R,
    C,  # aka "dim"
    eps,
    ROW_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK_SIZE + tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE)

    mask_row = row < R
    mask_col = cols < C

    # Simple indexing for contiguous data
    x = tl.load(
        X + row[:, None] * C + cols[None, :],
        mask=mask_row[:, None] & mask_col[None, :],
        other=0.0
    ).to(tl.float32)

    weight = tl.load(Weight + cols, mask=mask_col, other=0.0).to(tl.float32)
    bias = tl.load(Bias + cols, mask=mask_col, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=1) / C
    diff = tl.where(mask_row[:, None] & mask_col[None, :], x - mean[:, None], 0)
    var = tl.sum(diff * diff, axis=1) / C
    rstd = 1 / tl.sqrt(var + eps)

    y_hat = (x - mean[:, None]) * rstd[:, None]
    y = y_hat * weight[None, :] + bias[None, :]

    tl.store(
        Y + row[:, None] * C + cols[None, :],
        y,
        mask=mask_row[:, None] & mask_col[None, :]
    )


def get_optimal_config_ln(dim):
    config = None
    if torch.cuda.get_device_capability()[0] == 9:
        if (dim <= 256):
            config = (16, 1)
        elif dim <= 512:
            config = (16, 2)
        elif dim <= 1024:
            config = (16, 4)
        
    if not config:
        config = (16, 4)
    return config


def triton_layernorm_first(x, weight, bias, eps=1e-5, num_warps=None, ROW_BLOCK_SIZE=None):
    B, seq_len, seq_len2, dim = x.shape
    assert(seq_len == seq_len2)

    R = B * seq_len * seq_len
    C = dim

    out = torch.empty_like(x, dtype=torch.float16)

    if not num_warps or not ROW_BLOCK_SIZE:
        ROW_BLOCK_SIZE, num_warps = get_optimal_config_ln(dim)

    BLOCK_SIZE = triton.next_power_of_2(C)
    assert(BLOCK_SIZE <= 1024)

    def grid(meta):
        return (triton.cdiv(R, meta["ROW_BLOCK_SIZE"]),)

    layernorm_kernel_first[grid](
        x, out, weight, bias,
        R, C, eps,
        ROW_BLOCK_SIZE=ROW_BLOCK_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=3
    )

    return out

'''
def triton_layernorm_first(x, weight, bias, eps=1e-5):
    B, seq_len, seq_len2, dim = x.shape
    assert(seq_len == seq_len2)

    R = B * seq_len * seq_len
    C = dim

    out = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(C)
    assert(BLOCK_SIZE <= 1024)

    def grid(meta):
        return (triton.cdiv(R, meta["ROW_BLOCK_SIZE"]),)

    layernorm_kernel_first[grid](
        x, out, weight, bias,
        R, C, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
'''


@triton.autotune(
    [triton.Config({"ROW_BLOCK_SIZE": 16}, num_warps=1, num_stages=3)],
    key=[]
)
@triton.jit
def layernorm_kernel_eltwise(
    X,
    Y,
    Weight,
    Bias,
    OutGate,
    seq_len,
    stride_batch,
    stride_dim,
    R,
    C,  # aka "dim"
    eps,
    ROW_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK_SIZE + tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, BLOCK_SIZE)

    # Calculate base pointer for this batch of rows
    tl.device_assert(seq_len*seq_len % ROW_BLOCK_SIZE == 0)
    # batch_offset = (row // (stride_seq1 // stride_dim)) * stride_batch
    batch = tl.program_id(0) * ROW_BLOCK_SIZE // (seq_len * seq_len)
    seqs_off = row % (seq_len * seq_len) # TODO is this going to prevent vectorization

    off_r = batch * stride_batch + seqs_off
    off_c = cols * stride_dim

    mask_row = row < R
    mask_col = cols < C

    out_gate = tl.load(
        OutGate + row[:, None] * C + cols[None, :],
        mask = mask_row[:, None] & mask_col[None, :],
    )

    x = tl.load(
        X + off_r[:, None] + off_c[None, :],
        mask=mask_row[:, None] & mask_col[None, :],
        other=0.0
    ).to(tl.float32)

    weight = tl.load(Weight + cols, mask=mask_col, other=0.0).to(tl.float32)
    bias = tl.load(Bias + cols, mask=mask_col, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=1) / C
    diff = tl.where(mask_row[:, None] & mask_col[None, :], x - mean[:, None], 0)
    var = tl.sum(diff * diff, axis=1) / C
    rstd = 1 / tl.sqrt(var + eps)

    y_hat = (x - mean[:, None]) * rstd[:, None]
    y = y_hat * weight[None, :] + bias[None, :]

    tl.store(
        Y + row[:, None] * C + cols[None, :],
        y * out_gate,
        mask=mask_row[:, None] & mask_col[None, :]
    )


def triton_layernorm_eltwise(x, weight, bias, out_gate, eps=1e-5):
    B, seq_len, seq_len2, dim = x.shape
    assert(seq_len == seq_len2)
    R = B * seq_len * seq_len
    assert(x.stride(3) == seq_len*seq_len)
    assert(out_gate.is_contiguous())
    C = dim

    out = torch.empty_like(out_gate, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(C)
    assert(BLOCK_SIZE == 128)

    def grid(meta):
        return (triton.cdiv(R, meta["ROW_BLOCK_SIZE"]),)

    layernorm_kernel_eltwise[grid](
        x, out, weight, bias, out_gate,
        seq_len,
        x.stride(0), x.stride(3),
        R, C, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out


def kernel_global(data: input_t) -> output_t:
    """
    Reference implementation of TriMul using PyTorch.
    
    Args:
        data: Tuple of (input: torch.Tensor, mask: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
            - input: Input tensor of shape [batch_size, seq_len, seq_len, dim]
            - mask: Mask tensor of shape [batch_size, seq_len, seq_len]
            - weights: Dictionary containing model weights
            - config: Dictionary containing model configuration parameters
    """
    input_tensor, mask, weights, config = data

    left_proj_weight = weights["left_proj.weight"].to(torch.float16)
    right_proj_weight = weights["right_proj.weight"].to(torch.float16)
    left_gate_weight = weights["left_gate.weight"].to(torch.float16)
    right_gate_weight = weights["right_gate.weight"].to(torch.float16)
    out_gate_weight = weights["out_gate.weight"].to(torch.float16)

    hidden_dim = config["hidden_dim"]
    # trimul = TriMul(dim=config["dim"], hidden_dim=config["hidden_dim"]).to(input_tensor.device)

    x = input_tensor

    batch_size, seq_len, _, dim = x.shape

    x = triton_layernorm_first(x, weights['norm.weight'], weights['norm.bias'])
    # x = torch.nn.functional.layer_norm(x, (dim,), eps=1e-5, weight=weights['norm.weight'], bias=weights['norm.bias'])

    left, right, out_gate = two_mm(x, left_proj_weight, right_proj_weight, left_gate_weight, right_gate_weight, out_gate_weight, mask)
    # left = torch.nn.functional.linear(x, weights['left_proj.weight'].to(torch.float16))
    # right = torch.nn.functional.linear(x, weights['right_proj.weight'].to(torch.float16))

    # left = left * mask.unsqueeze(-1)
    # right = right * mask.unsqueeze(-1)

    '''
    left = left.to(torch.float32)
    right = right.to(torch.float32)
    x = x.to(torch.float32)

    left_gate = left_gate.sigmoid()
    right_gate = right_gate.sigmoid()
    out_gate = out_gate.sigmoid()
    '''

    # Elementwise multiplication now handled in kernel
    # left = left * left_gate
    # right = right * right_gate

    # out = einsum('... i k d, ... j k d -> ... i j d', left, right)
    out = torch.bmm(left.permute(0, 3, 1, 2).view(-1, left.shape[1], left.shape[2]), right.permute(0, 3, 2, 1).view(-1, right.shape[2], right.shape[1]))
    out = out.view(batch_size, hidden_dim, seq_len, seq_len).permute(0, 2, 3, 1)

    # out = torch.compile(second_layernorm_mul, dynamic=False)(out, hidden_dim, weights['to_out_norm.weight'], weights['to_out_norm.bias'], out_gate)
    out = triton_layernorm_eltwise(out, weights['to_out_norm.weight'], weights['to_out_norm.bias'], out_gate)
    # out = torch.nn.functional.layer_norm(out, (hidden_dim,), eps=1e-5, weight=weights['to_out_norm.weight'].to(out.dtype), bias=weights['to_out_norm.bias'].to(out.dtype))
    # out = out * out_gate
    return torch.nn.functional.linear(out, weights['to_out.weight'])

    '''
    # Fill in the given weights of the model
    trimul.norm.weight = nn.Parameter(weights['norm.weight'])
    trimul.norm.bias = nn.Parameter(weights['norm.bias'])
    trimul.left_proj.weight = nn.Parameter(weights['left_proj.weight'])
    trimul.right_proj.weight = nn.Parameter(weights['right_proj.weight'])
    trimul.left_gate.weight = nn.Parameter(weights['left_gate.weight'])
    trimul.right_gate.weight = nn.Parameter(weights['right_gate.weight'])
    trimul.out_gate.weight = nn.Parameter(weights['out_gate.weight'])
    trimul.to_out_norm.weight = nn.Parameter(weights['to_out_norm.weight'])
    trimul.to_out_norm.bias = nn.Parameter(weights['to_out_norm.bias'])
    trimul.to_out.weight = nn.Parameter(weights['to_out.weight'])

    output = trimul(input_tensor, mask)

    return output
    '''