import torch
import pytest

from quantization_gptq import gemm_int4_forward

def unpack_weight_packed_for_cpu(packed_qweight: torch.Tensor, block_n: int = 32):
    """
    Inverse of convert_weight_packed_for_cpu.
    packed_qweight: (N, K//2) uint8, each byte = (high<<4)|low, both 4-bit values in 0..15
    returns: qweight_final (N, K) uint8 with original 4-bit values (0..15)
    """
    assert packed_qweight.dtype == torch.uint8
    assert packed_qweight.dim() == 2
    N, K_half = packed_qweight.shape
    assert N % block_n == 0
    BIT_COUNT = block_n  # 32
    # reshape to rows of 32 packed bytes
    qw = packed_qweight.reshape(-1, BIT_COUNT)           # [(N//block_n)*K_half, 32]
    low  = (qw & 0x0F)
    high = (qw >> 4) & 0x0F
    # restore 64 nibbles (low first then high, matching original pack order)
    restored = torch.cat([low, high], dim=1)             # [..., 64]
    # reshape back (inverse of flatten)
    restored = restored.reshape(N // block_n, K_half, block_n, 2)  # [N/block_n, K//2, block_n, 2]
    # inverse transpose
    restored = restored.transpose(-3, -2)                # [N/block_n, block_n, K//2, 2]
    # final shape
    qweight_final = restored.reshape(N, K_half * 2).to(torch.uint8)
    return qweight_final

def ref_gemm_int4(x, packed_weight, zeros, scales, group_size):
    unpacked_weight = unpack_weight_packed_for_cpu(packed_weight)
    original_weight = (unpacked_weight - zeros.T.repeat_interleave(group_size, dim=1)) * scales.T.repeat_interleave(group_size, dim=1)
    res = torch.matmul(x, original_weight.T.to(x.dtype))
    return res

@pytest.mark.parametrize("M", [1, 31, 244, 1024, 2666])
@pytest.mark.parametrize("K", [2048, 4096, 14336])
@pytest.mark.parametrize("N", [1024, 4096, 7168])
@pytest.mark.parametrize("group_size", [64, 128])
def test_gptq(M, K, N, group_size):
    device = torch.device("cpu")
    dtype = torch.bfloat16

    assert K % group_size == 0
    assert K % 2 == 0

    x = torch.randn((M, K), device=device, dtype=dtype) * 0.1
    w = torch.randint(0, 15, (N, K // 2), device=device, dtype=torch.uint8)
    num_groups = K // group_size
    scales = torch.rand((num_groups, N), device=device, dtype=torch.bfloat16).pow(4.0)
    zeros = torch.randint(0, 15, (num_groups, N), device=device, dtype=torch.uint8)

    output = gemm_int4_forward(x, w, zeros, scales, group_size)
    ref_out = ref_gemm_int4(x, w, zeros, scales, group_size)

    torch.testing.assert_close(output, ref_out, atol=1e-1, rtol=1e-2)
