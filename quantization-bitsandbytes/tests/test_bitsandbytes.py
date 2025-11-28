import torch
import pytest

from quantization_bitsandbytes import gemm_4bit_forward

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
 
 
_NF4_QUANT_TABLE = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)

_FP4_QUANT_TABLE = torch.tensor(
    [
        0.0000,
        0.0052,
        0.6667,
        1.0000,
        0.3333,
        0.5000,
        0.1667,
        0.2500,
        0.0000,
        -0.0052,
        -0.6667,
        -1.0000,
        -0.3333,
        -0.5000,
        -0.1667,
        -0.2500,
    ],
    dtype=torch.float32,
)

def ref_gemm_4bit(x, packed_weight, scales, group_size, quant_type):
    unpacked_weight = unpack_weight_packed_for_cpu(packed_weight)
    shape = unpacked_weight.shape
    table = _FP4_QUANT_TABLE if quant_type == 1 else _NF4_QUANT_TABLE
    original_weight = table[unpacked_weight.reshape(-1).int()].reshape(shape) * scales.T.repeat_interleave(group_size, dim=1)
    res = torch.matmul(x, original_weight.T.to(x.dtype))
    return res

@pytest.mark.parametrize("M", [1, 4, 32, 128, 512, 1024])
@pytest.mark.parametrize("K", [2048, 4096])
@pytest.mark.parametrize("N", [2048, 4096])
@pytest.mark.parametrize("quant_type", [0, 1])
def test_bitsandbytes(M, K, N, quant_type):
    torch.manual_seed(100)
    device = torch.device("cpu")
    dtype = torch.bfloat16
    group_size = 64

    assert K % group_size == 0
    assert K % 2 == 0

    x = torch.randn((M, K), device=device, dtype=dtype) * 0.1
    w = torch.randint(0, 15, (N, K // 2), device=device, dtype=torch.uint8)
    num_groups = K // group_size
    scales = torch.rand((num_groups, N), device=device, dtype=torch.bfloat16).pow(4.0)

    output = gemm_4bit_forward(x, w, scales, group_size, quant_type)
    ref_out = ref_gemm_4bit(x, w, scales, group_size, quant_type)

    torch.testing.assert_close(output, ref_out, atol=1e-1, rtol=1e-2)
