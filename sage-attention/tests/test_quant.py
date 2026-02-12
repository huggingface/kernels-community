import math
import pytest
import torch

import sage_attention as sa


cuda_available = torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_per_block_int8_shapes_and_types(tensor_layout):
    device = "cuda"
    dtype = torch.float16

    if tensor_layout == "HND":
        q = torch.randn(2, 4, 129, 128, dtype=dtype, device=device)
        k = torch.randn(2, 4, 257, 128, dtype=dtype, device=device)
        expected_q_scale_shape = (2, 4, math.ceil(129 / 128))
        expected_k_scale_shape = (2, 4, math.ceil(257 / 64))
    else:
        q = torch.randn(2, 129, 4, 128, dtype=dtype, device=device)
        k = torch.randn(2, 257, 4, 128, dtype=dtype, device=device)
        expected_q_scale_shape = (2, 4, math.ceil(129 / 128))
        expected_k_scale_shape = (2, 4, math.ceil(257 / 64))

    km = (
        torch.randn(2, 4, 128, dtype=dtype, device=device)
        if tensor_layout == "HND"
        else torch.randn(2, 4, 128, dtype=dtype, device=device)
    )

    q_int8, q_scale, k_int8, k_scale = sa.per_block_int8(
        q, k, km, tensor_layout=tensor_layout
    )

    assert q_int8.shape == q.shape and q_int8.dtype == torch.int8
    assert k_int8.shape == k.shape and k_int8.dtype == torch.int8
    assert q_scale.shape == expected_q_scale_shape and q_scale.dtype == torch.float32
    assert k_scale.shape == expected_k_scale_shape and k_scale.dtype == torch.float32
    assert q_int8.device == q.device == k.device == q_scale.device == k_scale.device
    assert torch.isfinite(q_scale).all()
    assert torch.isfinite(k_scale).all()


@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_per_warp_int8_shapes_and_types(tensor_layout, head_dim):
    device = "cuda"
    dtype = torch.float16

    if tensor_layout == "HND":
        q = torch.randn(1, 2, 130, head_dim, dtype=dtype, device=device)
        k = torch.randn(1, 2, 70, head_dim, dtype=dtype, device=device)
        expected_q_scale_shape = (
            1,
            2,
            math.ceil(130 / 128) * (128 // (16 if head_dim == 128 else 32)),
        )
        expected_k_scale_shape = (1, 2, math.ceil(70 / 64))
    else:
        q = torch.randn(1, 130, 2, head_dim, dtype=dtype, device=device)
        k = torch.randn(1, 70, 2, head_dim, dtype=dtype, device=device)
        expected_q_scale_shape = (
            1,
            2,
            math.ceil(130 / 128) * (128 // (16 if head_dim == 128 else 32)),
        )
        expected_k_scale_shape = (1, 2, math.ceil(70 / 64))

    q_int8, q_scale, k_int8, k_scale = sa.per_warp_int8(
        q,
        k,
        tensor_layout=tensor_layout,
        BLKQ=128,
        WARPQ=(16 if head_dim == 128 else 32),
        BLKK=64,
    )

    assert q_int8.shape == q.shape and q_int8.dtype == torch.int8
    assert k_int8.shape == k.shape and k_int8.dtype == torch.int8
    assert q_scale.shape == expected_q_scale_shape and q_scale.dtype == torch.float32
    assert k_scale.shape == expected_k_scale_shape and k_scale.dtype == torch.float32
    assert torch.isfinite(q_scale).all()
    assert torch.isfinite(k_scale).all()


@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
def test_sub_mean_properties(tensor_layout):
    device = "cuda"
    dtype = torch.float16

    if tensor_layout == "HND":
        v = torch.randn(2, 3, 65, 128, dtype=dtype, device=device)
        seq_dim = 2
        nh_dim = 1
    else:
        v = torch.randn(2, 65, 3, 128, dtype=dtype, device=device)
        seq_dim = 1
        nh_dim = 2

    v_smoothed, vm = sa.sub_mean(v, tensor_layout=tensor_layout)

    assert v_smoothed.shape == v.shape and v_smoothed.dtype == torch.float16
    assert vm.shape == (v.size(0), v.size(nh_dim), v.size(-1)) and vm.dtype == v.dtype
    # The mean along the sequence dimension of smoothed v should be ~0 (in fp16)
    mean_after = v_smoothed.mean(dim=seq_dim)
    assert torch.isfinite(mean_after).all()
    assert (mean_after.abs() < 1e-1).all()


@pytest.mark.skipif(not cuda_available, reason="CUDA is required")
@pytest.mark.parametrize("tensor_layout", ["HND", "NHD"])
@pytest.mark.parametrize("smooth_v", [True, False])
def test_per_channel_fp8_shapes_and_outputs(tensor_layout, smooth_v):
    device = "cuda"
    dtype = torch.float16

    if tensor_layout == "HND":
        v = torch.randn(2, 3, 77, 128, dtype=dtype, device=device)
        kv_len = v.size(2)
    else:
        v = torch.randn(2, 77, 3, 128, dtype=dtype, device=device)
        kv_len = v.size(1)

    v_fp8, v_scale, vm = sa.per_channel_fp8(
        v, tensor_layout=tensor_layout, smooth_v=smooth_v
    )

    assert v_fp8.dtype == torch.float8_e4m3fn
    assert v_scale.shape == (2, 3, 128)
    if smooth_v:
        assert vm is not None and vm.shape == (2, 3, 128) and vm.dtype == torch.float32
    else:
        assert vm is None

    # Padded seq len should be multiple of 64
    padded_len = ((kv_len + 63) // 64) * 64
    if tensor_layout == "HND":
        assert v_fp8.shape == (2, 3, 128, padded_len)
    else:
        assert v_fp8.shape == (2, 128, 3, padded_len)
    assert torch.isfinite(v_scale).all()
