"""`affine_qmm_t` against a dequantize-then-matmul reference, and against MLX itself.

The reference unpacks MLX's affine layout in plain torch: each row of `w` is a little-endian bit
stream of `bits`-wide values, and each `group_size` run shares a scale and a bias. When `mlx` is
installed, quantization goes through `mx.quantize`, so the layout under test is MLX's own, and the
output is also compared with `mx.quantized_matmul` -- which, for the same shape on the same GPU,
runs the same kernel and must agree exactly.

Every branch of the dispatch is reached by steering `MLX_METAL_GPU_ARCH`, as MLX itself allows.
"""

import os
from contextlib import contextmanager

import pytest
import torch

import mlx_quantization_metal_kernels as mq
from mlx_quantization_metal_kernels._ops import ops


pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

try:
    import mlx.core as mx
    import numpy as np
except ImportError:
    mx = None

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
BITS = [2, 3, 4, 5, 6, 8]
GROUP_SIZES = [32, 64, 128]


@contextmanager
def arch(name):
    old = os.environ.get("MLX_METAL_GPU_ARCH")
    os.environ["MLX_METAL_GPU_ARCH"] = name
    try:
        yield
    finally:
        if old is None:
            del os.environ["MLX_METAL_GPU_ARCH"]
        else:
            os.environ["MLX_METAL_GPU_ARCH"] = old


def quantize(w, group_size, bits):
    """[N, K] float -> (w_q uint32 [N, K*bits/32], scales [N, K/gs], biases [N, K/gs]), MLX's layout."""
    if mx is not None:
        wq, s, b = mx.quantize(mx.array(w.float().numpy()), group_size=group_size, bits=bits)
        to_t = lambda a: torch.from_numpy(np.array(a))  # noqa: E731
        return to_t(wq).view(torch.uint32), to_t(s).float(), to_t(b).float()
    # MLX's affine scheme, for when mlx is not installed
    N, K = w.shape
    g = w.float().reshape(N, K // group_size, group_size)
    w_min, w_max = g.amin(-1), g.amax(-1)
    n_bins = 2**bits - 1
    scales = ((w_max - w_min) / n_bins).clamp_min(1e-7)
    side = w_min.abs() > w_max.abs()
    scales = torch.where(side, scales, -scales)
    edge = torch.where(side, w_min, w_max)
    q0 = (edge / scales).round()
    scales = torch.where(q0 != 0, edge / q0, scales)
    biases = torch.where(q0 == 0, torch.zeros_like(q0), edge)
    q = ((g - biases[..., None]) / scales[..., None]).round().clamp(0, n_bins).to(torch.int64).reshape(N, K)
    # pack as a little-endian bit stream
    shifts = torch.arange(bits)
    stream = ((q[..., None] >> shifts) & 1).reshape(N, K * bits // 8, 8)
    packed = (stream << torch.arange(8)).sum(-1).to(torch.uint8)
    return packed.view(torch.int32).view(torch.uint32), scales, biases


def dequantize(wq, scales, biases, group_size, bits):
    """MLX's affine layout -> [N, K] float32, in plain torch."""
    N = wq.shape[0]
    bytes_ = wq.cpu().view(torch.uint8).to(torch.int64)
    stream = ((bytes_[..., None] >> torch.arange(8)) & 1).reshape(N, -1, bits)
    q = (stream << torch.arange(bits)).sum(-1).float()
    K = q.shape[1]
    q = q.reshape(N, K // group_size, group_size)
    return (q * scales.cpu().float()[..., None] + biases.cpu().float()[..., None]).reshape(N, K)


def make(N, K, group_size, bits, dtype, seed=0):
    torch.manual_seed(seed)
    wq, s, b = quantize(torch.randn(N, K), group_size, bits)
    return wq.to("mps"), s.to("mps", dtype), b.to("mps", dtype)


def check(y, x, wq, s, b, group_size, bits):
    """`y` against x @ dequantize(w).T in float32, within a rounding bound.

    The kernels accumulate in float32 and round once to x's dtype, so the error is bounded by a few
    units in the last place of the output's dtype times sum_k |x_k * w_k| -- the bound that also
    covers the cancellation a plain relative tolerance trips on.
    """
    w = dequantize(wq, s, b, group_size, bits)
    xf = x.cpu().float().reshape(-1, x.shape[-1])
    ref = xf @ w.T
    scale = xf.abs() @ w.abs().T
    unit = {torch.float32: 2.0**-24, torch.float16: 2.0**-11, torch.bfloat16: 2.0**-8}[x.dtype]
    assert y.shape == (*x.shape[:-1], w.shape[0]) and y.dtype == x.dtype
    yf = y.cpu().float().reshape(ref.shape)
    assert torch.isfinite(yf).all()
    err = (yf - ref).abs()
    bound = 2 * unit * scale + 1e-6 * scale
    assert (err <= bound).all(), f"max err {err.max():.3g}, worst err/bound {(err / bound).max():.3g}"


# --- the reference itself --------------------------------------------------------------------------


@pytest.mark.skipif(mx is None, reason="needs mlx")
@pytest.mark.parametrize("bits", BITS)
def test_reference_dequantize_matches_mlx(bits):
    wq, s, b = quantize(torch.randn(64, 256), 64, bits)
    ours = dequantize(wq, s, b, 64, bits)
    to_mx = lambda t: mx.array(t.numpy())  # noqa: E731
    theirs = mx.dequantize(to_mx(wq.view(torch.int32)).view(mx.uint32), to_mx(s), to_mx(b), group_size=64, bits=bits)
    torch.testing.assert_close(ours, torch.from_numpy(np.array(theirs)))


# --- every branch of the dispatch ------------------------------------------------------------------

# (arch, M, N, K, expected kernel prefix). Limits from get_qmv_batch_limit: g14s at <=2048 is 14,
# at <=4096 is 10; g15s routes 2+ rows to qmv_wide.
BRANCHES = [
    ("applegpu_g14s", 1, 512, 64, "affine_qmv_quad_"),
    ("applegpu_g14s", 3, 512, 128, "affine_qmv_quad_"),
    ("applegpu_g14s", 1, 1024, 1024, "affine_qmv_fast_"),
    ("applegpu_g14s", 5, 1004, 1024, "affine_qmv_"),  # N % 8 != 0 -> not fast
    ("applegpu_g14s", 2, 1024, 1024 + 64, "affine_qmv_"),  # K off the fast alignment
    ("applegpu_g15s", 2, 1024, 1024, "affine_qmv_wide_"),
    ("applegpu_g15s", 5, 1024, 1024, "affine_qmv_wide_"),
    ("applegpu_g15s", 12, 1000, 1024, "affine_qmv_wide_"),
    ("applegpu_g14s", 16, 1024, 1024, "affine_qmm_t_splitk_"),
    ("applegpu_g14s", 16, 1000, 1024, "affine_qmm_t_splitk_"),  # alN_false
    ("applegpu_g14s", 300, 1024, 1024, "affine_qmm_t_"),
    ("applegpu_g14s", 300, 1000, 1024, "affine_qmm_t_"),  # alN_false
    ("applegpu_g14s", 17, 1024, 1024 + 64, "affine_qmm_t_"),  # split-K cannot divide K -> qmm
]


@pytest.mark.parametrize("arch_name,M,N,K,expected", BRANCHES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_every_branch(arch_name, M, N, K, expected, dtype):
    group_size, bits = 64, 4
    with arch(arch_name):
        name = ops.kernel_for(M, N, K, group_size, bits, dtype)
        assert name.startswith(expected), name
        if expected == "affine_qmv_":
            assert not name.startswith("affine_qmv_fast_")
        if expected == "affine_qmm_t_":
            assert "splitk" not in name
        wq, s, b = make(N, K, group_size, bits, dtype)
        x = torch.randn(M, K, device="mps", dtype=dtype)
        check(mq.affine_qmm_t(x, wq, s, b, group_size, bits), x, wq, s, b, group_size, bits)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("M", [1, 7, 64])
def test_bits_group_sizes_dtypes(bits, group_size, dtype, M):
    N, K = 256, 512
    wq, s, b = make(N, K, group_size, bits, dtype)
    x = torch.randn(M, K, device="mps", dtype=dtype)
    check(mq.affine_qmm_t(x, wq, s, b, group_size, bits), x, wq, s, b, group_size, bits)


# --- parity with MLX -------------------------------------------------------------------------------


@pytest.mark.skipif(mx is None, reason="needs mlx")
@pytest.mark.parametrize("M", [1, 3, 9, 16, 33, 200, 600])
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_mlx(M, bits, dtype):
    """Against `mx.quantized_matmul` on the same weights, which runs the same kernels on this GPU.

    Bit for bit, except split-K: MLX sums its K partitions in the output dtype, this package in
    float32 (torch's sum), so there the requirement is to be at least as close to an exact result.
    """
    assert "MLX_METAL_GPU_ARCH" not in os.environ
    N, K, group_size = 1024, 2048, 64
    mx.random.seed(0)
    wq_mx, s_mx, b_mx = mx.quantize(mx.random.normal((N, K)), group_size=group_size, bits=bits)
    mx_dtype = {torch.float32: mx.float32, torch.float16: mx.float16, torch.bfloat16: mx.bfloat16}[dtype]
    s_mx, b_mx = s_mx.astype(mx_dtype), b_mx.astype(mx_dtype)
    torch.manual_seed(0)
    x = torch.randn(M, K).to(dtype)
    x_mx = mx.array(x.float().numpy()).astype(mx_dtype)
    theirs = mx.quantized_matmul(x_mx, wq_mx, s_mx, b_mx, transpose=True, group_size=group_size, bits=bits)

    to_t = lambda a: torch.from_numpy(np.array(a.astype(mx.float32)))  # noqa: E731
    wq = torch.from_numpy(np.array(wq_mx)).view(torch.uint32)
    s, b = to_t(s_mx).to(dtype), to_t(b_mx).to(dtype)
    ours = mq.affine_qmm_t(x.to("mps"), wq.to("mps"), s.to("mps"), b.to("mps"), group_size, bits).cpu()
    theirs = to_t(theirs)

    if "splitk" not in ops.kernel_for(M, N, K, group_size, bits, dtype):
        torch.testing.assert_close(ours.float(), theirs, rtol=0, atol=0)
    else:
        exact = x.float() @ dequantize(wq, s, b, group_size, bits).T
        ours_err = (ours.float() - exact).abs().mean()
        theirs_err = (theirs - exact).abs().mean()
        assert ours_err <= theirs_err * 1.01, (ours_err, theirs_err)


# --- shapes and layouts ----------------------------------------------------------------------------


@pytest.mark.parametrize("M_per_row", [1, 3, 40])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_input_matches_per_row(M_per_row, dtype):
    """huggingface/transformers#49337: every row of a 3D input must equal that row run alone."""
    N, K, group_size, bits = 256, 512, 64, 8
    wq, s, b = make(N, K, group_size, bits, dtype)
    x = torch.randn(4, M_per_row, K, device="mps", dtype=dtype)
    batched = mq.affine_qmm_t(x, wq, s, b, group_size, bits)
    per_row = torch.stack([mq.affine_qmm_t(r, wq, s, b, group_size, bits) for r in x])
    assert batched.shape == (4, M_per_row, N)
    torch.testing.assert_close(batched, per_row, rtol=0, atol=0)


def test_1d_and_4d_inputs():
    N, K = 128, 256
    wq, s, b = make(N, K, 64, 4, torch.float16)
    v = torch.randn(K, device="mps", dtype=torch.float16)
    assert mq.affine_qmm_t(v, wq, s, b, 64, 4).shape == (N,)
    x = torch.randn(2, 3, 5, K, device="mps", dtype=torch.float16)
    y = mq.affine_qmm_t(x, wq, s, b, 64, 4)
    torch.testing.assert_close(y, mq.affine_qmm_t(x.reshape(-1, K), wq, s, b, 64, 4).reshape(2, 3, 5, N))


@pytest.mark.parametrize("M", [2, 8, 100])
def test_non_contiguous_input(M):
    N, K = 256, 512
    wq, s, b = make(N, K, 64, 4, torch.float16)
    xt = torch.randn(K, M, device="mps", dtype=torch.float16).t()
    assert not xt.is_contiguous()
    torch.testing.assert_close(
        mq.affine_qmm_t(xt, wq, s, b, 64, 4), mq.affine_qmm_t(xt.contiguous(), wq, s, b, 64, 4), rtol=0, atol=0
    )


def test_input_with_storage_offset():
    N, K = 256, 512
    wq, s, b = make(N, K, 64, 4, torch.float16)
    big = torch.randn(10, K, device="mps", dtype=torch.float16)
    torch.testing.assert_close(
        mq.affine_qmm_t(big[3:7], wq, s, b, 64, 4), mq.affine_qmm_t(big[3:7].clone(), wq, s, b, 64, 4), rtol=0, atol=0
    )


def test_empty_input():
    wq, s, b = make(128, 256, 64, 4, torch.float16)
    y = mq.affine_qmm_t(torch.empty(0, 256, device="mps", dtype=torch.float16), wq, s, b, 64, 4)
    assert y.shape == (0, 128)


# --- inputs the kernels cannot take fail loudly ------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    ["scales_dtype", "w_dtype", "w_shape", "scales_shape", "group_size", "bits", "cpu"],
)
def test_bad_inputs_raise(case):
    N, K = 128, 256
    wq, s, b = make(N, K, 64, 4, torch.float16)
    x = torch.randn(2, K, device="mps", dtype=torch.float16)
    gs, bits = 64, 4
    if case == "scales_dtype":
        s, b = s.float(), b.float()
    elif case == "w_dtype":
        wq = wq.view(torch.int32)
    elif case == "w_shape":
        wq = wq[:, :-1]
    elif case == "scales_shape":
        s = s[:, :-1]
    elif case == "group_size":
        gs = 16
    elif case == "bits":
        bits = 7
    elif case == "cpu":
        x = x.cpu()
    with pytest.raises(RuntimeError, match="mlx-quantization-metal-kernels"):
        mq.affine_qmm_t(x, wq, s, b, gs, bits)
