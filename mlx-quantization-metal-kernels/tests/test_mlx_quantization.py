"""The ops against MLX itself.

Every op is compared with its `mlx.core` counterpart on the same inputs: since both run the same
kernels with the same dispatch on the same GPU, results must agree bit for bit. Which kernels run is
checked through `trace_*`, which plans a call without launching it; `MLX_METAL_GPU_ARCH` (as in MLX)
steers that plan to other GPU generations.

Needs MPS, and `mlx` at the version vendor/UPSTREAM pins (the parity tests skip without it).
"""

import os
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import kernels


mq = kernels.get_kernel("kernels-community/mlx-quantization-metal-kernels", version=2)
ops = mq.ops

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")

try:
    import mlx.core as mx
    import numpy as np
except ImportError:
    mx = None

needs_mlx = pytest.mark.skipif(mx is None, reason="needs mlx")

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
MODES = {"affine": (64, 4), "mxfp4": (32, 4), "mxfp8": (32, 8), "nvfp4": (16, 4)}
AFFINE = [(gs, b) for gs in (32, 64, 128) for b in (2, 3, 4, 5, 6, 8)]


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


# --- conversions --------------------------------------------------------------------------------


def mx_dtype(dtype):
    return {torch.float32: mx.float32, torch.float16: mx.float16, torch.bfloat16: mx.bfloat16}[dtype]


def to_mx(t):
    t = t.detach().cpu()
    if t.dtype == torch.uint32:
        return mx.array(t.view(torch.int32).numpy()).view(mx.uint32)
    if t.dtype in (torch.float16, torch.bfloat16):
        return mx.array(t.float().numpy()).astype(mx_dtype(t.dtype))
    return mx.array(t.numpy())


def mx_indices(t):
    return mx.array(t.cpu().to(torch.int32).numpy()).astype(mx.uint32)


def to_t(a, device="mps"):
    if a.dtype == mx.uint32:
        return torch.from_numpy(np.array(a.view(mx.int32))).view(torch.uint32).to(device)
    if a.dtype in (mx.float16, mx.bfloat16):
        dtype = torch.float16 if a.dtype == mx.float16 else torch.bfloat16
        return torch.from_numpy(np.array(a.astype(mx.float32))).to(device, dtype)
    return torch.from_numpy(np.array(a)).to(device)


def assert_same(ours, theirs):
    """Bit for bit, compared as float32 (exact for every dtype here)."""
    theirs = to_t(theirs, "cpu")
    assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
    assert ours.dtype == theirs.dtype, (ours.dtype, theirs.dtype)
    torch.testing.assert_close(ours.cpu().float(), theirs.float(), rtol=0, atol=0, equal_nan=True)


def quantized(shape, dtype, mode, group_size=None, bits=None, seed=0):
    """MLX-quantized weights, as (torch tensors, mlx arrays); affine scales/biases in `dtype`."""
    mx.random.seed(seed)
    w = mx.random.normal(shape).astype(mx_dtype(dtype))
    q = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
    if mode != "affine":
        q = [*q, None]
    return [None if a is None else to_t(a) for a in q], list(q)


def gs_bits(mode, group_size=None, bits=None):
    d = MODES[mode]
    return group_size or d[0], bits or d[1]


# --- quantize / dequantize --------------------------------------------------------------------


@needs_mlx
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_quantize_matches_mlx(mode, dtype):
    torch.manual_seed(0)
    w = torch.randn(64, 512, dtype=dtype, device="mps")
    ours = mq.quantize(w, mode=mode)
    theirs = mx.quantize(to_mx(w), mode=mode)
    assert len(ours) == len(theirs)
    for o, t in zip(ours, theirs):
        if t.dtype in (mx.uint32, mx.uint8):
            assert torch.equal(o.cpu(), to_t(t, "cpu"))
        else:
            assert_same(o, t)


@needs_mlx
@pytest.mark.parametrize("group_size,bits", AFFINE)
def test_affine_quantize_dequantize_match_mlx(group_size, bits):
    torch.manual_seed(0)
    w = torch.randn(3, 32, 384, dtype=torch.bfloat16, device="mps")
    wq, s, b = mq.quantize(w, group_size, bits)
    wq_m, s_m, b_m = mx.quantize(to_mx(w), group_size=group_size, bits=bits)
    assert torch.equal(wq.cpu(), to_t(wq_m, "cpu"))
    assert_same(s, s_m)
    assert_same(b, b_m)
    assert_same(mq.dequantize(wq, s, b, group_size, bits), mx.dequantize(wq_m, s_m, b_m, group_size, bits))


@needs_mlx
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ["mxfp4", "mxfp8", "nvfp4"])
def test_fp_dequantize_matches_mlx(mode, dtype):
    (wq, s, _), (wq_m, s_m, _) = quantized((32, 256), torch.float32, mode)
    assert_same(
        mq.dequantize(wq, s, mode=mode, dtype=dtype),
        mx.dequantize(wq_m, s_m, mode=mode, dtype=mx_dtype(dtype)),
    )
    assert_same(mq.dequantize(wq, s, mode=mode), mx.dequantize(wq_m, s_m, mode=mode))  # bfloat16 default


@needs_mlx
def test_nvfp4_global_scale_matches_mlx():
    torch.manual_seed(0)
    w = torch.randn(32, 256, dtype=torch.bfloat16, device="mps")
    g = torch.tensor(3.5, device="mps")
    wq, s = mq.quantize(w, mode="nvfp4", global_scale=g)
    wq_m, s_m = mx.quantize(to_mx(w), mode="nvfp4", global_scale=to_mx(g))
    assert torch.equal(wq.cpu(), to_t(wq_m, "cpu")) and torch.equal(s.cpu(), to_t(s_m, "cpu"))
    assert_same(
        mq.dequantize(wq, s, mode="nvfp4", global_scale=g),
        mx.dequantize(wq_m, s_m, mode="nvfp4", global_scale=to_mx(g)),
    )


# --- quantized_matmul -------------------------------------------------------------------------

# (transpose, M, N, K): row counts either side of the qmv/qmm crossover, K of 64 for qmv_quad,
# N off the 8/32 alignments, and K of 512 / 2048 for qvm vs qvm_split_k.
MATMUL_SHAPES = [
    (True, 1, 1024, 2048),
    (True, 3, 1000, 2048),
    (True, 9, 1024, 2048),
    (True, 16, 1024, 2048),
    (True, 33, 1000, 2048),
    (True, 200, 1024, 2048),
    (True, 600, 512, 1024),
    (True, 2, 512, 64),
    (False, 1, 512, 512),
    (False, 3, 512, 2048),
    (False, 8, 256, 512),
    (False, 40, 512, 1024),
]


@needs_mlx
@pytest.mark.parametrize("transpose,M,N,K", MATMUL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", MODES)
def test_quantized_matmul_matches_mlx(mode, dtype, transpose, M, N, K):
    group_size, bits = gs_bits(mode)
    if K % group_size or (not transpose and N % group_size):
        pytest.skip("shape not divisible by the group size")
    qt, qm = quantized((N, K) if transpose else (K, N), dtype, mode)
    torch.manual_seed(1)
    x = torch.randn(M, K, dtype=dtype, device="mps")
    ours = mq.quantized_matmul(x, *qt, transpose=transpose, mode=mode)
    theirs = mx.quantized_matmul(to_mx(x), *qm, transpose=transpose, mode=mode)
    assert_same(ours, theirs)


@needs_mlx
@pytest.mark.parametrize("group_size,bits", AFFINE)
@pytest.mark.parametrize("M", [1, 5, 64])
@pytest.mark.parametrize("transpose", [True, False])
def test_affine_bits_and_group_sizes_match_mlx(group_size, bits, M, transpose):
    N, K = 256, 1024
    qt, qm = quantized((N, K) if transpose else (K, N), torch.float16, "affine", group_size, bits)
    torch.manual_seed(1)
    x = torch.randn(M, K, dtype=torch.float16, device="mps")
    assert_same(
        mq.quantized_matmul(x, *qt, transpose=transpose, group_size=group_size, bits=bits),
        mx.quantized_matmul(to_mx(x), *qm, transpose=transpose, group_size=group_size, bits=bits),
    )


@needs_mlx
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("M", [1, 3, 40])
@pytest.mark.parametrize("mode", ["affine", "mxfp4"])
def test_batched_weights_match_mlx(mode, M, transpose):
    """3D weights, broadcast against a 4D x; MLX runs the batched kernels for this."""
    N, K = 256, 512
    qt, qm = quantized((3, N, K) if transpose else (3, K, N), torch.float16, mode)
    torch.manual_seed(1)
    x = torch.randn(2, 1, M, K, dtype=torch.float16, device="mps")
    assert_same(
        mq.quantized_matmul(x, *qt, transpose=transpose, mode=mode),
        mx.quantized_matmul(to_mx(x), *qm, transpose=transpose, mode=mode),
    )


@needs_mlx
@pytest.mark.parametrize("reduce_shape", [(16, 1024, 2048), (16, 256, 4096), (8, 32, 32768)])
@pytest.mark.parametrize("dtype", DTYPES)
def test_split_k_sums_match_mlx(reduce_shape, dtype):
    """qmm_t_splitk followed by each of upstream's column sums: small, looped and two-pass."""
    M, N, K = reduce_shape
    qt, qm = quantized((N, K), dtype, "affine")
    x = torch.randn(M, K, dtype=dtype, device="mps")
    trace = ops.trace_quantized_matmul(x, *qt, True, None, None, "affine")
    assert any("splitk" in k for k in trace) and any(k.startswith("col_reduce_") for k in trace), trace
    assert_same(mq.quantized_matmul(x, *qt), mx.quantized_matmul(to_mx(x), *qm))


# --- gather_qmm ---------------------------------------------------------------------------------


@needs_mlx
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "case,transpose,M,expect",
    [
        ("lhs_rhs", True, 1, "gather_qmv"),
        ("lhs_rhs", True, 20, "gather_qmm_t"),
        ("lhs_rhs", False, 1, "gather_qvm"),
        ("lhs_rhs", False, 20, "gather_qmm_n"),
        ("rhs_only", True, 1, "gather_qmv"),
        ("sorted", True, 1, "gather_qmm_rhs_nt"),
        ("sorted", False, 1, "gather_qmm_rhs_nn"),
    ],
)
def test_gather_qmm_matches_mlx(mode, dtype, case, transpose, M, expect):
    E, N, K, T = 8, 256, 512, 64
    qt, qm = quantized((E, N, K) if transpose else (E, K, N), dtype, mode)
    torch.manual_seed(2)
    rhs = torch.randint(0, E, (T,), device="mps")
    if case == "sorted":
        rhs = rhs.sort().values
    if case == "lhs_rhs":
        x = torch.randn(4, M, K, dtype=dtype, device="mps")
        lhs = torch.randint(0, 4, (T,), device="mps")
        kw = dict(lhs_indices=lhs, rhs_indices=rhs)
        kw_m = dict(lhs_indices=mx_indices(lhs), rhs_indices=mx_indices(rhs))
    else:
        x = torch.randn(T, M, K, dtype=dtype, device="mps")
        kw = dict(rhs_indices=rhs, sorted_indices=case == "sorted")
        kw_m = dict(rhs_indices=mx_indices(rhs), sorted_indices=case == "sorted")
    trace = ops.trace_gather_qmm(
        x, *qt, kw.get("lhs_indices"), rhs, transpose, None, None, mode, None, case == "sorted"
    )
    assert any(k.startswith(f"{mode}_{expect}_") for k in trace), trace
    assert_same(
        mq.gather_qmm(x, *qt, transpose=transpose, mode=mode, **kw),
        mx.gather_qmm(to_mx(x), *qm, transpose=transpose, mode=mode, **kw_m),
    )


@needs_mlx
def test_gather_qmm_nvfp4_global_scale_matches_mlx():
    E, N, K, T = 4, 128, 256, 16
    (wq, s, _), (wq_m, s_m, _) = quantized((E, N, K), torch.bfloat16, "nvfp4")
    g = torch.rand(E, device="mps") + 0.5
    x = torch.randn(T, 1, K, dtype=torch.bfloat16, device="mps")
    rhs = torch.randint(0, E, (T,), device="mps")
    assert_same(
        mq.gather_qmm(x, wq, s, rhs_indices=rhs, mode="nvfp4", global_scale=g),
        mx.gather_qmm(to_mx(x), wq_m, s_m, rhs_indices=mx_indices(rhs), mode="nvfp4", global_scale=to_mx(g)),
    )


# --- which kernels run --------------------------------------------------------------------------


def meta_quantized(shape, mode, dtype=torch.float16):
    group_size, bits = gs_bits(mode)
    *batch, rows, cols = shape
    wq = torch.empty(*batch, rows, cols * bits // 32, dtype=torch.uint32, device="meta")
    if mode == "affine":
        s = torch.empty(*batch, rows, cols // group_size, dtype=dtype, device="meta")
        return wq, s, s.clone()
    return wq, torch.empty(*batch, rows, cols // group_size, dtype=torch.uint8, device="meta"), None


# (arch, transpose, M, N, K, expected first kernel). Limits from get_qmv_batch_limit: g14s is 14 up
# to 2048, 10 up to 4096; g15s routes 2+ rows to qmv_wide; g17s (M5) runs NAX for qmm.
BRANCHES = [
    ("applegpu_g14s", True, 1, 512, 64, "affine_qmv_quad_"),
    ("applegpu_g14s", True, 1, 1024, 1024, "affine_qmv_fast_"),
    ("applegpu_g14s", True, 5, 1004, 1024, "affine_qmv_float"),
    ("applegpu_g15s", True, 5, 1024, 1024, "affine_qmv_wide_"),
    ("applegpu_g14s", True, 16, 1024, 1024, "affine_qmm_t_splitk_"),
    ("applegpu_g14s", True, 300, 1024, 1024, "affine_qmm_t_float"),
    ("applegpu_g14s", False, 2, 1024, 512, "affine_qvm_float"),
    ("applegpu_g14s", False, 2, 1024, 2048, "affine_qvm_split_k_"),
    ("applegpu_g14s", False, 4, 1024, 512, "affine_qmm_n_float"),
    ("applegpu_g17s", True, 300, 1024, 1024, "affine_qmm_t_nax_"),
    ("applegpu_g17s", False, 300, 1024, 1024, "affine_qmm_n_nax_"),
]


@pytest.mark.kernels_ci
@pytest.mark.parametrize("arch_name,transpose,M,N,K,expected", BRANCHES)
def test_dispatch_branches(arch_name, transpose, M, N, K, expected):
    w, s, b = meta_quantized((N, K) if transpose else (K, N), "affine")
    x = torch.empty(M, K, dtype=torch.float16, device="meta")
    with arch(arch_name):
        trace = ops.trace_quantized_matmul(x, w, s, b, transpose, None, None, "affine")
    assert trace[0].startswith(expected), trace


def test_every_traced_kernel_is_in_the_metallib():
    """Names the dispatch builds for other GPUs (NAX on M5, qmv_wide on M3+) must exist in the build,
    even where this GPU cannot run them. The metallib is embedded in the extension."""
    so = next(Path(mq.__file__).parent.glob("_mlx_quantization_metal_kernels*.so"))
    blob = so.read_bytes()
    names = set()
    for arch_name in ("applegpu_g14s", "applegpu_g15s", "applegpu_g17s"):
        with arch(arch_name):
            for mode in MODES:
                for transpose, M, N, K in [
                    (True, 1, 1024, 1024),
                    (True, 5, 1024, 1024),
                    (True, 300, 1024, 1024),
                    (False, 2, 1024, 2048),
                    (False, 300, 1024, 1024),
                ]:
                    w, s, b = meta_quantized((N, K) if transpose else (K, N), mode)
                    x = torch.empty(M, K, dtype=torch.bfloat16, device="meta")
                    names.update(ops.trace_quantized_matmul(x, w, s, b, transpose, None, None, mode))
                w, s, b = meta_quantized((8, 1024, 1024), mode)
                rhs = torch.zeros(64, dtype=torch.int32, device="meta")
                for M, sort in [(1, True), (1, False), (40, False)]:
                    x = torch.empty(64, M, 1024, dtype=torch.bfloat16, device="meta")
                    names.update(ops.trace_gather_qmm(x, w, s, b, None, rhs, True, None, None, mode, None, sort))
    missing = sorted(n for n in names if n.encode() not in blob)
    assert len(names) > 40 and not missing, missing


# --- the version 1 API ------------------------------------------------------------------------


@pytest.mark.kernels_ci
def test_v1_functions():
    torch.manual_seed(0)
    N, K = 256, 512
    x = torch.randn(4, K, dtype=torch.float16, device="mps")
    w = torch.randn(N, K, dtype=torch.float16, device="mps")
    wq, s, b = mq.quantize(w, 128, 4)
    y = mq.quantized_matmul(x, wq, s, b, group_size=128)
    for f in (mq.affine_qmm_t, mq.affine_qmm_t_nax):
        torch.testing.assert_close(f(x, wq, s, b), y, rtol=0, atol=0)
    torch.testing.assert_close(mq.affine_qmv(x, wq, s, b, N), y, rtol=0, atol=0)

    wq_n, s_n, b_n = mq.quantize(w.T.contiguous(), 128, 4)  # [K, N]
    y_n = mq.quantized_matmul(x, wq_n, s_n, b_n, transpose=False, group_size=128)
    for f in (mq.affine_qmm_n, mq.affine_qmm_n_nax):
        torch.testing.assert_close(f(x, wq_n, s_n, b_n, N), y_n, rtol=0, atol=0)

    wq4, s4 = mq.quantize(w, mode="mxfp4")
    torch.testing.assert_close(mq.mxfp4_qmv(x, wq4, s4, N), mq.quantized_matmul(x, wq4, s4, mode="mxfp4"))
    wq4n, s4n = mq.quantize(w.T.contiguous(), mode="mxfp4")
    torch.testing.assert_close(
        mq.mxfp4_qmm_n(x, wq4n, s4n, N), mq.quantized_matmul(x, wq4n, s4n, transpose=False, mode="mxfp4")
    )

    we = torch.randn(4, N, K, dtype=torch.float16, device="mps")
    wqe, se, be = mq.quantize(we, 128, 4)
    idx = torch.tensor([3, 0, 0, 2], device="mps")
    y_g = mq.affine_gather_qmm_rhs_nax(x, wqe, se, be, idx, N)
    for m in range(4):
        e = int(idx[m])
        torch.testing.assert_close(y_g[m], mq.quantized_matmul(x[m], wqe[e], se[e], be[e], group_size=128))

    with pytest.raises(ValueError, match="output_features"):
        mq.affine_qmv(x, wq, s, b, N + 1)


# --- shapes and layouts ---------------------------------------------------------------------------


@pytest.mark.kernels_ci
@pytest.mark.parametrize("rows", [1, 3, 40])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_input_matches_per_row(rows, dtype):
    """huggingface/transformers#49337: every row of a 3D input must equal that row run alone."""
    w = torch.randn(256, 512, dtype=dtype, device="mps")
    wq, s, b = mq.quantize(w, 64, 8)
    x = torch.randn(4, rows, 512, dtype=dtype, device="mps")
    batched = mq.quantized_matmul(x, wq, s, b, bits=8)
    per_row = torch.stack([mq.quantized_matmul(r, wq, s, b, bits=8) for r in x])
    assert batched.shape == (4, rows, 256)
    torch.testing.assert_close(batched, per_row, rtol=0, atol=0)


def test_layouts():
    w = torch.randn(256, 512, dtype=torch.float16, device="mps")
    wq, s, b = mq.quantize(w)
    big = torch.randn(10, 512, dtype=torch.float16, device="mps")
    y = mq.quantized_matmul(big[3:7].clone(), wq, s, b)
    torch.testing.assert_close(mq.quantized_matmul(big[3:7], wq, s, b), y, rtol=0, atol=0)  # storage offset
    xt = big[3:7].clone().T.contiguous().T  # same values, column-major
    assert not xt.is_contiguous()
    torch.testing.assert_close(mq.quantized_matmul(xt, wq, s, b), y, rtol=0, atol=0)
    assert mq.quantized_matmul(big[3], wq, s, b).shape == (256,)  # 1D x
    assert mq.quantized_matmul(big[:0], wq, s, b).shape == (0, 256)  # empty


@pytest.mark.kernels_ci
@pytest.mark.parametrize(
    "case", ["scales_dtype", "w_dtype", "w_shape", "group_size", "bits", "mode", "biases_fp", "cpu"]
)
def test_bad_inputs_raise(case):
    w = torch.randn(128, 256, dtype=torch.float16, device="mps")
    wq, s, b = mq.quantize(w)
    x = torch.randn(2, 256, dtype=torch.float16, device="mps")
    kw = dict(group_size=None, bits=None, mode="affine")
    if case == "scales_dtype":
        s, b = s.to(torch.int32), b.to(torch.int32)
    elif case == "w_dtype":
        wq = wq.view(torch.int32)
    elif case == "w_shape":
        wq = wq[:, :-1]
    elif case == "group_size":
        kw["group_size"] = 16
    elif case == "bits":
        kw["bits"] = 7
    elif case == "mode":
        kw["mode"] = "int4"
    elif case == "biases_fp":
        kw["mode"] = "mxfp4"
    elif case == "cpu":
        x = x.cpu()
    with pytest.raises(RuntimeError, match="mlx-quantization-metal-kernels"):
        mq.quantized_matmul(x, wq, s, b, **kw)


# --- qqmm / gather_qqmm -------------------------------------------------------------------------

QQ_MODES = ["nvfp4", "mxfp8", "mxfp4"]


def global_scale(t):
    """An nvfp4 global scale for `t`, as a float32 scalar on mps (amax / (fp8 max * fp4 max))."""
    return (t.float().abs().max() / (448.0 * 6.0)).reshape(())


def qq_weight(w, mode, quantize_w, gw=None):
    """(ours, mlx) weight arguments for qqmm: as is, or quantized by MLX (with `gw` for nvfp4)."""
    if not quantize_w:
        return (w, None), (to_mx(w), None)
    kw = {} if gw is None else dict(global_scale=to_mx(gw))
    wq, s = mx.quantize(to_mx(w), mode=mode, **kw)
    return (to_t(wq), to_t(s)), (wq, s)


@needs_mlx
@pytest.mark.parametrize("quantize_w", [True, False])
@pytest.mark.parametrize("M", [1, 4, 32, 512])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", QQ_MODES)
def test_qqmm_matches_mlx(mode, dtype, M, quantize_w):
    N, K = 256, 512
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=dtype, device="mps")
    w = torch.randn(N, K, dtype=dtype, device="mps")
    (wt, st), (wm, sm) = qq_weight(w, mode, quantize_w)
    assert_same(mq.qqmm(x, wt, st, mode=mode), mx.qqmm(to_mx(x), wm, sm, mode=mode))


@needs_mlx
@pytest.mark.parametrize("quantize_w", [True, False])
@pytest.mark.parametrize("M", [1, 4, 32, 512])
@pytest.mark.parametrize("dtype", DTYPES)
def test_qqmm_nvfp4_global_scales_match_mlx(dtype, M, quantize_w):
    """With both global scales; a quantized bf16 weight takes qmm past the qmv limit, qmv otherwise."""
    N, K = 256, 512
    torch.manual_seed(1)
    x = torch.randn(M, K, dtype=dtype, device="mps")
    w = torch.randn(N, K, dtype=dtype, device="mps")
    gx, gw = global_scale(x), global_scale(w)
    (wt, st), (wm, sm) = qq_weight(w, "nvfp4", quantize_w, gw)
    with arch("applegpu_g14s"):  # qmv up to 17 rows at this size (get_qmv_batch_limit)
        trace = ops.trace_qqmm(x, wt, st, None, None, "nvfp4", gx, gw)
    qmm = quantize_w and dtype == torch.bfloat16 and M >= 32
    assert any(k.startswith("nvfp4_qmm_t_") and k.endswith("_hgs") for k in trace) == qmm, trace
    assert_same(
        mq.qqmm(x, wt, st, mode="nvfp4", global_scale_x=gx, global_scale_w=gw),
        mx.qqmm(to_mx(x), wm, sm, mode="nvfp4", global_scale_x=to_mx(gx), global_scale_w=to_mx(gw)),
    )


@needs_mlx
@pytest.mark.parametrize("shape", [(256,), (2, 3, 256), (2, 1, 5, 256)])
def test_qqmm_flattens_x_like_mlx(shape):
    torch.manual_seed(2)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="mps")
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="mps")
    (wt, st), (wm, sm) = qq_weight(w, "nvfp4", True)
    assert_same(mq.qqmm(x, wt, st), mx.qqmm(to_mx(x), wm, sm))


@needs_mlx
@pytest.mark.parametrize("quantize_w", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "mode,global_scales", [("nvfp4", False), ("nvfp4", True), ("mxfp8", False), ("mxfp4", False)]
)
@pytest.mark.parametrize(
    "case,M",
    [("lhs_rhs", 1), ("lhs_rhs", 32), ("rhs_only", 1), ("rhs_only", 32), ("sorted", 1)],
)
def test_gather_qqmm_matches_mlx(case, M, mode, global_scales, dtype, quantize_w):
    E, N, K, T = 8, 256, 512, 64
    torch.manual_seed(3)
    w = torch.randn(E, N, K, dtype=dtype, device="mps")
    rhs = torch.randint(0, E, (T,), device="mps")
    if case == "sorted":
        rhs = rhs.sort().values
    if case == "lhs_rhs":
        x = torch.randn(4, M, K, dtype=dtype, device="mps")
        lhs = torch.randint(0, 4, (T,), device="mps")
        kw, kw_m = dict(lhs_indices=lhs, rhs_indices=rhs), dict(lhs_indices=mx_indices(lhs), rhs_indices=mx_indices(rhs))
    else:
        x = torch.randn(T, M, K, dtype=dtype, device="mps")
        lhs = None
        kw = dict(rhs_indices=rhs, sorted_indices=case == "sorted")
        kw_m = dict(rhs_indices=mx_indices(rhs), sorted_indices=case == "sorted")
    gx = gw = None
    if global_scales:
        gx, gw = global_scale(x), global_scale(w)
        kw.update(global_scale_x=gx, global_scale_w=gw)
        kw_m.update(global_scale_x=to_mx(gx), global_scale_w=to_mx(gw))
    (wt, st), (wm, sm) = qq_weight(w, mode, quantize_w, gw)

    # the matrix kernels need both global scales, a quantized bf16 weight and K % 32 == 0
    matrix = global_scales and quantize_w and dtype == torch.bfloat16
    expect = "gather_qmv_" if not matrix else {"sorted": "gather_qmm_rhs_nt_", "lhs_rhs": None}.get(case, None)
    if matrix and expect is None:
        expect = "gather_qmm_t_" if M == 32 else "gather_qmv_"
    trace = ops.trace_gather_qqmm(x, wt, st, lhs, rhs, None, None, mode, gx, gw, case == "sorted")
    assert any(k.startswith(f"{mode}_{expect}") for k in trace), trace
    assert_same(
        mq.gather_qqmm(x, wt, st, mode=mode, **kw),
        mx.gather_qqmm(to_mx(x), wm, sm, mode=mode, **kw_m),
    )


@needs_mlx
def test_gather_qqmm_default_indices_match_mlx():
    torch.manual_seed(4)
    x = torch.randn(4, 3, 256, dtype=torch.bfloat16, device="mps")
    w = torch.randn(4, 64, 256, dtype=torch.bfloat16, device="mps")
    (wt, st), (wm, sm) = qq_weight(w, "nvfp4", True)
    assert_same(mq.gather_qqmm(x, wt, st), mx.gather_qqmm(to_mx(x), wm, sm))


@pytest.mark.kernels_ci
@pytest.mark.parametrize("mode", QQ_MODES)
def test_qqmm_quantizes_an_unquantized_weight_like_quantize(mode):
    """A plain `w` is quantized on the fly with `quantize`'s kernel, so the result is the same."""
    torch.manual_seed(5)
    x = torch.randn(3, 256, dtype=torch.bfloat16, device="mps")
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="mps")
    wq, s = mq.quantize(w, mode=mode)
    ours = mq.qqmm(x, w, mode=mode)
    assert ours.shape == (3, 64) and ours.dtype == torch.bfloat16
    assert torch.equal(ours, mq.qqmm(x, wq, s, mode=mode))
    # and close to the unquantized product (both sides lose precision)
    ref = x.float() @ w.float().T
    assert (ours.float() - ref).abs().max() < 0.5 * ref.abs().max()


@pytest.mark.kernels_ci
def test_qqmm_dispatch():
    """The kernels each path launches: quantize the weight if needed, quantize-dequantize x, matmul."""
    w = torch.empty(1024, 1024, dtype=torch.bfloat16, device="meta")
    wq = torch.empty(1024, 1024 * 4 // 32, dtype=torch.uint32, device="meta")
    s = torch.empty(1024, 1024 // 16, dtype=torch.uint8, device="meta")
    g = torch.empty((), dtype=torch.float32, device="meta")
    with arch("applegpu_g14s"):
        x = torch.empty(1, 1024, dtype=torch.bfloat16, device="meta")
        assert ops.trace_qqmm(x, w, None, None, None, "nvfp4", None, None) == [
            "nvfp4_quantize_bfloat16_t_gs_16_b_4_hgs_false",
            "nvfp4_quantize_dequantize_bfloat16_t_gs_16_b_4_hgs_false",
            "nvfp4_qmv_fast_bfloat16_t_gs_16_b_4_batch_0",
        ]
        x = torch.empty(300, 1024, dtype=torch.bfloat16, device="meta")
        assert ops.trace_qqmm(x, wq, s, None, None, "nvfp4", g, g) == [
            "nvfp4_quantize_dequantize_bfloat16_t_gs_16_b_4_hgs_true",
            "nvfp4_qmm_t_bfloat16_t_gs_16_b_4_alN_true_batch_0_hgs",
        ]
        # without the global scales the matrix kernel is not used, however large M is
        assert ops.trace_qqmm(x, wq, s, None, None, "nvfp4", None, None)[-1].startswith("nvfp4_qmv_wide_")


def test_every_traced_qqmm_kernel_is_in_the_metallib():
    so = next(Path(mq.__file__).parent.glob("_mlx_quantization_metal_kernels*.so"))
    blob = so.read_bytes()
    names = set()
    for arch_name in ("applegpu_g14s", "applegpu_g15s", "applegpu_g17s"):
        with arch(arch_name):
            for mode in QQ_MODES:
                g = torch.empty((), dtype=torch.float32, device="meta") if mode == "nvfp4" else None
                for dtype in DTYPES:
                    for M in (1, 5, 300):
                        x = torch.empty(M, 1024, dtype=dtype, device="meta")
                        w = torch.empty(1024, 1024, dtype=dtype, device="meta")
                        names.update(ops.trace_qqmm(x, w, None, None, None, mode, g, g))
                        wq, s, _ = meta_quantized((1024, 1024), mode)
                        names.update(ops.trace_qqmm(x, wq, s, None, None, mode, g, g))
                    wq, s, _ = meta_quantized((8, 1024, 1024), mode)
                    rhs = torch.zeros(64, dtype=torch.int32, device="meta")
                    for M, sort in [(1, True), (1, False), (40, False)]:
                        x = torch.empty(64, M, 1024, dtype=dtype, device="meta")
                        names.update(ops.trace_gather_qqmm(x, wq, s, None, rhs, None, None, mode, g, g, sort))
    missing = sorted(n for n in names if n.encode() not in blob)
    assert len(names) > 40 and not missing, missing


@pytest.mark.kernels_ci
@pytest.mark.parametrize("case", ["affine", "w_3d", "one_global_scale", "global_scale_mxfp8", "no_scales", "groups"])
def test_qqmm_bad_inputs_raise(case):
    x = torch.randn(2, 256, dtype=torch.bfloat16, device="mps")
    w = torch.randn(64, 256, dtype=torch.bfloat16, device="mps")
    g = torch.ones((), dtype=torch.float32, device="mps")
    args, kw = (x, w), {}
    if case == "affine":
        kw = dict(mode="affine")
    elif case == "w_3d":
        args = (x, w[None])
    elif case == "one_global_scale":
        kw = dict(global_scale_x=g)
    elif case == "global_scale_mxfp8":
        kw = dict(mode="mxfp8", global_scale_x=g, global_scale_w=g)
    elif case == "no_scales":
        args = (x, mq.quantize(w, mode="nvfp4")[0])
    elif case == "groups":
        args = (x[:, :250], w[:, :250])
    with pytest.raises(RuntimeError, match="mlx-quantization-metal-kernels"):
        mq.qqmm(*args, **kw)
