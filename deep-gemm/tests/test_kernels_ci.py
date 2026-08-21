# Smoke tests for the kernels-community CI runner (`nix run .#ci-test`, which
# runs `pytest -m kernels_ci`).
#
# The rest of the suite is vendored from upstream: it sweeps hundreds of shapes
# and benchmarks each one with `bench_kineto`, which is far too slow for CI. The
# tests here are deliberately standalone -- one small shape per kernel family, no
# benchmarking, no shared helpers -- so that CI runtime is dominated by
# DeepGEMM's runtime JIT compilation rather than by the GEMMs themselves.
#
# One test per kernel family: the cuBLASLt fallback, the layout kernels, and the
# BF16, FP8 and grouped GEMMs that are DeepGEMM's reason to exist. Everything
# except cuBLASLt is skipped below sm90 -- see `requires_hopper` -- so on a
# pre-Hopper runner this file is close to a load-and-call check.

import kernels
import pytest
import torch

# `kernels-community/deep-gemm` is `[general.hub] repo-id` and 2 is
# `[general] version` in `build.toml`.
deep_gemm = kernels.get_kernel("kernels-community/deep-gemm", version=2)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

# DeepGEMM needs Hopper or newer on two counts: the GEMM entry points dispatch
# on `arch_major` being 9 (Hopper) or 10 (Blackwell) and assert otherwise, and
# the runtime JIT always compiles with `-arch=sm_<cc>a`, a suffix nvcc only
# accepts from sm90 on ("Unsupported gpu architecture 'sm_89a'"). That leaves
# cuBLASLt -- which calls the library directly and JITs nothing -- as the only
# kernel this file can exercise on an older GPU.
requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="DeepGEMM requires Hopper (sm90) or newer",
)

# Small enough to keep the kernels themselves negligible, large enough to stay
# on the block sizes the heuristics pick for real shapes.
M, N, K = 128, 512, 512


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Upstream's error metric: 1 - cosine similarity, so that it is meaningful
    for the quantized kernels too."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    return 0.0 if denominator == 0 else (1 - 2 * (x * y).sum() / denominator).item()


def randn_ab(m: int = M, n: int = N, k: int = K):
    """`[m, k] @ [n, k].T` operands plus the FP32 reference product."""
    torch.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    return a, b, d, a.float() @ b.float().t()


@pytest.mark.kernels_ci
@requires_cuda
def test_cublaslt_gemm_nt() -> None:
    a, b, d, ref_d = randn_ab()

    deep_gemm.cublaslt_gemm_nt(a, b, d)

    diff = calc_diff(d, ref_d)
    assert diff < 1e-5, f'{diff:.7f}'


@pytest.mark.kernels_ci
@requires_cuda
@requires_hopper
def test_sf_layout_transpose() -> None:
    # `get_mn_major_tma_aligned_tensor` is architecture-agnostic
    # (`smxx_layout`) and is the cheapest kernel that goes through the runtime
    # JIT compiler, so this doubles as a check that JIT compilation works.
    torch.manual_seed(0)
    x = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    _, fp32_sf = deep_gemm.utils.per_token_cast_to_fp8(x, use_ue8m0=False)

    transposed_sf = deep_gemm.get_mn_major_tma_aligned_tensor(fp32_sf)

    # The kernel only restrides the scaling factors into an MN-major, TMA
    # aligned layout, so the values must come back untouched.
    mn, sf_k = fp32_sf.shape
    assert transposed_sf.shape == (mn, sf_k)
    assert transposed_sf.stride() == (1, deep_gemm.get_tma_aligned_size(mn, 4))
    assert torch.equal(fp32_sf, transposed_sf)


@pytest.mark.kernels_ci
@requires_cuda
@requires_hopper
def test_bf16_gemm_nt() -> None:
    a, b, d, ref_d = randn_ab()

    deep_gemm.bf16_gemm_nt(a, b, d)

    diff = calc_diff(d, ref_d)
    assert diff < 1e-5, f'{diff:.7f}'


@pytest.mark.kernels_ci
@requires_cuda
@requires_hopper
def test_fp8_gemm_nt() -> None:
    a, b, d, ref_d = randn_ab()

    # Hopper runs the 1D2D kernel with FP32 scaling factors, Blackwell the 1D1D
    # kernel with UE8M0 ones. Leaving the recipes unset lets the kernel pick the
    # default for the scaling factor dtypes it is handed.
    use_ue8m0 = torch.cuda.get_device_capability()[0] >= 10
    a_fp8 = deep_gemm.utils.per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = deep_gemm.utils.per_block_cast_to_fp8(b, use_ue8m0=use_ue8m0)

    deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d, disable_ue8m0_cast=not use_ue8m0)

    # Compared against the unquantized reference, so the tolerance has to absorb
    # the FP8 quantization error -- same threshold the upstream sweep uses.
    diff = calc_diff(d, ref_d)
    assert diff < 0.001, f'{diff:.7f}'


@pytest.mark.kernels_ci
@requires_cuda
@requires_hopper
def test_m_grouped_bf16_gemm_nt_contiguous() -> None:
    num_groups = 2

    # Rows of `a` are grouped in blocks of this alignment; using exactly one
    # block per group keeps the layout free of padding rows (which would be
    # marked with -1 in `grouped_layout`).
    m_per_group = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(m_per_group)

    a, b, d, _ = randn_ab(m=num_groups * m_per_group)
    b = b.unsqueeze(0).repeat(num_groups, 1, 1).contiguous()
    b[1] = -b[1]  # so that a wrong group index cannot pass unnoticed
    grouped_layout = torch.arange(
        num_groups, device='cuda', dtype=torch.int32
    ).repeat_interleave(m_per_group)
    ref_d = torch.cat([
        a[i * m_per_group:(i + 1) * m_per_group].float() @ b[i].float().t()
        for i in range(num_groups)
    ])

    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout)

    diff = calc_diff(d, ref_d)
    assert diff < 1e-5, f'{diff:.7f}'
