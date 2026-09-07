# Smoke tests for the kernels-community CI runner (`nix run .#ci-test`, which
# runs `pytest -m kernels_ci`).
#
# The rest of the suite is vendored from upstream: it sweeps hundreds of shapes
# and benchmarks each one with `bench_kineto`, which is far too slow for CI. The
# tests here are deliberately standalone -- small shapes, no benchmarking, no
# shared helpers -- and they all run on any CUDA device.
#
# That last point constrains what can be covered. DeepGEMM's GEMMs are Hopper
# and Blackwell only, on two counts: the entry points dispatch on `arch_major`
# being 9 or 10 and assert otherwise, and the runtime JIT always compiles with
# `-arch=sm_<cc>a`, a suffix nvcc only accepts from sm90 on ("Unsupported gpu
# architecture 'sm_89a'"). So this file covers what is reachable without a
# Hopper GPU and without JIT compilation:
#
#   - the cuBLASLt GEMMs and the cuBLASLt einsum path, which call the library
#     directly;
#   - the non-JIT paths of the scaling-factor layout API;
#   - the runtime configuration surface (SM count, TC util, PDL, alignment);
#   - the operator/fake-tensor registration of the Hopper-only GEMMs, which is
#     checked under `FakeTensorMode` and so needs no kernel launch.

import kernels
import pytest
import torch

# `kernels-community/deep-gemm` is `[general.hub] repo-id` and 2 is
# `[general] version` in `build.toml`.
deep_gemm = kernels.get_kernel("kernels-community/deep-gemm", version=2)

pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]

M, N, K = 128, 512, 512


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Upstream's error metric: 1 - cosine similarity."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    return 0.0 if denominator == 0 else (1 - 2 * (x * y).sum() / denominator).item()


def tma_aligned(mn: int, element_size: int = 4) -> int:
    """Rounds `mn` up to a 16-byte TMA boundary.

    Mirrors the kernel's `get_tma_aligned_size`, which cannot be called from
    Python: it takes only ints but is registered for `torch::kCUDA`, so the
    dispatcher has no tensor argument to pick a backend from and raises.
    """
    elems_per_16b = 16 // element_size
    return -(-mn // elems_per_16b) * elems_per_16b


# ── cuBLASLt GEMMs ─────────────────────────────────────────────────────────
# `nn`/`tn`/`tt` are thin transposing wrappers around `nt`, so the operands are
# built to make every variant compute the same `[M, K] @ [N, K].T` product.

@pytest.mark.parametrize("variant", ["nt", "nn", "tn", "tt"])
def test_cublaslt_gemm(variant: str) -> None:
    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
    d = torch.empty((M, N), device='cuda', dtype=torch.bfloat16)
    ref_d = a.float() @ b.float().t()

    operands = {
        'nt': (a, b),
        'nn': (a, b.t().contiguous()),
        'tn': (a.t().contiguous(), b.t().contiguous()),
        'tt': (a.t().contiguous(), b),
    }[variant]
    getattr(deep_gemm, f'cublaslt_gemm_{variant}')(*operands, d)

    diff = calc_diff(d, ref_d)
    assert diff < 1e-5, f'{diff:.7f}'


# ── einsum, cuBLASLt path ──────────────────────────────────────────────────
# `bhr,hdr->bhd` and `bhd,hdr->bhr` check `use_cublaslt` before dispatching on
# the architecture, so they are the only reachable einsum expressions here
# (`bmk,bnk->mn` has no cuBLASLt path and asserts on anything below sm90).

@pytest.mark.parametrize("expr", ["bhr,hdr->bhd", "bhd,hdr->bhr"])
def test_einsum_cublaslt(expr: str) -> None:
    torch.manual_seed(0)
    b_, h, r, d_ = 4, 8, 512, 128
    lhs_last, out_last = (r, d_) if expr == "bhr,hdr->bhd" else (d_, r)

    x = torch.randn((b_, h, lhs_last), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((h, d_, r), device='cuda', dtype=torch.bfloat16)
    z = torch.empty((b_, h, out_last), device='cuda', dtype=torch.bfloat16)

    deep_gemm.einsum(expr, x, y, z, use_cublaslt=True)

    diff = calc_diff(z, torch.einsum(expr, x, y))
    assert diff < 1e-5, f'{diff:.7f}'


# ── scaling-factor layout, non-JIT paths ───────────────────────────────────
# `get_mn_major_tma_aligned_tensor` restrides scaling factors into an MN-major,
# TMA-aligned layout. It returns the input untouched when it already has that
# layout, and falls back to a PyTorch copy when the input is not contiguous;
# only the contiguous-but-unaligned case needs the JIT. `mn` is deliberately not
# a multiple of the 4-element TMA boundary so the padding is exercised.

def test_mn_major_tma_aligned_tensor_already_aligned() -> None:
    mn, sf_k = 129, 4
    storage = torch.randn((sf_k, tma_aligned(mn)), device='cuda', dtype=torch.float)
    sf = storage[:, :mn].t()
    assert sf.stride() == (1, tma_aligned(mn))

    out = deep_gemm.get_mn_major_tma_aligned_tensor(sf)

    assert out.data_ptr() == sf.data_ptr(), "should have been returned as-is"
    assert torch.equal(out, sf)


@pytest.mark.parametrize("num_groups", [1, 2])
def test_mn_major_tma_aligned_tensor_non_contiguous(num_groups: int) -> None:
    mn, sf_k = 129, 4
    shape = (mn, sf_k + 1) if num_groups == 1 else (num_groups, mn, sf_k + 1)
    sf = torch.randn(shape, device='cuda', dtype=torch.float)[..., :sf_k]
    assert not sf.is_contiguous()

    out = deep_gemm.get_mn_major_tma_aligned_tensor(sf)

    aligned_mn = tma_aligned(mn)
    assert out.shape == sf.shape
    assert out.stride()[-2:] == (1, aligned_mn)
    if num_groups > 1:
        assert out.stride(0) == aligned_mn * sf_k
    assert torch.equal(out, sf)


# ── runtime configuration ──────────────────────────────────────────────────

def test_num_sms() -> None:
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    assert deep_gemm.get_num_sms() == total

    deep_gemm.set_num_sms(total // 2)
    assert deep_gemm.get_num_sms() == total // 2
    deep_gemm.set_num_sms(total)

    with pytest.raises(RuntimeError):
        deep_gemm.set_num_sms(total + 1)
    assert deep_gemm.get_num_sms() == total


def test_tc_util() -> None:
    original = deep_gemm.get_tc_util()
    deep_gemm.set_tc_util(50)
    assert deep_gemm.get_tc_util() == 50
    deep_gemm.set_tc_util(original)

    with pytest.raises(RuntimeError):
        deep_gemm.set_tc_util(101)
    assert deep_gemm.get_tc_util() == original


def test_pdl() -> None:
    original = deep_gemm.get_pdl()
    deep_gemm.set_pdl(not original)
    assert deep_gemm.get_pdl() is not original
    deep_gemm.set_pdl(original)
    assert deep_gemm.get_pdl() is original


def test_mk_alignment_for_contiguous_layout() -> None:
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    assert alignment > 0 and alignment % 8 == 0
    assert deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(256) > 0

    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    assert deep_gemm.get_mk_alignment_for_contiguous_layout() == alignment
    # Upstream aliases, kept for backwards compatibility.
    assert deep_gemm.get_m_alignment_for_contiguous_layout() == alignment
    assert deep_gemm.get_k_alignment_for_contiguous_layout() == alignment


# ── operator registration of the Hopper-only GEMMs ─────────────────────────
# `FakeTensorMode` runs the registered fake implementations instead of the
# kernels, so this checks the op schemas and the `register_fake` block in the
# kernel's `__init__.py` on a GPU that could never launch them.

def test_gemm_ops_registered() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        a = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
        b = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
        d = torch.empty((M, N), device='cuda', dtype=torch.bfloat16)

        deep_gemm.bf16_gemm_nt(a, b, d)
        deep_gemm.bf16_gemm_nn(a, b.t(), d)

        a_fp8 = (torch.empty((M, K), device='cuda', dtype=torch.float8_e4m3fn),
                 torch.empty((M, K // 128), device='cuda', dtype=torch.float))
        b_fp8 = (torch.empty((N, K), device='cuda', dtype=torch.float8_e4m3fn),
                 torch.empty((N // 128, K // 128), device='cuda', dtype=torch.float))
        deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d)

        num_groups = 2
        grouped_a = torch.randn((num_groups * M, K), device='cuda', dtype=torch.bfloat16)
        grouped_b = torch.randn((num_groups, N, K), device='cuda', dtype=torch.bfloat16)
        grouped_d = torch.empty((num_groups * M, N), device='cuda', dtype=torch.bfloat16)
        layout = torch.zeros((num_groups * M,), device='cuda', dtype=torch.int32)
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(grouped_a, grouped_b, grouped_d, layout)
