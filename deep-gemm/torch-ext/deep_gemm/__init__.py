import os
import subprocess
import torch

# Import the compiled extension
from ._ops import ops
from . import utils

__version__ = "2.3.0"


# Runtime


def set_num_sms(num_sms: int):
    ops.set_num_sms(num_sms)


def get_num_sms() -> int:
    return ops.get_num_sms()


def set_tc_util(tc_util: int):
    ops.set_tc_util(tc_util)


def get_tc_util() -> int:
    return ops.get_tc_util()


def get_mk_alignment_for_contiguous_layout() -> int:
    return ops.get_mk_alignment_for_contiguous_layout()


# Layout utilities


def get_tma_aligned_size(mn: int, element_size: int) -> int:
    return ops.get_tma_aligned_size(mn, element_size).item()


def get_mn_major_tma_aligned_tensor(sf):
    return ops.get_mn_major_tma_aligned_tensor(sf)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf):
    return ops.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks):
    ks_int = torch.tensor(ks, dtype=torch.int32, device="cpu")
    return ops.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        sf, ks_tensor, ks_int
    )


def transform_sf_into_required_layout(
    sf,
    mn,
    k,
    recipe=None,
    recipe_ab=None,
    num_groups=None,
    is_sfa=False,
    disable_ue8m0_cast=False,
):
    has_recipe = recipe is not None
    r0, r1, r2 = recipe if has_recipe else (0, 0, 0)
    has_recipe_ab = recipe_ab is not None
    rab0, rab1 = recipe_ab if has_recipe_ab else (0, 0)
    has_ng = num_groups is not None
    ng = num_groups if has_ng else 0
    return ops.transform_sf_into_required_layout(
        sf,
        mn,
        k,
        r0,
        r1,
        r2,
        has_recipe,
        rab0,
        rab1,
        has_recipe_ab,
        ng,
        has_ng,
        is_sfa,
        disable_ue8m0_cast,
    )


# Aliases for contiguous layout alignment
get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout


# Helper to flatten recipe args


def _flatten_recipe(recipe, recipe_a=None, recipe_b=None):
    has_recipe = recipe is not None
    r0, r1, r2 = recipe if has_recipe else (0, 0, 0)
    has_ra = recipe_a is not None
    ra0, ra1 = recipe_a if has_ra else (0, 0)
    has_rb = recipe_b is not None
    rb0, rb1 = recipe_b if has_rb else (0, 0)
    return r0, r1, r2, has_recipe, ra0, ra1, has_ra, rb0, rb1, has_rb


# FP8/FP4 GEMM ops


def fp8_fp4_gemm_nt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2, hr, ra0, ra1, hra, rb0, rb1, hrb = _flatten_recipe(
        recipe, recipe_a, recipe_b
    )
    ops.fp8_fp4_gemm_nt(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        c,
        r0,
        r1,
        r2,
        hr,
        ra0,
        ra1,
        hra,
        rb0,
        rb1,
        hrb,
        compiled_dims,
        disable_ue8m0_cast,
    )


def fp8_fp4_gemm_nn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2, hr, ra0, ra1, hra, rb0, rb1, hrb = _flatten_recipe(
        recipe, recipe_a, recipe_b
    )
    ops.fp8_fp4_gemm_nn(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        c,
        r0,
        r1,
        r2,
        hr,
        ra0,
        ra1,
        hra,
        rb0,
        rb1,
        hrb,
        compiled_dims,
        disable_ue8m0_cast,
    )


def fp8_fp4_gemm_tn(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="mn",
    disable_ue8m0_cast=False,
):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2, hr, ra0, ra1, hra, rb0, rb1, hrb = _flatten_recipe(
        recipe, recipe_a, recipe_b
    )
    ops.fp8_fp4_gemm_tn(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        c,
        r0,
        r1,
        r2,
        hr,
        ra0,
        ra1,
        hra,
        rb0,
        rb1,
        hrb,
        compiled_dims,
        disable_ue8m0_cast,
    )


def fp8_fp4_gemm_tt(
    a,
    b,
    d,
    c=None,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="mn",
    disable_ue8m0_cast=False,
):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2, hr, ra0, ra1, hra, rb0, rb1, hrb = _flatten_recipe(
        recipe, recipe_a, recipe_b
    )
    ops.fp8_fp4_gemm_tt(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        c,
        r0,
        r1,
        r2,
        hr,
        ra0,
        ra1,
        hra,
        rb0,
        rb1,
        hrb,
        compiled_dims,
        disable_ue8m0_cast,
    )


# FP8 aliases (same as FP8/FP4)
fp8_gemm_nt = fp8_fp4_gemm_nt
fp8_gemm_nn = fp8_fp4_gemm_nn
fp8_gemm_tn = fp8_fp4_gemm_tn
fp8_gemm_tt = fp8_fp4_gemm_tt


# M-grouped FP8/FP4 GEMM ops


def m_grouped_fp8_fp4_gemm_nt_contiguous(
    a,
    b,
    d,
    grouped_layout,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
    use_psum_layout=False,
    expected_m_for_psum_layout=None,
):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2, hr, ra0, ra1, hra, rb0, rb1, hrb = _flatten_recipe(
        recipe, recipe_a, recipe_b
    )
    has_em = expected_m_for_psum_layout is not None
    em = expected_m_for_psum_layout if has_em else 0
    ops.m_grouped_fp8_fp4_gemm_nt_contiguous(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        grouped_layout,
        r0,
        r1,
        r2,
        hr,
        ra0,
        ra1,
        hra,
        rb0,
        rb1,
        hrb,
        compiled_dims,
        disable_ue8m0_cast,
        use_psum_layout,
        em,
        has_em,
    )


def m_grouped_fp8_fp4_gemm_nn_contiguous(
    a,
    b,
    d,
    grouped_layout,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
    use_psum_layout=False,
):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2, hr, ra0, ra1, hra, rb0, rb1, hrb = _flatten_recipe(
        recipe, recipe_a, recipe_b
    )
    ops.m_grouped_fp8_fp4_gemm_nn_contiguous(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        grouped_layout,
        r0,
        r1,
        r2,
        hr,
        ra0,
        ra1,
        hra,
        rb0,
        rb1,
        hrb,
        compiled_dims,
        disable_ue8m0_cast,
        use_psum_layout,
    )


def m_grouped_fp8_fp4_gemm_nt_masked(
    a,
    b,
    d,
    masked_m,
    expected_m,
    recipe=None,
    recipe_a=None,
    recipe_b=None,
    compiled_dims="nk",
    disable_ue8m0_cast=False,
):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2, hr, ra0, ra1, hra, rb0, rb1, hrb = _flatten_recipe(
        recipe, recipe_a, recipe_b
    )
    ops.m_grouped_fp8_fp4_gemm_nt_masked(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        masked_m,
        expected_m,
        r0,
        r1,
        r2,
        hr,
        ra0,
        ra1,
        hra,
        rb0,
        rb1,
        hrb,
        compiled_dims,
        disable_ue8m0_cast,
    )


# M-grouped FP8 aliases
m_grouped_fp8_gemm_nt_contiguous = m_grouped_fp8_fp4_gemm_nt_contiguous
m_grouped_fp8_gemm_nn_contiguous = m_grouped_fp8_fp4_gemm_nn_contiguous
m_grouped_fp8_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked

# Legacy aliases
fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_fp4_gemm_nt_masked


# K-grouped FP8 GEMM ops


def k_grouped_fp8_gemm_tn_contiguous(
    a, b, d, ks, ks_tensor, c=None, recipe=(1, 1, 128), compiled_dims="mn"
):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2 = recipe
    ops.k_grouped_fp8_gemm_tn_contiguous(
        a_data, a_sf, b_data, b_sf, d, ks_tensor, c, r0, r1, r2, compiled_dims
    )


def k_grouped_fp8_gemm_nt_contiguous(
    a, b, d, ks, ks_tensor, c=None, recipe=(1, 1, 128), compiled_dims="mn"
):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2 = recipe
    ops.k_grouped_fp8_gemm_nt_contiguous(
        a_data, a_sf, b_data, b_sf, d, ks_tensor, c, r0, r1, r2, compiled_dims
    )


# BF16 GEMM ops


def bf16_gemm_nt(a, b, d, c=None, compiled_dims="nk"):
    ops.bf16_gemm_nt(a, b, d, c, compiled_dims)


def bf16_gemm_nn(a, b, d, c=None, compiled_dims="nk"):
    ops.bf16_gemm_nn(a, b, d, c, compiled_dims)


def bf16_gemm_tn(a, b, d, c=None, compiled_dims="mn"):
    ops.bf16_gemm_tn(a, b, d, c, compiled_dims)


def bf16_gemm_tt(a, b, d, c=None, compiled_dims="mn"):
    ops.bf16_gemm_tt(a, b, d, c, compiled_dims)


# M-grouped BF16 GEMM ops


def m_grouped_bf16_gemm_nt_contiguous(
    a,
    b,
    d,
    grouped_layout,
    compiled_dims="nk",
    use_psum_layout=False,
    expected_m_for_psum_layout=None,
):
    has_em = expected_m_for_psum_layout is not None
    em = expected_m_for_psum_layout if has_em else 0
    ops.m_grouped_bf16_gemm_nt_contiguous(
        a, b, d, grouped_layout, compiled_dims, use_psum_layout, em, has_em
    )


def m_grouped_bf16_gemm_nn_contiguous(
    a, b, d, grouped_layout, compiled_dims="nk", use_psum_layout=False
):
    ops.m_grouped_bf16_gemm_nn_contiguous(
        a, b, d, grouped_layout, compiled_dims, use_psum_layout
    )


def m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims="nk"):
    ops.m_grouped_bf16_gemm_nt_masked(a, b, d, masked_m, expected_m, compiled_dims)


# Legacy alias
bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked


# K-grouped BF16 GEMM ops


def k_grouped_bf16_gemm_tn_contiguous(
    a, b, d, ks, ks_tensor, c=None, compiled_dims="mn"
):
    ops.k_grouped_bf16_gemm_tn_contiguous(a, b, d, ks_tensor, c, compiled_dims)


# cuBLASLt GEMM ops


def cublaslt_gemm_nt(a, b, d, c=None):
    ops.cublaslt_gemm_nt(a, b, d, c)


def cublaslt_gemm_nn(a, b, d, c=None):
    ops.cublaslt_gemm_nn(a, b, d, c)


def cublaslt_gemm_tn(a, b, d, c=None):
    ops.cublaslt_gemm_tn(a, b, d, c)


def cublaslt_gemm_tt(a, b, d, c=None):
    ops.cublaslt_gemm_tt(a, b, d, c)


# Attention ops


def fp8_gemm_nt_skip_head_mid(
    a, b, d, head_splits, recipe=None, compiled_dims="nk", disable_ue8m0_cast=False
):
    a_data, a_sf = a
    b_data, b_sf = b
    left, mid, right = head_splits
    has_recipe = recipe is not None
    r0, r1, r2 = recipe if has_recipe else (0, 0, 0)
    ops.fp8_gemm_nt_skip_head_mid(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        left,
        mid,
        right,
        r0,
        r1,
        r2,
        has_recipe,
        compiled_dims,
        disable_ue8m0_cast,
    )


def fp8_mqa_logits(
    q,
    kv,
    weights,
    cu_seq_len_k_start,
    cu_seq_len_k_end,
    clean_logits=True,
    max_seqlen_k=0,
):
    kv_data, kv_sf = kv
    return ops.fp8_mqa_logits(
        q,
        kv_data,
        kv_sf,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        clean_logits,
        max_seqlen_k,
    )


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms):
    return ops.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms)


def fp8_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    clean_logits=False,
):
    return ops.fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
    )


# Einsum ops


def einsum(expr, a, b, d, c=None, use_cublaslt=False):
    ops.einsum(expr, a, b, d, c, use_cublaslt)


def fp8_einsum(expr, a, b, d, c=None, recipe=(1, 128, 128)):
    a_data, a_sf = a
    b_data, b_sf = b
    r0, r1, r2 = recipe
    ops.fp8_einsum(expr, a_data, a_sf, b_data, b_sf, d, c, r0, r1, r2)


# Hyperconnection ops


def tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits=None):
    has_ns = num_splits is not None
    ns = num_splits if has_ns else 0
    ops.tf32_hc_prenorm_gemm(a, b, d, sqr_sum, ns, has_ns)


# Initialize the C++ runtime


def _find_cuda_home() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        try:
            with open(os.devnull, "w") as devnull:
                nvcc = (
                    subprocess.check_output(["which", "nvcc"], stderr=devnull)
                    .decode()
                    .rstrip("\r\n")
                )
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = "/usr/local/cuda"
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None, "Could not find CUDA installation"
    return cuda_home


# Find the library root for JIT headers
# In development: use the repo's deep_gemm/ directory
# In installed wheel: use this package's directory
_lib_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deep_gemm"
)
if not os.path.isdir(os.path.join(_lib_root, "include")):
    # Fallback: try the parent package
    _lib_root = os.path.dirname(os.path.abspath(__file__))

_initialized = False

# Set DG_CUTLASS_INCLUDE for JIT kernel compilation (if not already set by user)
if "DG_CUTLASS_INCLUDE" not in os.environ:
    _include = os.path.join(_lib_root, "include")
    if os.path.isdir(os.path.join(_include, "cutlass")):
        # Bundled CUTLASS headers (from kernel-builder bundle-dep-includes)
        os.environ["DG_CUTLASS_INCLUDE"] = _include
    else:
        # Fall back to nvidia-cutlass pip package
        try:
            import nvidia.cutlass as _nc
            os.environ["DG_CUTLASS_INCLUDE"] = os.path.join(
                os.path.dirname(_nc.__file__), "include"
            )
        except ImportError:
            pass

def _ensure_initialized():
    global _initialized
    if _initialized:
        return
    _initialized = True
    ops.init(_lib_root, _find_cuda_home())


# Try to initialize eagerly, but don't fail if CUDA is not found
# (e.g., during build-time import checks). init() will be called
# lazily on first actual kernel use.
try:
    _ensure_initialized()
except (AssertionError, RuntimeError):
    pass
