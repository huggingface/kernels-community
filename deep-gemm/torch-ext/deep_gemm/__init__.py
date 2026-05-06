import os
import subprocess
import torch

# Import the compiled extension
from ._ops import ops, add_op_namespace_prefix
from . import utils

# Optional legacy Triton kernels (require triton)
try:
    from . import legacy
except Exception:
    legacy = None

__version__ = "2.3.0"


# ── Register fake tensor implementations for torch.compile ──────────────────
# All GEMM ops mutate the output tensor `d` in-place and return void.
# The fake implementations are no-ops since `d` is pre-allocated by the caller.


for _op in [
    "fp8_fp4_gemm_nt",
    "fp8_fp4_gemm_nn",
    "fp8_fp4_gemm_tn",
    "fp8_fp4_gemm_tt",
    "m_grouped_fp8_fp4_gemm_nt_contiguous",
    "m_grouped_fp8_fp4_gemm_nn_contiguous",
    "m_grouped_fp8_fp4_gemm_nt_masked",
    "k_grouped_fp8_gemm_nt_contiguous",
    "k_grouped_fp8_gemm_tn_contiguous",
    "bf16_gemm_nt",
    "bf16_gemm_nn",
    "bf16_gemm_tn",
    "bf16_gemm_tt",
    "m_grouped_bf16_gemm_nt_contiguous",
    "m_grouped_bf16_gemm_nn_contiguous",
    "m_grouped_bf16_gemm_nt_masked",
    "k_grouped_bf16_gemm_tn_contiguous",
    "fp8_gemm_nt_skip_head_mid",
    "tf32_hc_prenorm_gemm",
    "einsum",
    "fp8_einsum",
    "cublaslt_gemm_nt",
    "cublaslt_gemm_nn",
    "cublaslt_gemm_tn",
    "cublaslt_gemm_tt",
]:

    @torch.library.register_fake(add_op_namespace_prefix(_op))
    def _fake(*args, **kwargs):
        pass


# ── Fakes for tensor-returning attention ops ────────────────────────────────
# Output: [seq_len_q, seq_len_kv] for non-paged, [batch*next_n, max_ctx_len] for paged.


@torch.library.register_fake(add_op_namespace_prefix("fp8_mqa_logits"))
def _fake_fp8_mqa_logits(q, kv_data, kv_sf, weights, cu_start, cu_end, clean_logits, max_seqlen_k):
    seq_len_q = q.shape[0]
    seq_len_kv = kv_data.shape[0]
    return torch.empty((seq_len_q, seq_len_kv), dtype=torch.float32, device=q.device)


@torch.library.register_fake(add_op_namespace_prefix("fp8_fp4_mqa_logits"))
def _fake_fp8_fp4_mqa_logits(q_fp, q_sf, kv_fp, kv_sf, weights, cu_start, cu_end,
                             clean_logits, max_seqlen_k, logits_dtype_int):
    seq_len_q = q_fp.shape[0]
    seq_len_kv = kv_fp.shape[0]
    dtype = torch.float32 if logits_dtype_int == 6 else torch.bfloat16
    return torch.empty((seq_len_q, seq_len_kv), dtype=dtype, device=q_fp.device)


@torch.library.register_fake(add_op_namespace_prefix("get_paged_mqa_logits_metadata"))
def _fake_get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices):
    return torch.empty((num_sms + 1, 2), dtype=context_lens.dtype, device=context_lens.device)


@torch.library.register_fake(add_op_namespace_prefix("fp8_paged_mqa_logits"))
def _fake_fp8_paged_mqa_logits(q, kv_cache, weights, ctx_lens, blk_tbl, sched, max_ctx_len, clean, indices):
    bsz, next_n = q.shape[0], q.shape[1]
    return torch.empty((bsz * next_n, int(max_ctx_len)), dtype=torch.float32, device=q.device)


@torch.library.register_fake(add_op_namespace_prefix("fp8_fp4_paged_mqa_logits"))
def _fake_fp8_fp4_paged_mqa_logits(q_fp, q_sf, kv_cache, weights, ctx_lens, blk_tbl, sched,
                                   max_ctx_len, clean, logits_dtype_int, indices):
    bsz, next_n = q_fp.shape[0], q_fp.shape[1]
    dtype = torch.float32 if logits_dtype_int == 6 else torch.bfloat16
    return torch.empty((bsz * next_n, int(max_ctx_len)), dtype=dtype, device=q_fp.device)


@torch.library.register_fake(add_op_namespace_prefix("transform_sf_into_required_layout"))
def _fake_transform_sf_into_required_layout(sf, mn, k, *args, **kwargs):
    # Output is a transformed scale-factor tensor; concrete shape depends on layout heuristics.
    # Conservatively return same shape/dtype/device as input — sufficient for tracing.
    return torch.empty_like(sf)


@torch.library.register_fake(add_op_namespace_prefix("get_tma_aligned_size"))
def _fake_get_tma_aligned_size(mn, element_size):
    return torch.empty((), dtype=torch.int64, device="cpu")


@torch.library.register_fake(add_op_namespace_prefix("get_mn_major_tma_aligned_tensor"))
def _fake_get_mn_major_tma_aligned_tensor(sf):
    return torch.empty_like(sf)


@torch.library.register_fake(add_op_namespace_prefix("get_mn_major_tma_aligned_packed_ue8m0_tensor"))
def _fake_get_mn_major_tma_aligned_packed_ue8m0_tensor(sf):
    return torch.empty_like(sf)


@torch.library.register_fake(add_op_namespace_prefix("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor"))
def _fake_get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks_int_tensor, gran_k):
    return torch.empty_like(sf)


# Runtime


def set_num_sms(num_sms: int):
    ops.set_num_sms(num_sms)


def get_num_sms() -> int:
    return ops.get_num_sms()


def set_tc_util(tc_util: int):
    ops.set_tc_util(tc_util)


def get_tc_util() -> int:
    return ops.get_tc_util()


def set_pdl(enable: bool) -> None:
    ops.set_pdl(enable)


def get_pdl() -> bool:
    return ops.get_pdl()


def set_ignore_compile_dims(value: bool) -> None:
    ops.set_ignore_compile_dims(value)


def set_block_size_multiple_of(value) -> None:
    if isinstance(value, tuple):
        x, y = value
    else:
        x, y = int(value), -1
    ops.set_block_size_multiple_of(int(x), int(y))


def get_mk_alignment_for_contiguous_layout() -> int:
    return ops.get_mk_alignment_for_contiguous_layout()


def set_mk_alignment_for_contiguous_layout(value: int) -> None:
    ops.set_mk_alignment_for_contiguous_layout(int(value))


def get_theoretical_mk_alignment_for_contiguous_layout(expected_m=None) -> int:
    return ops.get_theoretical_mk_alignment_for_contiguous_layout(
        -1 if expected_m is None else int(expected_m)
    )


def get_token_alignment_for_mega_moe() -> int:
    # Mega MoE is SM_100 / Blackwell, CUDA 12.8+. The C++ binding is only
    # compiled on toolchains that support it; otherwise raise a clear error.
    if not hasattr(ops, "get_token_alignment_for_mega_moe"):
        raise RuntimeError(
            "Mega MoE is unavailable: this build of deep-gemm was compiled "
            "with a CUDA toolchain older than 12.8 (Blackwell intrinsics "
            "such as __fmul2_rn are required)."
        )
    return ops.get_token_alignment_for_mega_moe()


# Layout utilities


def get_tma_aligned_size(mn: int, element_size: int) -> int:
    return ops.get_tma_aligned_size(mn, element_size).item()


def get_mn_major_tma_aligned_tensor(sf):
    return ops.get_mn_major_tma_aligned_tensor(sf)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf):
    return ops.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks, gran_k: int = 128):
    ks_int = torch.tensor(ks, dtype=torch.int32, device="cpu")
    return ops.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        sf, ks_tensor, ks_int, gran_k
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


_DTYPE_TO_INT = {
    torch.float32: 6,    # at::ScalarType::Float
    torch.float: 6,
    torch.bfloat16: 15,  # at::ScalarType::BFloat16
}


def _scalar_type_int(dtype):
    if dtype is None:
        return _DTYPE_TO_INT[torch.float32]
    return _DTYPE_TO_INT[dtype] if dtype in _DTYPE_TO_INT else int(dtype)


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


def fp8_fp4_mqa_logits(
    q,
    kv,
    weights,
    cu_seq_len_k_start,
    cu_seq_len_k_end,
    clean_logits=True,
    max_seqlen_k=0,
    logits_dtype=torch.float32,
):
    # q: (q_fp, optional q_sf); kv: (kv_fp, kv_sf)
    if isinstance(q, tuple):
        q_fp, q_sf = q
    else:
        q_fp, q_sf = q, None
    kv_fp, kv_sf = kv
    return ops.fp8_fp4_mqa_logits(
        q_fp, q_sf, kv_fp, kv_sf, weights,
        cu_seq_len_k_start, cu_seq_len_k_end,
        clean_logits, max_seqlen_k, _scalar_type_int(logits_dtype),
    )


def get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices=None):
    return ops.get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices)


def fp8_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    clean_logits=False,
    indices=None,
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
        indices,
    )


def fp8_fp4_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    clean_logits=False,
    logits_dtype=torch.float32,
    indices=None,
):
    if isinstance(q, tuple):
        q_fp, q_sf = q
    else:
        q_fp, q_sf = q, None
    return ops.fp8_fp4_paged_mqa_logits(
        q_fp, q_sf, kv_cache, weights, context_lens, block_table, schedule_meta,
        max_context_len, clean_logits, _scalar_type_int(logits_dtype), indices,
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
    """Find a CUDA toolkit directory (containing `bin/nvcc`) without requiring
    the user to set any environment variable. Search order:

      1. CUDA_HOME / CUDA_PATH env (explicit override).
      2. `nvcc` on PATH.
      3. pip-installed nvidia-cuda-nvcc-cu12 (only if it actually ships nvcc).
      4. Conventional system locations (/usr/local/cuda, /opt/cuda, /opt/nvidia).
    """
    def _has_nvcc(d):
        return d and os.path.isfile(os.path.join(d, "bin", "nvcc"))

    # 1. Explicit override (still honoured for power users / CI).
    for var in ("CUDA_HOME", "CUDA_PATH"):
        cand = os.environ.get(var)
        if _has_nvcc(cand):
            return cand

    # 2. nvcc on PATH (covers system installs that put nvcc in /usr/bin).
    try:
        nvcc = subprocess.check_output(
            ["which", "nvcc"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if nvcc:
            home = os.path.dirname(os.path.dirname(nvcc))
            if _has_nvcc(home):
                return home
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    # 3. pip-installed `nvidia-cuda-nvcc-cu12`. Note: in some package versions
    #    the binary `nvcc` is missing and only `ptxas` ships — skip those.
    try:
        import nvidia.cuda_nvcc as _nvcc_pkg  # type: ignore
        cand = os.path.dirname(_nvcc_pkg.__file__)
        if _has_nvcc(cand):
            return cand
    except ImportError:
        pass

    # 4. Conventional system layouts (DGX images often have one of these).
    for cand in (
        "/usr/local/cuda",
        "/opt/cuda",
        "/opt/nvidia/cuda",
        "/usr/lib/cuda",
    ):
        if _has_nvcc(cand):
            return cand
    # Versioned siblings: /usr/local/cuda-12.9, etc. — pick the highest.
    import glob
    versioned = sorted(
        (d for d in glob.glob("/usr/local/cuda-*") if _has_nvcc(d)),
        reverse=True,
    )
    if versioned:
        return versioned[0]

    raise RuntimeError(
        "Could not find a CUDA installation with nvcc. DeepGEMM JIT-compiles "
        "kernels at runtime and needs nvcc available. Set CUDA_HOME to a "
        "toolkit directory, install `nvidia-cuda-nvcc-cu12`, or install the "
        "system CUDA toolkit."
    )


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
    _cutlass_include_candidates = [
        _include,  # legacy layout: include/cutlass
        os.path.join(_include, "third-party", "cutlass", "include"),  # submodule layout
    ]
    for _cutlass_include in _cutlass_include_candidates:
        if os.path.isdir(os.path.join(_cutlass_include, "cutlass")):
            os.environ["DG_CUTLASS_INCLUDE"] = _cutlass_include
            break
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
