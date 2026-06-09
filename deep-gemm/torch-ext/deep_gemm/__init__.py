import os
import subprocess
import sysconfig
import torch
from typing import Optional

# Avoid holding a CUDA tensor in DeepGEMM's process-lifetime runtime singleton.
# In packaged/lazy-loaded use, that can outlive PyTorch's CUDA teardown and crash
# during interpreter shutdown.
os.environ.setdefault("DG_USE_TEMP_CUBLASLT_WORKSPACE", "1")

# Import the compiled extension
from ._ops import ops as _ops, add_op_namespace_prefix
from . import utils

__version__ = "2.5.0"


# ── Register fake tensor implementations for torch.compile ──────────────────
# All GEMM ops mutate the output tensor `d` in-place and return void.
# The fake implementations are no-ops since `d` is pre-allocated by the caller.
#
# Five ops are intentionally absent from this list — see the
# ``@torch.library.custom_op`` block below. Those ops are wrapped Python-side
# because the C++ ``Tensor! d`` mutation schema is rejected by torch's
# functionalize pass for custom (non-ATen) ops once the call sits inside a
# nested compile region (``@torch.compiler.nested_compile_region`` →
# ``invoke_subgraph`` → ``auto_functionalized_v2``). The Python ``custom_op``
# expresses the same mutation via ``mutates_args=(...)`` which functionalize
# does accept. See the C++ ``_raw_`` rename in ``torch_binding.cpp``.
# ``custom_op`` auto-provides their fakes, so they don't need an entry here.


for _op in [
    "fp8_fp4_gemm_nt",
    "fp8_fp4_gemm_nn",
    "fp8_fp4_gemm_tn",
    "fp8_fp4_gemm_tt",
    "m_grouped_fp8_fp4_gemm_nt_masked",
    "k_grouped_fp8_gemm_nt_contiguous",
    "k_grouped_fp8_gemm_tn_contiguous",
    "bf16_gemm_nt",
    "bf16_gemm_nn",
    "bf16_gemm_tn",
    "bf16_gemm_tt",
    "m_grouped_bf16_gemm_nt_masked",
    "fp8_gemm_nt_skip_head_mid",
]:

    @torch.library.register_fake(add_op_namespace_prefix(_op))
    def _fake(*args, **kwargs):
        pass


# ── Functionalize-safe wrappers for the in-place mutating ops ───────────────
# Public name → ``@torch.library.custom_op(mutates_args=(...))``. The body
# dispatches to the C++ ``_raw_X`` torch op (same TORCH_LIBRARY namespace,
# schema with no ``Tensor!`` annotation — see ``torch_binding.cpp``).
#
# Why this exists: when a mutating op is reached inside
# ``@torch.compiler.nested_compile_region`` (or any compiled region that hits
# functionalize), torch's functionalize pass wraps the call in
# ``auto_functionalized_v2`` and then tries to decompose it. For custom
# (non-ATen) ops registered with ``Tensor! d`` it has no decomposition rule
# and the assertion ``auto_functionalized_v2 was not removed`` fires.
# ``mutates_args=`` is the modern primitive functionalize understands — it
# autogenerates the clone-then-mutate-on-clone rewrite at compile time.
#
# Public wrappers further down keep calling the unprefixed name via
# ``ops.X(...)``; that hits the custom_op, which then hits the renamed C++ op
# at runtime. No call-site changes needed.


@torch.library.custom_op(
    add_op_namespace_prefix("m_grouped_fp8_fp4_gemm_nt_contiguous"),
    mutates_args=("d",),
)
def _m_grouped_fp8_fp4_gemm_nt_contiguous(
    a_data: torch.Tensor,
    a_sf: torch.Tensor,
    b_data: torch.Tensor,
    b_sf: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe_0: int,
    recipe_1: int,
    recipe_2: int,
    has_recipe: bool,
    recipe_a_0: int,
    recipe_a_1: int,
    has_recipe_a: bool,
    recipe_b_0: int,
    recipe_b_1: int,
    has_recipe_b: bool,
    compiled_dims: str,
    disable_ue8m0_cast: bool,
    use_psum_layout: bool,
    expected_m_for_psum_layout: int,
    has_expected_m_for_psum_layout: bool,
) -> None:
    _ops._raw_m_grouped_fp8_fp4_gemm_nt_contiguous(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        grouped_layout,
        recipe_0,
        recipe_1,
        recipe_2,
        has_recipe,
        recipe_a_0,
        recipe_a_1,
        has_recipe_a,
        recipe_b_0,
        recipe_b_1,
        has_recipe_b,
        compiled_dims,
        disable_ue8m0_cast,
        use_psum_layout,
        expected_m_for_psum_layout,
        has_expected_m_for_psum_layout,
    )


@torch.library.custom_op(
    add_op_namespace_prefix("m_grouped_fp8_fp4_gemm_nn_contiguous"),
    mutates_args=("d",),
)
def _m_grouped_fp8_fp4_gemm_nn_contiguous(
    a_data: torch.Tensor,
    a_sf: torch.Tensor,
    b_data: torch.Tensor,
    b_sf: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    recipe_0: int,
    recipe_1: int,
    recipe_2: int,
    has_recipe: bool,
    recipe_a_0: int,
    recipe_a_1: int,
    has_recipe_a: bool,
    recipe_b_0: int,
    recipe_b_1: int,
    has_recipe_b: bool,
    compiled_dims: str,
    disable_ue8m0_cast: bool,
    use_psum_layout: bool,
) -> None:
    _ops._raw_m_grouped_fp8_fp4_gemm_nn_contiguous(
        a_data,
        a_sf,
        b_data,
        b_sf,
        d,
        grouped_layout,
        recipe_0,
        recipe_1,
        recipe_2,
        has_recipe,
        recipe_a_0,
        recipe_a_1,
        has_recipe_a,
        recipe_b_0,
        recipe_b_1,
        has_recipe_b,
        compiled_dims,
        disable_ue8m0_cast,
        use_psum_layout,
    )


@torch.library.custom_op(
    add_op_namespace_prefix("m_grouped_bf16_gemm_nt_contiguous"),
    mutates_args=("d",),
)
def _m_grouped_bf16_gemm_nt_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str,
    use_psum_layout: bool,
    expected_m_for_psum_layout: int,
    has_expected_m_for_psum_layout: bool,
) -> None:
    _ops._raw_m_grouped_bf16_gemm_nt_contiguous(
        a,
        b,
        d,
        grouped_layout,
        compiled_dims,
        use_psum_layout,
        expected_m_for_psum_layout,
        has_expected_m_for_psum_layout,
    )


@torch.library.custom_op(
    add_op_namespace_prefix("m_grouped_bf16_gemm_nn_contiguous"),
    mutates_args=("d",),
)
def _m_grouped_bf16_gemm_nn_contiguous(
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor,
    grouped_layout: torch.Tensor,
    compiled_dims: str,
    use_psum_layout: bool,
) -> None:
    _ops._raw_m_grouped_bf16_gemm_nn_contiguous(
        a,
        b,
        d,
        grouped_layout,
        compiled_dims,
        use_psum_layout,
    )


@torch.library.custom_op(
    add_op_namespace_prefix("fp8_fp4_mega_moe"),
    mutates_args=("y",),
)
def _fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: torch.Tensor,
    l1_weights_sf: torch.Tensor,
    l2_weights: torch.Tensor,
    l2_weights_sf: torch.Tensor,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    sym_buffer: torch.Tensor,
    sym_buffer_ptrs: list[int],
    rank_idx: int,
    num_max_tokens_per_rank: int,
    num_experts: int,
    num_topk: int,
    recipe_0: int,
    recipe_1: int,
    recipe_2: int,
    activation: str,
    activation_clamp: Optional[float],
    fast_math: bool,
) -> None:
    _ops._raw_fp8_fp4_mega_moe(
        y,
        l1_weights,
        l1_weights_sf,
        l2_weights,
        l2_weights_sf,
        cumulative_local_expert_recv_stats,
        sym_buffer,
        sym_buffer_ptrs,
        rank_idx,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        recipe_0,
        recipe_1,
        recipe_2,
        activation,
        activation_clamp,
        fast_math,
    )


# Runtime


def set_num_sms(num_sms: int):
    ops.set_num_sms(num_sms)


def get_num_sms() -> int:
    return ops.get_num_sms()


def set_tc_util(tc_util: int):
    ops.set_tc_util(tc_util)


def get_tc_util() -> int:
    return ops.get_tc_util()


def set_ignore_compile_dims(value: bool):
    ops.set_ignore_compile_dims(value)


def set_block_size_multiple_of(value):
    if isinstance(value, tuple):
        block_m, block_n = value
    else:
        block_m = block_n = value
    ops.set_block_size_multiple_of(block_m, block_n)


def set_pdl(enable_pdl: bool):
    ops.set_pdl(enable_pdl)


def get_pdl() -> bool:
    return ops.get_pdl()


def set_mk_alignment_for_contiguous_layout(alignment: int):
    ops.set_mk_alignment_for_contiguous_layout(alignment)


def get_mk_alignment_for_contiguous_layout() -> int:
    return ops.get_mk_alignment_for_contiguous_layout()


def get_theoretical_mk_alignment_for_contiguous_layout(expected_m=None) -> int:
    return ops.get_theoretical_mk_alignment_for_contiguous_layout(
        0 if expected_m is None else expected_m,
        expected_m is not None,
    )


# Layout utilities


def get_tma_aligned_size(mn: int, element_size: int) -> int:
    return ops.get_tma_aligned_size(mn, element_size).item()


def get_mn_major_tma_aligned_tensor(sf):
    return ops.get_mn_major_tma_aligned_tensor(sf)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf):
    return ops.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks, gran_k):
    ks_int = torch.tensor(ks, dtype=torch.int32, device="cpu")
    return ops.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        sf, ks_tensor, ks_int, gran_k
    )


def transform_sf_into_required_layout(
    sf,
    mn,
    k,
    recipe,
    num_groups=None,
    is_sfa=None,
    disable_ue8m0_cast=False,
):
    if len(recipe) == 3:
        r0, r1, r2 = recipe
        recipe_len = 3
    elif len(recipe) == 2:
        r0, r1 = recipe
        r2 = 0
        recipe_len = 2
    else:
        raise ValueError("recipe must have length 2 or 3")
    has_ng = num_groups is not None
    ng = num_groups if has_ng else 0
    return ops.transform_sf_into_required_layout(
        sf,
        mn,
        k,
        r0,
        r1,
        r2,
        recipe_len,
        ng,
        has_ng,
        False if is_sfa is None else is_sfa,
        is_sfa is not None,
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
    if isinstance(q, tuple):
        q_data, q_sf = q
    else:
        q_data, q_sf = q, None
    kv_data, kv_sf = kv
    return ops.fp8_fp4_mqa_logits(
        q_data,
        q_sf,
        kv_data,
        kv_sf,
        weights,
        cu_seq_len_k_start,
        cu_seq_len_k_end,
        clean_logits,
        max_seqlen_k,
        logits_dtype,
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
        q_data, q_sf = q
    else:
        q_data, q_sf = q, None
    return ops.fp8_fp4_paged_mqa_logits(
        q_data,
        q_sf,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits,
        logits_dtype,
        indices,
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


from .mega import (
    SymmBuffer,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
)


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
    _cutlass_include_candidates = [
        _include,  # legacy layout: include/cutlass
        os.path.join(_include, "third-party", "cutlass", "include"),  # submodule layout
    ]
    for _site_packages in {
        sysconfig.get_paths().get("purelib"),
        sysconfig.get_paths().get("platlib"),
    }:
        if _site_packages:
            _cutlass_include_candidates.append(
                os.path.join(_site_packages, "cutlass_library", "source", "include")
            )
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
    _ops.init(_lib_root, _find_cuda_home())
    _initialized = True


class _InitializedOps:
    def __init__(self, raw_ops):
        self._raw_ops = raw_ops

    def __getattr__(self, name):
        if name != "init":
            _ensure_initialized()
        return getattr(self._raw_ops, name)


ops = _InitializedOps(_ops)


# Try to initialize eagerly, but don't fail if CUDA is not found
# (e.g., during build-time import checks). init() will be called
# lazily on first actual kernel use.
try:
    _ensure_initialized()
except (AssertionError, RuntimeError):
    pass
