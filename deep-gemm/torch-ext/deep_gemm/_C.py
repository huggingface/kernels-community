import torch

from ._ops import ops


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


def set_mk_alignment_for_contiguous_layout(value: int):
    ops.set_mk_alignment_for_contiguous_layout(value)


def get_mk_alignment_for_contiguous_layout() -> int:
    return ops.get_mk_alignment_for_contiguous_layout()


def get_theoretical_mk_alignment_for_contiguous_layout(expected_m=None) -> int:
    return ops.get_theoretical_mk_alignment_for_contiguous_layout(
        0 if expected_m is None else expected_m,
        expected_m is not None,
    )


def get_tma_aligned_size(mn: int, element_size: int) -> int:
    return ops.get_tma_aligned_size(mn, element_size).item()


def get_mn_major_tma_aligned_tensor(sf):
    return ops.get_mn_major_tma_aligned_tensor(sf)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf):
    return ops.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
    sf, ks_tensor, ks, gran_k
):
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

    return ops.transform_sf_into_required_layout(
        sf,
        mn,
        k,
        r0,
        r1,
        r2,
        recipe_len,
        0 if num_groups is None else num_groups,
        num_groups is not None,
        False if is_sfa is None else is_sfa,
        is_sfa is not None,
        disable_ue8m0_cast,
    )


def get_token_alignment_for_mega_moe() -> int:
    return ops.get_token_alignment_for_mega_moe()


def get_symm_buffer_size_for_mega_moe(
    num_ranks,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    use_fp8_dispatch=True,
    activation="swiglu",
):
    num_bytes = ops.get_symm_buffer_size_for_mega_moe(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch,
        activation,
    )

    def slice_input_buffers(buffer):
        return tuple(
            ops.get_symm_buffer_views_for_mega_moe(
                buffer,
                num_ranks,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                use_fp8_dispatch,
                activation,
            )
        )

    return num_bytes, slice_input_buffers


def fp8_fp4_mega_moe(
    y,
    l1_weights,
    l2_weights,
    cumulative_local_expert_recv_stats,
    sym_buffer,
    sym_buffer_ptrs,
    rank_idx,
    num_max_tokens_per_rank,
    num_experts,
    num_topk,
    recipe,
    activation,
    activation_clamp,
    fast_math,
):
    l1_weights_data, l1_weights_sf = l1_weights
    l2_weights_data, l2_weights_sf = l2_weights
    r0, r1, r2 = recipe
    ops.fp8_fp4_mega_moe(
        y,
        l1_weights_data,
        l1_weights_sf,
        l2_weights_data,
        l2_weights_sf,
        cumulative_local_expert_recv_stats,
        sym_buffer,
        sym_buffer_ptrs,
        rank_idx,
        num_max_tokens_per_rank,
        num_experts,
        num_topk,
        r0,
        r1,
        r2,
        activation,
        activation_clamp,
        fast_math,
    )
