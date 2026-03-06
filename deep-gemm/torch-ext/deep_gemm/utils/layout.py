from .._ops import ops


def get_mk_alignment_for_contiguous_layout():
    return ops.get_mk_alignment_for_contiguous_layout()


def get_tma_aligned_size(mn: int, element_size: int):
    return ops.get_tma_aligned_size(mn, element_size).item()


def get_mn_major_tma_aligned_tensor(sf):
    return ops.get_mn_major_tma_aligned_tensor(sf)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf):
    return ops.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks):
    return ops.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks)


get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
