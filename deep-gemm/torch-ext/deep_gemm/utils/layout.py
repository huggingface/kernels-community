import torch

from .._ops import ops


def get_tma_aligned_size(mn: int, element_size: int) -> int:
    return ops.get_tma_aligned_size(mn, element_size).item()


def get_mn_major_tma_aligned_tensor(sf: torch.Tensor) -> torch.Tensor:
    return ops.get_mn_major_tma_aligned_tensor(sf)


def get_mn_major_tma_aligned_packed_ue8m0_tensor(sf: torch.Tensor) -> torch.Tensor:
    return ops.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)


def get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks, gran_k: int = 128):
    ks_int = torch.tensor(ks, dtype=torch.int32, device="cpu")
    return ops.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks_int, gran_k)


def set_mk_alignment_for_contiguous_layout(value: int) -> None:
    ops.set_mk_alignment_for_contiguous_layout(value)


def get_mk_alignment_for_contiguous_layout() -> int:
    return ops.get_mk_alignment_for_contiguous_layout()


def get_theoretical_mk_alignment_for_contiguous_layout(expected_m=None) -> int:
    return ops.get_theoretical_mk_alignment_for_contiguous_layout(
        -1 if expected_m is None else int(expected_m)
    )


# Aliases
get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
