import torch

from ._ops import ops


@torch.no_grad()
def count_cumsum(
    x: torch.Tensor, E: int, do_cumsum: bool = True
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Count expert assignments and optionally compute cumulative sum.

    Args:
        x: 1D tensor of expert indices (int32 or int64).
        E: Number of experts. Must be divisible by 4 and <= 50000.
        do_cumsum: Whether to also compute the cumulative sum.

    Returns:
        If do_cumsum: (count_output, cumsum_output)
        Otherwise: count_output
    """
    assert x.dim() == 1, "x should be 1-dimensional"
    assert x.dtype in [torch.int32, torch.long]

    count_output = torch.empty(E, dtype=torch.int32, device=x.device)
    cumsum_output = torch.empty(E, dtype=torch.int32, device=x.device)

    ops.count_cumsum(x, count_output, cumsum_output, do_cumsum)

    if do_cumsum:
        return count_output, cumsum_output
    else:
        return count_output
