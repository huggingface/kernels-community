"""Kernel dependencies.

Resolved eagerly while the kernel is loaded (`get_kernel_dep` only works then);
submodules import from here.
"""

import kernels

_einops = kernels.get_kernel_dep("kernels-community/einops")

einsum = _einops.einsum
rearrange = _einops.rearrange
reduce = _einops.reduce
repeat = _einops.repeat

__all__ = ["einsum", "rearrange", "reduce", "repeat"]
