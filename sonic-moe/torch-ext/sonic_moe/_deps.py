"""Kernel dependencies.

Resolved eagerly while the kernel is loaded (`get_kernel_dep` only works then);
submodules import from here.
"""

import kernels

_einops = kernels.get_kernel_dep("kernels-community/einops")

rearrange = _einops.rearrange

__all__ = ["rearrange"]
