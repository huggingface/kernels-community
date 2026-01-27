import torch
import triton

from .matmul_ogs_details._common import _constexpr_function
from triton.runtime import driver

def current_target():
    try:
        active_driver = driver.active
    except RuntimeError:
        # If there is no active driver, return None
        return None
    return active_driver.get_current_target()

current_target.__triton_builtin__ = True


@_constexpr_function
def is_cuda():
    target = current_target()
    return target is not None and target.backend == "cuda"


@_constexpr_function
def is_hip():
    target = current_target()
    return target is not None and target.backend == "hip"


@_constexpr_function
def is_xpu():
    target = current_target()
    return target is not None and target.backend == "xpu"


@_constexpr_function
def is_hip_cdna3():
    target = current_target()
    return target is not None and target.arch == "gfx942"


@_constexpr_function
def is_hip_cdna4():
    target = current_target()
    return target is not None and target.arch == "gfx950"


@_constexpr_function
def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    target = current_target()
    if target is None or target.backend != "cuda":
        return False
    assert isinstance(target.arch, int)
    return target.arch >= major * 10 + minor


@_constexpr_function
def get_cdna_version():
    """
    Gets the AMD architecture version, i.e. CDNA3 or CDNA4, currently
    only supports 3 (gfx942) or 4 (gfx950). Returns -1 if it is not AMD
    hardware or unsupported architecture
    """
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != 'hip':
        return -1
    if target.arch == 'gfx942':
        return 3
    if target.arch == 'gfx950':
        return 4
    return -1


@_constexpr_function
def has_tma_gather():
    return cuda_capability_geq(10, 0)


@_constexpr_function
def has_native_mxfp():
    return cuda_capability_geq(10, 0)


def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties(0).multi_processor_count
    if is_xpu():
        return torch.xpu.get_device_properties(0).max_compute_units
