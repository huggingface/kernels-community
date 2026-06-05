import triton

# Detect the GPU arch lazily: querying the triton driver at import time fails
# in headless environments (e.g. the kernel-builder ABI check sandbox has no
# GPU), and the original JAX fallback pulled in an unrelated runtime dep. The
# arch is only actually needed when a GMM kernel is dispatched, so resolve and
# cache on first call.
_CACHED_ARCH = None


def get_arch():
    global _CACHED_ARCH
    if _CACHED_ARCH is not None:
        return _CACHED_ARCH
    try:
        _CACHED_ARCH = triton.runtime.driver.active.get_current_target().arch
    except RuntimeError:
        try:
            from jax._src.lib import gpu_triton as triton_kernel_call_lib
            _CACHED_ARCH = triton_kernel_call_lib.get_arch_details("0").split(":")[0]
        except ImportError as e:
            raise RuntimeError(
                "Cannot determine GPU arch: triton driver is inactive and "
                "JAX is not available. A GPU is required for grouped GEMM."
            ) from e
    return _CACHED_ARCH


def is_gluon_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_fp4_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_fp8_avail():
    return get_arch() in ("gfx942", "gfx950", "gfx1250", "gfx1200", "gfx1201")


def is_mx_scale_preshuffling_avail():
    return get_arch() in ("gfx950", "gfx1250")


def is_tdm_avail():
    return get_arch() in ("gfx1250",)
