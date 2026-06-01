import triton


_CACHED_ARCH = None


def get_arch():
    # Lazy-detect on first call so the module is importable in environments without an
    # active GPU driver (e.g. the kernel-builder Nix build sandbox running
    # `get_local_kernel` for layout checks).
    global _CACHED_ARCH
    if _CACHED_ARCH is None:
        try:
            _CACHED_ARCH = triton.runtime.driver.active.get_current_target().arch
        except RuntimeError:
            try:
                from jax._src.lib import gpu_triton as triton_kernel_call_lib

                _CACHED_ARCH = triton_kernel_call_lib.get_arch_details("0").split(":")[0]
            except ImportError:
                _CACHED_ARCH = ""
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
