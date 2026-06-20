import functools


@functools.lru_cache(maxsize=1)
def _detect_arch() -> str:
    """Resolve the active GPU arch lazily so module import succeeds on
    machines with no GPU / no Triton driver (e.g. kernels-community CI build
    sandboxes)."""
    try:
        import triton

        return triton.runtime.driver.active.get_current_target().arch
    except Exception:
        pass
    try:
        from jax._src.lib import gpu_triton as triton_kernel_call_lib

        return triton_kernel_call_lib.get_arch_details("0").split(":")[0]
    except Exception:
        pass
    return ""


def get_arch():
    return _detect_arch()


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


_LDS_CAP_BYTES = {"gfx950": 163840, "gfx942": 65536}
