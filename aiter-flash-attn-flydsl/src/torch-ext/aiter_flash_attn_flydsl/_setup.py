# SPDX-License-Identifier: MIT
"""Actionable diagnostics for the FlyDSL JIT toolchain."""

import os
import re
from functools import lru_cache
from pathlib import Path

import kernels
flydsl = kernels.get_kernel_dep("kernels-sdk/flydsl")
RocmBackend = flydsl.compiler.backends.rocm.RocmBackend
get_rocm_arch = flydsl.runtime.device.get_rocm_arch


def _toolkit_root():
    # The Nix wheel sets toolkit explicitly in the compilation pipeline. An
    # environment-only check would incorrectly reject that supported setup.
    backend = RocmBackend(RocmBackend.make_target("gfx1201"))
    pipeline = ",".join(backend.pipeline_fragments(compile_hints={}))
    match = re.search(r"\btoolkit=([^\s}]+)", pipeline)
    return Path(match.group(1) if match else os.environ.get("ROCM_PATH", "/opt/rocm"))


def _detected_arch():
    return get_rocm_arch()


@lru_cache(maxsize=1)
def check_runtime_setup():
    """Check once before compilation; failures can be retried after setup."""
    toolkit = _toolkit_root()
    linker = toolkit / "llvm/bin/ld.lld"
    if not linker.is_file() or not os.access(linker, os.X_OK):
        raise RuntimeError(
            "AITER FlyDSL attention requires the ROCm development linker "
            f"ld.lld, but it is missing or not executable at {linker}. "
            "For AMD's split ROCm wheels, install 'rocm[devel]' matching your "
            "PyTorch ROCm version from the AMD wheel index, then run "
            "'rocm-sdk init'. Set ROCM_PATH to the output of "
            "'rocm-sdk path --root' and add $ROCM_PATH/lib/llvm/bin and "
            "$ROCM_PATH/bin to PATH. A system ROCm development SDK is also "
            "supported. Installing FlyDSL alone does not install this linker."
        )
    arch = _detected_arch()
    if arch != "gfx1201":
        raise RuntimeError(
            "AITER FlyDSL attention requires gfx1201, but FlyDSL detected "
            f"{arch!r}. Make the ROCm 'rocm_agent_enumerator' tool available "
            "on PATH and restart the Python process, or set "
            "FLYDSL_GPU_ARCH=gfx1201 for this gfx1201-only "
            "package. FlyDSL can default to gfx942 when discovery tools "
            "are missing."
        )
