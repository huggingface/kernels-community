"""
kernels-community: runtime requirement checks for the Hub build of DeepEP.

Upstream checks NCCL and initializes its JIT at import time. A Hub kernel must
import cleanly everywhere (including on build machines without a GPU, CUDA
toolkit or recent NCCL), so these checks run when the first buffer is created
instead, and fail with an actionable message:

- PyTorch and DeepEP must use the same NCCL library. The Hub kernel does not
  link or ship NCCL: its NCCL symbols resolve to the library PyTorch loaded.
- That NCCL must be >= 2.32.3 (DeepEP 2.5 uses the NCCL device API and GIN
  features introduced in 2.31/2.32).
- DeepEP compiles its kernels at runtime with DeepJIT, which needs the NCCL
  headers matching the loaded library, a CUDA toolkit >= 13.1 (`nvcc`), and a
  host C++20 compiler.

Set `EP_SUPPRESS_NCCL_CHECK=1` to skip the NCCL checks (as upstream) and
`EP_SUPPRESS_NVCC_CHECK=1` to skip the toolkit check.
"""

import functools
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from . import _C
from .utils.find_pkgs import find_pkg_root

MIN_NCCL_VERSION = (2, 32, 3)
MIN_NVCC_VERSION = (13, 1)


class DeepEPRuntimeError(RuntimeError):
    pass


def _version_code(major: int, minor: int, patch: int) -> int:
    # NCCL >= 2.9 encodes versions as `major * 10000 + minor * 100 + patch`
    return major * 10000 + minor * 100 + patch


def _format_version(code: int) -> str:
    return f'{code // 10000}.{(code % 10000) // 100}.{code % 100}'


def _env_flag(name: str) -> bool:
    return bool(int(os.environ.get(name, '0') or '0'))


def get_loaded_nccl_libraries() -> Tuple[str, ...]:
    """Paths of the NCCL libraries mapped into this process, in load order."""
    libraries = []
    with open('/proc/self/maps', 'r') as f:
        for line in f:
            path = line.strip().split(' ')[-1]
            if 'libnccl' in os.path.basename(path) and '.so' in path and path not in libraries:
                libraries.append(path)
    return tuple(libraries)


def _has_nccl_headers(root: Path) -> bool:
    return (root / 'include' / 'nccl.h').is_file() and (root / 'include' / 'nccl_device.h').is_file()


def _read_header_version(root: Path) -> Optional[int]:
    text = (root / 'include' / 'nccl.h').read_text(errors='ignore')
    values = {}
    for name in ('NCCL_MAJOR', 'NCCL_MINOR', 'NCCL_PATCH'):
        match = re.search(rf'#define\s+{name}\s+(\d+)', text)
        if match is None:
            return None
        values[name] = int(match.group(1))
    return _version_code(values['NCCL_MAJOR'], values['NCCL_MINOR'], values['NCCL_PATCH'])


@functools.lru_cache()
def find_nccl_root() -> str:
    """
    The NCCL installation (`include/` and `lib/`) used for JIT compilation.

    Resolution order: `EP_NCCL_ROOT_DIR`, `NCCL_DIR`, the installation of the
    NCCL library PyTorch loaded, then an installed `nvidia-nccl-*` package.
    """
    for name in ('EP_NCCL_ROOT_DIR', 'NCCL_DIR'):
        if os.environ.get(name):
            return os.environ[name]

    for library in get_loaded_nccl_libraries():
        root = Path(library).resolve().parent.parent
        if _has_nccl_headers(root):
            return str(root)

    root = find_pkg_root('nccl', lib_name='libnccl.so', optional=True)
    if root is not None:
        return root

    raise DeepEPRuntimeError(
        'Cannot find the NCCL headers that DeepEP needs to JIT-compile its kernels. Install the NCCL package '
        'matching the NCCL library used by PyTorch (e.g. `pip install "nvidia-nccl-cu13>=2.32.3"` for CUDA 13), '
        'or set `EP_NCCL_ROOT_DIR` to a directory containing `include/` and `lib/`.')


def check_nccl() -> None:
    """Check that one NCCL >= 2.32.3 is loaded, and that the JIT headers match it."""
    if _env_flag('EP_SUPPRESS_NCCL_CHECK'):
        return

    import torch

    loaded = get_loaded_nccl_libraries()
    if len(loaded) > 1:
        raise DeepEPRuntimeError(
            f'Multiple NCCL libraries are loaded in this process: {", ".join(loaded)}. '
            'PyTorch and DeepEP must use the same NCCL library.')

    runtime_version = _C.get_nccl_runtime_version()
    if runtime_version < _version_code(*MIN_NCCL_VERSION):
        raise DeepEPRuntimeError(
            f'DeepEP requires NCCL >= {".".join(map(str, MIN_NCCL_VERSION))}, but PyTorch '
            f'{torch.__version__} uses NCCL {_format_version(runtime_version)}'
            f'{" (" + loaded[0] + ")" if loaded else ""}. DeepEP uses the NCCL library loaded by PyTorch, so '
            'upgrade the NCCL that PyTorch loads, e.g. `pip install "nvidia-nccl-cu13>=2.32.3"` '
            '(`nvidia-nccl-cu12` for CUDA 12 builds of PyTorch), or use a PyTorch build that ships it.')

    root = Path(find_nccl_root())
    if not _has_nccl_headers(root):
        raise DeepEPRuntimeError(
            f'NCCL root `{root}` has no `include/nccl.h` and `include/nccl_device.h`, which DeepEP needs to '
            'JIT-compile its kernels. Set `EP_NCCL_ROOT_DIR` to the NCCL installation used by PyTorch.')
    header_version = _read_header_version(root)
    if header_version is not None and header_version != runtime_version:
        raise DeepEPRuntimeError(
            f'The NCCL headers in `{root}` are version {_format_version(header_version)}, but the loaded NCCL '
            f'library is {_format_version(runtime_version)}. Set `EP_NCCL_ROOT_DIR` to the installation of '
            'the NCCL library loaded by PyTorch.')


def find_cuda_home() -> Optional[str]:
    """The CUDA toolkit DeepJIT will use, with the same lookup order as DeepJIT."""
    for name in ('CUDA_HOME', 'CUDA_PATH'):
        if os.environ.get(name):
            return os.environ[name]
    nvcc = shutil.which('nvcc')
    if nvcc is not None:
        return str(Path(nvcc).resolve().parent.parent)
    if os.path.exists('/usr/local/cuda'):
        return '/usr/local/cuda'
    return None


def check_cuda_toolkit() -> None:
    """Check that DeepJIT can find `nvcc` >= 13.1 and a host compiler."""
    if _env_flag('EP_SUPPRESS_NVCC_CHECK'):
        return

    required = '.'.join(map(str, MIN_NVCC_VERSION))
    cuda_home = find_cuda_home()
    nvcc = os.environ.get('EP_JIT_NVCC_COMPILER') or os.environ.get('DJ_JIT_NVCC_COMPILER')
    if not nvcc and cuda_home is not None:
        nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    if not nvcc or not os.access(nvcc, os.X_OK):
        raise DeepEPRuntimeError(
            f'DeepEP compiles its kernels at runtime and needs a CUDA toolkit >= {required} with `nvcc`, but none '
            f'was found (looked at `CUDA_HOME`, `CUDA_PATH`, `nvcc` on `PATH` and `/usr/local/cuda`). Install one '
            f'(e.g. `pip install "nvidia-cuda-nvcc>={required}" "nvidia-cuda-cccl>={required}" '
            f'"nvidia-cuda-crt>={required}" "nvidia-nvvm>={required}" "nvidia-cuda-runtime>={required}"`, then set '
            f'`CUDA_HOME` to its `nvidia/cu13` directory).')

    try:
        output = subprocess.check_output([nvcc, '--version'], stderr=subprocess.STDOUT, text=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise DeepEPRuntimeError(f'Failed to run `{nvcc} --version`: {e}') from e
    match = re.search(r'release (\d+)\.(\d+)', output)
    if match is None or (int(match.group(1)), int(match.group(2))) < MIN_NVCC_VERSION:
        found = f'{match.group(1)}.{match.group(2)}' if match else 'an unknown version'
        raise DeepEPRuntimeError(
            f'DeepEP needs `nvcc` >= {required} to compile its kernels, but `{nvcc}` is {found}. '
            'Point `CUDA_HOME` (or `EP_JIT_NVCC_COMPILER`) at a newer CUDA toolkit.')

    if not (os.environ.get('NVCC_CCBIN') or shutil.which('c++') or shutil.which('g++')):
        raise DeepEPRuntimeError(
            'DeepEP compiles its kernels at runtime and needs a host C++20 compiler (`c++`/`g++` on `PATH`, '
            'or `NVCC_CCBIN`).')


_initialized = False


def ensure_initialized() -> None:
    """Run the runtime checks and initialize DeepJIT, once."""
    global _initialized
    if _initialized:
        return
    check_nccl()
    check_cuda_toolkit()
    _C.init_jit(os.path.dirname(os.path.abspath(__file__)), find_nccl_root())
    _initialized = True
