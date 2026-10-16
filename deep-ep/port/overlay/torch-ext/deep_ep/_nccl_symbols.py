"""
kernels-community: make PyTorch's NCCL symbols visible to the DeepEP extension.

Upstream links `deep_ep._C` against NCCL. The Hub kernel must use the very
same NCCL library as PyTorch (DeepEP shares PyTorch's communicators), so it
neither links nor ships NCCL; its NCCL symbols are left undefined. PyTorch
loads its NCCL privately (as a dependency of `libtorch_cuda`), so before the
extension is loaded, re-open that already-loaded library with `RTLD_GLOBAL`
(`RTLD_NOLOAD` guarantees that no second copy is ever loaded).

This module must be imported before `._ops`, and must not import it.
"""

import ctypes
import os

import torch  # noqa: F401  (loads libtorch_cuda and with it PyTorch's NCCL)


def _loaded_nccl_libraries():
    libraries = []
    with open('/proc/self/maps', 'r') as f:
        for line in f:
            path = line.strip().split(' ')[-1]
            if os.path.basename(path).startswith('libnccl') and '.so' in path and path not in libraries:
                libraries.append(path)
    return libraries


def expose_torch_nccl() -> str:
    libraries = _loaded_nccl_libraries()
    if not libraries:
        raise ImportError(
            'DeepEP uses the NCCL library loaded by PyTorch, but no NCCL library is loaded in this process. '
            'DeepEP requires a CUDA build of PyTorch with NCCL support.')
    # NOTES: if several are loaded, the runtime checks report it when a buffer is created
    ctypes.CDLL(libraries[0], mode=os.RTLD_NOLOAD | os.RTLD_GLOBAL)
    return libraries[0]


torch_nccl_library = expose_torch_nccl()
