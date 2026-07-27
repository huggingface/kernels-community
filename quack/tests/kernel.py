"""Access to the kernel under test.

Tests must exercise the kernel the way consumers do, i.e. through
``kernels.get_kernel`` rather than a direct ``import quack``. The
``kernel-builder`` devshells (``kernel-builder devshell`` /
``kernel-builder testshell``) set ``LOCAL_KERNELS`` so that this resolves to
the local build instead of downloading from the Hub.

``get_kernel`` registers the package in ``sys.modules`` under a randomized
name, so submodules cannot be imported by their upstream dotted path. Use
:func:`submodule` to reach into the package.
"""

import importlib
from types import ModuleType

import kernels

quack = kernels.get_kernel("kernels-community/quack", version=0)


def submodule(name: str) -> ModuleType:
    """Import ``quack.<name>`` from the loaded kernel.

    For example, ``submodule("blockscaled.utils")`` returns what upstream
    calls ``quack.blockscaled.utils``.
    """
    return importlib.import_module(f"{quack.__name__}.{name}")
