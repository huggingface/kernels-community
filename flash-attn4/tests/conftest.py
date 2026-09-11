"""Resolve ``flash_attn4`` to the built kernel for the vendored upstream suite.

The upstream tests under ``tests/cute`` import ``flash_attn4`` as a top-level
module, but a Hub kernel is loaded under a unique name (``_flash_attn4_cuda_``
plus a hash) so that several versions can coexist in one process. Without help,
every vendored module fails to import and ``pytest -m kernels_ci`` dies during
collection before it reaches ``test_kernels_ci.py``.

Rather than rewriting all of the vendored files -- which would make future
upstream syncs painful -- install an import alias so that ``flash_attn4`` and
``flash_attn4.<submodule>`` resolve to the module ``get_kernel`` returned. The
vendored suite then exercises the packaged kernel rather than the source tree,
which is what the kernel guidelines ask for.

When the kernel cannot be loaded (for example a local checkout run with
``PYTHONPATH=torch-ext``), no alias is installed and a real ``flash_attn4`` on
``sys.path`` is used as-is.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys

_ALIAS = "flash_attn4"


def _load_kernel():
    try:
        import kernels
    except ImportError:
        return None
    try:
        return kernels.get_kernel("kernels-community/flash-attn4", version=0)
    except Exception:
        return None


class _AliasLoader(importlib.abc.Loader):
    """Bind an already-imported module under a second name."""

    def __init__(self, real_name):
        self._real_name = real_name

    def create_module(self, spec):
        return importlib.import_module(self._real_name)

    def exec_module(self, module):
        # The real module was executed when it was first imported.
        pass


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Map ``flash_attn4[.submodule]`` onto the loaded kernel's module tree."""

    def __init__(self, real_root):
        self._real_root = real_root

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _ALIAS and not fullname.startswith(_ALIAS + "."):
            return None
        real_name = self._real_root + fullname[len(_ALIAS) :]
        try:
            module = importlib.import_module(real_name)
        except ImportError:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _AliasLoader(real_name),
            is_package=hasattr(module, "__path__"),
        )


def pytest_ignore_collect(collection_path, config):
    """Skip the vendored suite when running the CI selection.

    ``nix run .#ci-test`` invokes ``pytest tests -m kernels_ci``. None of the
    vendored tests carry that marker, so collecting their ~466k
    parametrizations costs about 35 seconds and selects nothing. Developer runs
    (no ``-m kernels_ci``) collect them as usual.
    """
    if "kernels_ci" in (config.option.markexpr or ""):
        return "cute" in collection_path.parts
    return None


def _install_alias():
    if _ALIAS in sys.modules:
        return
    try:
        importlib.import_module(_ALIAS)
    except ImportError:
        pass
    else:
        # A real flash_attn4 is importable; leave it alone.
        return

    kernel = _load_kernel()
    if kernel is None:
        return
    # Insert ahead of PathFinder so the alias wins for flash_attn4.* names.
    sys.meta_path.insert(0, _AliasFinder(kernel.__name__))
    sys.modules[_ALIAS] = kernel


_install_alias()
