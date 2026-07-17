#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

from ._environment import HAS_LIBNATTEN
from .backends import (
    get_bwd_configs_for_cutlass_blackwell_fmha,
    get_bwd_configs_for_cutlass_blackwell_fna,
    get_bwd_configs_for_cutlass_fmha,
    get_bwd_configs_for_cutlass_fna,
    get_bwd_configs_for_cutlass_hopper_fmha,
    get_bwd_configs_for_cutlass_hopper_fna,
    get_configs_for_cutlass_blackwell_fmha,
    get_configs_for_cutlass_blackwell_fna,
    get_configs_for_cutlass_fmha,
    get_configs_for_cutlass_fna,
    get_configs_for_cutlass_hopper_fmha,
    get_configs_for_cutlass_hopper_fna,
    get_configs_for_flex_fmha,
    get_configs_for_flex_fna,
)
from .context import (
    allow_flex_compile,
    allow_flex_compile_backprop,
    are_deterministic_algorithms_enabled,
    disable_flex_compile,
    disable_flex_compile_backprop,
    get_memory_usage_preference,
    is_flex_compile_allowed,
    is_flex_compile_backprop_allowed,
    is_kv_parallelism_in_fused_na_enabled,
    is_memory_usage_default,
    is_memory_usage_strict,
    is_memory_usage_unrestricted,
    set_memory_usage_preference,
    use_deterministic_algorithms,
    use_kv_parallelism_in_fused_na,
)
from .functional import attention, merge_attentions, na1d, na2d, na3d
from .modules import (
    NeighborhoodAttention1D,
    NeighborhoodAttention2D,
    NeighborhoodAttention3D,
)
from .version import __version__

# kernel-builder port: the package contents are installed flat into the build
# variant directory, so a module literally named `types` would shadow the
# standard library `types` module whenever that directory is on PYTHONPATH
# (e.g. kernel-builder test shells and CI runners), breaking interpreter
# startup. The module therefore lives in `_types`; alias it here so
# `natten.types` keeps working like upstream.
import sys as _sys

from . import _types as types

_sys.modules[__name__ + ".types"] = types

# kernel-builder's compat shim (`natten/__init__.py` inside the build variant
# directory) executes this package under a path-derived module name and copies
# our globals into a `natten` module whose __path__ contains no submodules.
# Attribute access (`natten.functional`) works there, but real submodule
# imports (`from natten.functional import na2d`, `import natten.utils.testing`)
# would either fail with ModuleNotFoundError or — when resolvable through a
# parent package's __path__ — re-execute the module under a second name,
# duplicating module state. Bridge this with a meta-path finder that resolves
# any `natten.*` import to our already-loaded module objects. Only installed
# when `natten` in sys.modules is *our* compat shim, so a real `natten`
# distribution in the same environment is never hijacked.
if __name__ != "natten":
    from pathlib import Path as _Path

    _compat = _sys.modules.get("natten")
    _is_our_compat = (
        _compat is not None
        and getattr(_compat, "__file__", None) is not None
        and _Path(_compat.__file__).resolve()
        == _Path(__file__).resolve().parent / "natten" / "__init__.py"
    )

    if _is_our_compat:
        import importlib as _importlib
        from importlib.abc import Loader as _Loader
        from importlib.abc import MetaPathFinder as _MetaPathFinder
        from importlib.util import spec_from_loader as _spec_from_loader

        _real_root = __name__

        class _NattenAliasLoader(_Loader):
            def __init__(self, module):
                self._module = module
                self._spec = getattr(module, "__spec__", None)
                self._loader = getattr(module, "__loader__", None)

            def create_module(self, spec):
                return self._module

            def exec_module(self, module):
                # The import machinery stamped the alias spec onto the real
                # module in module_from_spec; restore its original identity.
                module.__spec__ = self._spec
                module.__loader__ = self._loader

        class _NattenAliasFinder(_MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if not fullname.startswith("natten."):
                    return None
                real_name = _real_root + fullname[len("natten") :]
                try:
                    module = _importlib.import_module(real_name)
                except ImportError:
                    return None
                return _spec_from_loader(fullname, _NattenAliasLoader(module))

        # Must precede PathFinder, which would otherwise re-execute
        # submodules reachable through a real parent package's __path__.
        _sys.meta_path.insert(0, _NattenAliasFinder())

__all__ = [
    "__version__",
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "NeighborhoodAttention3D",
    "are_deterministic_algorithms_enabled",
    "use_deterministic_algorithms",
    "use_kv_parallelism_in_fused_na",
    "is_kv_parallelism_in_fused_na_enabled",
    "set_memory_usage_preference",
    "get_memory_usage_preference",
    "is_memory_usage_default",
    "is_memory_usage_strict",
    "is_memory_usage_unrestricted",
    "is_flex_compile_allowed",
    "is_flex_compile_backprop_allowed",
    "allow_flex_compile",
    "allow_flex_compile_backprop",
    "disable_flex_compile",
    "disable_flex_compile_backprop",
    "get_bwd_configs_for_cutlass_fmha",
    "get_bwd_configs_for_cutlass_fna",
    "get_configs_for_cutlass_fmha",
    "get_configs_for_cutlass_fna",
    "get_configs_for_cutlass_hopper_fmha",
    "get_bwd_configs_for_cutlass_hopper_fmha",
    "get_configs_for_cutlass_hopper_fna",
    "get_bwd_configs_for_cutlass_hopper_fna",
    "get_bwd_configs_for_cutlass_blackwell_fmha",
    "get_bwd_configs_for_cutlass_blackwell_fna",
    "get_configs_for_cutlass_blackwell_fmha",
    "get_configs_for_cutlass_blackwell_fna",
    "get_configs_for_flex_fmha",
    "get_configs_for_flex_fna",
    "HAS_LIBNATTEN",
    "na1d",
    "na2d",
    "na3d",
    "attention",
    "merge_attentions",
]
