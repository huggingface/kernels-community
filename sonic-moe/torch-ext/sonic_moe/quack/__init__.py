__version__ = "0.5.0"

import os

# Try the binary shim first (it patches both libnvvm and ptxas inside the
# CUTLASS DSL `.so` with a single mprotect+trampoline pass, no per-compile
# Python overhead). The shim is off by default and activates only when
# QUACK_CUTE_DSL_SHIM=1 is set.
#
# If the shim activates, the legacy Python `cute_dsl_ptxas` hook is skipped
# — otherwise it falls back when CUTE_DSL_PTXAS_PATH is set.
from .dsl import cute_dsl_shim as _cute_dsl_shim

_shim_active = _cute_dsl_shim.try_activate()

if not _shim_active and os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from .dsl import cute_dsl_ptxas as _cute_dsl_ptxas

    # Patch before importing any modules that instantiate CuTeDSL. The patch
    # forces PTX dumping so the CUDA library loader can replace CUTLASS DSL's
    # embedded ptxas-library cubin with one assembled by system ptxas.
    _cute_dsl_ptxas.patch()
