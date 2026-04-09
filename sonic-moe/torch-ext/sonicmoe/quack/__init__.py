__version__ = "0.2.5"

import os

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from . import cute_dsl_ptxas

    cute_dsl_ptxas.patch()
