__version__ = "0.3.11"

import os

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from . import cute_dsl_ptxas  # noqa: F401

    cute_dsl_ptxas.patch()
