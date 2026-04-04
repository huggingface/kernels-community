__version__ = "0.2.5"

import os

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    import quack.cute_dsl_ptxas  # noqa: F401

    quack.cute_dsl_ptxas.patch()
