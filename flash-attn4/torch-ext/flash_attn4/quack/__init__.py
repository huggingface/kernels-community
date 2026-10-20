"""Vendored subset of quack (https://github.com/Dao-AILab/quack).

Importing `.dsl` installs quack's CuTe tensor indexing patch. It must happen
here, exactly as upstream's `quack/__init__.py` does it, because
`quack.copy_utils` indexes tensors with `...` and CuTe does not support that
without the patch.
"""

from . import dsl  # noqa: F401
