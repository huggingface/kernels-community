"""CuTe DSL helpers and integration hooks.

Importing this package installs quack's Pythonic CuTe tensor indexing, which
monkey-patches CuTe's tensor classes process-wide so that `...` and `:` work in
`__getitem__`/`__setitem__`. `quack.copy_utils` relies on that sugar (e.g.
`tRS_sC[..., dst_idx]`), and the sm90 backward kernel reaches it.

Upstream's `quack.dsl` also exports `cute_op` from `torch_library_op`; nothing
vendored here uses it, so it is not copied.
"""

from . import cute_tensor_indexing  # noqa: F401
