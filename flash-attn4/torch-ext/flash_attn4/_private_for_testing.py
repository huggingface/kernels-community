"""Internal symbols reachable for the test suite.

Modules re-exported here are **not** public API. They carry no stability or
versioning guarantee and must not be used by downstream code -- they are
exposed only so that the tests can reach them through ``get_kernel``, which
loads the kernel under a generated module name.

Anything genuinely public belongs in ``__init__.py``'s ``__all__`` instead,
where changing it requires a kernel version bump.
"""

from . import compute_block_sparsity  # noqa: F401

__all__ = [
    "compute_block_sparsity",
]
