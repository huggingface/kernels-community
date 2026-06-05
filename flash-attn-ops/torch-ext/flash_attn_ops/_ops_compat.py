"""Compatibility helpers for op namespacing in source and built layouts.

In the built (Hub) layout, kernel-builder generates an `_ops` module that
exposes `add_op_namespace_prefix`, which prefixes op names with a unique,
build-hashed namespace so custom ops never collide across kernels/versions. When
running directly from source there is no generated `_ops`, so we fall back to a
fixed namespace.
"""

try:
    from ._ops import add_op_namespace_prefix as _generated_add_op_namespace_prefix
except ImportError:
    def _generated_add_op_namespace_prefix(name: str) -> str:
        return name if "::" in name else f"flash_attn_ops::{name}"


def add_op_namespace_prefix(name: str) -> str:
    return _generated_add_op_namespace_prefix(name)
