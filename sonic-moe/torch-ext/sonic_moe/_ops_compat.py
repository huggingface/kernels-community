"""Compatibility helpers for op namespacing in source and built layouts."""

try:
    from ._ops import add_op_namespace_prefix as _generated_add_op_namespace_prefix
except ImportError:
    def _generated_add_op_namespace_prefix(name: str) -> str:
        return name if "::" in name else f"sonicmoe::{name}"

def add_op_namespace_prefix(name: str) -> str:
    return _generated_add_op_namespace_prefix(name)
