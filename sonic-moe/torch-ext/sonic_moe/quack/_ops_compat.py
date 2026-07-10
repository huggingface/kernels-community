from .._ops import add_op_namespace_prefix as _add_op_namespace_prefix



# For quack we need to prefix the function name because some names
# overlap between quack and sonic-moe itself. Name the function the
# same as the function it is wrapping for the prefix check to be
# happy.
def add_op_namespace_prefix(name: str) -> str:
    return _add_op_namespace_prefix(f"quack__{name}")
