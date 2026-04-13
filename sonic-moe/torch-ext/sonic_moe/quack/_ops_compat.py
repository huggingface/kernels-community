from .._ops_compat import add_op_namespace_prefix

def add_quack_op_namespace_prefix(name: str) -> str:
    return add_op_namespace_prefix(f"quack__{name}")
