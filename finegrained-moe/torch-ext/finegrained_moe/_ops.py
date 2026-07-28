"""Source-tree fallback for the build-generated ``_ops`` module.

When the package is built via the kernels-community pipeline, the build emits a
``_ops`` module that wires ``ops`` to the namespaced ``torch.ops`` entry and sets
``add_op_namespace_prefix`` to use the build-time namespace (which includes a
hash). This stub lets the package import unbuilt — needed for running the test
suite against the source tree directly. The ``@triton_op`` decorators register
themselves into the same namespace via ``add_op_namespace_prefix``, so the
fallback only needs both names to agree on whatever string we pick.
"""

import torch

_NAMESPACE = "finegrained_moe"


def add_op_namespace_prefix(name: str) -> str:
    return f"{_NAMESPACE}::{name}"


ops = getattr(torch.ops, _NAMESPACE)
