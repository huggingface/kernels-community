# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

_NAMESPACE = "finegrained_kernels"


def add_op_namespace_prefix(name: str) -> str:
    return f"{_NAMESPACE}::{name}"


ops = getattr(torch.ops, _NAMESPACE)
