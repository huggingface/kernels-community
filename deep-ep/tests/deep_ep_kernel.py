"""
kernels-community: load DeepEP through `kernels.get_kernel` for the upstream tests.

The upstream tests `import deep_ep` and its submodules by name. Load the Hub
kernel instead (in CI and `kernel-builder testshell`, `LOCAL_KERNELS` points
`get_kernel` at the local build) and register it, with its already-imported
submodules, under the upstream package name.
"""

import sys

import kernels

# `kernels-community/deep-ep` is `[general.hub] repo-id` and 1 is `[general] version` in `build.toml`
deep_ep = kernels.get_kernel("kernels-community/deep-ep", version=1)

for _name, _module in list(sys.modules.items()):
    if _name == deep_ep.__name__ or _name.startswith(deep_ep.__name__ + "."):
        sys.modules["deep_ep" + _name[len(deep_ep.__name__):]] = _module
