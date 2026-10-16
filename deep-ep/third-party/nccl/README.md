# NCCL headers

Public NCCL 2.32.3 headers, reproduced by `port/port.kdl` from NVIDIA/nccl
commit `12df1a11afad322be5a204a2db890161cbf8131d` (tag `v2.32.3-1`), including
its DOCA/EFA device headers and version-template substitutions. These match
the headers in the `nvidia-nccl-cu13==2.32.3` wheel. `LICENSE.txt` comes from
the same Git commit. The headers are a build-only dependency of the DeepEP
host extension until kernel-builder provides NCCL headers.

The extension does not link or ship NCCL: at runtime it uses the NCCL library
that PyTorch loads (see `torch-ext/deep_ep/_nccl_symbols.py`), and DeepEP's
runtime JIT compiles against the headers of that library, not these.
