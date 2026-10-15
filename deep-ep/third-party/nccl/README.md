# NCCL headers

The `include/` directory of the `nvidia-nccl-cu13==2.32.3` wheel (NCCL
2.32.3, https://github.com/NVIDIA/nccl tag `v2.32.3-1`; `LICENSE.txt` is that
tag's license), vendored as a header-only build dependency of the DeepEP host
extension until kernel-builder provides NCCL headers.

The extension does not link or ship NCCL: at runtime it uses the NCCL library
that PyTorch loads (see `torch-ext/deep_ep/_nccl_symbols.py`), and DeepEP's
runtime JIT compiles against the headers of that library, not these.
