# Kernel-specific instructions

## flash-attn3

When the user asks to sync a flash-attn3 release, carry out the following
steps:

- Fetch the upstream Git repository from https://github.com/Dao-AILab/flash-attention.git
- Check out the tag that the user specified.
- Flash Attention 3 is in the directory `hopper` of the upstream repo.
- Copy upstream Flash Attention 3 CUDA files, as well as the stable API (`flash_api_stable.cpp`) to
  `flash-attn3/flash-attn`.
- Copy the Flash Attention 3 upstream `flash_attn_interface` Python modules to
  `flash-attn3/torch-ext/flash_attn3`.
- Copy the Python files prefixed with `test_` to `flash-attn3/tests`.
- In every copied or updated test file, rewrite imports of `flash_attn_interface` to use
  `flash_attn3`. For instance, `import flash_attn_interface` → `import flash_attn3.flash_attn_interface as flash_attn_interface`
- Relativize imports of local test helpers and modules (files placed under `flash-attn3/tests`) so
  they import from the local package rather than as top-level modules.
- Add the following imports to `flash-attn3/torch-ext/flash_attn_interface.py`:

  ```
  from ._ops import ops as flash_attn_3_cuda
  from ._ops import add_op_namespace_prefix
  ```

- Validate that all native (op) calls in `flash-attn3/torch-ext/flash_attn_interface.py` are
  dispatched to `flash_attn3_cuda`.
- Do not modify `flash-attn3/torch-ext/flash-attn3/__init__.py`, this is a local re-export of the
  flash-attn3 functions. However, do check if anything vital is missing and if so, report to the
  user.
- Check whether any Torch custom ops are defined in `flash-attn3/torch-ext/flash_attn_interface.py`
  (look for `torch.library.custom_op`, `torch.library.define`, etc.). If any are found, update them
  to use `add_op_namespace_prefix` for the op name. For example, a definition like
  `@torch.library.custom_op("_flash_attn_forward", mutates_args=(), device_types="cuda")`
  should become
  `@torch.library.custom_op(add_op_namespace_prefix("_flash_attn_forward"), mutates_args=(), device_types="cuda")`.
  `add_op_namespace_prefix` is imported from `._ops` (see
  `flash-attn3/torch-ext/flash_attn3/flash_attn_interface.py` prior to the update for an example).

## flash-attn4

When the user asks to sync a flash-attn4 release, carry out the following
steps:

- Fetch the upstream Git repository from https://github.com/Dao-AILab/flash-attention.git
- Check out the tag that the user specified.
- Flash Attention 4 is in the directory `flash_attn/cute` of the upstream repo.
- Copy Flash Attention 4 upstream files to `flash-attn4/torch-ext/flash_attn4`.
- Copy tests from the tests from the upstream directory `tests/cute` to
  `flash-attn4/tests/cute`.
- Check in `flash_attn/cute/pyproject.toml` upstream what version of quack is
  required.
- Get this version of quack from https://github.com/Dao-AILab/quack.git
- Copy the `quack` directory from quack to `flash-attn4/torch-ext/flash_attn4/quack`
- Now make all imports of Flash Attention 4 and quack in
  `flash-attn4/torch-ext/flash_attn4` and `flash-attn4/torch-ext/flash_attn4/quack`
  relative imports.
- Remove all quack files in `flash-attn4/torch-ext/flash_attn4/quack` that are not used.
- Update imports of `flash_attn.cute` in `flash-attn4/tests/cute` to `flash_attn4`.
- Set `__version__` in `flash-attn4/torch-ext/flash_attn4/__init__.py` to the
  version from the tag (e.g. for tag `fa4-v4.0.0.beta8` set it to
  `"4.0.0.beta8"`). Remove any `importlib.metadata` version lookup code.
- Check whether any Torch custom ops are defined in `flash-attn4/torch-ext/flash_attn4`
  or `flash-attn4/torch-ext/flash_attn4/quack` (look for `torch.library.custom_op`,
  `torch.library.define`, etc.). If any are found, update them to use
  `add_op_namespace_prefix` for the op name. For example, a definition like
  `@torch.library.custom_op("_flash_attn_forward", mutates_args=(), device_types="cuda")`
  should become
  `@torch.library.custom_op(add_op_namespace_prefix("_flash_attn_forward"), mutates_args=(), device_types="cuda")`.
  `add_op_namespace_prefix` is imported from `._ops` (see
  `flash-attn3/torch-ext/flash_attn3/flash_attn_interface.py` for an example).

If the user did not specify the version tag, stop and ask which tag to sync
from.

## msa

When the user asks to sync an MSA (MiniMax Sparse Attention) release, carry
out the following steps:

- Fetch the upstream Git repository from https://github.com/MiniMax-AI/MSA.git
- Check out the tag that the user specified.
- Only the CuTe-DSL sparse stack in `python/fmha_sm100/cute` upstream is
  mirrored. The dense FMHA stack (`python/fmha_sm100/csrc`, `api.py`,
  `jit.py`) is intentionally excluded — it relies on runtime `nvcc`/jinja JIT
  compilation, which is incompatible with kernel-builder.
- Copy the upstream `cute` top-level modules (`interface.py`,
  `fp4_indexer_interface.py`, `quantize.py`, `sparse_index_utils.py`) and the
  `cute/src` tree to `msa/torch-ext/msa/`. Do not copy `example.py`,
  `Makefile`, `pytest.ini`, or `.gitignore`.
- The two CUDA C++ helper extensions are precompiled Torch ops in this repo
  instead of `torch.utils.cpp_extension.load` JIT modules:
  - `cute/src/sm100/build_k2q_csr/build_k2q_csr.cu` → `msa/csrc/build_k2q_csr.cu`
  - `cute/src/sm100/fwd_decode/build_decode_schedule/build_decode_schedule.cu`
    → `msa/csrc/build_decode_schedule.cu`
  When syncing, diff the upstream `.cu` files against the local copies and
  re-apply the local porting changes: `torch/extension.h` → `torch/all.h`,
  no pybind11 (ops are registered in `msa/torch-ext/torch_binding.cpp`), and
  `build_decode_schedule` returns a tuple of tensors plus a fixed-order
  `int[]` scalar summary instead of a `py::dict`. The corresponding Python
  wrapper `__init__.py` files under `msa/torch-ext/msa/src/sm100/` dispatch
  through `._ops` and must keep their public signatures in sync with
  upstream.
- Check in `python/fmha_sm100/cute/requirements.txt` (or the upstream
  `pyproject.toml`) which version of quack is required.
- Get this version of quack from https://github.com/Dao-AILab/quack.git
- Vendor only the quack modules the stack imports (currently
  `layout_utils`, `copy_utils`, `cute_dsl_utils`, `activation`,
  `compile_utils`) into `msa/torch-ext/msa/quack/`, with an empty
  `__init__.py` and the quack `LICENSE`.
- Now make all imports of the cute stack and quack in `msa/torch-ext/msa`
  relative imports. In particular:
  - `from src.<x> import ...` → relative import of `.src.<x>` (dots according
    to the depth of the importing file).
  - `from quack import ...` / `from quack.<x> import ...` → relative imports
    of the vendored `quack` package.
  - `import quack.activation` → `from <dots>quack import activation`, with
    `quack.activation.` references renamed to `activation.`.
  - `import src.common.utils as utils` → `from <dots>src.common import utils`.
- Copy `cute/test_sparse_atten.py` and `cute/test_fp4_indexer.py` to
  `msa/tests/` and rewrite the top-level imports (`interface`,
  `sparse_index_utils`, `fp4_indexer_interface`, `quantize`, `src.*`,
  `quack.*`) to import from the `msa` package.
- Keep `msa/torch-ext/msa/__init__.py` in sync with the public API surface
  re-exported by upstream `python/fmha_sm100/sparse.py`, and set
  `__version__` to the upstream package version from `pyproject.toml`.
- Check whether any Torch custom ops are defined in the synced files (look
  for `torch.library.custom_op`, `torch.library.define`, etc.). If any are
  found, update them to use `add_op_namespace_prefix` for the op name,
  importing it from `._ops`. Upstream MSA does not currently define any such
  ops, so this step is usually a no-op.

If the user did not specify the version tag, stop and ask which tag to sync
from.

## liger-kernels

When the user asks to sync a Liger-Kernel release, carry out the following
steps:

- Fetch the upstream Git repository from https://github.com/linkedin/Liger-Kernel.git
- Check out the tag that the user specified (e.g. `v0.8.0`).
- Liger-Kernel ops are in the directory `src/liger_kernel/ops` of the upstream
  repo. There is no `cute` / `quack` dependency — only the Triton ops are
  mirrored.
- For each file already present in `liger-kernels/torch-ext/liger_kernels/`,
  except `__init__.py` and `layers.py` (which are local additions, not
  upstream), copy the matching file from `src/liger_kernel/ops/` upstream,
  overwriting the local copy. Do not introduce new files that were not
  previously synced — the local tree intentionally tracks a subset of upstream
  ops.
- Some helpers used by the ops live in the upstream top-level module
  `src/liger_kernel/utils.py` (e.g. `is_npu_available`) rather than in
  `src/liger_kernel/ops/utils.py`. If a synced op imports such a helper, fold
  the helper into `liger-kernels/torch-ext/liger_kernels/utils.py` instead of
  adding a separate file.
- Now make all imports of Liger-Kernel in
  `liger-kernels/torch-ext/liger_kernels` relative imports. In particular:
  - `from liger_kernel.ops.<name> import ...` → `from .<name> import ...`
  - `from liger_kernel.ops.utils import ...` → `from .utils import ...`
  - `from liger_kernel.utils import ...` → `from .utils import ...` (after
    folding the helper into the local `utils.py`).
- Do not modify `liger-kernels/torch-ext/liger_kernels/__init__.py` or
  `layers.py` — these are local `torch.nn.Module` wrappers that are not
  derived from upstream.
- Check whether any Torch custom ops are defined in the synced files (look
  for `torch.library.custom_op`, `torch.library.define`, etc.). If any are
  found, update them to use `add_op_namespace_prefix` for the op name,
  importing it from `._ops` (see
  `flash-attn3/torch-ext/flash_attn3/flash_attn_interface.py` for an
  example). Upstream Liger-Kernel does not currently define any such ops, so
  this step is usually a no-op.

If the user did not specify the version tag, stop and ask which tag to sync
from.
