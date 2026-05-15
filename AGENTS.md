# Kernel-specific instructions

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
