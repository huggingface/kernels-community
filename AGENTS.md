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
