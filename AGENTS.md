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

## aiter-kernels

This package mirrors the **Triton subset** of `ROCm/aiter` upstream as a
single Hub kernel — same idea as `liger-kernels` for `linkedin/Liger-Kernel`.
Flash Attention is intentionally out of scope here; it lives in
`aiter-flash-attn`. When the user asks to sync an aiter-kernels release,
carry out the following steps:

- Fetch the upstream Git repository from https://github.com/ROCm/aiter.git
- Check out the tag the user specified (e.g. `v0.1.5`).
- The Triton ops live under `aiter/ops/triton/` upstream. Mirror that subtree
  into `aiter-kernels/torch-ext/aiter_kernels/`, but exclude everything
  Flash-Attention-related:
  - `aiter/ops/triton/attention/**`
  - `aiter/ops/triton/_triton_kernels/attention/**`
  - `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/**`
  - `aiter/ops/triton/_gluon_kernels/*/attention/**`
  - `aiter/ops/triton/gluon/mla_decode_gluon.py`,
    `pa_decode_gluon.py`, `pa_mqa_logits.py`
  - `aiter/ops/triton/configs/*MHA*`, `*EXTEND_ATTENTION*`,
    `*LEANATTN*`, `*MLA*`
  - `aiter/ops/triton/quant/sage_attention_quant_wrappers.py` (depends on
    excluded attention internals)
- A clean way to mirror is:
  ```bash
  rsync -av \
    --exclude='attention/' \
    --exclude='flash_attn_triton_amd/' \
    --exclude='mla_decode_gluon.py' \
    --exclude='pa_decode_gluon.py' \
    --exclude='pa_mqa_logits.py' \
    --exclude='configs/*MHA*' \
    --exclude='configs/*EXTEND_ATTENTION*' \
    --exclude='configs/*LEANATTN*' \
    --exclude='configs/*MLA*' \
    aiter/ops/triton/ aiter-kernels/torch-ext/aiter_kernels/
  rm -f aiter-kernels/torch-ext/aiter_kernels/quant/sage_attention_quant_wrappers.py
  ```
- After mirroring, also drop `utils/_triton/tunning/` — those are upstream
  CLI tuning scripts with top-level `argparse` that crash on package import.
- Rewrite imports to use **relative** form. The Hub kernel loader imports the
  package under the variant directory name (e.g. `torch-rocm/`), not
  `aiter_kernels/`, so absolute `from aiter_kernels.X` imports break at
  load time. Apply the global sed pass first to land on a single absolute
  namespace, then run the depth-aware rewriter to convert to relative:
  - `from aiter.ops.triton.<x>` → `from aiter_kernels.<x>` (intermediate)
  - `from aiter.ops.triton import <x>` → `from aiter_kernels import <x>`
  - `import aiter.ops.triton.<x> as Y` → `import aiter_kernels.<x> as Y`
  - `from aiter.jit.utils.torch_guard import ...` →
    `from aiter_kernels._aiter_compat.torch_guard import ...`
  - `from aiter.utility.triton.triton_metadata_redirect import ...` →
    `from aiter_kernels._aiter_compat.triton_metadata_redirect import ...`
  - `from aiter import dtypes` / `from aiter.utility import dtypes` →
    `from aiter_kernels._aiter_compat import dtypes`
  - Bare `import aiter` (with later use of `aiter.dtypes.*`) →
    `from aiter_kernels import _aiter_compat as aiter`
  Then convert every absolute `aiter_kernels.X` import to relative form
  based on the file's depth from the package root. A file at
  `aiter_kernels/<a>/<b>/.../file.py` needs `(depth + 1)` dots — one for
  "current package", plus one for each level climbed back to
  `aiter_kernels/`. Patterns to handle:
  - `from aiter_kernels.X.Y import Z` → `from <dots>X.Y import Z`
  - `from aiter_kernels import Z` → `from <dots> import Z`
  - `import aiter_kernels.X.Y.Z as W` → `from <dots>X.Y import Z as W`
    (the `import ... as ...` form has no relative equivalent; rewrite as
    `from ... import ... as ...`)
- Replace the upstream `aiter/ops/triton/__init__.py` with the local curated
  init — do **not** copy upstream's `__init__.py` over. The local init
  imports the subpackages and exposes `apply_rotary_transformers` via
  `aiter_kernels.rope`. Upstream's `_BACKWARD_COMPAT_MAP` shim is for the
  `aiter.ops.triton.*` namespace and is irrelevant here.
- Ensure every directory containing `.py` files has an `__init__.py` —
  upstream relies on Python's namespace-package behavior for some
  subdirectories; add empty inits where missing so module discovery is
  deterministic.
- Do not modify `aiter-kernels/torch-ext/aiter_kernels/_aiter_compat/` —
  this is a local shim layer for the cross-tree dependencies
  (`dtypes`, `chip_info.get_gfx`, `torch_compile_guard`,
  `AOTMetadataContext`). It is **not** derived from `aiter/ops/triton/` so
  the sync should never overwrite it. If a newly-synced op imports
  something not yet in `_aiter_compat`, add the symbol there rather than
  importing across the tree boundary.
- Do not modify the local `rope/__init__.py` — it carries the
  `apply_rotary_transformers` transformers-compat shim, which is a local
  addition that's not derived from upstream.
- Smoke-test the sync with a recursive `pkgutil.walk_packages` import: every
  module under `aiter_kernels.*` should import cleanly except
  `aiter_kernels.comms` (which requires the optional `iris` package).
- Check whether any Torch custom ops are defined in the synced files
  (`torch.library.custom_op`, `torch.library.define`). If any are found,
  update them to use `add_op_namespace_prefix` for the op name. Upstream's
  Triton ops do not currently register ops via `torch.library` directly —
  the `torch_compile_guard` decorator was the only path, and our shim makes
  it a no-op — so this step is usually a no-op.

If the user did not specify the version tag, stop and ask which tag to sync
from.
