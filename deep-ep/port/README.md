# DeepEP port recipe

`port.kdl` and `overlay/` are the source of truth for this port. They reproduce
the existing Torch ops/custom-class implementation and version-specific Torch
builds. The recipe does not convert to the Torch stable ABI.

Review the source pins, guarded recipe edits, and handwritten overlays.
Imported upstream sources are derived mechanically. Edit the recipe/overlays,
regenerate separately, and update the generated files together. Normal CI
builds the checked-in generated kernel and needs no recipe-time network access.

The recipe verifies clean input checkouts and exact edit counts, makes Hub
package imports relative, rewrites tests to load through `kernels.get_kernel`,
and generates the manifest and deterministic `port-provenance.json`. Overlays
hold only local bindings, runtime checks, CI tests, build metadata and docs.
The manifest op does not yet expose Torch binding flags directly, so a guarded
edit adds the original flags to its generated table.

## Reproduce

Use kernel-port from `huggingface/kernels` at
`f6bd904093ee680b77dbcee2b9c9292bb5f2a722`, built with
`cargo build -p kernel-port --release` in that checkout. The tool and its recipe
language are experimental; this recipe uses format 1.

Prepare clean checkouts with these origins and commits:

| Input | Origin | Commit |
|---|---|---|
| DeepEP | `https://github.com/deepseek-ai/DeepEP.git` | `def865146e95aa9ea223e277c0a8a4c6b81e5ac7` |
| DeepJIT | `https://github.com/deepseek-ai/DeepJIT.git` | `cba802df081985141eeee36803685775c183a047` |
| NCCL | `https://github.com/NVIDIA/nccl.git` | `12df1a11afad322be5a204a2db890161cbf8131d` |

DeepEP is also supplied as the `python` vendor to derive the Hub Python package
from the same source while retaining the upstream layout. NCCL's public headers,
including DOCA/EFA device dependencies, are assembled from pinned Git sources
and version-template substitutions. They match the previously vendored wheel
headers; no installed wheel supplies recipe inputs.

```sh
python deep-ep/port/regenerate.py \
  --kernel-port /path/to/kernels/target/release/kernel-port \
  --deep-ep /path/to/DeepEP --deepjit /path/to/DeepJIT \
  --nccl /path/to/nccl --output /tmp/deep-ep-generated --check
```

`--check` compares every generated file, including provenance, with this kernel,
excluding `port/` and Python/tool caches. Without `--check`, `--output` produces
a tree for inspection. Output must not overlap the kernel or input checkouts:
the runner replaces its output directory. After reviewing regenerated files,
sync them while preserving this recipe:

```sh
rsync -a --delete --exclude=port/ /tmp/deep-ep-generated/ deep-ep/
```

Then build normally with kernel-builder and run the existing CI/tests using
`LOCAL_KERNELS`. Changes to source pins or porting operations still require
normal code review and validation; successful regeneration alone is not a
correctness or performance check.
