---
tags:
- kernels
---

# FlyDSL

[FlyDSL](https://github.com/ROCm/FlyDSL) (ROCm Domain Specific Language for
layout algebra, with an embedded MLIR runtime) packaged as an SDK for the
`kernels` library. It has no torch dependency of its own.

## What is packaged

This is **not built from source**. It is made by repackaging the upstream
release wheel (`v0.5.0`, `cp314`, x86_64) into the kernel format. Modifications
are limited to:

- Replacing the absolute `flydsl` imports (and the `import flydsl` used to
  locate the package) by relative ones, since the kernel is not importable under
  the name `flydsl`.
- Replacing the RPATHs by `$ORIGIN`-relative ones. The vendored libraries
  (`libzstd`, ...) are in `flydsl.libs`.

## Build variants

There is a single variant, `sdk-rocm-7.0-abi3-x86_64-linux`.

**Note:** the compiled extensions are built for Python 3.14 (`cp314`) and do
not use the stable ABI. The SDK variant format cannot express a Python
version yet, so the variant is marked as `abi3` for now. It will fail to
import on other Python versions.

## Requirements and caveats

- ROCm 7 (`libamdhip64.so.7`) is needed to compile and run kernels.
- The upstream wheel is `manylinux_2_27`.
- The package is not registered under the name `flydsl`, so code that does
  `import flydsl` by name does not see it.
- The device architectures in the metadata are the GPUs that upstream lists as
  tested; kernels are compiled at runtime.

## Updating FlyDSL

1. Get the URL and hash of the `cp314` wheel from
   `https://pypi.org/pypi/flydsl/<version>/json` and convert the hash with
   `nix hash convert --hash-algo sha256 --to sri <hex>`.
2. Update `version` and `wheelFor` in `flake.nix`.
3. Rebuild and check that the `substituteInPlace` patches in
   `flydsl-kernel.nix` still apply (they use `--replace-fail`) and that no
   new absolute `flydsl` imports appeared.

## Testing

```bash
nix build .#ci-test && ./result/bin/ci-test
```
