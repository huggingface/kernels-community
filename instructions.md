# Handoff: kernels-community issue #1050 (smoke tests in CI, starting with `relu`)

## Task
Implement https://github.com/huggingface/kernels-community/issues/1050 in
`huggingface/kernels-community`. The issue asks to:
1. Convert kernel tests to use `get_kernel` instead of importing the built package directly.
2. Rely on `LOCAL_KERNELS` being set in CI to point at the freshly built kernel ("worth verifying").
3. Mark a subset of cheap tests with the `kernels_ci` marker.
Suggested first step: do `relu`, fire up a build, fix what breaks, then roll out.

## Work already done (uncommitted, on branch `main`)
Three modified files, `git diff --stat`:
```
 AGENTS.md               | 41 +++++++++++++++++++++++++++++++++++++++++
 relu/flake.lock         | 20 ++++++++++----------
 relu/tests/test_relu.py | 35 ++++++++++++++++-------------------
```

### 1. `relu/tests/test_relu.py` — converted to `get_kernel`
Replaced `import relu` with:
```python
import kernels
relu = kernels.get_kernel("kernels-community/relu", version=1)
```
(repo id from `[general.hub] repo-id`, version from `[general] version` in `relu/build.toml`).
Also factored the duplicated device-selection into a single `get_device()` helper — the two
tests had subtly different logic — and switched the Darwin branch from `platform.system()` to
`torch.backends.mps.is_available()`. Both tests keep their existing `@pytest.mark.kernels_ci`
markers. `relu.layers.ReLU` still resolves because `torch-ext/relu/__init__.py` does
`from . import layers`.

### 2. `relu/flake.lock` — bumped (this was the actual blocker)
Ran `python3 scripts/update_flakes.py relu`. `kernel-builder` went
`81580bb9…` (1783417137) -> `18c4d686…` (1787146545, kernels `main`).

**Why this was required:** at the old pin, `mkCiTests` in `nix-builder/lib/build.nix` built a
`ci-test` script that set only `PYTHONPATH` — it did **not** add the `kernels` Python package to
`testPython` and did **not** export `LOCAL_KERNELS`. So `import kernels` / `get_kernel` tests
would have failed at collection. On current `main` both are present:
```nix
export LOCAL_KERNELS="${repoId}=${extension}"   # only when build.toml has [general.hub] repo-id
${testPython}/bin/python3 -m pytest ${extension.src}/tests -m kernels_ci -p no:cacheprovider || test $? -eq 5
```
For reference, `einops` (added in #1053) already pins `d0610aa5…`, which does have both — that's
why its `get_kernel` tests work today.

Note `get_kernel()` checks the `LOCAL_KERNELS` override *before* resolving `version`, so passing
`version=1` is safe both locally and against the Hub.

### 3. `AGENTS.md` — new `## Tests` section
Added under "General instructions that apply to all kernels" (just before
"# Kernel-specific instructions"): documents get_kernel-over-direct-import, where repo-id/version
come from, the `LOCAL_KERNELS` + flake.lock-freshness requirement, and the `kernels_ci` marker
with the ~60s budget. This is the spec for the roll-out.

## What is NOT done — pick up here
The verification build never finished. `nix build -L .#ci-test` in `relu/` was started on a macOS
box where the **huggingface cachix substituter is not configured** (`nix store info` reports
`Trusted: 0`, and `/etc/nix/nix.conf` only has the Determinate substituters), so it fell back to
building 141 derivations from source (Rust crates for `kernel-abi-check`, Python deps). It was
killed before completing. No `relu/result` symlink exists.

**Next steps:**
1. On a box with the huggingface binary cache configured:
   ```bash
   cd relu && nix build -L .#ci-test && ./result/bin/ci-test
   ```
   Expect two passing tests: `test_relu` and `test_relu_layer`.
   To force a specific variant: `nix run .#ciTests.<variant>` (e.g.
   `torch210-cxx11-cpu-x86_64-linux`).
2. Fix whatever breaks.
3. Open a PR against `huggingface/kernels-community`. There is no branch yet — the changes are
   sitting uncommitted on `main`, so branch first.

## Roll-out survey (deliberately not started — the issue gates it on relu proving out)
Of 37 kernel dirs with a `tests/` dir:
- Fully converted: `einops` only.
- Use `get_kernel` in some test files but have **zero** `kernels_ci` markers (so `ci-test`
  selects nothing and exits 5, which the script swallows): `deep-gemm`, `flash-mla`,
  `flash-attn3`, `vllm-flash-attn3`.
- Have `kernels_ci` markers but still import the package directly: `finegrained-fp8`,
  `esmfold2-trimul`, `natten`, `sgl-flash-attn3`, plus `relu` before this change.
- Everything else (~28 kernels) has neither.
- `finegrained-fp8` is on the same stale `81580bb9…` pin and will need the same lock bump.
  Worth checking every kernel's `kernel-builder` pin against `lastModified >= 1785427612`
  (the `d0610aa5…` cutoff) as part of the roll-out.
- `rotary/tests/test_rotary.py` uses a different pattern — try `import rotary`, fall back to
  `get_local_kernel(repo_path=...)`. Should be normalized to plain `get_kernel`.

## Useful context
- CI path: `.github/workflows/build.yaml` — job `build-ci-test` runs `nix build -L .#ci-test`,
  exports the Nix closure as an artifact, and job `test-kernel-gpu` imports it on an
  `aws-g6-12xlarge-plus` runner and executes `$CI_TEST_PATH/bin/ci-test`. Both are gated on
  `inputs.mode == 'pr'`.
- Upstream docs for this exact convention: `docs/source/builder/writing-kernels.md` in
  `huggingface/kernels` ("Kernel tests ... must not use direct imports, but instead use
  `get_kernel`" + "Mark CI tests").
- `scripts/update_flakes.py <kernel>` is the supported way to bump a lock.
