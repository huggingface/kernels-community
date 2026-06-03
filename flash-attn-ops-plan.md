# Plan: `flash-attn-ops` — a `kernels`-compliant package for flash-attn's Triton ops

Tracking issue: [huggingface/kernels-community#900](https://github.com/huggingface/kernels-community/issues/900)
— *"Package flash-attn Triton ops (cross_entropy, rotary, etc.) as a kernel"* (no comments on the issue as of writing).

> **Motivating use case (verbatim from the issue):** integrating
> [dexmal/dexbotic](https://github.com/dexmal/dexbotic/issues/94), which imports
> **`flash_attn.losses.cross_entropy`** — a wrapper that relies on the Triton
> kernels bundled inside flash-attn that we don't currently expose. Note this is
> `flash_attn.losses`, a thin `nn.Module` layer *on top of*
> `flash_attn/ops/triton/cross_entropy.py` (see §4a).
>
> The issue proposes the name **`flash-attn-triton`**; this plan uses
> **`flash-attn-ops`** per the request, but the name is worth aligning with the
> issue author before publishing.

## 1. Goal

Flash Attention's main repo ships, alongside the attention kernels, a set of
general-purpose **Triton** ops under
[`flash_attn/ops/`](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/ops)
that are widely imported by training codebases (cross-entropy, RoPE, layer/RMS
norm, fused linear, MLP, activations). Our existing `flash-attn2` kernel only
exposes the attention forward/backward/kvcache ops; these utility ops are **not**
distributed in a `kernels`-compliant way.

We want a new top-level kernel directory **`flash-attn-ops`** that builds and
distributes these ops through the Hub (`kernels-community/flash-attn-ops`) so
projects can `get_kernel("kernels-community/flash-attn-ops")` and pull *only*
the utility kernels without the full flash-attn build.

### Scope (from the issue)

| Tier | Contents | Build type |
|------|----------|------------|
| **Minimum (Phase 1)** | `flash_attn/ops/triton/` — `cross_entropy.py`, `rotary.py`, `layer_norm.py`, `k_activations.py` (+ `flash_attn/losses/cross_entropy.py` wrapper) | Pure Triton, **no compilation** |
| **Phase 1b (follow-up)** | `flash_attn/ops/triton/` — `linear.py`, `mlp.py` | `linear.py` needs `triton.ops.matmul_perf_model` (removed in Triton ≥ 3.0 → real port, not just an import rewrite); `mlp.py` needs the compiled `fused_dense_lib` |
| **Ideal (Phase 2, optional)** | the rest of `flash_attn/ops/` — `fused_dense.py`, `layer_norm.py`, `rms_norm.py`, `activations.py` | These wrap **compiled CUDA** extensions (`csrc/fused_dense`, `csrc/layer_norm`) |

> **Implementation status (this commit):** Phase 1 is implemented under
> `flash-attn-ops/`, vendored from upstream commit
> `b02b07e1a10238fe12831b80a8937ed59b1353a5`. `linear.py`/`mlp.py` are deferred to
> Phase 1b for the dependency reasons above. Validated: every file compiles, no
> `flash_attn` import remains (fully self-contained), and all `__init__` exports
> resolve. Runtime/GPU pytest still needs a CUDA + Triton runner (not available
> in the authoring environment).

This plan delivers **Phase 1** (the actual ask) end-to-end and lays out Phase 2
as a follow-up, because Phase 2 is a much larger build surface (compiled CUDA),
not because of any cross-folder overlap — in this monorepo each folder is built
in isolation, so overlap with sibling kernels never blocks shipping (see §3).

## 2. Architectural decision: model on `liger-kernels`, not `flash-attn2`

The Triton ops compile at runtime via Triton's JIT — there is **no C++/CUDA to
build ahead of time**. So this kernel should follow the **pure-Triton packaging
pattern** already used in this repo by `liger-kernels`, *not* the compiled
pattern used by `flash-attn2`/`rotary`.

Concretely, pure-Triton kernels in this repo:

- have an **empty `[kernel]`** section in `build.toml` (no `src`, no
  `cuda-capabilities`, no `torch_binding.cpp`),
- declare `backends` only for portability metadata (`liger-kernels` uses
  `["cuda", "rocm", "xpu"]`),
- ship **all source as `.py` files** directly under
  `torch-ext/<package_name>/`,
- have **no `_ops`/`torch_binding.cpp`/`registration.h`** machinery (those exist
  only for compiled ops).

Reference: `liger-kernels/build.toml` (empty `[kernel]`) and
`liger-kernels/torch-ext/liger_kernels/*.py`.

## 3. Relationship to existing kernels (monorepo — each folder is isolated)

`kernels-community` is a **monorepo where every top-level folder is a fully
self-contained, independently built and published kernel**. `flash-attn-ops`
stands on its own: it vendors the flash-attn ops it needs, builds in isolation,
and does **not** depend on, defer to, or coordinate with any sibling folder.
Functional overlap between folders is expected and acceptable — it is **not** a
reason to omit an op or block a phase.

The issue's "audit for deduplication" note is therefore purely **informational**
here: at most we add a "Related kernels" cross-link in our README so users can
pick the implementation that fits them. It does not change what we ship.

| Sibling kernel | What it is | Same ground as | Note |
|----------------|-----------|----------------|------|
| `rotary` | CUDA/XPU `apply_rotary` (compiled) | our Triton `rotary.py` | Different impl & API (ours is varlen/seqlen-offset aware). README cross-link. |
| `liger-kernels` | Pure-Triton norm/rope/xent/etc. | our cross-entropy, rope, norms | Different upstream impl & API. README cross-link. |
| `layer-norm` | Compiled CUDA build of flash-attn `csrc/layer_norm` | Phase-2 `ops/layer_norm.py` | Independent of our Phase-1 Triton `layer_norm.py`. README cross-link. |
| `rmsnorm`, `activation` | Compiled norms & activations | Phase-2 `rms_norm.py`, `activations.py` | README cross-link. |

**Conclusion:** isolation means there is nothing to gate on. We ship the ops
this kernel needs (Phase 1 in full; Phase 2 when/if pursued), self-contained in
this folder, regardless of what sibling folders also provide.

## 4. Upstream source map (Phase 1)

From `flash_attn/ops/triton/` — vendor these, preserving filenames:

| File | Implements | Internal deps |
|------|-----------|---------------|
| `__init__.py` | (usually empty / re-exports) | — |
| `cross_entropy.py` | `CrossEntropyLoss` Triton fwd/bwd | torch, triton |
| `rotary.py` | `apply_rotary` (seqlen-offset / varlen aware) | torch, triton |
| `layer_norm.py` | fused LayerNorm **and** RMSNorm, residual/dropout add, `layer_norm_fn`, `rms_norm_fn` | torch, triton |
| `k_activations.py` | Triton activation primitives (gelu, etc.) | triton |
| `linear.py` | fused linear (matmul + bias + activation) | `k_activations` |
| `mlp.py` | fused MLP | `linear`, `k_activations` |

### 4a. Also vendor `flash_attn.losses.cross_entropy` (to satisfy the motivating use case)

The issue's concrete blocker is `from flash_attn.losses.cross_entropy import
CrossEntropyLoss`. That module (`flash_attn/losses/cross_entropy.py`) is a small
`nn.Module` wrapper that calls into `ops/triton/cross_entropy.py`. To make the
kernel a drop-in for the dexbotic-style import, vendor it too and expose the
`CrossEntropyLoss` class from our public API (e.g. as
`flash_attn_ops.CrossEntropyLoss`). It's pure Python + the Triton op, so it adds
no build surface — just rewrite its `from flash_attn.ops.triton.cross_entropy
import ...` to a relative import.

**Vendor the full `flash_attn` dependency closure.** This is the core
self-containment task, and it is mechanical, not risky: whatever the vendored ops
import from `flash_attn` (e.g. `from flash_attn.ops.triton.k_activations import
...`, `flash_attn.utils.*`), we **copy into this package** and rewrite to relative
imports — then repeat transitively on whatever *those* files import, until
nothing references `flash_attn`. There is no constraint to keep the vendored
surface "small"; pull in as much of `flash_attn` as the closure requires. The end
state is a package that builds and runs with `flash_attn` *not installed*. This
mirrors how `flash-attn3` relativizes imports (see `AGENTS.md`).

Procedure (repeat until fixpoint):

1. `grep -rn "flash_attn" torch-ext/flash_attn_ops/` — find every reference.
2. For each referenced `flash_attn.<x>` symbol, copy its source file into the
   package (preserving a sensible module path) and rewrite the import to relative.
3. Re-grep; new files may introduce new `flash_attn` imports — vendor those too.
4. Stop when the grep is empty. Final check: import the package in an env where
   `flash_attn` is uninstalled.

## 5. Directory layout to create

```
flash-attn-ops/
├── build.toml
├── flake.nix
├── flake.lock                 # generated: `nix flake lock`
├── README.md                  # Hub model card (yaml `tags: [kernel]`)
├── CARD.md                    # Jinja template (copy from rotary/flash-attn2)
├── torch-ext/
│   └── flash_attn_ops/
│       ├── __init__.py        # curated public API (see §7)
│       ├── _ops_compat.py     # add_op_namespace_prefix shim (built + source layouts)
│       ├── cross_entropy.py
│       ├── losses.py          # vendored flash_attn.losses.cross_entropy wrapper (§4a)
│       ├── rotary.py
│       ├── layer_norm.py
│       ├── k_activations.py
│       └── utils/             # mirrors upstream flash_attn/utils/ layout
│           ├── __init__.py
│           ├── torch.py       # custom_fwd / custom_bwd
│           └── library.py     # triton_op
├── tests/
│   ├── __init__.py
│   ├── test_cross_entropy.py
│   ├── test_rotary.py
│   └── test_layer_norm.py     # covers layer_norm_fn + rms_norm_fn
└── scripts/
    └── readme_example.py      # runnable usage snippet
```

> **Conventions followed (per `AGENTS.md`):**
> - Vendored helpers mirror **upstream module paths** — `flash_attn/utils/torch.py`
>   and `flash_attn/utils/library.py` become `utils/torch.py` and `utils/library.py`
>   (not flattened/renamed).
> - **Torch custom ops** (`layer_norm.py`'s two `@triton_op(...)` registrations)
>   use `add_op_namespace_prefix(<bare-name>)` instead of a hard-coded namespace,
>   so the published kernel gets a unique build-hashed op namespace and never
>   collides with an installed `flash_attn` (which registers under `flash_attn::`).
>   `add_op_namespace_prefix` comes from `_ops_compat.py`, which uses the generated
>   `._ops` in the built layout and falls back to `flash_attn_ops::` from source.
> - `linear.py`/`mlp.py` (Phase 1b) and `benchmarks/` are not created yet.

Package name: **`flash_attn_ops`** (Python import name). Hub repo-id:
**`kernels-community/flash-attn-ops`**.

## 6. `build.toml`

Pure-Triton → empty `[kernel]`, no `[torch]` sources, no per-backend blocks:

```toml
[general]
name = "flash-attn-ops"
license = "BSD-3-Clause"          # match flash-attn upstream
version = 1
backends = [
    "cuda",
    "rocm",
    "xpu",
    # "cpu",                       # include iff upstream + Triton-CPU validate (see below)
]

[general.hub]
repo-id = "kernels-community/flash-attn-ops"

[kernel]
```

**Backends should mirror upstream, not be artificially capped at CUDA.** Because
these are pure-Triton ops, the *same* `.py` source runs on every backend Triton
can target (CUDA, ROCm, Intel XPU, and CPU via triton-cpu) — there is no
per-backend code to write. So the `backends` list is a function of two things:

1. **What upstream flash-attn supports / intends** for these ops, and
2. **What we actually validate** in CI.

Default to the broadest set that upstream supports (mirroring `liger-kernels`,
which already ships `["cuda", "rocm", "xpu"]` from one Triton source), and add
`"cpu"` if upstream + triton-cpu validate. The only reason to *drop* a backend is
a verified runtime failure of a specific op on it — and in that case prefer a
per-op runtime guard with a clear error over silently narrowing the whole
kernel's advertised backends.

> Action item: before finalizing, check upstream for any backend-specific code
> paths or guards in the vendored ops (e.g. `is_hip()`, device-type asserts,
> CUDA-only intrinsics) and reflect exactly those constraints — at the op level
> where possible, in `backends` only as a last resort.

## 7. `flake.nix`

Copy `liger-kernels/flake.nix` verbatim except the `description` — it is the
minimal pure-Triton form:

```nix
{
  description = "Flake for flash-attn Triton ops";
  inputs.kernel-builder.url = "github:huggingface/kernels";
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
```

Then generate `flake.lock` with `nix flake lock` (or copy + `nix flake update`).
If tests need `einops`, add a `pythonCheckInputs` block like `flash-attn2`'s.

## 8. Python public API (`torch-ext/flash_attn_ops/__init__.py`)

Curate a stable, documented surface so `get_kernel(...)` returns something
ergonomic. Re-export the canonical entry points:

```python
from .cross_entropy import cross_entropy_loss            # low-level Triton op
from .losses import CrossEntropyLoss                     # nn.Module (the motivating import)
from .rotary import apply_rotary
from .layer_norm import layer_norm_fn, rms_norm_fn, RMSNorm, LayerNorm
from .linear import linear_func                           # name per upstream
from .mlp import fused_mlp_func, FusedMLP                 # name per upstream

__all__ = [
    "cross_entropy_loss", "CrossEntropyLoss",
    "apply_rotary",
    "layer_norm_fn", "rms_norm_fn", "RMSNorm", "LayerNorm",
    "linear_func",
    "fused_mlp_func", "FusedMLP",
]
```

(Exact symbol names to be confirmed against the pinned upstream commit during
implementation.) Keeping `__all__` curated is what populates the auto-generated
CARD "Available functions" list.

## 9. Tests

Port the relevant upstream tests from flash-attn's `tests/` (e.g.
`test_rotary.py`, layer-norm and cross-entropy tests) into `flash-attn-ops/tests/`,
rewriting imports to the local package (`from flash_attn_ops import ...`) exactly
as `AGENTS.md` prescribes for `flash-attn3`. Each test:

- runs on CUDA (guard/skip when no GPU),
- compares the Triton op against a pure-PyTorch reference (allclose at the
  appropriate dtype tolerances),
- covers fp16/bf16 and a couple of shapes.

Tests are run by the kernel-builder check (`pythonCheckInputs` must include
`einops`, `pytest`, etc.).

## 10. Docs

- `README.md`: Hub card with front-matter `tags: [kernel]` and
  `license: bsd-3-clause` (copy structure from `flash-attn2/README.md`); link to
  the upstream source, list the exposed ops, and optionally **add a "Related
  kernels" note** pointing to `rotary`, `liger-kernels`, `layer-norm` (purely a
  convenience cross-link — see §3; folders are independent).
- `CARD.md`: copy the Jinja template from `rotary/CARD.md` (it is generic and
  auto-fills `repo_id`/`functions`).
- `scripts/readme_example.py`: minimal runnable example used in the README.

## 11. Build & validation

```bash
cd flash-attn-ops
nix flake lock                       # produce flake.lock
nix run .#test                       # or: kernel-builder build/test per docs
# Local smoke test without Hub:
python -c "import sys; sys.path.insert(0,'torch-ext'); import flash_attn_ops as f; print(f.__all__)"
```

Follow kernel-builder docs:
[writing-kernels](https://github.com/huggingface/kernel-builder/blob/main/docs/writing-kernels.md)
and [nix](https://github.com/huggingface/kernel-builder/blob/main/docs/nix.md).
Because there is no compiled artifact, the "build" is largely packaging +
running the Triton ops under pytest on a GPU runner.

## 12. CI / nightly integration

The repo runs nightly kernel checks (`scripts/run_kernels_checks.py`,
`scripts/report_kernel_failures.py`). After the kernel is published:

- ensure `flash-attn-ops` is picked up by the nightly check (confirm whether the
  check auto-discovers directories or needs an explicit entry / config),
- verify it appears in any kernel inventory the checks consume.

## 13. Phasing & deliverables

**Phase 1 — Triton ops (this plan, the actual ask):**
1. Create `flash-attn-ops/` skeleton (§5).
2. Vendor the 6 Triton files from a **pinned** upstream commit; rewrite all
   `flash_attn.*` imports to relative + vendor `_utils.py` (§4).
3. Write `build.toml`, `flake.nix`, generate `flake.lock` (§6–7).
4. Curate `__init__.py` public API (§8).
5. Port tests + docs (§9–10).
6. `nix` build/test green on a GPU runner (§11).
7. Open PR titled `flash-attn-ops: add flash-attn Triton ops kernel`.

**Phase 2 — compiled `ops/` (deferred: larger build surface, not a dedup gate):**
- `fused_dense.py` (+ `csrc/fused_dense`), `activations.py`, `layer_norm.py`,
  `rms_norm.py`. This is a compiled CUDA build, so it would use the
  `flash-attn2`-style `build.toml` + `torch_binding.cpp` instead of the empty
  `[kernel]`. It is deferred purely because of that build cost — **not** because
  sibling kernels (`layer-norm`/`rmsnorm`/`activation`) cover similar ground;
  per §3, folder isolation means we ship our own self-contained versions
  regardless of what siblings provide.

## 14. Open questions / risks

1. **Self-containment (mechanical, not a blocker):** any `flash_attn.*` import in
   the vendored files (esp. `layer_norm.py`, `mlp.py`) is simply vendored in —
   copy the dependency closure transitively and rewrite to relative imports until
   nothing references `flash_attn` (§4). Verified by importing the package with
   `flash_attn` uninstalled. No size limit on what we pull in.
2. **Upstream pin:** choose and record an upstream commit/tag to vendor from, so
   future syncs are reproducible (add an `AGENTS.md` sync note like flash-attn3's).
3. **Backend coverage = upstream coverage.** These are pure-Triton ops, so one
   source serves all Triton-supported backends — mirror what upstream flash-attn
   supports (CUDA, ROCm, XPU, and CPU where validated) rather than capping at
   CUDA. Narrow a backend only on a verified per-op failure, and prefer an op-level
   runtime guard over dropping it from `build.toml`'s `backends` (§6).
4. **API naming:** confirm exact exported symbol names against the pinned commit
   before finalizing `__init__.py`/CARD.
5. **License:** confirm BSD-3-Clause matches upstream flash-attn at the pinned
   commit.
6. **Folder isolation:** keep this kernel fully self-contained — vendor what it
   needs and build in isolation. Do **not** make it depend on or defer to sibling
   folders; overlap with `rotary`/`liger-kernels`/`layer-norm`/`rmsnorm`/
   `activation` is fine and is at most a README cross-link (§3).
```
