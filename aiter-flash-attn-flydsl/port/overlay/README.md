---
license: mit
tags:
  - kernels
---
# AITER FlyDSL FlashAttention for gfx1201

BF16/FP16 and FP8 attention for AMD RDNA4, packaged from
[ROCm/aiter](https://github.com/ROCm/aiter). The kernels use wave32 WMMA and
FP32 accumulation. This package includes the FP8 quantization producer and
does not depend on an installed AITER package.

Sources are pinned to AITER commit
[`c1096fa40a907cf83cf95d6d87c4fad5823e7054`](https://github.com/ROCm/aiter/tree/c1096fa40a907cf83cf95d6d87c4fad5823e7054).
FlyDSL is a kernel dependency (`kernels-sdk/flydsl`, version 1), so it does not
need to be installed separately. The vendored kernel sources were written against
FlyDSL 0.4.2 (the upstream AITER requirement); the dependency currently provides 0.5.0. Kernel sources retain their upstream MIT or
Apache-2.0 notices; the Apache-2.0 license is included in `LICENSE-FlyDSL`.

## Usage

Install ROCm-enabled PyTorch and `kernels` on Linux x86_64.
FlyDSL JIT also needs the ROCm development SDK and linker. Set `ROCM_PATH` to
its root and include the LLVM tool directory in `PATH`.
Before the first GPU
compilation, the wrapper also checks the development linker and FlyDSL's GPU
architecture detection, with setup instructions if either check fails.

With AMD's split ROCm 10 wheels, install the device extras explicitly:

```bash
pip install "torch[device-gfx1201]==2.14.0+rocm10.0.0" \
  "rocm[devel,device-gfx1201]==10.0.0" \
  --index-url https://stable.repo.amd.com/rocm/whl-next \
  --extra-index-url https://pypi.org/simple
rocm-sdk init
# Set ROCM_PATH using the root printed by: rocm-sdk path --root
# Add $ROCM_PATH/lib/llvm/bin and $ROCM_PATH/bin to PATH.
```

```python
from kernels import get_kernel

attention = get_kernel("kernels-community/aiter-flash-attn-flydsl", version=1)
# q/k/v: [batch, sequence, heads, head_dim], BF16 or FP16
out = attention.attention(q, k, v, causal=False)
# Force BF16-to-FP8 attention, including Q/K Hadamard rotation:
out_fp8 = attention.attention(q, k, v, fp8=True)
# Or use the pinned xDiT shape crossover policy:
out_auto = attention.attention(q, k, v, fp8="auto")

# Explicit quantization permits reuse and separate timing.
q8, k8, v8, sq, sk, sv = attention.fp8_quant(q, k, v, rotation=True)
out8 = attention.flash_attn_func(
    q8, k8, v8, q_descale=sq, k_descale=sk, v_descale=sv,
)
```

## Contract

- gfx1201 only (RX 9070 series / Radeon AI PRO R9700), Linux x86_64.
- Attention accepts matching BF16, FP16, or E4M3FN FP8 Q/K/V. FP8 requires
  one-element FP32 device descales and returns BF16; other outputs match input dtype.
- Layout is BSHD. Batch, head count, and head dimension must match; K/V shapes
  must match. Non-causal cross-attention permits different Q and KV lengths.
- Head dimensions must be at least 64 and divisible by 32. Attention also
  enforces the gfx1201 LDS and buffer-index limits; not every larger dimension fits.
- Causal self-attention is supported. Causal cross-attention, arbitrary masks,
  GQA, dropout, variable-length packed inputs, LSE, and backward are not provided.
- `fp8_quant` accepts BF16 inputs with head dimensions 64–1024 divisible by 32.
  Rotation applies an orthonormal Hadamard transform to Q/K only and requires
  power-of-two dimensions. Other widths require `rotation=False` and use padding.
- Non-contiguous inputs can cause materialization. The wrappers respect the
  current device/stream; cross-stream consumers must establish dependencies.
- FP8 changes numerical results. Assess output quality for the intended model.

`attention` registers custom ops and fake implementations for `torch.compile`
(including full-graph mode). It is adapted from xDiT commit
[`4ac65db2fa4add8ff5426cfbb021d96d0be6eab0`](https://github.com/xdit-project/xDiT/tree/4ac65db2fa4add8ff5426cfbb021d96d0be6eab0/xfuser/core/attention/backends/aiter_flydsl),
whose Apache-2.0 license is included in `LICENSE-xDiT`. `fp8="auto"` retains
xDiT's measured non-causal shape table: causal attention, FP16, unknown shapes, and short conditioning K/V
retain input precision. These thresholds are inherited measurements, not a
performance guarantee for ROCm 10 or every gfx1201 board. `fp8=True` forces the
quantization path and requires BF16. Prequantized FP8 inputs can also use
`attention` with explicit descales. This package accepts BSHD directly; xDiT's
layout conversion, logging, distributed orchestration, and SDPA dispatch stay
in the integration layer.

The raw public functions preserve upstream `flydsl_flash_attn_func` and
`flydsl_fp8_quant` signatures; only their exported names differ.

## Build and test

This kernel is generated from upstream AITER by `port/port.kdl` (see the
`kernel-port` tool) into `src/`; edit the recipe or the files in `port/overlay`, not
`src/`. It depends on the `kernels-sdk/flydsl` kernel (`kernel-depends` in
`build.toml`), which is resolved when the kernel is loaded. Build with Nix from this
directory; evaluate with `nix eval` before building, as required by the builder
repository. Nix Git flakes omit untracked files, so add new files to Git first.

Tests load through `get_kernel`; use `LOCAL_KERNELS` for a local build. Run the
full suite on gfx1201 with `python -m pytest tests`. Benchmarks live in
`benchmarks/comparison.py`; they compare with published `amd/aiter-flash-attn` v2
and SDPA, record the resolved baseline revision, and include quantization cost.

## Upstream synchronization

The vendored files come from `aiter/ops/flydsl/kernels/fmha_gfx1201/`,
`fp8_quant.py`, `kernels/fp8_quant_gfx1201.py`, and their required helpers.
The internal directory is named `_kernels` so that adding a build variant to
`PYTHONPATH` does not shadow the Hugging Face `kernels` loader.
`fmha.py` retains only the gfx1201 functions from upstream `fmha_kernels.py`;
`tensor_shim.py` uses relative imports and omits the unused AITER-dependent
`check_e8m0` helper. Updates require an explicit source sync and validation.

The standard Hub benchmark implements `kernels.benchmark.Benchmark`. It verifies
BF16 using the runner's fixed allclose tolerance; FP8 is timed there and its
accuracy is assessed separately by the numerical tests and comparison script. For the
published-provider comparison, run `python benchmarks/comparison.py` with a
local kernel build. `--aiter-shapes --timing profiler --repeats 101 --warmup 2`
uses AITER's Flux/Wan/SD3.5 shapes and mean GPU activity duration; event timing
is also retained. The profiler mode reuses inputs and does not reproduce
AITER's automatic input rotation or IQR filtering. `sdpa_flash` preformats BHSD
inputs and forces FlashAttention; `sdpa` includes layout handling and automatic
backend selection. Both receive the same Q/K/V values.

## Measured performance

Measured on AMD Radeon AI Pro R9700S (gfx1201), Ubuntu 24.04, PyTorch
`2.14.0+rocm10.0.0`, FlyDSL `0.4.2`. The published Hub baseline is
`amd/aiter-flash-attn` v2 at `8a198c597373ea2beef2b17b283ba44568f0d456`.
All shapes below are B=1 BF16 non-causal self-attention. Values are mean GPU
activity milliseconds over 101 calls after two warmups. FP8 total includes
Hadamard Q/K rotation and Q/K/V quantization. SDPA Flash inputs are preformatted
to contiguous BHSD outside timing. Inputs are reused; these runs do not apply
AITER's automatic input rotation or IQR filtering.

| S | H | D | Hub Triton | SDPA Flash | FlyDSL BF16 | FP8 total | FP8 vs BF16 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536 | 24 | 128 | 0.355 | 0.322 | 0.317 | 0.321 | 0.99x |
| 4096 | 24 | 128 | 2.457 | 2.213 | 2.039 | 1.764 | 1.16x |
| 8192 | 12 | 128 | 4.768 | 4.368 | 4.049 | 3.252 | 1.24x |
| 32768 | 12 | 128 | 81.844 | 74.246 | 67.441 | 51.043 | 1.32x |
| 2048 | 38 | 64 | 0.552 | 0.554 | 0.472 | 0.477 | 0.99x |

FP8 total is faster for Flux S=4096 and both Wan shapes in this run. It is
slightly slower than BF16 for Flux S=1536 and SD3.5 S=2048. FP8 relative L2
error versus the FP32 reference is 0.0527–0.0546, with minimum row cosine
0.9943 or greater in these shapes. BF16 relative L2 error is below 0.0037.
These synthetic attention measurements do not establish diffusion model quality.
The separate 30-case event-timed sweep also covers FP16, causal attention,
short conditioning K/V, non-tile-aligned lengths, and D=64/128/256.
