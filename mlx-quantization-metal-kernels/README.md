## mlx-quantization-metal-kernels

[MLX](https://github.com/ml-explore/mlx)'s quantized kernels for torch tensors on Apple Silicon (MPS):
the same Metal kernels, chosen and launched the way MLX chooses and launches them, so every op
returns what its `mlx.core` counterpart returns, bit for bit.

- `quantize` / `dequantize` — MLX's packed layout (`mx.quantize` / `mx.dequantize`)
- `quantized_matmul` — `x @ dequantize(w).T`, or `x @ dequantize(w)` with `transpose=False`
- `gather_qmm` — `quantized_matmul` with per-row weights picked by index, for mixture-of-experts

Modes: `affine` (group size 32/64/128, 2/3/4/5/6/8 bits, default 64/4; scales and biases in the
activation dtype) and the fp formats `mxfp4` (32/4), `mxfp8` (32/8) and `nvfp4` (16/4; uint8 scales,
no biases, optional global scale).

## Usage

```python
import torch
from kernels import get_kernel

mq = get_kernel("kernels-community/mlx-quantization-metal-kernels", version=2)

w = torch.randn(4096, 4096, dtype=torch.bfloat16, device="mps")
x = torch.randn(1, 4096, dtype=torch.bfloat16, device="mps")

wq, scales, biases = mq.quantize(w)                         # affine, 64/4
y = mq.quantized_matmul(x, wq, scales, biases)              # (1, 4096), an nn.Linear without bias

wq4, s4 = mq.quantize(w, mode="mxfp4")
y4 = mq.quantized_matmul(x, wq4, s4, mode="mxfp4")
```

Version 1's functions (`affine_qmm_t`, `affine_qmv`, `mxfp4_qmv`, ...) are kept, on top of these.

## How it is put together

```
vendor/                       MLX at the release in vendor/UPSTREAM, as it ships
mlx_metal/
├── mlx_*.metal               one per upstream .metal: includes it under MLX's math mode
├── common.h                  the boundary: a minimal `array` and the three launchers
├── mlx_dispatch.mm           Metal side: MLX's launchers, transcribed (quantized.cpp, reduce.cpp)
└── mlx_quantization.cpp      torch side: MLX's op layer, transcribed (ops.cpp)
torch-ext/                    schema, registration, Python API
```

The kernels are compiled as they ship. What cannot be vendored is the host code that launches them,
which upstream writes against its own `array` and device types; `mlx_dispatch.mm` and
`mlx_quantization.cpp` transcribe it function by function, keeping upstream's names, and helpers
marked verbatim are upstream's text.

Two build details: kernel-builder compiles every `.metal` it is given with its own flags and has no
way to pass MLX's `-fno-fast-math`, so each wrapper sets the equivalent `#pragma METAL fp
math_mode(safe)` and includes the upstream file, which `vendor.py` stores as `.metal.h` so it is not
compiled a second time. On M5-class GPUs MLX's dispatch picks the NAX kernels, as upstream does.

## Updating MLX

```bash
python vendor.py --rev v0.32.3      # clones MLX next to this file unless --src is given
python -m pytest tests/test_vendor_drift.py
```

`test_vendor_drift.py` needs no GPU. It fails when upstream changes anything the transcription relies
on — a verbatim helper, a threshold, a grid, a kernel name or a buffer index — and says what.

## Building and testing

```bash
nix run .#build-and-copy -L
LOCAL_KERNELS=kernels-community/mlx-quantization-metal-kernels=$PWD/build python -m pytest tests
```

The tests compare every op with `mlx.core` (install the `mlx` release `vendor/UPSTREAM` pins; those
tests skip without it) and check which kernels each call plans, including for other GPU generations.
