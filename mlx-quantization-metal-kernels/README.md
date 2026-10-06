# mlx-quantization-metal-kernels

MLX's affine quantized matmul kernels for torch tensors on Apple Silicon (MPS).

The Metal kernels are [MLX](https://github.com/ml-explore/mlx)'s own, vendored at a pinned release
(`vendor/UPSTREAM`) and compiled as they ship. The host-side dispatch in `mlx_metal/mlx_dispatch.mm`
transcribes MLX's `QuantizedMatmul::eval_gpu`, so a given shape runs the kernel it would under
`mlx.core.quantized_matmul`: matrix-vector kernels (`qmv_fast`, `qmv`, `qmv_quad`, `qmv_wide`) for
decode-sized inputs, and split-K or tiled `qmm_t` above MLX's per-GPU crossover.

```python
from kernels import get_kernel

mq = get_kernel("kernels-community/mlx-quantization-metal-kernels", version=1)

# x: [..., K] float32/float16/bfloat16
# w: [N, K * bits / 32] uint32, scales/biases: [N, K / group_size] in x's dtype (MLX's layout)
y = mq.affine_qmm_t(x, w, scales, biases, group_size=64, bits=4)  # [..., N]
```

`affine_qmm_t` is the only API: it is what transformers' `MetalConfig` uses.

`group_size` is 32, 64 or 128; `bits` is 2, 3, 4, 5, 6 or 8. Results match `mx.quantized_matmul`
bit for bit, except on the split-K path, where the partial products are summed in float32 rather
than in the output dtype.

Not covered: the fp modes (mxfp4/nvfp4/mxfp8), batched weights, gather (MoE) matmuls, and MLX's NAX
path for M5-class GPUs, which needs Metal 4 features the builder does not enable.

## Updating MLX

```bash
python vendor.py --rev v0.32.3   # clones MLX next to this file if --src is not given
python -m pytest tests/test_vendor_drift.py
```

`test_vendor_drift.py` needs no GPU. It fails when upstream changes something the dispatch
transcribes -- a helper, a threshold, a grid, a kernel name or a buffer index -- and says which.

## Building and testing

```bash
nix run .#build-and-copy -L
python -m pytest tests   # needs MPS; parity tests against MLX need `pip install mlx==<pinned version>`
```
