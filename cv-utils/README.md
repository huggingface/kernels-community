---
license: apache-2.0
tags:
  - kernels
---

# cv_utils

Kernels for computer vision pre- and post-processing.

| function | computes | backend |
|---|---|---|
| `cc_2d` | connected-component labels of binary masks, optionally with component areas | CUDA |
| `generic_nms` | non-maximum suppression of boxes | CUDA |
| `resize_normalize` | antialiased resize, center crop, rescale and normalize of uint8 images into `(N, C, H, W)` | Triton |
| `resize_normalize_patchify` | the same, written in the Qwen2-VL `pixel_values` patch layout, for images and videos | Triton |

## Usage

```python
from kernels import get_kernel

cv_utils = get_kernel("kernels-community/cv-utils", version=2)
pixel_values = cv_utils.resize_normalize(
    images, (224, 224), image_mean, image_std, 1 / 255, "bicubic", True, round_to_uint8=True
)
```

`transformers` image processors call these kernels when loaded with `use_kernels=True`.

## How it works

The resize is separable. One launch computes the normalized filter weights of every output row and column of every
image into a table. A horizontal pass then resizes the width of each image into a scratch buffer, and a vertical pass
resizes the height, rescales, normalizes and writes the output. Images of different sizes share each launch.

The filter weights follow `torch.nn.functional.interpolate(..., antialias=True)`. With `round_to_uint8=True`, each
pass rounds to uint8, like torchvision on uint8 tensors.

`resize_normalize_patchify` takes one target size per frame and a list of frame indices per image or video, and writes
each value directly at its place in the patch layout.

## Limitations

- Inputs are uint8 `(C, H, W)` tensors on one CUDA device, with the same channel count.
- Bilinear and bicubic only. The output is float32.
- `resize_normalize` supports a square resize with an optional crop, or a shortest-edge resize with a crop.
- `resize_normalize_patchify` needs target sizes that are multiples of `patch_size * merge_size`, and the frames of
  one video must share a target size.
- With `round_to_uint8=True`, results can differ from torch by one uint8 level where a value falls on a rounding
  boundary.
