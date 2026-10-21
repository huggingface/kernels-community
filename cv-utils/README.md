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
| `connected_component_areas` | the component area of every pixel of a mask of any height and width | CUDA |
| `generic_nms` | non-maximum suppression of boxes | CUDA |
| `resize_normalize` | antialiased resize to one size per image, center crop, rescale and normalize of uint8 images | Triton |
| `resize_normalize_patchify` | the same, written in the Qwen2-VL `pixel_values` patch layout, for images and videos | Triton |
| `upsample_argmax` | the class of every pixel of bilinearly upsampled segmentation logits, without storing the upsampled logits | Triton |

## Usage

```python
from kernels import get_kernel

cv_utils = get_kernel("kernels-community/cv-utils", version=2)
pixel_values = cv_utils.resize_normalize(images, [(224, 224)] * len(images), image_mean, image_std, 1 / 255, "bicubic")
```

`transformers` image processors call these kernels when loaded with `use_kernels=True`.

## How it works

The resize is separable. A horizontal pass resizes the width of every image into a scratch buffer, and a vertical pass
resizes the height, rescales, normalizes and writes the output. Images of different sizes share each launch. The
horizontal pass is skipped when no image changes width.

The per-image sizes, offsets and normalized filter weights depend only on the shapes and sizes of the batch. They are
computed once per batch layout, with one copy to the device and one launch, and cached. A call with a known layout
then launches only the two passes.

The filter weights follow `torch.nn.functional.interpolate(..., antialias=True)`. By default (`round_to_uint8=True`),
each pass rounds to uint8, like torchvision on uint8 tensors.

`resize_normalize_patchify` takes uint8 `(T, C, H, W)` inputs, with `T=1` for an image, and one target size
per input. It writes each value directly at its place in the patch layout, in `block_size x block_size` blocks of
patches, and returns `grid_thw` as an int64 tensor.

## Limitations

- `upsample_argmax` runs on tensor cores in float16 when both sides are upsampled at least 2x. Then a few pixels
  at near-ties between two classes can get another class than in float32 (about 1e-4 of pixels on ADE20K, with no
  change in mIoU). Smaller upsampling factors run in float32.
- Inputs are uint8 `(C, H, W)` tensors on one CUDA device, with the same channel count.
- Bilinear and bicubic only. The output is float32.
- `resize_normalize` takes the resized size of each image from the caller, and an optional crop no larger than
  any of them. It returns one `(C, H, W)` tensor per image.
- `resize_normalize_patchify` needs target sizes that are multiples of `patch_size * block_size`.
- With `round_to_uint8=True`, results can differ from torch by one uint8 level where a value falls on a rounding
  boundary.
- The metadata of a batch layout (input sizes, target sizes, crop) is cached, up to 1024 layouts. A new layout costs
  one extra copy and one extra launch, so batches of photos that all have different sizes pay that cost on every call.
