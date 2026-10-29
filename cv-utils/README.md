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

### `upsample_argmax`

Semantic segmentation models predict logits at a low resolution, for example 150 classes at 128x128. The class map is
`interpolate(logits, size=(H, W), mode="bilinear").argmax(0)`. In torch, that stores every class at full size before
the argmax: 150 x 480 x 640 float32 values, or 184 MB for one photo. The kernel computes each output pixel from 4
logits per class and keeps only the best class. It writes one int64 per pixel, which is 2.4 MB for the same photo.

One output pixel, 2x2 logits upsampled to 4x4, output pixel `(1, 2)`:

```
source position:  row = (1 + 0.5) * 2/4 - 0.5 = 0.25      column = (2 + 0.5) * 2/4 - 0.5 = 0.75

weights of the 4 neighbors          class 0 logits        class 1 logits
   (1-0.25)(1-0.75)  (1-0.25)(0.75)     2    0                0    1
   (0.25)(1-0.75)    (0.25)(0.75)       1    3                2    1
 = 0.1875   0.5625
   0.0625   0.1875

class 0: 0.1875*2 + 0.5625*0 + 0.0625*1 + 0.1875*3 = 1.000   <- best so far, class 0
class 1: 0.1875*0 + 0.5625*1 + 0.0625*2 + 0.1875*1 = 0.875      not better, keep class 0
```

The loop over classes keeps only the best value and its class. The upsampled logits are never stored. When both sides
grow at least 2x, a program computes a 32x64 output tile for each class as `row_weights @ patch @ column_weights`
on tensor cores, where `patch` is the block of logits under the tile.

A batch has one output size per image. The work is cut into tiles of output pixels: 256 pixels, or 32x64 on tensor
cores. All tiles of all images go to one launch. For example, with images of 4, 2 and 3 tiles:

```
tiles of the launch       0  1  2  3 | 4  5 | 6  7  8
image                     0  0  0  0 | 1  1 | 2  2  2
last tile + 1 per image            4 |    6 |       9

program 7: images whose tiles end at or before 7 -> 2 (ends 4 and 6) -> image 2
           first tile of image 2 is 6            -> tile 1 of image 2
```

The table of tile ends and the output sizes and offsets of each image are copied once and cached per batch layout.
When all images have the same output size, as with one image, there is no table: the launch grid has one row per
image and the output size is a kernel argument.

A photo of 480x640 is 150 tiles on tensor cores, fewer than one wave on an A100. With one launch per image, most of
the GPU waits at the end of each launch. With one launch per batch, the tiles of the next image fill it.

A100, real logits, 16 COCO photos of different sizes, median ms:

| model | one launch per image | one launch per batch | speedup |
|---|---|---|---|
| SegFormer ADE20K (150 classes) | 1.36 | 0.86 | 1.58x |
| RF-DETR-Seg large (91 classes) | 0.83 | 0.43 | 1.92x |
| SegFormer Cityscapes (19 classes) | 0.74 | 0.46 | 1.63x |
| Mask2Former ADE20K (float32 path) | 2.71 | 2.03 | 1.34x |

Batches with one output size run at the same speed as before.

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
- The metadata of a batch layout is cached, up to 1024 layouts. The key is the tuple of all sizes in the batch, so a
  batch of photos with different sizes almost never hits the cache. A miss costs one extra copy to the device, plus
  one extra launch for the resize. Measured on an A100:

  | call | known layout | new layout | reference |
  |---|---|---|---|
  | `resize_normalize`, 32 photos of different sizes | 0.73 ms | 0.78 ms | torchvision 6.3 ms |
  | `resize_normalize`, 1 photo | 0.16 ms | 0.49 ms | torchvision 0.43 ms |
  | `upsample_argmax`, 16 photos of different sizes | 0.86 ms | 1.14 ms | one launch per image 1.36 ms |
  | `upsample_argmax`, 1 image | 0.13 ms | 0.15 ms | one launch per image 0.13 ms |

  At batch 1 with a new size on every call, `resize_normalize` is about 13% slower than torchvision. `upsample_argmax`
  keeps no device metadata when all output sizes are equal, so a new size there adds only about 0.02 ms on the CPU.
