"""Tests for the cv-utils resize kernels, checked against `torch.nn.functional.interpolate`.

The whole suite is small and fast, so every test is part of the CI subset.
"""

import kernels
import pytest
import torch
import torch.nn.functional as F


cv_utils = kernels.get_kernel("kernels-community/cv-utils", version=2)

pytestmark = pytest.mark.kernels_ci

DEVICE = "cuda"

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def random_images(sizes):
    generator = torch.Generator().manual_seed(0)
    return [
        torch.randint(0, 256, (3, height, width), generator=generator, dtype=torch.uint8).to(DEVICE)
        for height, width in sizes
    ]


def reference(image, size, resample, antialias=True):
    resized = F.interpolate(image[None].float(), size=size, mode=resample, antialias=antialias, align_corners=False)[0]
    return (resized / 255 - torch.tensor(MEAN, device=image.device)[:, None, None]) / torch.tensor(
        STD, device=image.device
    )[:, None, None]


@pytest.mark.parametrize("antialias", [True, False])
@pytest.mark.parametrize("resample", ["bilinear", "bicubic"])
@pytest.mark.parametrize(
    "size, shortest_edge, crop_size",
    [((224, 224), None, None), ((256, 256), None, (224, 224)), (None, 256, (224, 224))],
)
def test_resize_normalize_matches_interpolate(antialias, resample, size, shortest_edge, crop_size):
    images = random_images([(480, 640), (300, 300), (1024, 768)])
    output = cv_utils.resize_normalize(
        images,
        MEAN,
        STD,
        1 / 255,
        resample,
        size=size,
        shortest_edge=shortest_edge,
        crop_size=crop_size,
        antialias=antialias,
        round_to_uint8=False,
    )
    for image, result in zip(images, output):
        height, width = image.shape[1:]
        if shortest_edge is not None:
            size_of_image = (
                (shortest_edge, int(width * shortest_edge / height))
                if height <= width
                else (int(height * shortest_edge / width), shortest_edge)
            )
        else:
            size_of_image = size
        expected = reference(image, size_of_image, resample, antialias)
        crop_height, crop_width = crop_size or size_of_image
        top, left = (size_of_image[0] - crop_height) // 2, (size_of_image[1] - crop_width) // 2
        torch.testing.assert_close(
            result, expected[:, top : top + crop_height, left : left + crop_width], atol=2e-3, rtol=0
        )


def test_resize_normalize_patchify_matches_reference():
    frames = random_images([(280, 420), (140, 140), (140, 140), (140, 140)])
    target_sizes = [(224, 336), (112, 112), (112, 112), (112, 112)]
    items = [[0], [1, 2, 3]]
    patch, merge, temporal = 14, 2, 2
    pixel_values, grid_thw = cv_utils.resize_normalize_patchify(
        frames, target_sizes, MEAN, STD, 1 / 255, "bicubic", patch, merge, temporal, items=items, round_to_uint8=False
    )

    expected = []
    for item in items:
        clip = torch.stack([reference(frames[index], target_sizes[index], "bicubic") for index in item])
        clip = torch.cat([clip, clip[-1:].expand(-clip.shape[0] % temporal, -1, -1, -1)])
        grid_t, (height, width) = clip.shape[0] // temporal, clip.shape[-2:]
        patches = clip.view(
            grid_t, temporal, 3, height // patch // merge, merge, patch, width // patch // merge, merge, patch
        )
        expected.append(patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(-1, 3 * temporal * patch * patch))
    assert grid_thw.dtype == torch.int64
    assert grid_thw.tolist() == [[1, 16, 24], [2, 8, 8]]
    torch.testing.assert_close(pixel_values, torch.cat(expected), atol=2e-3, rtol=0)


@pytest.mark.parametrize("resample", ["bilinear", "bicubic"])
def test_resize_normalize_rounds_like_two_uint8_passes(resample):
    images = random_images([(480, 640), (1024, 768)])
    output = cv_utils.resize_normalize(images, MEAN, STD, 1 / 255, resample, size=(224, 224))
    for image, result in zip(images, output):
        wide = F.interpolate(image[None].float(), size=(image.shape[1], 224), mode=resample, antialias=True)
        wide = wide.round().clamp(0, 255)
        resized = F.interpolate(wide, size=(224, 224), mode=resample, antialias=True).round().clamp(0, 255)[0]
        mean = torch.tensor(MEAN, device=image.device)[:, None, None]
        std = torch.tensor(STD, device=image.device)[:, None, None]
        levels = ((result * std + mean) * 255 - resized).abs()
        assert levels.max().item() <= 1.01
        assert (levels > 0.5).float().mean().item() < 0.05


def test_invalid_inputs_raise():
    image = random_images([(64, 64)])[0]
    with pytest.raises(ValueError, match="uint8"):
        cv_utils.resize_normalize([image.float()], MEAN, STD, 1 / 255, "bilinear", size=(32, 32))
    with pytest.raises(ValueError, match="crop_size"):
        cv_utils.resize_normalize([image], MEAN, STD, 1 / 255, "bilinear", size=(32, 32), crop_size=(48, 48))
    with pytest.raises(ValueError, match="exactly one"):
        cv_utils.resize_normalize([image], MEAN, STD, 1 / 255, "bilinear", size=(32, 32), shortest_edge=32)
    with pytest.raises(ValueError, match="needs a `crop_size`"):
        cv_utils.resize_normalize([image], MEAN, STD, 1 / 255, "bilinear", shortest_edge=32)
    with pytest.raises(ValueError, match="one value per channel"):
        cv_utils.resize_normalize([image], [0.5, 0.5], [0.5, 0.5], 1 / 255, "bilinear", size=(32, 32))
    with pytest.raises(ValueError, match="multiple"):
        cv_utils.resize_normalize_patchify([image], [(42, 56)], MEAN, STD, 1 / 255, "bicubic", 14, 2, 2)
    with pytest.raises(ValueError, match="share a target size"):
        cv_utils.resize_normalize_patchify(
            [image, image], [(56, 56), (28, 28)], MEAN, STD, 1 / 255, "bicubic", 14, 2, 2, items=[[0, 1]]
        )


def test_single_mean_and_std_apply_to_every_channel():
    images = random_images([(64, 64)])
    single = cv_utils.resize_normalize(images, 0.5, 0.5, 1 / 255, "bilinear", size=(32, 32))
    per_channel = cv_utils.resize_normalize(images, [0.5] * 3, [0.5] * 3, 1 / 255, "bilinear", size=(32, 32))
    torch.testing.assert_close(single, per_channel, atol=0, rtol=0)


@pytest.mark.parametrize("patch, target_sizes", [(14, [(280, 420), (504, 336)]), (16, [(288, 448), (512, 320)])])
def test_resize_normalize_patchify_row_by_row_with_merge_size_one(patch, target_sizes):
    frames = random_images([(300, 450), (500, 333)])
    pixel_values, grid_thw = cv_utils.resize_normalize_patchify(
        frames, target_sizes, MEAN, STD, 1 / 255, "bicubic", patch, 1, 1, round_to_uint8=False
    )

    expected = []
    for frame, (height, width) in zip(frames, target_sizes):
        image = reference(frame, (height, width), "bicubic")
        patches = image.view(3, height // patch, patch, width // patch, patch).permute(1, 3, 0, 2, 4)
        expected.append(patches.reshape(-1, 3 * patch * patch))
    assert grid_thw.tolist() == [[1, height // patch, width // patch] for height, width in target_sizes]
    torch.testing.assert_close(pixel_values, torch.cat(expected), atol=2e-3, rtol=0)
