import pytest
import torch

import cv_utils


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cv_utils requires CUDA"
)


@pytest.mark.kernels_ci
@requires_cuda
@torch.inference_mode()
def test_cc_2d_single_component():
    inputs = torch.ones((1, 1, 4, 4), dtype=torch.uint8, device="cuda")

    labels, counts = cv_utils.cc_2d(inputs, True)

    torch.testing.assert_close(labels, torch.ones_like(inputs, dtype=torch.int32))
    torch.testing.assert_close(counts, torch.full_like(labels, 16))


@pytest.mark.kernels_ci
@requires_cuda
@torch.inference_mode()
def test_cc_2d_without_counts_and_empty_foreground():
    inputs = torch.zeros((1, 1, 4, 4), dtype=torch.uint8, device="cuda")

    labels, counts = cv_utils.cc_2d(inputs, False)

    torch.testing.assert_close(labels, torch.zeros_like(inputs, dtype=torch.int32))
    torch.testing.assert_close(counts, torch.zeros_like(labels))


@pytest.mark.kernels_ci
@requires_cuda
@torch.inference_mode()
def test_generic_nms_boxes():
    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [5.0, 5.0, 6.0, 6.0]],
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")

    result = cv_utils.generic_nms(boxes, scores, 0.5, False)

    torch.testing.assert_close(result, torch.tensor([0, 2], device="cuda"))


@pytest.mark.kernels_ci
@requires_cuda
@torch.inference_mode()
def test_generic_nms_iou_matrix():
    iou_matrix = torch.tensor(
        [[0.0, 0.6, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.0]], device="cuda"
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")

    result = cv_utils.generic_nms(iou_matrix, scores, 0.5, True)

    torch.testing.assert_close(result, torch.tensor([0, 2], device="cuda"))


@pytest.mark.kernels_ci
@requires_cuda
def test_input_validation():
    with pytest.raises(RuntimeError, match="height"):
        cv_utils.cc_2d(
            torch.zeros((1, 1, 3, 4), dtype=torch.uint8, device="cuda"), True
        )

    with pytest.raises(RuntimeError, match="boxes"):
        cv_utils.generic_nms(
            torch.zeros((2, 3), device="cuda"),
            torch.zeros(2, device="cuda"),
            0.5,
            False,
        )
