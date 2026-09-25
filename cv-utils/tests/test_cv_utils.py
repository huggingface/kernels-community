"""Tests for the cv-utils kernels, checked against pure-PyTorch/Python references.

The whole suite is small and fast, so every test is part of the CI subset.
"""

from collections import deque

import kernels
import pytest
import torch


cv_utils = kernels.get_kernel("kernels-community/cv-utils", version=1)

pytestmark = pytest.mark.kernels_ci

DEVICE = "cuda"


def random_boxes(n, dtype, generator):
    # Integer coordinates are exactly representable in every supported dtype
    # (including float16), so the kernel and the reference compute identical
    # IoUs and ties at the threshold resolve identically.
    xy = torch.randint(0, 64, (n, 2), generator=generator)
    wh = torch.randint(1, 32, (n, 2), generator=generator)
    return torch.cat([xy, xy + wh], dim=1).to(dtype)


def box_iou_ref(boxes):
    """Pairwise IoU, mirroring the arithmetic of the kernel's `devIoU`."""
    # float16 is accumulated in float32 by the kernel (at::acc_type).
    b = boxes.float() if boxes.dtype == torch.float16 else boxes
    lt = torch.maximum(b[:, None, :2], b[None, :, :2])
    rb = torch.minimum(b[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area[:, None] + area[None, :] - inter)


def nms_ref(iou, scores, iou_threshold):
    """Greedy NMS on a precomputed IoU matrix; returns kept indices by score."""
    order = torch.sort(scores, stable=True, descending=True).indices
    iou_sorted = iou[order][:, order]
    n = scores.numel()
    suppressed = torch.zeros(n, dtype=torch.bool)
    keep = []
    for i in range(n):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou_sorted[i] > iou_threshold
    return order[torch.tensor(keep, dtype=torch.long)]


# The thresholds are exactly representable in float16/float32, so the
# comparison in the kernel (float) and in the reference agree.
@pytest.mark.parametrize("iou_threshold", [0.25, 0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("n", [1, 7, 64, 65, 300, 1000])
def test_generic_nms_boxes(n, dtype, iou_threshold):
    g = torch.Generator().manual_seed(n)
    boxes = random_boxes(n, dtype, g)
    scores = torch.rand(n, generator=g)

    keep = cv_utils.generic_nms(
        boxes.to(DEVICE), scores.to(DEVICE), iou_threshold, False
    )
    keep_ref = nms_ref(box_iou_ref(boxes), scores, iou_threshold)

    assert keep.dtype == torch.long
    assert keep.device.type == DEVICE
    torch.testing.assert_close(keep.cpu(), keep_ref, rtol=0, atol=0)


@pytest.mark.parametrize("iou_threshold", [0.25, 0.5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("n", [1, 7, 64, 65, 300])
def test_generic_nms_iou_matrix(n, dtype, iou_threshold):
    g = torch.Generator().manual_seed(n + 1)
    boxes = random_boxes(n, torch.float32, g)
    scores = torch.rand(n, generator=g)
    iou = box_iou_ref(boxes).to(dtype)

    keep = cv_utils.generic_nms(iou.to(DEVICE), scores.to(DEVICE), iou_threshold, True)
    keep_ref = nms_ref(iou, scores, iou_threshold)

    assert keep.dtype == torch.long
    torch.testing.assert_close(keep.cpu(), keep_ref, rtol=0, atol=0)


def test_generic_nms_matches_between_modes():
    g = torch.Generator().manual_seed(42)
    boxes = random_boxes(500, torch.float32, g).to(DEVICE)
    scores = torch.rand(500, generator=g).to(DEVICE)

    keep_boxes = cv_utils.generic_nms(boxes, scores, 0.5, False)
    iou = box_iou_ref(boxes.cpu()).to(DEVICE)
    keep_iou = cv_utils.generic_nms(iou, scores, 0.5, True)

    torch.testing.assert_close(keep_boxes, keep_iou, rtol=0, atol=0)


def test_generic_nms_non_contiguous():
    g = torch.Generator().manual_seed(7)
    boxes = random_boxes(200, torch.float32, g)
    scores = torch.rand(200, generator=g)
    # Column-major boxes and strided scores.
    boxes_nc = boxes.t().contiguous().t().to(DEVICE)
    scores_nc = torch.stack([scores, scores], dim=1).to(DEVICE)[:, 0]
    assert not boxes_nc.is_contiguous() and not scores_nc.is_contiguous()

    keep = cv_utils.generic_nms(boxes_nc, scores_nc, 0.5, False)
    keep_ref = nms_ref(box_iou_ref(boxes), scores, 0.5)
    torch.testing.assert_close(keep.cpu(), keep_ref, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_generic_nms_other_device():
    g = torch.Generator().manual_seed(8)
    boxes = random_boxes(200, torch.float32, g)
    scores = torch.rand(200, generator=g)
    with torch.cuda.device(0):
        keep = cv_utils.generic_nms(boxes.to("cuda:1"), scores.to("cuda:1"), 0.5, False)
    assert keep.device == torch.device("cuda:1")
    keep_ref = nms_ref(box_iou_ref(boxes), scores, 0.5)
    torch.testing.assert_close(keep.cpu(), keep_ref, rtol=0, atol=0)


def test_generic_nms_empty():
    boxes = torch.empty(0, 4, device=DEVICE)
    scores = torch.empty(0, device=DEVICE)
    keep = cv_utils.generic_nms(boxes, scores, 0.5, False)
    assert keep.shape == (0,)
    assert keep.dtype == torch.long


def test_generic_nms_invalid_inputs():
    boxes = torch.rand(8, 4, device=DEVICE)
    scores = torch.rand(8, device=DEVICE)
    with pytest.raises(RuntimeError, match="CUDA"):
        cv_utils.generic_nms(boxes.cpu(), scores.cpu(), 0.5, False)
    with pytest.raises(RuntimeError, match="same number of elements"):
        cv_utils.generic_nms(boxes, scores[:4], 0.5, False)
    with pytest.raises(RuntimeError, match=r"\[N,4\]"):
        cv_utils.generic_nms(torch.rand(8, 5, device=DEVICE), scores, 0.5, False)
    with pytest.raises(RuntimeError, match=r"\[N,N\]"):
        cv_utils.generic_nms(boxes, scores, 0.5, True)


def cc_ref(image):
    """8-connected component labels and component sizes for one HxW image.

    Returns (labels, sizes) as nested lists; labels are arbitrary positive ids
    per component (0 for background), sizes the pixel count of each pixel's
    component (0 for background).
    """
    img = image.tolist()
    h, w = len(img), len(img[0])
    labels = [[0] * w for _ in range(h)]
    sizes = [[0] * w for _ in range(h)]
    next_label = 1
    for r0 in range(h):
        for c0 in range(w):
            if not img[r0][c0] or labels[r0][c0]:
                continue
            component = []
            labels[r0][c0] = next_label
            queue = deque([(r0, c0)])
            while queue:
                r, c = queue.popleft()
                component.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if (
                            0 <= rr < h
                            and 0 <= cc < w
                            and img[rr][cc]
                            and not labels[rr][cc]
                        ):
                            labels[rr][cc] = next_label
                            queue.append((rr, cc))
            for r, c in component:
                sizes[r][c] = len(component)
            next_label += 1
    return labels, sizes


def assert_same_partition(labels, labels_ref):
    """Labels may differ in value, but must induce the same components."""
    flat = [x for row in labels for x in row]
    flat_ref = [x for row in labels_ref for x in row]
    for x, x_ref in zip(flat, flat_ref):
        assert (x == 0) == (x_ref == 0), "foreground/background mismatch"
    pairs = {(x, x_ref) for x, x_ref in zip(flat, flat_ref) if x_ref != 0}
    assert len(pairs) == len({x for x, _ in pairs}) == len({x for _, x in pairs})


def check_cc(inputs):
    labels, counts = cv_utils.cc_2d(inputs.to(DEVICE), True)
    labels_nc, counts_nc = cv_utils.cc_2d(inputs.to(DEVICE), False)

    assert labels.shape == inputs.shape and counts.shape == inputs.shape
    assert labels.dtype == torch.int32 and counts.dtype == torch.int32
    torch.testing.assert_close(labels_nc, labels, rtol=0, atol=0)
    assert not counts_nc.any(), "counts must be zero when get_counts=False"

    for n in range(inputs.shape[0]):
        labels_ref, sizes_ref = cc_ref(inputs[n, 0])
        assert_same_partition(labels[n, 0].tolist(), labels_ref)
        assert counts[n, 0].tolist() == sizes_ref


@pytest.mark.parametrize("density", [0.2, 0.45, 0.7])
@pytest.mark.parametrize("shape", [(1, 1, 2, 2), (2, 1, 16, 16), (3, 1, 30, 46), (1, 1, 64, 130)])
def test_cc_2d_random(shape, density):
    g = torch.Generator().manual_seed(sum(shape))
    inputs = (torch.rand(shape, generator=g) < density).to(torch.uint8)
    check_cc(inputs)


def test_cc_2d_patterns():
    h, w = 34, 66
    patterns = torch.zeros(5, 1, h, w, dtype=torch.uint8)
    # All background.
    # All foreground: a single component.
    patterns[1] = 1
    # Diagonal line: only 8-connected.
    idx = torch.arange(min(h, w))
    patterns[2, 0, idx, idx] = 1
    # Checkerboard: a single 8-connected component.
    patterns[3, 0] = (torch.arange(h)[:, None] + torch.arange(w)[None, :]) % 2 == 0
    # Isolated pixels at odd positions, which fall into the lower-right corner
    # of the kernel's 2x2 blocks.
    patterns[4, 0, 1::4, 1::4] = 1
    check_cc(patterns)


def test_cc_2d_non_zero_values_are_foreground():
    inputs = torch.tensor([[[[0, 7], [255, 0]]]], dtype=torch.uint8)
    labels, counts = cv_utils.cc_2d(inputs.to(DEVICE), True)
    labels = labels.cpu()
    assert labels[0, 0, 0, 1] == labels[0, 0, 1, 0] != 0
    assert counts.cpu().tolist() == [[[[0, 2], [2, 0]]]]


def test_cc_2d_non_contiguous():
    g = torch.Generator().manual_seed(9)
    base = (torch.rand(2, 1, 40, 30, generator=g) < 0.45).to(torch.uint8)
    # A transposed view: logically [N, 1, 30, 40], but not contiguous.
    inputs = base.transpose(2, 3)
    assert not inputs.is_contiguous()
    check_cc(inputs)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
def test_cc_2d_other_device():
    g = torch.Generator().manual_seed(10)
    inputs = (torch.rand(2, 1, 30, 46, generator=g) < 0.45).to(torch.uint8)
    with torch.cuda.device(0):
        labels, counts = cv_utils.cc_2d(inputs.to("cuda:1"), True)
    assert labels.device == counts.device == torch.device("cuda:1")
    for n in range(inputs.shape[0]):
        labels_ref, sizes_ref = cc_ref(inputs[n, 0])
        assert_same_partition(labels[n, 0].tolist(), labels_ref)
        assert counts[n, 0].tolist() == sizes_ref


def test_cc_2d_empty():
    inputs = torch.zeros(0, 1, 4, 4, dtype=torch.uint8, device=DEVICE)
    labels, counts = cv_utils.cc_2d(inputs, True)
    assert labels.shape == (0, 1, 4, 4) and counts.shape == (0, 1, 4, 4)


def test_cc_2d_invalid_inputs():
    with pytest.raises(RuntimeError, match="uint8"):
        cv_utils.cc_2d(torch.zeros(1, 1, 4, 4, device=DEVICE), True)
    with pytest.raises(RuntimeError, match="even"):
        cv_utils.cc_2d(torch.zeros(1, 1, 4, 5, dtype=torch.uint8, device=DEVICE), True)
    with pytest.raises(RuntimeError, match="shape"):
        cv_utils.cc_2d(torch.zeros(1, 2, 4, 4, dtype=torch.uint8, device=DEVICE), True)
    with pytest.raises(RuntimeError, match="CUDA"):
        cv_utils.cc_2d(torch.zeros(1, 1, 4, 4, dtype=torch.uint8), True)
