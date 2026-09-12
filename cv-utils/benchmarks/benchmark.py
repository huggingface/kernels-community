import torch

from kernels.benchmark import Benchmark


def reference_nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Small Torch reference for the box-NMS path."""
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break

        remaining = order[1:]
        current_box = boxes[current]
        other_boxes = boxes[remaining]

        top_left = torch.maximum(current_box[:2], other_boxes[:, :2])
        bottom_right = torch.minimum(current_box[2:], other_boxes[:, 2:])
        intersection_size = (bottom_right - top_left).clamp_min(0)
        intersection = intersection_size[:, 0] * intersection_size[:, 1]

        current_area = (current_box[2] - current_box[0]) * (
            current_box[3] - current_box[1]
        )
        other_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (
            other_boxes[:, 3] - other_boxes[:, 1]
        )
        union = current_area + other_area - intersection
        iou = intersection / union
        order = remaining[iou <= iou_threshold]

    return torch.stack(keep) if keep else order


def reference_cc_2d(inputs: torch.Tensor) -> torch.Tensor:
    """Reference for the single foreground component used by the benchmark."""
    labels = inputs.to(torch.int32)
    component_size = inputs.sum(dim=(-2, -1), keepdim=True).to(torch.int32)
    counts = torch.where(labels != 0, component_size, torch.zeros_like(labels))
    return torch.cat((labels.flatten(), counts.flatten()))


class GenericNMSBenchmark(Benchmark):
    seed: int = 42

    def _setup(self, num_boxes: int):
        x1y1 = torch.rand(num_boxes, 2, device=self.device)
        sizes = 0.05 + 0.2 * torch.rand(num_boxes, 2, device=self.device)
        self.boxes = torch.cat((x1y1, x1y1 + sizes), dim=1)
        self.scores = torch.rand(num_boxes, device=self.device)
        self.iou_threshold = 0.5
        self.out = torch.empty(num_boxes, dtype=torch.long, device=self.device)

    def setup(self):
        self._setup(512)

    def benchmark_base(self):
        self.out = self.kernel.generic_nms(
            self.boxes, self.scores, self.iou_threshold, False
        )

    def verify_base(self) -> torch.Tensor:
        return reference_nms(self.boxes, self.scores, self.iou_threshold)

    def setup_large(self):
        self._setup(2048)

    def benchmark_large(self):
        self.out = self.kernel.generic_nms(
            self.boxes, self.scores, self.iou_threshold, False
        )

    def verify_large(self) -> torch.Tensor:
        return reference_nms(self.boxes, self.scores, self.iou_threshold)


class ConnectedComponentsBenchmark(Benchmark):
    seed: int = 42

    def _setup(self, size: int):
        self.inputs = torch.ones(
            (1, 1, size, size), dtype=torch.uint8, device=self.device
        )
        self.out = torch.empty(2 * size * size, dtype=torch.int32, device=self.device)

    def setup(self):
        self._setup(64)

    def benchmark_base(self):
        labels, counts = self.kernel.cc_2d(self.inputs, True)
        self.out = torch.cat((labels.flatten(), counts.flatten()))

    def verify_base(self) -> torch.Tensor:
        return reference_cc_2d(self.inputs)

    def setup_large(self):
        self._setup(256)

    def benchmark_large(self):
        labels, counts = self.kernel.cc_2d(self.inputs, True)
        self.out = torch.cat((labels.flatten(), counts.flatten()))

    def verify_large(self) -> torch.Tensor:
        return reference_cc_2d(self.inputs)
