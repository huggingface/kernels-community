import torch

from kernels.benchmark import Benchmark

MIN_RANK_CUSTOM = 16


def use_cutlass_shrink(lora_rank: int) -> bool:
    return lora_rank < MIN_RANK_CUSTOM


def lora_ref_impl(
    x: torch.Tensor,
    wa: torch.Tensor,
    wb: torch.Tensor,
    s_start: torch.Tensor,
    s_end: torch.Tensor,
    layer_idx: int,
    lora_rank: int,
) -> torch.Tensor:
    y = torch.zeros(x.shape[0], wb.shape[2], dtype=x.dtype, device=x.device)

    for i in range(len(s_start)):
        start = s_start[i].item()
        end = s_end[i].item()
        if end - start <= 0:
            continue

        xi = x[start:end]
        wai = wa[layer_idx]
        wbi = wb[layer_idx]

        # Layout differs based on rank
        if not use_cutlass_shrink(lora_rank):
            wai = wai.t()  # (r, H) -> (H, r)

        # Compute: y[start:end] += x[start:end] @ wa @ wb
        tmp = xi @ wai  # (seq, H) @ (H, r) -> (seq, r)
        y[start:end] = tmp @ wbi  # (seq, r) @ (r, H) -> (seq, H)

    return y


class PunicaSgmvBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        batch_size = 8
        hidden_dim = 1024
        lora_rank = 32
        num_layers = 2
        num_segments = 2
        layer_idx = 0
        dtype = torch.float16

        self.lora_rank = lora_rank
        self.layer_idx = layer_idx

        # Input tensor
        self.x = torch.randn(batch_size, hidden_dim, device=self.device, dtype=dtype)

        # LoRA weights - layout depends on rank
        if use_cutlass_shrink(lora_rank):
            # cutlass shrink uses (num_layers, H, r) layout
            self.wa = torch.randn(
                num_layers, hidden_dim, lora_rank, device=self.device, dtype=dtype
            )
        else:
            # custom kernel uses (num_layers, r, H) layout
            self.wa = torch.randn(
                num_layers, lora_rank, hidden_dim, device=self.device, dtype=dtype
            )

        self.wb = torch.randn(
            num_layers, lora_rank, hidden_dim, device=self.device, dtype=dtype
        )

        # Segment indices - split batch into segments
        self.s_start = torch.tensor([0, 4], dtype=torch.int32, device=self.device)
        self.s_end = torch.tensor([4, 8], dtype=torch.int32, device=self.device)

        # Pointers to LoRA weights (both segments use same weights for simplicity)
        self.wa_ptr = torch.tensor(
            [self.wa.data_ptr()] * num_segments, dtype=torch.int64, device=self.device
        )
        self.wb_ptr = torch.tensor(
            [self.wb.data_ptr()] * num_segments, dtype=torch.int64, device=self.device
        )

        # Get temporary buffers
        self.tmp_shrink, self.tmp_expand = self.kernel.get_tmp_tensors(
            num_segments, lora_rank, torch.device("cuda")
        )

        # Output tensor
        self.out = torch.zeros(batch_size, hidden_dim, device=self.device, dtype=dtype)

    def benchmark_base(self):
        self.out.zero_()
        v = self.kernel.lora_a_sgmv_cutlass(
            self.x,
            self.tmp_shrink,
            self.wa_ptr,
            self.s_start,
            self.s_end,
            self.layer_idx,
            self.lora_rank,
        )
        self.kernel.lora_b_sgmv_cutlass(
            self.out,
            v,
            self.tmp_expand,
            self.wb_ptr,
            self.s_start,
            self.s_end,
            self.layer_idx,
        )

    def verify_base(self) -> torch.Tensor:
        return lora_ref_impl(
            self.x,
            self.wa,
            self.wb,
            self.s_start,
            self.s_end,
            self.layer_idx,
            self.lora_rank,
        )

    def setup_large(self):
        batch_size = 64
        hidden_dim = 4096
        lora_rank = 64
        num_layers = 4
        num_segments = 8
        layer_idx = 0
        dtype = torch.float16

        self.lora_rank = lora_rank
        self.layer_idx = layer_idx

        # Scale inputs to keep outputs in reasonable range for float16 precision
        scale = 0.01
        self.x = torch.randn(batch_size, hidden_dim, device=self.device, dtype=dtype) * scale

        if use_cutlass_shrink(lora_rank):
            self.wa = (
                torch.randn(
                    num_layers, hidden_dim, lora_rank, device=self.device, dtype=dtype
                )
                * scale
            )
        else:
            self.wa = (
                torch.randn(
                    num_layers, lora_rank, hidden_dim, device=self.device, dtype=dtype
                )
                * scale
            )

        self.wb = (
            torch.randn(num_layers, lora_rank, hidden_dim, device=self.device, dtype=dtype)
            * scale
        )

        # Split batch evenly across segments
        seg_size = batch_size // num_segments
        self.s_start = torch.tensor(
            [i * seg_size for i in range(num_segments)],
            dtype=torch.int32,
            device=self.device,
        )
        self.s_end = torch.tensor(
            [(i + 1) * seg_size for i in range(num_segments)],
            dtype=torch.int32,
            device=self.device,
        )

        self.wa_ptr = torch.tensor(
            [self.wa.data_ptr()] * num_segments, dtype=torch.int64, device=self.device
        )
        self.wb_ptr = torch.tensor(
            [self.wb.data_ptr()] * num_segments, dtype=torch.int64, device=self.device
        )

        self.tmp_shrink, self.tmp_expand = self.kernel.get_tmp_tensors(
            num_segments, lora_rank, torch.device("cuda")
        )

        self.out = torch.zeros(batch_size, hidden_dim, device=self.device, dtype=dtype)

    def benchmark_large(self):
        self.out.zero_()
        v = self.kernel.lora_a_sgmv_cutlass(
            self.x,
            self.tmp_shrink,
            self.wa_ptr,
            self.s_start,
            self.s_end,
            self.layer_idx,
            self.lora_rank,
        )
        self.kernel.lora_b_sgmv_cutlass(
            self.out,
            v,
            self.tmp_expand,
            self.wb_ptr,
            self.s_start,
            self.s_end,
            self.layer_idx,
        )

    def verify_large(self) -> torch.Tensor:
        return lora_ref_impl(
            self.x,
            self.wa,
            self.wb,
            self.s_start,
            self.s_end,
            self.layer_idx,
            self.lora_rank,
        )
