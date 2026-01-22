import torch
import torch.nn.functional as F

from kernels.benchmark import Benchmark


def ms_deform_attn_reference(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    batch, _, num_heads, channels = value.shape
    _, num_query, _, num_levels, num_points, _ = sampling_locations.shape

    # Split value by levels
    value_list = []
    for level_id in range(num_levels):
        H, W = spatial_shapes[level_id]
        start_idx = level_start_index[level_id]
        end_idx = (
            level_start_index[level_id + 1]
            if level_id < num_levels - 1
            else value.shape[1]
        )
        # (batch, H*W, num_heads, channels) -> (batch, num_heads, channels, H, W)
        value_level = value[:, start_idx:end_idx, :, :].view(
            batch, H, W, num_heads, channels
        )
        value_level = value_level.permute(0, 3, 4, 1, 2).contiguous()
        value_list.append(value_level)

    # Sample from each level
    output = torch.zeros(
        batch, num_query, num_heads, channels, device=value.device, dtype=value.dtype
    )

    for level_id in range(num_levels):
        H, W = spatial_shapes[level_id]
        value_level = value_list[level_id]  # (batch, num_heads, channels, H, W)

        # Get sampling locations for this level: (batch, num_query, num_heads, num_points, 2)
        sampling_loc_level = sampling_locations[:, :, :, level_id, :, :]

        # Convert from [0, 1] to [-1, 1] for grid_sample
        grid = (
            2.0 * sampling_loc_level - 1.0
        )  # (batch, num_query, num_heads, num_points, 2)

        # Reshape for grid_sample: need (batch * num_heads, channels, H, W) and (batch * num_heads, num_query, num_points, 2)
        value_level = value_level.view(batch * num_heads, channels, H.item(), W.item())
        grid = grid.permute(
            0, 2, 1, 3, 4
        ).contiguous()  # (batch, num_heads, num_query, num_points, 2)
        grid = grid.view(batch * num_heads, num_query, num_points, 2)

        # Sample: output is (batch * num_heads, channels, num_query, num_points)
        sampled = F.grid_sample(
            value_level,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # Reshape back: (batch, num_heads, channels, num_query, num_points)
        sampled = sampled.view(batch, num_heads, channels, num_query, num_points)
        # -> (batch, num_query, num_heads, num_points, channels)
        sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()

        # Get attention weights for this level: (batch, num_query, num_heads, num_points)
        attn_level = attention_weights[:, :, :, level_id, :]

        # Weighted sum over points: (batch, num_query, num_heads, channels)
        output += (sampled * attn_level.unsqueeze(-1)).sum(dim=3)

    # Reshape to (batch, num_query, num_heads * channels)
    output = output.view(batch, num_query, num_heads * channels)
    return output


class MSDeformAttnBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        batch = 2
        num_heads = 8
        channels = 32  # embed_dim = num_heads * channels = 256
        num_levels = 4
        num_query = 300
        num_points = 4
        im2col_step = 64

        # Spatial shapes for 4 levels: 64x64, 32x32, 16x16, 8x8
        spatial_shapes = torch.tensor(
            [[64, 64], [32, 32], [16, 16], [8, 8]], dtype=torch.int64, device="cuda"
        )
        # Calculate spatial_size = sum of H*W for all levels
        spatial_size = (64 * 64) + (32 * 32) + (16 * 16) + (8 * 8)  # 5440

        # Level start indices
        level_start_index = torch.tensor(
            [0, 64 * 64, 64 * 64 + 32 * 32, 64 * 64 + 32 * 32 + 16 * 16],
            dtype=torch.int64,
            device="cuda",
        )

        self.value = torch.randn(
            batch, spatial_size, num_heads, channels, device="cuda", dtype=torch.float32
        )
        self.spatial_shapes = spatial_shapes
        self.level_start_index = level_start_index
        self.sampling_loc = torch.rand(
            batch,
            num_query,
            num_heads,
            num_levels,
            num_points,
            2,
            device="cuda",
            dtype=torch.float32,
        )
        self.attn_weight = torch.rand(
            batch,
            num_query,
            num_heads,
            num_levels,
            num_points,
            device="cuda",
            dtype=torch.float32,
        )
        # Normalize attention weights
        self.attn_weight = self.attn_weight / self.attn_weight.sum(-1, keepdim=True)
        self.im2col_step = im2col_step

        self.out = torch.empty(
            batch, num_query, num_heads * channels, device="cuda", dtype=torch.float32
        )

    def benchmark_forward(self):
        self.out = self.kernel.ms_deform_attn_forward(
            self.value,
            self.spatial_shapes,
            self.level_start_index,
            self.sampling_loc,
            self.attn_weight,
            self.im2col_step,
        )

    def verify_forward(self) -> torch.Tensor:
        return ms_deform_attn_reference(
            self.value,
            self.spatial_shapes,
            self.level_start_index,
            self.sampling_loc,
            self.attn_weight,
        )

    def setup_large(self):
        batch = 8
        num_heads = 8
        channels = 32
        num_levels = 4
        num_query = 900
        num_points = 4
        im2col_step = 64

        spatial_shapes = torch.tensor(
            [[64, 64], [32, 32], [16, 16], [8, 8]], dtype=torch.int64, device="cuda"
        )
        spatial_size = (64 * 64) + (32 * 32) + (16 * 16) + (8 * 8)

        level_start_index = torch.tensor(
            [0, 64 * 64, 64 * 64 + 32 * 32, 64 * 64 + 32 * 32 + 16 * 16],
            dtype=torch.int64,
            device="cuda",
        )

        self.value = torch.randn(
            batch, spatial_size, num_heads, channels, device="cuda", dtype=torch.float32
        )
        self.spatial_shapes = spatial_shapes
        self.level_start_index = level_start_index
        self.sampling_loc = torch.rand(
            batch,
            num_query,
            num_heads,
            num_levels,
            num_points,
            2,
            device="cuda",
            dtype=torch.float32,
        )
        self.attn_weight = torch.rand(
            batch,
            num_query,
            num_heads,
            num_levels,
            num_points,
            device="cuda",
            dtype=torch.float32,
        )
        self.attn_weight = self.attn_weight / self.attn_weight.sum(-1, keepdim=True)
        self.im2col_step = im2col_step

        self.out = torch.empty(
            batch, num_query, num_heads * channels, device="cuda", dtype=torch.float32
        )

    def benchmark_large(self):
        self.out = self.kernel.ms_deform_attn_forward(
            self.value,
            self.spatial_shapes,
            self.level_start_index,
            self.sampling_loc,
            self.attn_weight,
            self.im2col_step,
        )

    def verify_large(self) -> torch.Tensor:
        return ms_deform_attn_reference(
            self.value,
            self.spatial_shapes,
            self.level_start_index,
            self.sampling_loc,
            self.attn_weight,
        )
