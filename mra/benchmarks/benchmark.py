import torch

from kernels.benchmark import Benchmark


def mm_to_sparse_reference(
    dense_A: torch.Tensor,
    dense_B: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    batch_size = dense_A.size(0)
    A_num_block = dense_A.size(1)
    B_num_block = dense_B.size(1)
    dim = dense_A.size(2)
    num_block = indices.size(1)

    # Output: (batch_size, num_block, 32, 32)
    sparse_C = torch.zeros(
        batch_size, num_block, 32, 32, device=dense_A.device, dtype=dense_A.dtype
    )

    for b in range(batch_size):
        for blk in range(num_block):
            AB_idx = indices[b, blk].item()
            A_idx = AB_idx // B_num_block
            B_idx = AB_idx % B_num_block

            A_block = dense_A[b, A_idx]  # (dim, 32)
            B_block = dense_B[b, B_idx]  # (dim, 32)

            # Kernel computes C = B.T @ A: (32, dim) @ (dim, 32) = (32, 32)
            sparse_C[b, blk] = B_block.T @ A_block

    return sparse_C


class MRABenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        # Config matching the kernel's expected format
        batch_size = 2
        num_heads = 8
        head_dim = 64
        block_size = 32  # Fixed by kernel

        A_num_block = 4
        B_num_block = 4
        total_blocks = A_num_block * B_num_block
        indices_per_block = 4  # Must be divisible by 4

        self.batch_heads = batch_size * num_heads

        # dense_A: [batch_size, A_num_block, dim, 32]
        self.dense_a = torch.randn(
            self.batch_heads,
            A_num_block,
            head_dim,
            block_size,
            device="cuda",
            dtype=torch.float32,
        )
        # dense_B: [batch_size, B_num_block, dim, 32]
        self.dense_b = torch.randn(
            self.batch_heads,
            B_num_block,
            head_dim,
            block_size,
            device="cuda",
            dtype=torch.float32,
        )
        # indices: [batch_size, num_block]
        self.indices = torch.randint(
            0,
            total_blocks,
            (self.batch_heads, indices_per_block),
            device="cuda",
            dtype=torch.int32,
        )

    def benchmark_base(self):
        self.out = self.kernel.mm_to_sparse(self.dense_a, self.dense_b, self.indices)

    def verify_base(self) -> torch.Tensor:
        return mm_to_sparse_reference(self.dense_a, self.dense_b, self.indices)

    def setup_large(self):
        batch_size = 4
        num_heads = 8
        head_dim = 64
        block_size = 32

        A_num_block = 8
        B_num_block = 8
        total_blocks = A_num_block * B_num_block
        indices_per_block = 8  # Must be divisible by 4

        self.batch_heads = batch_size * num_heads

        self.dense_a = torch.randn(
            self.batch_heads,
            A_num_block,
            head_dim,
            block_size,
            device="cuda",
            dtype=torch.float32,
        )
        self.dense_b = torch.randn(
            self.batch_heads,
            B_num_block,
            head_dim,
            block_size,
            device="cuda",
            dtype=torch.float32,
        )
        self.indices = torch.randint(
            0,
            total_blocks,
            (self.batch_heads, indices_per_block),
            device="cuda",
            dtype=torch.int32,
        )

    def benchmark_large(self):
        self.out = self.kernel.mm_to_sparse(self.dense_a, self.dense_b, self.indices)

    def verify_large(self) -> torch.Tensor:
        return mm_to_sparse_reference(self.dense_a, self.dense_b, self.indices)
