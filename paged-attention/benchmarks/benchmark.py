import torch

from kernels.benchmark import Benchmark


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    # query: (q, h, d), key: (k, h, d), value: (k, h, d)
    # Transpose to (h, q, d) and (h, k, d) for batched matmul
    q = query.transpose(0, 1)  # (h, q, d)
    k = key.transpose(0, 1)  # (h, k, d)
    v = value.transpose(0, 1)  # (h, k, d)

    # Compute attention scores: (h, q, d) @ (h, d, k) -> (h, q, k)
    attn_weights = (scale * torch.matmul(q, k.transpose(-1, -2))).float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)

    # Compute output: (h, q, k) @ (h, k, d) -> (h, q, d)
    out = torch.matmul(attn_weights, v)

    # Transpose back to (q, h, d)
    return out.transpose(0, 1)


def ref_paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]
    block_size = value_cache.shape[3]
    max_seq_len = int(seq_lens.max().item())

    # Create position indices for all sequences up to max_seq_len
    positions = torch.arange(max_seq_len, device=query.device)
    block_indices = positions // block_size  # (max_seq_len,)
    block_offsets = positions % block_size  # (max_seq_len,)

    # Gather block numbers for all sequences: (num_seqs, max_seq_len)
    block_numbers = block_tables[:, block_indices.long()]

    # Flatten for gathering: (num_seqs * max_seq_len,)
    flat_block_numbers = block_numbers.reshape(-1)
    flat_offsets = block_offsets.repeat(num_seqs)

    # Gather keys: key_cache is (num_blocks, num_heads, head_size // x, block_size, x)
    # Index into [block_number, :, :, offset, :] and reshape
    keys = key_cache[flat_block_numbers, :, :, flat_offsets, :]
    keys = keys.reshape(num_seqs, max_seq_len, num_heads, head_size)
    keys = keys.transpose(1, 2)  # (num_seqs, num_heads, max_seq_len, head_size)

    # Gather values: value_cache is (num_blocks, num_heads, head_size, block_size)
    values = value_cache[flat_block_numbers, :, :, flat_offsets]
    values = values.reshape(num_seqs, max_seq_len, num_heads, head_size)
    values = values.transpose(1, 2)  # (num_seqs, num_heads, max_seq_len, head_size)

    # Query: (num_seqs, num_heads, head_size) -> (num_seqs, num_heads, 1, head_size)
    q = query.unsqueeze(2)

    # Compute attention scores: (num_seqs, num_heads, 1, head_size) @ (num_seqs, num_heads, head_size, max_seq_len)
    attn_weights = (scale * torch.matmul(q, keys.transpose(-1, -2))).float()

    # Create causal mask for variable sequence lengths
    # Mask out positions beyond seq_len for each sequence
    seq_mask = positions.unsqueeze(0) >= seq_lens.unsqueeze(
        1
    )  # (num_seqs, max_seq_len)
    seq_mask = seq_mask.unsqueeze(1).unsqueeze(2)  # (num_seqs, 1, 1, max_seq_len)
    attn_weights = attn_weights.masked_fill(seq_mask, float("-inf"))

    attn_weights = torch.softmax(attn_weights, dim=-1).to(values.dtype)

    # Compute output: (num_seqs, num_heads, 1, max_seq_len) @ (num_seqs, num_heads, max_seq_len, head_size)
    out = torch.matmul(attn_weights, values)

    return out.squeeze(2)  # (num_seqs, num_heads, head_size)


class PagedAttentionBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        num_seqs = 4
        num_heads = 8
        head_size = 64
        block_size = 16
        max_seq_len = 128
        num_blocks = 64
        dtype = torch.float16

        self.num_heads = num_heads
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / (head_size**0.5)

        # Query tensor (current token)
        self.query = torch.randn(
            num_seqs, num_heads, head_size, device=self.device, dtype=dtype
        )

        # KV cache with proper layout for the kernel
        # x = 16 // element_size, for float16 x = 8
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        self.key_cache = torch.randn(
            num_blocks,
            num_heads,
            head_size // x,
            block_size,
            x,
            device=self.device,
            dtype=dtype,
        )
        self.value_cache = torch.randn(
            num_blocks, num_heads, head_size, block_size, device=self.device, dtype=dtype
        )

        # Block tables: mapping from sequences to memory blocks
        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        self.block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            device=self.device,
            dtype=torch.int32,
        )

        # Sequence lengths
        self.seq_lens = torch.tensor(
            [64, 96, 48, 128], device=self.device, dtype=torch.int32
        )

        # KV scales
        self.k_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.v_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        # Output tensor
        self.out = torch.empty_like(self.query)

    def benchmark_base(self):
        self.kernel.paged_attention_v1(
            self.out,
            self.query,
            self.key_cache,
            self.value_cache,
            num_kv_heads=self.num_heads,
            scale=self.scale,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            block_size=self.block_size,
            max_seq_len=self.max_seq_len,
            alibi_slopes=None,
            kv_cache_dtype="auto",
            k_scale=self.k_scale,
            v_scale=self.v_scale,
        )

    def verify_base(self) -> torch.Tensor:
        return ref_paged_attention(
            self.query,
            self.key_cache,
            self.value_cache,
            self.block_tables,
            self.seq_lens,
            self.scale,
        )

    def setup_large(self):
        num_seqs = 16
        num_heads = 32
        head_size = 128
        block_size = 16
        max_seq_len = 512
        num_blocks = 256
        dtype = torch.float16

        self.num_heads = num_heads
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / (head_size**0.5)

        self.query = torch.randn(
            num_seqs, num_heads, head_size, device=self.device, dtype=dtype
        )

        x = 16 // torch.tensor([], dtype=dtype).element_size()
        self.key_cache = torch.randn(
            num_blocks,
            num_heads,
            head_size // x,
            block_size,
            x,
            device=self.device,
            dtype=dtype,
        )
        self.value_cache = torch.randn(
            num_blocks, num_heads, head_size, block_size, device=self.device, dtype=dtype
        )

        max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        self.block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            device=self.device,
            dtype=torch.int32,
        )

        # Variable sequence lengths
        self.seq_lens = torch.randint(
            64, max_seq_len + 1, (num_seqs,), device=self.device, dtype=torch.int32
        )

        self.k_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        self.v_scale = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        self.out = torch.empty_like(self.query)

    def benchmark_large(self):
        self.kernel.paged_attention_v1(
            self.out,
            self.query,
            self.key_cache,
            self.value_cache,
            num_kv_heads=self.num_heads,
            scale=self.scale,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            block_size=self.block_size,
            max_seq_len=self.max_seq_len,
            alibi_slopes=None,
            kv_cache_dtype="auto",
            k_scale=self.k_scale,
            v_scale=self.v_scale,
        )

    def verify_large(self) -> torch.Tensor:
        return ref_paged_attention(
            self.query,
            self.key_cache,
            self.value_cache,
            self.block_tables,
            self.seq_lens,
            self.scale,
        )
