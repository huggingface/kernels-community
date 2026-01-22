import torch

from kernels.benchmark import Benchmark


def lsh_weighted_cumulation_reference(
    query_mask: torch.Tensor,
    query_hash_code: torch.Tensor,
    query_weight: torch.Tensor,
    key_mask: torch.Tensor,
    key_hash_code: torch.Tensor,
    key_weight: torch.Tensor,
    value: torch.Tensor,
    hashtable_capacity: int,
) -> torch.Tensor:
    batch_size, num_query, num_hash_f = query_hash_code.shape
    _, num_key, value_dim = value.shape
    weight_dim = query_weight.shape[2]
    device = value.device
    dtype = value.dtype

    output = torch.zeros(batch_size, num_query, value_dim, device=device, dtype=dtype)

    for b in range(batch_size):
        for weight_idx in range(weight_dim):
            # Build hashtables for all hash functions
            hashtables = torch.zeros(
                num_hash_f, hashtable_capacity, value_dim, device=device, dtype=dtype
            )

            k_mask = key_mask[b, :].float()  # [num_key]
            k_weight_val = key_weight[b, :, weight_idx]  # [num_key]

            for h in range(num_hash_f):
                k_hash = key_hash_code[b, :, h].long()  # [num_key]
                # Weighted values: [num_key, value_dim]
                weighted_values = (
                    k_mask.unsqueeze(-1) * k_weight_val.unsqueeze(-1) * value[b]
                )
                k_hash_expanded = k_hash.unsqueeze(-1).expand(-1, value_dim)
                hashtables[h].scatter_add_(0, k_hash_expanded, weighted_values)

            # Query: sum over all hash functions
            q_mask = query_mask[b, :].float()  # [num_query]
            q_weight_val = query_weight[b, :, weight_idx]  # [num_query]

            sum_val = torch.zeros(num_query, value_dim, device=device, dtype=dtype)
            for h in range(num_hash_f):
                q_hash = query_hash_code[b, :, h].long()  # [num_query]
                gathered = hashtables[h][q_hash]  # [num_query, value_dim]
                sum_val += gathered

            # Apply query weight and divide by num_hash_f
            output[b] += (
                q_mask.unsqueeze(-1) * q_weight_val.unsqueeze(-1) * sum_val / num_hash_f
            )

    return output


class YosoBenchmark(Benchmark):
    seed: int = 42

    def setup(self):
        batch_size = 2
        num_query = 128
        num_key = 128
        dim = 64
        self.num_hash_f = 32
        self.hash_code_len = 9
        self.weight_dim = self.num_hash_f
        self.value_dim = dim
        self.hashtable_capacity = 1 << self.hash_code_len

        self.query_mask = torch.ones(
            batch_size, num_query, device="cuda", dtype=torch.int32
        )
        self.query_vector = torch.randn(
            batch_size, num_query, dim, device="cuda", dtype=torch.float32
        )
        self.key_mask = torch.ones(
            batch_size, num_key, device="cuda", dtype=torch.int32
        )
        self.key_vector = torch.randn(
            batch_size, num_key, dim, device="cuda", dtype=torch.float32
        )
        self.value = torch.randn(
            batch_size, num_key, self.value_dim, device="cuda", dtype=torch.float32
        )
        self.query_weight = torch.randn(
            batch_size, num_query, self.weight_dim, device="cuda", dtype=torch.float32
        )
        self.key_weight = torch.randn(
            batch_size, num_key, self.weight_dim, device="cuda", dtype=torch.float32
        )

        # Pre-compute hash codes for cumulation benchmarks
        hash_result = self.kernel.fast_hash(
            self.query_mask,
            self.query_vector,
            self.key_mask,
            self.key_vector,
            self.num_hash_f,
            self.hash_code_len,
            True,
            1,
        )
        self.query_hash_code = hash_result[0]
        self.key_hash_code = hash_result[1]

        self.out = torch.empty(
            batch_size, num_query, self.value_dim, device="cuda", dtype=torch.float32
        )

    def benchmark_base(self):
        self.out = self.kernel.lsh_weighted_cumulation(
            self.query_mask,
            self.query_hash_code,
            self.query_weight,
            self.key_mask,
            self.key_hash_code,
            self.key_weight,
            self.value,
            self.hashtable_capacity,
            True,
            1,
        )

    def verify_base(self) -> torch.Tensor:
        return lsh_weighted_cumulation_reference(
            self.query_mask,
            self.query_hash_code,
            self.query_weight,
            self.key_mask,
            self.key_hash_code,
            self.key_weight,
            self.value,
            self.hashtable_capacity,
        )

    def setup_large(self):
        batch_size = 4
        num_query = 512
        num_key = 512
        dim = 128
        self.num_hash_f = 32
        self.hash_code_len = 9
        self.weight_dim = self.num_hash_f
        self.value_dim = dim
        self.hashtable_capacity = 1 << self.hash_code_len

        self.query_mask = torch.ones(
            batch_size, num_query, device="cuda", dtype=torch.int32
        )
        self.query_vector = torch.randn(
            batch_size, num_query, dim, device="cuda", dtype=torch.float32
        )
        self.key_mask = torch.ones(
            batch_size, num_key, device="cuda", dtype=torch.int32
        )
        self.key_vector = torch.randn(
            batch_size, num_key, dim, device="cuda", dtype=torch.float32
        )
        self.value = torch.randn(
            batch_size, num_key, self.value_dim, device="cuda", dtype=torch.float32
        )
        self.query_weight = torch.randn(
            batch_size, num_query, self.weight_dim, device="cuda", dtype=torch.float32
        )
        self.key_weight = torch.randn(
            batch_size, num_key, self.weight_dim, device="cuda", dtype=torch.float32
        )

        hash_result = self.kernel.fast_hash(
            self.query_mask,
            self.query_vector,
            self.key_mask,
            self.key_vector,
            self.num_hash_f,
            self.hash_code_len,
            True,
            1,
        )
        self.query_hash_code = hash_result[0]
        self.key_hash_code = hash_result[1]

        self.out = torch.empty(
            batch_size, num_query, self.value_dim, device="cuda", dtype=torch.float32
        )

    def benchmark_large(self):
        self.out = self.kernel.lsh_weighted_cumulation(
            self.query_mask,
            self.query_hash_code,
            self.query_weight,
            self.key_mask,
            self.key_hash_code,
            self.key_weight,
            self.value,
            self.hashtable_capacity,
            True,
            1,
        )

    def verify_large(self) -> torch.Tensor:
        return lsh_weighted_cumulation_reference(
            self.query_mask,
            self.query_hash_code,
            self.query_weight,
            self.key_mask,
            self.key_hash_code,
            self.key_weight,
            self.value,
            self.hashtable_capacity,
        )
