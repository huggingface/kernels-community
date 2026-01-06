# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "kernels"
# ]
# ///
import torch
from kernels import get_kernel

# Setup
torch.manual_seed(42)
yoso = get_kernel("kernels-community/yoso")
device = torch.device("cuda")

# YOSO: You Only Sample Once
# Efficient attention using Locality Sensitive Hashing (LSH)
batch_size = 2
num_query = 128
num_key = 128
vector_dim = 64
value_dim = 64
num_hash_f = 32  # number of hash functions
hash_code_len = 9  # bits per hash code
hashtable_capacity = 512

# Query and key masks (1 = valid, 0 = padding)
query_mask = torch.ones(batch_size, num_query, device=device, dtype=torch.float32)
key_mask = torch.ones(batch_size, num_key, device=device, dtype=torch.float32)

# Query and key vectors for hashing
query_vector = torch.randn(
    batch_size, num_query, vector_dim, device=device, dtype=torch.float32
)
key_vector = torch.randn(
    batch_size, num_key, vector_dim, device=device, dtype=torch.float32
)

# Value tensor
value = torch.randn(
    batch_size, num_key, value_dim, device=device, dtype=torch.float32
)

# Step 1: Fast hash to get hash codes for queries and keys
hash_codes = yoso.fast_hash(
    query_mask, query_vector,
    key_mask, key_vector,
    num_hash_f, hash_code_len,
    True,  # use_cuda
    1      # version
)
query_hash_code, key_hash_code = hash_codes

print(f"Query vector shape: {query_vector.shape}")
print(f"Key vector shape: {key_vector.shape}")
print(f"Query hash code shape: {query_hash_code.shape}")
print(f"Key hash code shape: {key_hash_code.shape}")

# Step 2: LSH cumulation - approximate attention via hashing
output = yoso.lsh_cumulation(
    query_mask, query_hash_code,
    key_mask, key_hash_code,
    value,
    hashtable_capacity,
    True,  # use_cuda
    1      # version
)

print(f"Value shape: {value.shape}")
print(f"Output shape: {output.shape}")
print(f"Output dtype: {output.dtype}")
# Query vector shape: torch.Size([2, 128, 64])
# Key vector shape: torch.Size([2, 128, 64])
# Query hash code shape: torch.Size([2, 128, 32])
# Key hash code shape: torch.Size([2, 128, 32])
# Value shape: torch.Size([2, 128, 64])
# Output shape: torch.Size([2, 128, 64])
# Output dtype: torch.float32
