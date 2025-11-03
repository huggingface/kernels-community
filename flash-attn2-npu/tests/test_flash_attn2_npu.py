# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch_npu
from flash_attn2_npu import (
    flash_attn_func,
    flash_attn_varlen_func,
)


def test_flash_attention():
    device = torch.device("npu")
    dtype = torch.bfloat16
    batch_size = 2
    seq_length = 128
    num_heads = 8
    head_dim = 64

    query = torch.randn(
        batch_size, seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn(
        batch_size, seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    value = torch.randn(
        batch_size, seq_length, num_heads, head_dim, device=device, dtype=dtype
    )

    output = flash_attn_func(query, key, value)

    scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    ref_out = torch.matmul(attn_weights, value)

    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=1e-2)


def test_flash_attention_varlen():
    device = torch.device("npu")
    dtype = torch.bfloat16
    batch_size = 2
    max_seq_length = 128
    num_heads = 8
    head_dim = 64

    query = torch.randn(
        batch_size, max_seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    key = torch.randn(
        batch_size, max_seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    value = torch.randn(
        batch_size, max_seq_length, num_heads, head_dim, device=device, dtype=dtype
    )
    cu_seqlens_q = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)
    total_q_len = cu_seqlens_q[-1].item()
    total_k_len = cu_seqlens_k[-1].item()

    output = flash_attn_varlen_func(
        query[:total_q_len],
        key[:total_k_len],
        value[:total_k_len],
        cu_seqlens_q,
        cu_seqlens_k,
    )

    ref_out = torch.zeros_like(output)
    for i in range(batch_size):
        q_start = cu_seqlens_q[i].item()
        q_end = cu_seqlens_q[i + 1].item()
        k_start = cu_seqlens_k[i].item()
        k_end = cu_seqlens_k[i + 1].item()

        q_slice = query[q_start:q_end]
        k_slice = key[k_start:k_end]
        v_slice = value[k_start:k_end]

        scores = torch.matmul(q_slice, k_slice.transpose(-2, -1)) / (head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        ref_out[q_start:q_end] = torch.matmul(attn_weights, v_slice)

    torch.testing.assert_close(output, ref_out, atol=1e-2, rtol=1e-2)
