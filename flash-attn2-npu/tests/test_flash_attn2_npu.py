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


# Reference implementation using PyTorch SDPA
def reference_attention(query, key, value, causal=False):
    query, key, value = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=causal
        )
    return out.transpose(1, 2).contiguous()


def var_reference_attention(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False
):
    batch_size = cu_seqlens_q.shape[0] - 1
    # Return output in packed format (same as flash attention)
    total_tokens_q = q.shape[0]
    out = torch.zeros(
        (total_tokens_q, q.shape[1], q.shape[2]), device=q.device, dtype=q.dtype
    )

    for b in range(batch_size):
        start_q, end_q = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        start_k, end_k = cu_seqlens_k[b], cu_seqlens_k[b + 1]

        # Extract slices for this batch
        q_slice = q[start_q:end_q]  # Shape: (seq_len_q, H, D)
        k_slice = k[start_k:end_k]  # Shape: (seq_len_k, H, D)
        v_slice = v[start_k:end_k]  # Shape: (seq_len_k, H, D)

        # Add batch dimension for reference_attention
        q_slice = q_slice.unsqueeze(0)  # Shape: (1, seq_len_q, H, D)
        k_slice = k_slice.unsqueeze(0)  # Shape: (1, seq_len_k, H, D)
        v_slice = v_slice.unsqueeze(0)  # Shape: (1, seq_len_k, H, D)

        # Compute attention and remove batch dimension
        attn_out = reference_attention(q_slice, k_slice, v_slice, causal=causal)
        attn_out = attn_out.squeeze(0)  # Shape: (seq_len_q, H, D)

        # Place result in output tensor (packed format)
        out[start_q:end_q] = attn_out

    return out


def test_flash_attention():
    device = torch.device("npu")
    dtype = torch.float16

    # batch_size, seq_len, head_nums, head_dim
    B, S, H, D = 2, 5, 4, 8
    query = torch.randn(B, S, H, D, device=device, dtype=dtype)
    key = torch.randn(B, S, H, D, device=device, dtype=dtype)
    value = torch.randn(B, S, H, D, device=device, dtype=dtype)

    out_ref = reference_attention(query, key, value, causal=False)
    output = flash_attn_func(query, key, value, causal=False)

    torch.allclose(output, out_ref, atol=1e-2, rtol=1e-2)


def test_flash_attention_varlen():
    device = torch.device("npu")
    dtype = torch.float16

    # batch_size, seq_len, head_nums, head_dim
    B, S, H, D = 2, 5, 4, 8
    q_var = torch.randn(10, H, D, device=device, dtype=torch.float16)  # total_q=10
    k_var = v_var = torch.randn(
        12, H, D, device=device, dtype=torch.float16
    )  # total_k=12
    cu_q = torch.tensor(
        [0, 3, 7, 10], device=device, dtype=torch.int32
    )  # cumulative sequence lengths
    cu_k = torch.tensor([0, 4, 9, 12], device=device, dtype=torch.int32)

    out_var_ref = var_reference_attention(
        q_var,
        k_var,
        v_var,
        cu_q,
        cu_k,
        max_seqlen_q=4,
        max_seqlen_k=5,
        causal=False,
    )
    output_var = flash_attn_varlen_func(
        q_var, k_var, v_var, cu_q, cu_k, max_seqlen_q=4, max_seqlen_k=5, causal=False
    )

    torch.allclose(output_var, out_var_ref, atol=1e-2, rtol=1e-2)
