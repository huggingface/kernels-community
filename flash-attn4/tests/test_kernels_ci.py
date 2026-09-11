"""Fast smoke tests for the kernels-community CI runner.

The vendored upstream suite under ``tests/cute`` is intentionally exhaustive and
compiles thousands of CuTe DSL kernel variants. These tests cover the two
functions flash-attn4 exports with small inputs and a handful of JIT
compilations, so ``pytest -m kernels_ci`` stays inexpensive.

Coverage is bounded by the CI runner, which is an sm89 (L4) device:

- The forward kernels have an sm80 fallback path, so fixed-length and
  variable-length forwards run here.
- The backward kernels assert ``9.x <= compute capability <= 12.x``, and the
  split-KV combine kernel emits ``griddepcontrol.wait`` (Programmatic Dependent
  Launch) unconditionally, which NVVM cannot lower below sm90. Neither can be
  smoke-tested until CI gains a Hopper or Blackwell runner.
"""

import kernels
import pytest
import torch
import torch.nn.functional as F


flash_attn4 = kernels.get_kernel("kernels-community/flash-attn4", version=0)

cuda_major = (
    torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else None
)
forward_supported = cuda_major in (8, 9, 10, 11, 12)

pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(
        not forward_supported,
        reason="flash-attn4 requires an sm80 or later CUDA device",
    ),
]


def reference_attention(q, k, v, causal=False):
    """Compute attention in float32 for an independent numerical reference."""
    q_ref = q.transpose(1, 2).float()
    k_ref = k.transpose(1, 2).float()
    v_ref = v.transpose(1, 2).float()
    if q_ref.shape[1] != k_ref.shape[1]:
        groups = q_ref.shape[1] // k_ref.shape[1]
        k_ref = k_ref.repeat_interleave(groups, dim=1)
        v_ref = v_ref.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=causal
    ).transpose(1, 2)


def assert_close(actual, expected):
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_kv_heads", [4, 2], ids=["mha", "gqa"])
def test_flash_attn_func(causal, num_kv_heads):
    """Exercise fixed-length MHA/GQA forward dispatch."""
    torch.manual_seed(0)
    batch, seqlen, num_heads, head_dim = 2, 128, 4, 64

    q = torch.randn(
        batch, seqlen, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        batch, seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)

    out, lse = flash_attn4.flash_attn_func(q, k, v, causal=causal, return_lse=True)
    out_ref = reference_attention(q, k, v, causal=causal)

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert lse.shape == (batch, num_heads, seqlen)
    assert_close(out, out_ref)


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_varlen_func(causal):
    """Exercise the packed variable-length path with unequal sequences."""
    torch.manual_seed(1)
    lengths = (128, 64)
    num_heads, head_dim = 4, 64
    total = sum(lengths)

    q, k, v = [
        torch.randn(total, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    cu_seqlens = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int32)

    out, lse = flash_attn4.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        causal=causal,
        return_lse=True,
    )
    out_ref = torch.cat(
        [
            reference_attention(
                q[start:end].unsqueeze(0),
                k[start:end].unsqueeze(0),
                v[start:end].unsqueeze(0),
                causal=causal,
            ).squeeze(0)
            for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
        ]
    )

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert lse.shape == (num_heads, total)
    assert_close(out, out_ref)


def test_lse_omitted_by_default():
    """``return_lse=False`` (the default) leaves the second result unset."""
    torch.manual_seed(2)
    q, k, v = [
        torch.randn(2, 128, 4, 64, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]

    out, lse = flash_attn4.flash_attn_func(q, k, v, causal=True)

    assert lse is None
    assert_close(out, reference_attention(q, k, v, causal=True))
