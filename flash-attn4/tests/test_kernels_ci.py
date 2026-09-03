import itertools
import math

import kernels
import pytest
import torch

# `kernels-community/flash-attn4` is `[general.hub] repo-id` and 0 is
# `[general] version` in `build.toml`.
flash_attn4 = kernels.get_kernel("kernels-community/flash-attn4", version=0)

pytestmark = [
    pytest.mark.kernels_ci,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
]

DEVICE = "cuda"
BATCH, HEADS = 2, 4


def reference(q, k, v, causal=False, upcast=True):
    """Attention over `(batch, seqlen, heads, head_dim)` tensors, in plain PyTorch.

    With `upcast` the softmax runs in float32, which is what the kernel is
    expected to approximate; without it the whole thing runs in the input dtype,
    which gives a same-precision baseline to measure the kernel's error against.
    """
    dtype_out = q.dtype
    dtype = torch.float32 if upcast else dtype_out
    q, k, v = (x.to(dtype).transpose(1, 2) for x in (q, k, v))  # to (b, h, s, d)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        # Flash Attention aligns the causal mask to the bottom right, so a query
        # attends to every key when seqlen_q == 1 regardless of seqlen_k.
        row = torch.arange(q.shape[-2], device=q.device).unsqueeze(-1)
        col = torch.arange(k.shape[-2], device=q.device)
        masked = col > row + k.shape[-2] - q.shape[-2]
        scores = scores.masked_fill(masked, float("-inf"))
    out = torch.softmax(scores, dim=-1).to(v.dtype) @ v
    return out.transpose(1, 2).to(dtype_out)


def assert_no_worse_than_pytorch(out, out_ref, out_pt):
    """The kernel's error must stay within twice PyTorch's own at the same dtype.

    Upstream's criterion. `out_ref` is the float32 answer rounded to the input
    dtype, so `out_ref + 0.3 - 0.3 - out_ref` measures one rounding step of that
    dtype and gives a floor for shapes where PyTorch happens to be exact.
    """
    atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    kernel_error = (out - out_ref).abs().max().item()
    pytorch_error = (out_pt - out_ref).abs().max().item()
    assert kernel_error <= 2 * pytorch_error + atol, (
        f"kernel error {kernel_error:.6f} exceeds "
        f"2 * {pytorch_error:.6f} (pytorch) + {atol:.6f}"
    )


def randn(*shape, dtype):
    return torch.randn(shape, device=DEVICE, dtype=dtype)


# ── forward pass ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "dtype,head_dim,seqlen_q,seqlen_k,causal",
    [
        (torch.bfloat16, 64, 256, 256, False),
        (torch.bfloat16, 64, 128, 256, True),
        (torch.bfloat16, 128, 192, 192, False),
        (torch.float16, 64, 256, 256, False),
        (torch.float16, 128, 128, 256, True),
    ],
)
def test_forward_matches_reference(dtype, head_dim, seqlen_q, seqlen_k, causal):
    torch.manual_seed(0)
    q = randn(BATCH, seqlen_q, HEADS, head_dim, dtype=dtype)
    k = randn(BATCH, seqlen_k, HEADS, head_dim, dtype=dtype)
    v = randn(BATCH, seqlen_k, HEADS, head_dim, dtype=dtype)

    out, _ = flash_attn4.flash_attn_func(q, k, v, causal=causal)

    assert out.shape == q.shape
    assert out.dtype == dtype
    assert_no_worse_than_pytorch(
        out,
        reference(q, k, v, causal=causal),
        reference(q, k, v, causal=causal, upcast=False),
    )


# ── varlen forward pass ────────────────────────────────────────────────────
# Sequences are packed back to back and delimited by `cu_seqlens`, so the
# reference runs one sequence at a time. seqlen_q <= seqlen_k throughout: with a
# bottom-right causal mask a longer query than key would leave the first rows
# with nothing to attend to, and their softmax undefined.

@pytest.mark.parametrize("causal", [False, True])
def test_varlen_forward_matches_reference(causal):
    torch.manual_seed(0)
    dtype, head_dim = torch.bfloat16, 64
    seqlens_q, seqlens_k = [96, 1, 160], [128, 64, 160]

    def cu_seqlens(seqlens):
        return torch.tensor(
            [0, *itertools.accumulate(seqlens)], dtype=torch.int32, device=DEVICE
        )

    cu_q, cu_k = cu_seqlens(seqlens_q), cu_seqlens(seqlens_k)
    q = randn(sum(seqlens_q), HEADS, head_dim, dtype=dtype)
    k = randn(sum(seqlens_k), HEADS, head_dim, dtype=dtype)
    v = randn(sum(seqlens_k), HEADS, head_dim, dtype=dtype)

    out, _ = flash_attn4.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max(seqlens_q),
        max_seqlen_k=max(seqlens_k),
        causal=causal,
    )

    assert out.shape == q.shape
    for i in range(len(seqlens_q)):
        q_i = q[cu_q[i] : cu_q[i + 1]].unsqueeze(0)
        k_i = k[cu_k[i] : cu_k[i + 1]].unsqueeze(0)
        v_i = v[cu_k[i] : cu_k[i + 1]].unsqueeze(0)
        assert_no_worse_than_pytorch(
            out[cu_q[i] : cu_q[i + 1]].unsqueeze(0),
            reference(q_i, k_i, v_i, causal=causal),
            reference(q_i, k_i, v_i, causal=causal, upcast=False),
        )


# ── log-sum-exp output ─────────────────────────────────────────────────────
# `return_lse` is part of the compile key, so this is its own kernel rather than
# a second look at the one `test_forward_matches_reference` builds. The LSE is
# accumulated in float32 and compared against a float32 reference directly.

def test_return_lse_matches_reference():
    torch.manual_seed(0)
    dtype, head_dim, seqlen = torch.bfloat16, 64, 256
    q = randn(BATCH, seqlen, HEADS, head_dim, dtype=dtype)
    k = randn(BATCH, seqlen, HEADS, head_dim, dtype=dtype)
    v = randn(BATCH, seqlen, HEADS, head_dim, dtype=dtype)

    out, lse = flash_attn4.flash_attn_func(q, k, v, return_lse=True)

    scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) / math.sqrt(head_dim)
    lse_ref = torch.logsumexp(scores, dim=-1)

    assert lse.shape == (BATCH, HEADS, seqlen)
    assert lse.dtype == torch.float32
    torch.testing.assert_close(lse, lse_ref, atol=2e-2, rtol=1e-3)
    assert_no_worse_than_pytorch(
        out, reference(q, k, v), reference(q, k, v, upcast=False)
    )


# ── package surface ────────────────────────────────────────────────────────
# No kernel launch, so this is the one check that survives a GPU whose
# architecture the forward pass does not support.

def test_public_api():
    assert flash_attn4.__version__
    assert callable(flash_attn4.flash_attn_func)
    assert callable(flash_attn4.flash_attn_varlen_func)
