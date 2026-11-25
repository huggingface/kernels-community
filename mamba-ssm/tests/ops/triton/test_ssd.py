import math

import kernels

import torch
import torch.nn.functional as F

import pytest

from einops import rearrange, repeat


mamba_ssm = kernels.get_kernel("kernels-community/mamba-ssm", version=5)

chunk_state = mamba_ssm.ops.triton.ssd_chunk_state.chunk_state
_chunk_cumsum_fwd = mamba_ssm.ops.triton.ssd_chunk_state._chunk_cumsum_fwd
_chunk_state_fwd = mamba_ssm.ops.triton.ssd_chunk_state._chunk_state_fwd
chunk_state_varlen = mamba_ssm.ops.triton.ssd_chunk_state.chunk_state_varlen
_state_passing_fwd = mamba_ssm.ops.triton.ssd_state_passing._state_passing_fwd
mamba_chunk_scan_combined = mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined
_chunk_scan_chunk_state_bwd_dx_kernel = mamba_ssm.ops.triton.ssd_combined._chunk_scan_chunk_state_bwd_dx_kernel


def detach_clone(*args):
    return tuple([arg.detach().clone().requires_grad_() if arg is not None else None for arg in args])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize('ngroups', [1, 2, 8, "max"])
# @pytest.mark.parametrize('ngroups', [1])
@pytest.mark.parametrize('chunk_size', [64, 128])
# @pytest.mark.parametrize('chunk_size', [128])
@pytest.mark.kernels_ci
def test_chunk_state_varlen(chunk_size, ngroups, dtype):
    device = 'cuda'
    rtol, atol = (1e-2, 3e-3)
    # set seed
    torch.random.manual_seed(chunk_size + (ngroups if ngroups != "max" else 64))
    batch = 300
    seqlens = torch.randint(1, 200, (batch,), device=device)
    # batch = 3
    # seqlens = torch.tensor([201, 56, 5], device=device)
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0))
    total_seqlen = seqlens.sum().item()
    seq_idx = torch.cat([torch.full((s,), i, dtype=torch.int32, device=device) for i, s in enumerate(seqlens)], dim=0).unsqueeze(0)
    dim = 4096
    # dim = 64
    headdim = 64
    # dim = 32
    dstate = 32
    assert dim % headdim == 0
    nheads = dim // headdim
    if ngroups == "max":
        ngroups = nheads
    assert nheads % ngroups == 0
    B = torch.randn(total_seqlen, ngroups, dstate, dtype=dtype, device=device) / 5
    x = torch.randn(total_seqlen, nheads, headdim, dtype=dtype, device=device)
    A = -0.1 * (torch.rand(nheads, device=device))
    dt = F.softplus(torch.randn(total_seqlen, nheads, device=device, dtype=torch.float32) - 4)
    dA_cumsum, dt_rounded = _chunk_cumsum_fwd(dt.unsqueeze(0), A, chunk_size)
    chunk_states = _chunk_state_fwd(B.unsqueeze(0), x.unsqueeze(0), dt_rounded, dA_cumsum, seq_idx=seq_idx)
    chunk_states, _ = _state_passing_fwd(rearrange(chunk_states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                         seq_idx=seq_idx, chunk_size=chunk_size)
    chunk_states = rearrange(chunk_states, "... (p n) -> ... p n", n=dstate)
    chunk_states = chunk_states.squeeze(0)
    dA_cumsum = dA_cumsum.squeeze(0)
    dt_rounded = dt_rounded.squeeze(0)
    out = chunk_state_varlen(B, x, dt_rounded, dA_cumsum, cu_seqlens, chunk_states)
    out_ref = []
    for b in range(batch):
        x_s = x[cu_seqlens[b]:cu_seqlens[b + 1]].unsqueeze(0)
        B_s = B[cu_seqlens[b]:cu_seqlens[b + 1]].unsqueeze(0)
        dt_s = dt[cu_seqlens[b]:cu_seqlens[b + 1]].unsqueeze(0)
        dA_cumsum_s, dt_rounded_s = _chunk_cumsum_fwd(dt_s, A, chunk_size)
        states = chunk_state(B_s, x_s, dt_rounded_s, dA_cumsum_s)
        _, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum_s[:, :, :, -1],
                                             chunk_size=chunk_size)
        final_states = rearrange(final_states, "... (p n) -> ... p n", n=dstate)
        out_ref.append(final_states)
    out_ref = torch.cat(out_ref, dim=0)
    print(f"Max diff = {(out - out_ref).abs().max().item()}")
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("block_size_n", [32, 64, 128])
def test_dD_head_split_across_blocks(block_size_n, monkeypatch):
    # Without z, out = y + D * x, so the grad of a per-head D is sum(dout * x) over everything but the head.
    # With headdim 128, BLOCK_SIZE_N < 128 splits each head across several blocks.
    configs = [c for c in _chunk_scan_chunk_state_bwd_dx_kernel.configs if c.kwargs["BLOCK_SIZE_N"] == block_size_n]
    monkeypatch.setattr(_chunk_scan_chunk_state_bwd_dx_kernel, "configs", configs[:1])
    monkeypatch.setattr(_chunk_scan_chunk_state_bwd_dx_kernel, "cache", {})
    device = "cuda"
    torch.random.manual_seed(0)
    batch, seqlen, nheads, headdim, dstate, chunk_size = 2, 256, 4, 128, 64, 64
    x = torch.randn(batch, seqlen, nheads, headdim, device=device)
    dt = F.softplus(torch.randn(batch, seqlen, nheads, device=device) - 4)
    A = -torch.rand(nheads, device=device)
    B = torch.randn(batch, seqlen, 1, dstate, device=device)
    C = torch.randn(batch, seqlen, 1, dstate, device=device)
    D = torch.randn(nheads, device=device, requires_grad=True)
    out = mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=D)
    dout = torch.randn_like(out)
    out.backward(dout)
    torch.testing.assert_close(D.grad, (dout * x).sum(dim=(0, 1, 3)), rtol=1e-4, atol=1e-3)
