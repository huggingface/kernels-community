# Smoke test for gdn and kda -> pass fwd and bwd with similar numbers as in fla original tests

import kernels
import pytest
import torch
import torch.nn.functional as F

from einops import rearrange


fla = kernels.get_kernel("kernels-community/fla", version=1)


def naive_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    """
    Reference PyTorch implementation of chunk gated delta rule.

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        v: [B, T, H, V]
        g: [B, T, H]
        beta: [B, T, H]
        chunk_size: int
        scale: float, optional
        initial_state: [B, H, K, V], optional
        output_final_state: bool

    Returns:
        o: [B, T, H, V]
        final_state: [B, H, K, V] if output_final_state else None
    """
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    q, k, v, beta, g = map(lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g])

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))

    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % chunk_size == 0

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size),
        [q, k, v, k_beta, decay.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)
    decay_exp = decay.exp()[..., None]
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumsum = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)
    v = k_cumsum

    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state.to(torch.float32)

    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = S * decay[:, :, i, -1, None, None].exp() + (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()
                                                        [..., None]).transpose(-1, -2) @ v_new
    if not output_final_state:
        S = None

    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


def naive_chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, HV, V]``. ``HV`` must be divisible by ``H``.
        g (torch.Tensor):
            Per-dimension decay gates (log-space) of shape ``[B, T, HV, K]``.
        beta (torch.Tensor):
            Beta scalars of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scale factor. Defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[B, HV, K, V]``.
        output_final_state (bool):
            Whether to return the final state.
        chunk_size (int):
            Chunk size for the chunked computation. Default: 64.

    Returns:
        A tuple ``(o, S)`` where ``o`` has shape ``[B, T, HV, V]`` and
        ``S`` has shape ``[B, HV, K, V]`` if ``output_final_state`` else ``None``.
    """
    dtype = v.dtype
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    G = HV // H
    BT = chunk_size
    NT = T // BT
    if scale is None:
        scale = K ** -0.5
    assert T % BT == 0

    # Rearrange into chunks: [B, head, NT, BT, ...]
    q, k = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [q, k]]
    v, g, beta = [rearrange(x, 'b (n c) h ... -> b h n c ...', c=BT).to(torch.float) for x in [v, g, beta]]
    # Expand q/k to value head dim for GVA: [B, H, ...] -> [B, HV, ...]
    q = q.repeat_interleave(G, dim=1) * scale  # [B, HV, NT, BT, K]
    k = k.repeat_interleave(G, dim=1)          # [B, HV, NT, BT, K]
    g = g.cumsum(-2)

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    # Akk uses k (expanded to HV) and g (per value head)
    A = torch.zeros(*g.shape[:-1], BT, dtype=torch.float, device=q.device)
    for i in range(BT):
        k_i = k[..., i, :]
        g_i = g[..., i:i+1, :]
        A[..., i] = torch.einsum('... c d, ... d -> ... c', k * (g - g_i).exp(), k_i)
    A = A * beta[..., None]

    A = -A.masked_fill(mask, 0)
    for i in range(1, BT):
        A[..., i, :i] = A[..., i, :i].clone() + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2)
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta[..., None, :]

    w = A @ (g.exp() * k)
    u = A @ v

    S = k.new_zeros(B, HV, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, NT):
        # [B, HV, BT, ...]
        q_i = q[:, :, i]      # [B, HV, BT, K]
        k_i = k[:, :, i]      # [B, HV, BT, K]
        u_i = u[:, :, i]        # [B, HV, BT, V]
        g_i = g[:, :, i]        # [B, HV, BT, K]
        w_i = w[:, :, i]        # [B, HV, BT, K]
        # Aqk: per value head (q from qk head, g from value head, k from qk head)
        Aqk = torch.zeros(B, HV, BT, BT, dtype=torch.float, device=q.device)
        for j in range(BT):
            k_j = k[:, :, i, j]
            g_j = g[:, :, i, j:j+1, :]
            Aqk[..., j] = torch.einsum('... c d, ... d -> ... c', q_i * (g_i - g_j).exp(), k_j)
        Aqk = Aqk.masked_fill(mask, 0)
        v_i = u_i - w_i @ S
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * rearrange(g_i[:, :, -1].exp(), 'b h k -> b h k 1')
        S += rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i
    if not output_final_state:
        S = None
    return rearrange(o, 'b h n c d -> b (n c) h d').to(dtype), S


@pytest.mark.kernels_ci
def test_gated_delta_rule():
    if not torch.cuda.is_available():
         pytest.skip("only sanity checking cuda for now")

    torch.manual_seed(42)

    B, T, H, D = 1, 64, 1, 64
    dtype = torch.float32
    device = "cuda"

    q = torch.rand(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    g = (
        F.logsigmoid(torch.randn(B, T, H, dtype=dtype, device=device))
        .requires_grad_()
    )
    beta = (
        torch.randn(B, T, H, dtype=dtype, device=device)
        .sigmoid()
        .requires_grad_()
    )

    ref, _ = naive_chunk_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        beta=beta,
        g=g,
    )

    do = torch.randn_like(ref)
    (ref * do).sum().backward()

    ref_grads = [
        x.grad.clone()
        for x in (q, k, v, g, beta)
    ]

    q.grad = k.grad = v.grad = g.grad = beta.grad = None

    out, _ = fla.chunk_gated_delta_rule(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        beta=beta,
        g=g,
    )

    (out * do).sum().backward()

    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    for grad, ref_grad in zip(
        (q.grad, k.grad, v.grad, g.grad, beta.grad),
        ref_grads,
    ):
        torch.testing.assert_close(
            grad,
            ref_grad,
            rtol=2e-2,
            atol=2e-2,
        )


@pytest.mark.kernels_ci
def test_kimi_delta_attention():
    if not torch.cuda.is_available():
         pytest.skip("only sanity checking cuda for now")

    torch.manual_seed(42)

    B, T, H, D = 1, 64, 1, 64
    dtype = torch.float32
    device = "cuda"

    q = torch.rand(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.rand(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.rand(B, T, H, D, dtype=dtype, device=device, requires_grad=True)
    g = (
        F.logsigmoid(
            torch.randn(B, T, H, D, dtype=dtype, device=device)
        )
        .requires_grad_()
    )
    beta = (
        torch.randn(B, T, H, dtype=dtype, device=device)
        .sigmoid()
        .requires_grad_()
    )

    ref, _ = naive_chunk_kda(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        g=g,
        beta=beta,
    )

    do = torch.randn_like(ref)
    (ref * do).sum().backward()

    ref_grads = [
        x.grad.clone()
        for x in (q, k, v, g, beta)
    ]

    q.grad = k.grad = v.grad = g.grad = beta.grad = None

    out, _ = fla.chunk_kda(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        g=g,
        beta=beta,
    )

    (out * do).sum().backward()

    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)

    for grad, ref_grad in zip(
        (q.grad, k.grad, v.grad, g.grad, beta.grad),
        ref_grads,
    ):
        torch.testing.assert_close(
            grad,
            ref_grad,
            rtol=2e-2,
            atol=2e-2,
        )
