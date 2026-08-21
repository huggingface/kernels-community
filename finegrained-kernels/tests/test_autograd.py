"""Backward for the quantized ops — dgrad only, kernels and end-to-end.

Two levels in one file, because they fail differently. The KERNEL tests hand-build operands, never
call ``.backward()``, and measure against the QUANTIZATION FLOOR (dequantize the exact weight the
kernel read, contract in fp32): that isolates "is this contracting the right axis" from "the weight
is lossy", which for a 4-bit weight is ~100x the thing under test. The END-TO-END tests call the
public op and ``.backward()``.

Registration is never asserted directly — it is implied: a recipe whose op lost its formula fails
``out.requires_grad`` in the parametrized e2e test, across every recipe in ``WEIGHTS`` rather than
against a hand-kept list of op names that goes stale.

``dX = dY @ W`` contracting N. No recipe needs its own arm: a weight reaches the kernel as three
numbers — ``(scale_group_k, scale_row_div, values_per_byte)`` — so MX (per-row K-groups) and
block-FP8 (128x128 blocks) are one code path at different divisors. Only dgrad exists; the
quantized weight takes no gradient, because an fp8/fp4 tensor cannot be an autograd leaf.
"""

import pytest
import torch
from utils import (
    SUPPORTS_FP8,
    TEST_DEVICE,
    WEIGHTS,
    dequantize_weight,
    make_weights,
    quant_dequant_a,
)

import finegrained_kernels as fg
from finegrained_kernels.autograd import (
    dgrad_matmul_2d,
    dgrad_matmul_batched,
    dgrad_matmul_grouped,
    glu_backward,
)


pytestmark = pytest.mark.skipif(not SUPPORTS_FP8, reason="FP8 kernels require SM90+")

M, N, K = 512, 1024, 768  # 128-aligned on every axis so every recipe's blocks divide evenly


def _rel(x: torch.Tensor, ref: torch.Tensor) -> float:
    return ((x.float() - ref.float()).norm() / ref.float().norm().clamp(min=1e-9)).item()


def _floor_mx(Wq, Ws, group):
    """Dequantize exactly what the kernel read: UE8M0 is a biased exponent, 2^(e-127)."""
    s = torch.pow(2.0, Ws.view(torch.uint8).float() - 127)
    return Wq.float() * s.repeat_interleave(group, dim=-1)[:, : Wq.shape[-1]]


def test_dgrad_mxfp8_contracts_n():
    """MXFP8 weight (group-32 along K, one scale row per output row): scale_group_k=32,
    scale_row_div=1. The weight tile is read in its natural (N, K) layout."""
    torch.manual_seed(0)
    dY = torch.randn(M, N, device=TEST_DEVICE, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    Wq, Ws = fg.mxfp8_act_quant(W)

    dX = dgrad_matmul_2d(dY, Wq, Ws, 32, 1, output_dtype=torch.float32)
    assert dX.shape == (M, K), f"expected {(M, K)}, got {tuple(dX.shape)}"

    floor = dY.float() @ _floor_mx(Wq, Ws, 32)
    assert _rel(dX, floor) < 5e-3, f"dgrad diverges from its own floor: {_rel(dX, floor):.2e}"


def test_dgrad_mxfp4_packed_weight():
    """MXFP4: the same path with values_per_byte=2 — the tile spans BK//2 bytes and the
    column-unpack happens in-register, so a 4-bit frozen weight differentiates with no
    transposed copy. This is the QLoRA case."""
    torch.manual_seed(0)
    dY = torch.randn(M, N, device=TEST_DEVICE, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    Wq, Ws = fg.mxfp4_act_quant(W)

    dX = dgrad_matmul_2d(dY, Wq.view(torch.uint8), Ws, 32, 1, output_dtype=torch.float32)
    assert dX.shape == (M, K), f"expected {(M, K)}, got {tuple(dX.shape)}"

    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=TEST_DEVICE,
    )
    b = Wq.view(torch.uint8)
    vals = torch.stack([lut[(b & 0xF).long()], lut[(b >> 4).long()]], -1).reshape(N, K)
    s = torch.pow(2.0, Ws.view(torch.uint8).float() - 127).repeat_interleave(32, -1)[:, :K]
    floor = dY.float() @ (vals * s)
    assert _rel(dX, floor) < 5e-3, f"dgrad diverges from its own floor: {_rel(dX, floor):.2e}"


def test_dgrad_recipe_is_only_divisors():
    """block-FP8 reaches the same kernel at scale_group_k=128, scale_row_div=128 — a 128x128
    block grid is the per-row K-group layout with both divisors widened, so adding a recipe whose
    scales tile this way needs no new arm."""
    torch.manual_seed(0)
    dY = torch.randn(M, N, device=TEST_DEVICE, dtype=torch.bfloat16)
    W = torch.randn(N, K, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    blk = torch.zeros(N // 128, K // 128, device=TEST_DEVICE, dtype=torch.float32)
    Wq = torch.empty_like(W, dtype=torch.float8_e4m3fn)
    for i in range(N // 128):
        for j in range(K // 128):
            tile = W[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128].float()
            s = tile.abs().max().clamp(min=1e-12) / 448.0
            blk[i, j] = s
            Wq[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = (tile / s).to(
                torch.float8_e4m3fn
            )

    dX = dgrad_matmul_2d(dY, Wq, blk, 128, 128, output_dtype=torch.float32)
    floor = dY.float() @ (
        Wq.float() * blk.repeat_interleave(128, 0).repeat_interleave(128, 1)[:N, :K]
    )
    assert _rel(dX, floor) < 5e-3, f"dgrad diverges from its own floor: {_rel(dX, floor):.2e}"


def test_glu_backward_matches_autograd():
    """``glu_backward`` against torch autograd on the interleaved GLU. The reference builds H the
    way the kernels do — gate at even columns, up at odd — so this also pins the interleaved
    convention, not just the derivative."""
    torch.manual_seed(0)
    I = 128
    Z = torch.randn(64, 2 * I, device=TEST_DEVICE, dtype=torch.float32, requires_grad=True)
    dH = torch.randn(64, I, device=TEST_DEVICE, dtype=torch.float32)

    H = torch.nn.functional.silu(Z[..., 0::2]) * Z[..., 1::2]
    H.backward(dH)

    dZ = glu_backward(dH, Z.detach(), act_fn="silu")
    assert dZ.shape == Z.shape, f"expected {tuple(Z.shape)}, got {tuple(dZ.shape)}"
    assert _rel(dZ, Z.grad) < 1e-5, f"GLU backward diverges from autograd: {_rel(dZ, Z.grad):.2e}"


def test_dgrad_grouped_routed():
    """Routed MoE dgrad: the kernel is handed the FORWARD's gather/scatter maps unchanged and
    swaps their roles itself. Reference contracts each token against its own expert's weight."""
    torch.manual_seed(0)
    E, S, Ne, Ke = 4, 128, 256, 256
    dY = torch.randn(S, Ne, device=TEST_DEVICE, dtype=torch.bfloat16)
    W = torch.randn(E, Ne, Ke, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    Wq, Ws = fg.mxfp8_act_quant(W.reshape(E * Ne, Ke))
    Wq, Ws = Wq.reshape(E, Ne, Ke), Ws.reshape(E, Ne, -1)

    # expert-sorted positions, evenly split; identity routing maps (the sort is virtual)
    expert_start = torch.arange(0, E + 1, device=TEST_DEVICE, dtype=torch.int32) * (S // E)
    gather_idx = torch.arange(S, device=TEST_DEVICE, dtype=torch.int32)
    scatter_idx = torch.arange(S, device=TEST_DEVICE, dtype=torch.int32)

    dX = dgrad_matmul_grouped(
        dY, Wq, Ws, expert_start, 32, 1,
        gather_idx=gather_idx, scatter_idx=scatter_idx, output_dtype=torch.float32,
    )
    assert dX.shape == (S, Ke), f"expected {(S, Ke)}, got {tuple(dX.shape)}"

    Wdeq = (
        Wq.float()
        * torch.pow(2.0, Ws.view(torch.uint8).float() - 127).repeat_interleave(32, -1)[..., :Ke]
    )
    rows = torch.arange(S, device=TEST_DEVICE)
    expert_of_row = torch.bucketize(rows, expert_start[1:].long(), right=True).clamp(max=E - 1)
    floor = torch.einsum("sn,snk->sk", dY.float(), Wdeq[expert_of_row])
    assert _rel(dX, floor) < 5e-3, f"grouped dgrad diverges from its floor: {_rel(dX, floor):.2e}"


@pytest.mark.parametrize(
    "act_fn,alpha,limit",
    [("silu", None, None), ("gelu", None, None), ("relu", None, None),
     ("silu", 1.702, None), ("silu", 1.702, 7.0), ("silu", None, 7.0)],
)
def test_glu_backward_arms_match_autograd(act_fn, alpha, limit):
    """Every GLU arm against torch autograd on the interleaved forward.

    The alpha arm is ``(up + 1) * gate * sigmoid(ALPHA * gate)`` — a torch reference that drops the
    ``+1`` still matches on every OTHER arm, so this parametrization is what catches it.
    ``limit`` saturates the forward, which must pass zero gradient."""
    torch.manual_seed(0)
    I = 128
    Z = torch.randn(64, 2 * I, device=TEST_DEVICE, dtype=torch.float32) * 3.0
    Z.requires_grad_(True)
    dH = torch.randn(64, I, device=TEST_DEVICE, dtype=torch.float32)

    g, u = Z[..., 0::2], Z[..., 1::2]
    if limit is not None:
        g = g.clamp(max=limit)
        u = u.clamp(min=-limit, max=limit)
    if alpha is not None:
        H = (u + 1.0) * g * torch.sigmoid(alpha * g)
    elif act_fn == "silu":
        H = torch.nn.functional.silu(g) * u
    elif act_fn == "relu":
        H = torch.relu(g) * u
    else:
        H = torch.nn.functional.gelu(g, approximate="none") * u
    H.backward(dH)

    dZ = glu_backward(dH, Z.detach(), act_fn=act_fn, swiglu_alpha=alpha, swiglu_limit=limit)
    assert _rel(dZ, Z.grad) < 1e-5, f"{act_fn} a={alpha} lim={limit}: {_rel(dZ, Z.grad):.2e}"


def test_dgrad_block_fp8_transposed_views():
    """dA = dY @ B on the forward kernel: the weight and its 128x128 scale grid both arrive as
    transposed VIEWS — no copy, no re-quantization, because a block scale spans both axes."""
    torch.manual_seed(0)
    B, Bs = make_weights(N, K, TEST_DEVICE, [128, 128])
    dY = torch.randn(M, N, device=TEST_DEVICE, dtype=torch.bfloat16)

    dA = fg.matmul_2d(dY, B.t(), None, Bs.t(), output_dtype=torch.bfloat16)
    assert dA.shape == (M, K), f"expected {(M, K)}, got {tuple(dA.shape)}"

    # Floor = what the kernel itself had to work with: the dequantized weight AND dY put
    # through the kernel's own inline activation quant. Comparing against raw bf16 dY instead
    # folds in the activation's rounding and inflates the error by ~5x, which is the trap here.
    B_deq = WEIGHTS["fp8_128x128"]["dequant"](B, Bs).reshape(N, K)
    floor = quant_dequant_a(dY, 128) @ B_deq
    assert _rel(dA, floor) < 5e-3, f"dgrad diverges from its own quantization floor: {_rel(dA, floor):.2e}"


# ── end to end: the public op, .backward(), and the gradient that comes out ──────

def test_quantized_linear_autograd():
    """Gradients arrive through `.backward()` and match a bf16 reference to fp8 tolerance.
    The weight stays a constant: an E4M3 tensor cannot be an autograd leaf (no fp8 `add`)."""
    torch.manual_seed(0)
    B, Bs = make_weights(N, K, TEST_DEVICE, [128, 128])
    A = torch.randn(M, K, device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(M, N, device=TEST_DEVICE, dtype=torch.bfloat16)

    out = fg.matmul_2d(A, B, None, Bs, output_dtype=torch.bfloat16)
    assert out.requires_grad, "matmul_2d output is not attached to the graph"
    out.backward(g)
    assert A.grad is not None and A.grad.shape == (M, K) and A.grad.dtype == A.dtype
    assert torch.isfinite(A.grad.float()).all()

    A_ref = A.detach().clone().requires_grad_(True)
    B_deq = WEIGHTS["fp8_128x128"]["dequant"](B, Bs).reshape(N, K).to(torch.bfloat16)
    (A_ref @ B_deq.t()).backward(g)
    assert _rel(A.grad, A_ref.grad) < 1e-1, f"dA vs bf16 autograd: {_rel(A.grad, A_ref.grad):.2e}"


def test_inference_path_is_untouched():
    """`matmul_2d` itself stays a plain forward call — the differentiable entry point is
    separate, so inference keeps the eager fast path with no autograd machinery."""
    B, Bs = make_weights(N, K, TEST_DEVICE, [128, 128])
    A = torch.randn(M, K, device=TEST_DEVICE, dtype=torch.bfloat16)
    out = fg.matmul_2d(A, B, None, Bs, output_dtype=torch.bfloat16)
    assert not out.requires_grad and torch.isfinite(out.float()).all()


QUANT_RECIPES = [r for r in WEIGHTS if r not in ("bf16", "fp16")]


@pytest.mark.parametrize("recipe", QUANT_RECIPES)
def test_dgrad_every_quant_recipe(recipe):
    """Every quantized recipe must produce a usable activation gradient. Recipes whose scale grid
    can be reoriented (per-tensor, block) take the quantized transposed-view route on the forward
    kernel; the per-row-group and packed-E2M1 ones contract N against the weight's natural tile
    (``backward.py``). Both must land near the bf16 reference — this pins COVERAGE, so a recipe
    can never silently lose its backward."""
    torch.manual_seed(0)
    try:
        B, Bs, g = WEIGHTS[recipe]["make"](N, K, None)
    except TypeError:  # a few makes only build the expert-stacked form; take one expert
        B, Bs, g = WEIGHTS[recipe]["make"](N, K, 1)
        B, Bs = B[0], Bs[0]
    A = torch.randn(M, K, device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(M, N, device=TEST_DEVICE, dtype=torch.bfloat16)

    out = fg.matmul_2d(A, B, Bs=Bs, output_dtype=torch.bfloat16, b_global_scale=g)
    assert out.requires_grad, f"{recipe}: output not attached to the graph"
    out.backward(grad_out)
    assert A.grad is not None and A.grad.shape == (M, K)
    assert torch.isfinite(A.grad.float()).all(), f"{recipe}: non-finite dA"

    # reference: the same product against the dequantized weight
    W = dequantize_weight(B, Bs, global_scale=g).reshape(N, K)
    ref = grad_out.float() @ W
    assert _rel(A.grad, ref) < 2e-1, f"{recipe}: dA vs dequantized reference {_rel(A.grad, ref):.2e}"


def test_dgrad_batched_per_row_expert():
    """Batched dgrad: ``dX[s] = dY[s] @ W[expert_ids[s]]``. Rows are NOT expert-sorted here — each
    names its own expert — so this cannot reuse the grouped kernel. Registered because the batched
    experts forward is a selectable dispatch, not a decode-only path."""
    torch.manual_seed(0)
    E, S, Ne, Ke = 4, 32, 256, 256
    dY = torch.randn(S, Ne, device=TEST_DEVICE, dtype=torch.bfloat16)
    W = torch.randn(E, Ne, Ke, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    Wq, Ws = fg.mxfp8_act_quant(W.reshape(E * Ne, Ke))
    Wq, Ws = Wq.reshape(E, Ne, Ke), Ws.reshape(E, Ne, -1)
    expert_ids = torch.randint(0, E, (S,), device=TEST_DEVICE, dtype=torch.int32)

    dX = dgrad_matmul_batched(dY, Wq, Ws, expert_ids, 32, 1, output_dtype=torch.float32)
    assert dX.shape == (S, Ke), f"expected {(S, Ke)}, got {tuple(dX.shape)}"

    Wdeq = (
        Wq.float()
        * torch.pow(2.0, Ws.view(torch.uint8).float() - 127).repeat_interleave(32, -1)[..., :Ke]
    )
    floor = torch.einsum("sn,snk->sk", dY.float(), Wdeq[expert_ids.long()])
    assert _rel(dX, floor) < 5e-3, f"batched dgrad diverges from its floor: {_rel(dX, floor):.2e}"


def test_dgrad_grouped_accumulates_over_top_k():
    """Routed dgrad with a NON-IDENTITY gather and top_k > 1: each token is routed to several
    experts, so its gradient is a SUM over them.

    This is the case an identity gather cannot express. With `gather_idx = arange(S)` every source
    row is written exactly once, so a kernel that STORES and one that ACCUMULATES are
    indistinguishable — which is how a store survived the suite while silently dropping all but
    one expert's contribution per token on any real MoE."""
    from finegrained_kernels.autograd import dgrad_matmul_grouped

    torch.manual_seed(0)
    T, tk, E, Ne, Ke = 8, 2, 4, 256, 256
    S = T * tk

    # each token -> tk DISTINCT experts, then sort the (token, expert) pairs by expert so the
    # rows are expert-contiguous, which is the layout the grouped schedule guarantees
    experts = torch.stack([torch.randperm(E)[:tk] for _ in range(T)])       # (T, tk)
    pairs = [(int(e), t) for t in range(T) for e in experts[t]]
    pairs.sort()
    gather_idx = torch.tensor([t for _, t in pairs], device=TEST_DEVICE, dtype=torch.int32)
    counts = torch.bincount(torch.tensor([e for e, _ in pairs]), minlength=E)
    expert_start = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).to(
        TEST_DEVICE, torch.int32)
    assert int(expert_start[-1]) == S

    dY = torch.randn(S, Ne, device=TEST_DEVICE, dtype=torch.bfloat16)
    W = torch.randn(E, Ne, Ke, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    Wq, Ws = fg.mxfp8_act_quant(W.reshape(E * Ne, Ke))
    Wq, Ws = Wq.reshape(E, Ne, Ke), Ws.reshape(E, Ne, -1)

    dX = dgrad_matmul_grouped(
        dY, Wq, Ws, expert_start, 32, 1,
        gather_idx=gather_idx, scatter_idx=None, output_dtype=torch.float32,
        num_input_rows=T,
    )
    assert dX.shape == (T, Ke), f"gradient must be in SOURCE space {(T, Ke)}, got {tuple(dX.shape)}"

    Wdeq = (Wq.float()
            * torch.pow(2.0, Ws.view(torch.uint8).float() - 127).repeat_interleave(32, -1)[..., :Ke])
    floor = torch.zeros(T, Ke, device=TEST_DEVICE, dtype=torch.float32)
    for s, (e, t) in enumerate(pairs):          # the sum a store would collapse to one term
        floor[t] += dY[s].float() @ Wdeq[e]
    assert _rel(dX, floor) < 5e-3, f"top-k accumulation diverges: {_rel(dX, floor):.2e}"
