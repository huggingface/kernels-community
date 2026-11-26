# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
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
"""Backward for the quantized ops — dgrad only, kernels and end-to-end.

Two levels in one file, because they fail differently. The KERNEL tests hand-build operands, never
call ``.backward()``, and measure against the QUANTIZATION FLOOR (dequantize the exact weight the
kernel read, contract in fp32): that isolates "is this contracting the right axis" from "the weight
is lossy", which for a 4-bit weight is ~100x the thing under test. The END-TO-END tests call the
public op and ``.backward()``.

Registration is never asserted directly — it is implied: a format whose op lost its formula fails
``out.requires_grad`` in the parametrized e2e test, across every format in ``WEIGHTS`` rather than
against a hand-kept list of op names that goes stale.

``dX = dY @ W`` contracting N. No format needs its own arm: a weight reaches the kernel as three
numbers — ``(scale_group_k, scale_row_div, values_per_byte)`` — so MX (per-row K-groups) and
block-FP8 (128x128 blocks) are one code path at different divisors. Only dgrad exists; the
quantized weight takes no gradient, because an fp8/fp4 tensor cannot be an autograd leaf.
"""

import pytest
import torch
from utils import (  # type: ignore
    dequant_gate_up,
    dequantize_weight,
    finegrained_kernels,
    internals,
    is_memory_overflow,
    make_gptq_act_order,
    make_weights,
    quant_dequant_a,
    SUPPORTS_FP8,
    sweep_configs,
    TEST_DEVICE,
    WEIGHTS,
)

fg = finegrained_kernels
dgrad_matmul_2d = internals.backward.dgrad_matmul_2d
dgrad_matmul_batched = internals.backward.dgrad_matmul_batched
dgrad_matmul_grouped = internals.backward.dgrad_matmul_grouped
glu_backward = internals.backward.glu_backward
apply_glu = internals.epilogue.apply_glu
dgrad_matmul_grouped_kernel = internals.backward.dgrad_matmul_grouped_kernel
dgrad_matmul_2d_kernel = internals.backward.dgrad_matmul_2d_kernel


pytestmark = pytest.mark.skipif(not SUPPORTS_FP8, reason="FP8 kernels require SM89+")

M, N, K = 512, 1024, 768  # 128-aligned on every axis so every format's blocks divide evenly


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


def test_dgrad_format_is_only_divisors():
    """block-FP8 reaches the same kernel at scale_group_k=128, scale_row_div=128 — a 128x128
    block grid is the per-row K-group layout with both divisors widened, so adding a format whose
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
    inter = 128
    Z = torch.randn(64, 2 * inter, device=TEST_DEVICE, dtype=torch.float32, requires_grad=True)
    dH = torch.randn(64, inter, device=TEST_DEVICE, dtype=torch.float32)

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
    inter = 128
    Z = torch.randn(64, 2 * inter, device=TEST_DEVICE, dtype=torch.float32) * 3.0
    Z.requires_grad_(True)
    dH = torch.randn(64, inter, device=TEST_DEVICE, dtype=torch.float32)

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


QUANT_WEIGHTS = [r for r in WEIGHTS if r not in ("bf16", "fp16")]


@pytest.mark.parametrize("weights", QUANT_WEIGHTS)
def test_dgrad_every_quant_format(weights):
    """Every quantized format must produce a usable activation gradient. Formats whose scale grid
    can be reoriented (per-tensor, block) take the quantized transposed-view route on the forward
    kernel; the per-row-group and packed-E2M1 ones contract N against the weight's natural tile
    (``backward.py``). Both must land near the bf16 reference — this pins COVERAGE, so a format
    can never silently lose its backward."""
    torch.manual_seed(0)
    try:
        B, Bs, g = WEIGHTS[weights]["make"](N, K, None)
    except TypeError:  # a few makes only build the expert-stacked form; take one expert
        B, Bs, g = WEIGHTS[weights]["make"](N, K, 1)
        B, Bs = B[0], Bs[0]
    A = torch.randn(M, K, device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(M, N, device=TEST_DEVICE, dtype=torch.bfloat16)

    out = fg.matmul_2d(A, B, Bs=Bs, output_dtype=torch.bfloat16, b_global_scale=g)
    assert out.requires_grad, f"{weights}: output not attached to the graph"
    out.backward(grad_out)
    assert A.grad is not None and A.grad.shape == (M, K)
    assert torch.isfinite(A.grad.float()).all(), f"{weights}: non-finite dA"

    # reference: the same product against the dequantized weight
    W = dequantize_weight(B, Bs, global_scale=g).reshape(N, K)
    ref = grad_out.float() @ W
    assert _rel(A.grad, ref) < 2e-1, f"{weights}: dA vs dequantized reference {_rel(A.grad, ref):.2e}"


@pytest.mark.parametrize("route", ["2d", "grouped", "batched"])
def test_dgrad_int_act_order_zero_points(route):
    """GPTQ act-order INT4 weights with zero points: the dgrad decodes the words with their zero points and stores
    each column back through its expert's ``b_perm``, landing on ``dY @ W`` in A's natural K order."""
    torch.manual_seed(0)
    T, top_k, E = 64, 2, 4
    W, Ws, Wz, perm, ref_W = make_gptq_act_order(N, K, 1 if route == "2d" else E)
    A = torch.randn(T, K, device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    if route == "2d":
        grad_out = torch.randn(T, N, device=TEST_DEVICE, dtype=torch.bfloat16)
        out = fg.matmul_2d(A, W[0], Bs=Ws[0], b_zero=Wz[0], b_perm=perm[0], output_dtype=torch.bfloat16)
        ref = grad_out.float() @ ref_W[0]
    else:
        expert_ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(T)]).flatten().to(TEST_DEVICE, torch.int32)
        grad_out = torch.randn(T * top_k, N, device=TEST_DEVICE, dtype=torch.bfloat16)
        if route == "grouped":
            expert_start, gather_idx, scatter_idx = internals.scheduling.compute_grouped_scheduling(expert_ids, E, top_k)
            out = fg.matmul_grouped(A, W, Bs=Ws, b_zero=Wz, b_perm=perm, expert_start=expert_start,
                                    gather_idx=gather_idx, routed_idx=scatter_idx, output_dtype=torch.bfloat16)
            tokens = gather_idx.long()
            experts = torch.repeat_interleave(torch.arange(E, device=TEST_DEVICE), expert_start.diff().long())
        else:
            out = fg.matmul_batched(A, W, Bs=Ws, b_zero=Wz, b_perm=perm, expert_ids=expert_ids,
                                    output_dtype=torch.bfloat16)
            tokens = torch.arange(T * top_k, device=TEST_DEVICE) // top_k
            experts = expert_ids.long()
        ref = torch.zeros(T, K, device=TEST_DEVICE).index_add_(
            0, tokens, torch.einsum("sn,snk->sk", grad_out.float(), ref_W[experts]))
    out.backward(grad_out)
    assert _rel(A.grad, ref) < 1e-2, f"{route}: dA vs act-order reference {_rel(A.grad, ref):.2e}"


def test_dgrad_pre_quant_scale():
    """An input smoothed before quantization (``a_pre_quant_scale``: the GEMM reads ``A * s``) takes
    ``dA = (dY @ W) * s``."""
    torch.manual_seed(0)
    B, Bs, g = WEIGHTS["mxfp8"]["make"](N, K, None)
    s = torch.rand(K, device=TEST_DEVICE, dtype=torch.bfloat16) + 0.5
    A = torch.randn(M, K, device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(M, N, device=TEST_DEVICE, dtype=torch.bfloat16)
    fg.matmul_2d(A, B, Bs=Bs, a_pre_quant_scale=s, output_dtype=torch.bfloat16).backward(grad_out)
    ref = (grad_out.float() @ dequantize_weight(B, Bs).reshape(N, K)) * s.float()
    assert _rel(A.grad, ref) < 2e-2, f"dA vs (dY @ W) * s: {_rel(A.grad, ref):.2e}"


ROUTED_WEIGHTS = [*QUANT_WEIGHTS, "bf16"]


@pytest.mark.parametrize("route", ["grouped", "batched"])
@pytest.mark.parametrize("weights", ROUTED_WEIGHTS)
def test_routed_dgrad_every_format(weights, route):
    """Every routed GEMM format differentiates through ``matmul_grouped`` / ``matmul_batched`` on a gathered top_k
    launch (one input row per token, the gate|up's shape), so each token's top_k rows sum back: near the product
    against the dequantized weight."""
    torch.manual_seed(0)
    T, top_k, E = 64, 2, 4
    B, Bs, g = WEIGHTS[weights]["make"](N, K, E)
    W = WEIGHTS[weights]["dequant"](B, Bs, g).reshape(E, N, K)
    A = torch.randn(T, K, device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    expert_ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(T)]).flatten().to(TEST_DEVICE, torch.int32)
    grad_out = torch.randn(T * top_k, N, device=TEST_DEVICE, dtype=torch.bfloat16)
    if route == "grouped":
        expert_start, gather_idx, scatter_idx = internals.scheduling.compute_grouped_scheduling(expert_ids, E, top_k)
        out = fg.matmul_grouped(A, B, Bs=Bs, expert_start=expert_start, gather_idx=gather_idx, routed_idx=scatter_idx,
                                b_global_scale=g, output_dtype=torch.bfloat16)
        tokens = gather_idx.long()
        experts = torch.repeat_interleave(torch.arange(E, device=TEST_DEVICE), expert_start.diff().long())
    else:
        out = fg.matmul_batched(A, B, Bs=Bs, expert_ids=expert_ids, b_global_scale=g, output_dtype=torch.bfloat16)
        tokens = torch.arange(T * top_k, device=TEST_DEVICE) // top_k
        experts = expert_ids.long()
    assert out.requires_grad, f"{weights} {route}: output not attached to the graph"
    out.backward(grad_out)
    ref = torch.zeros(T, K, device=TEST_DEVICE).index_add_(
        0, tokens, torch.einsum("sn,snk->sk", grad_out.float(), W[experts]))
    assert A.grad is not None and A.grad.shape == (T, K) and torch.isfinite(A.grad.float()).all()
    assert _rel(A.grad, ref) < 2e-1, f"{weights} {route}: dA vs dequantized reference {_rel(A.grad, ref):.2e}"


@pytest.mark.parametrize("route", ["2d", "grouped", "batched"])
def test_dgrad_fp8_weight_only(route):
    """Block-FP8 weight-only (raw BF16 activations, ``activation_format="bf16"``) on every route: the dgrad kernel
    decodes the FP8 tile against its block grid with a BF16 ``dY``, no requantization."""
    torch.manual_seed(0)
    T, top_k, E = 64, 2, 4
    B, Bs, _ = WEIGHTS["fp8_128x128"]["make"](N, K, None if route == "2d" else E)
    W = WEIGHTS["fp8_128x128"]["dequant"](B, Bs)
    rows = T if route == "2d" else T * top_k
    A = torch.randn(T, K, device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    grad_out = torch.randn(rows, N, device=TEST_DEVICE, dtype=torch.bfloat16)
    if route == "2d":
        out = fg.matmul_2d(A, B, Bs=Bs, activation_format="bf16", output_dtype=torch.bfloat16)
        ref = grad_out.float() @ W.reshape(N, K)
    else:
        expert_ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(T)]).flatten().to(TEST_DEVICE, torch.int32)
        W = W.reshape(E, N, K)
        if route == "grouped":
            expert_start, gather_idx, scatter_idx = internals.scheduling.compute_grouped_scheduling(expert_ids, E, top_k)
            out = fg.matmul_grouped(A, B, Bs=Bs, expert_start=expert_start, gather_idx=gather_idx, routed_idx=scatter_idx,
                                    activation_format="bf16", output_dtype=torch.bfloat16)
            tokens = gather_idx.long()
            experts = torch.repeat_interleave(torch.arange(E, device=TEST_DEVICE), expert_start.diff().long())
        else:
            out = fg.matmul_batched(A, B, Bs=Bs, expert_ids=expert_ids, activation_format="bf16",
                                    output_dtype=torch.bfloat16)
            tokens = torch.arange(rows, device=TEST_DEVICE) // top_k
            experts = expert_ids.long()
        ref = torch.zeros(T, K, device=TEST_DEVICE).index_add_(
            0, tokens, torch.einsum("sn,snk->sk", grad_out.float(), W[experts]))
    out.backward(grad_out)
    assert _rel(A.grad, ref) < 1e-2, f"{route}: dA vs dequantized reference {_rel(A.grad, ref):.2e}"


GATED_WEIGHTS = ["fp8_128x128", "mxfp8", "int4", "bf16"]


@pytest.mark.parametrize("route", ["2d", "grouped", "batched"])
@pytest.mark.parametrize("weights", GATED_WEIGHTS)
def test_gated_dgrad(weights, route):
    """A fused gate|up GLU differentiates through its saved pre-activation (the dispatchers save it whenever the
    input takes a gradient): near torch autograd through ``apply_glu`` on the dequantized gate|up weight, gate rows
    at ``0::2`` and up rows at ``1::2``."""
    torch.manual_seed(0)
    T, top_k, E = 64, 2, 4
    B, Bs, g = WEIGHTS[weights]["make"](2 * N, K, 1 if route == "2d" else E)
    W = dequant_gate_up(WEIGHTS[weights], B, Bs, g).reshape(-1, 2 * N, K)
    if route == "2d":
        B, Bs = B[0], None if Bs is None else Bs[0]
        g = None if g is None else g.reshape(-1)[:1]
    A = torch.randn(T, K, device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    glu = dict(gate=True, act_fn="silu", output_dtype=torch.bfloat16)
    if route == "2d":
        out = fg.matmul_2d(A, B, Bs=Bs, b_global_scale=g, **glu)
        tokens = torch.arange(T, device=TEST_DEVICE)
        experts = torch.zeros(T, device=TEST_DEVICE, dtype=torch.long)
    else:
        expert_ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(T)]).flatten().to(TEST_DEVICE, torch.int32)
        if route == "grouped":
            expert_start, gather_idx, scatter_idx = internals.scheduling.compute_grouped_scheduling(expert_ids, E, top_k)
            out = fg.matmul_grouped(A, B, Bs=Bs, b_global_scale=g, expert_start=expert_start, gather_idx=gather_idx,
                                    routed_idx=scatter_idx, **glu)
            tokens = gather_idx.long()
            experts = torch.repeat_interleave(torch.arange(E, device=TEST_DEVICE), expert_start.diff().long())
        else:
            out = fg.matmul_batched(A, B, Bs=Bs, b_global_scale=g, expert_ids=expert_ids, **glu)
            tokens = torch.arange(T * top_k, device=TEST_DEVICE) // top_k
            experts = expert_ids.long()
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    A_ref = A.detach().float().requires_grad_(True)
    # the forward's activation round-trip (W8A8), straight through: Z sees what the kernel saw, the gradient is dA's
    quant, row = WEIGHTS[weights]["act_quant"][None], WEIGHTS[weights]
    A_seen = A_ref if quant is None else A_ref + (row["dq_act"](*quant(A.detach())) - A_ref).detach()
    Z = torch.einsum("sk,snk->sn", A_seen[tokens], W[experts])
    H = apply_glu(Z[:, 0::2], Z[:, 1::2], "silu", None, None)
    (H * grad_out.float()).sum().backward()
    assert A.grad is not None and torch.isfinite(A.grad.float()).all()
    assert _rel(A.grad, A_ref.grad) < 5e-2, f"{weights} {route}: gated dA vs autograd {_rel(A.grad, A_ref.grad):.2e}"


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
    dgrad_matmul_grouped = internals.backward.dgrad_matmul_grouped

    torch.manual_seed(0)
    T, tk, E, Ne, Ke = 8, 2, 4, 256, 256
    S = T * tk

    # each token -> tk DISTINCT experts, then sort the (token, expert) pairs by expert so the
    # rows are expert-contiguous, which is the layout the grouped schedule guarantees
    experts = torch.stack([torch.randperm(E)[:tk] for _ in range(T)])       # (T, tk)
    pairs = [(int(e), t) for t in range(T) for e in experts[t]]
    pairs.sort()
    gather_idx = torch.tensor([t for _, t in pairs], device=TEST_DEVICE, dtype=torch.int32)
    # each position's token-major routed row, t * tk + (the expert's slot among the token's)
    routed_idx = torch.tensor(
        [t * tk + experts[t].tolist().index(e) for e, t in pairs], device=TEST_DEVICE, dtype=torch.int32
    )
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
        num_input_rows=T, routed_idx=routed_idx,
    )
    assert dX.shape == (T, Ke), f"gradient must be in SOURCE space {(T, Ke)}, got {tuple(dX.shape)}"

    Wdeq = (Wq.float()
            * torch.pow(2.0, Ws.view(torch.uint8).float() - 127).repeat_interleave(32, -1)[..., :Ke])
    floor = torch.zeros(T, Ke, device=TEST_DEVICE, dtype=torch.float32)
    for s, (e, t) in enumerate(pairs):          # the sum a store would collapse to one term
        floor[t] += dY[s].float() @ Wdeq[e]
    assert _rel(dX, floor) < 5e-3, f"top-k accumulation diverges: {_rel(dX, floor):.2e}"


@pytest.mark.slow
def test_dgrad_grouped_every_admitted_config_accumulates():
    """Force-run EVERY admitted config on the top-k ACCUMULATE launch (non-identity gather,
    duplicate targets, no scatter) and hard-check each against the torch floor sum. The tuner
    benches configs by speed, so an arm that is wrong ONLY under these launch semantics gets
    crowned whenever it wins a near-tied tune — the descriptor-anchor bug this guards against
    read the wrong dY rows on exactly this launch while passing every identity-gather test."""
    dgrad_matmul_grouped = internals.backward.dgrad_matmul_grouped

    torch.manual_seed(0)
    T, tk, E, Ne, Ke = 8, 2, 4, 256, 256
    S = T * tk
    experts = torch.stack([torch.randperm(E)[:tk] for _ in range(T)])
    pairs = sorted((int(e), t) for t in range(T) for e in experts[t])
    gather_idx = torch.tensor([t for _, t in pairs], device=TEST_DEVICE, dtype=torch.int32)
    # each position's token-major routed row, t * tk + (the expert's slot among the token's)
    routed_idx = torch.tensor(
        [t * tk + experts[t].tolist().index(e) for e, t in pairs], device=TEST_DEVICE, dtype=torch.int32
    )
    counts = torch.bincount(torch.tensor([e for e, _ in pairs]), minlength=E)
    expert_start = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]).to(
        TEST_DEVICE, torch.int32)
    dY = torch.randn(S, Ne, device=TEST_DEVICE, dtype=torch.bfloat16)
    W = torch.randn(E, Ne, Ke, device=TEST_DEVICE, dtype=torch.bfloat16) * 0.1
    Wq, Ws = fg.mxfp8_act_quant(W.reshape(E * Ne, Ke))
    Wq, Ws = Wq.reshape(E, Ne, Ke), Ws.reshape(E, Ne, -1)
    Wdeq = (Wq.float()
            * torch.pow(2.0, Ws.view(torch.uint8).float() - 127).repeat_interleave(32, -1)[..., :Ke])
    floor = torch.zeros(T, Ke, device=TEST_DEVICE, dtype=torch.float32)
    for s, (e, t) in enumerate(pairs):
        floor[t] += dY[s].float() @ Wdeq[e]

    def launch():
        return dgrad_matmul_grouped(
            dY, Wq, Ws, expert_start, 32, 1,
            gather_idx=gather_idx, scatter_idx=None, output_dtype=torch.float32,
            num_input_rows=T, routed_idx=routed_idx,
        )

    _assert_every_admitted_config(dgrad_matmul_grouped_kernel, launch, floor, "break top-k accumulation")


def _assert_every_admitted_config(tuner, launch, floor, what):
    """Force-run every config the pruners admit for ``launch`` and hard-check each against ``floor``. The tuner benches
    configs by speed, so an arm that is wrong only on some launch gets crowned whenever it wins a near-tied tune."""
    admitted = {}
    original_prune, original_configs = tuner.early_config_prune, tuner.configs

    def spy(configs, named_args, **kwargs):
        kept = original_prune(configs, named_args, **kwargs) if original_prune else configs
        admitted["configs"] = sweep_configs(kept)
        return kept[:1]  # this launch only captures the list; every config runs below, so skip the tune

    tuner.early_config_prune = spy
    tuner.cache.clear()  # an earlier test's crown takes the fast path and prune (the spy) never runs
    try:
        launch()
    finally:
        tuner.early_config_prune = original_prune

    wrong, failed = [], []
    try:
        for config in admitted["configs"]:
            tuner.configs = [config]
            tuner.cache.clear()
            try:
                out = launch()
                torch.accelerator.synchronize()
            except Exception as e:
                # strict fails on anything but a compiled memory overflow (the compiler's per-shape choice)
                if not is_memory_overflow(str(e)):
                    failed.append((config.all_kwargs(), f"{type(e).__name__}: {str(e)[:120]}"))
                continue
            rel = _rel(out, floor)
            if rel > 5e-3:
                wrong.append((rel, config.all_kwargs()))
    finally:
        tuner.configs = original_configs
        tuner.cache.clear()
    assert not wrong, f"{len(wrong)} config(s) {what}, e.g. {wrong[0]}"
    # strict (the test default, see conftest): the pruners admitted a config that cannot run
    assert not failed or tuner.max_failures != 0, f"{len(failed)} admitted config(s) failed to run, e.g. {failed[0]}"


@pytest.mark.slow
@pytest.mark.parametrize("rows", [1, 16, 64, 512])  # swapped / split decode, the split edge, grouped prefill order
@pytest.mark.parametrize("weights", ["fp8_128x128", "mxfp4", "int4"])
def test_dgrad_2d_every_admitted_config_is_correct(weights, rows):
    """Every admitted 2D dgrad config against the quantization floor: block, group-32 and integer scale grids across the
    decode (swapped, reduction-split) and prefill (grouped tile order) bands."""
    torch.manual_seed(0)
    Nw, Kw = 512, 768
    B, Bs, _ = WEIGHTS[weights]["make"](Nw, Kw, 1)
    B, Bs = B[0], Bs[0]
    if B.dtype == torch.uint32:  # integer words go in as stored, like the op's backward (``_int_dgrad_2d``)
        W, Ws, group, row_div, int_bits = B, Bs, Kw // Bs.shape[-1], 1, 32 // (Kw // B.shape[-1])
    else:
        (W, Ws, group, row_div), int_bits = internals.backward._dgrad_operands(B, Bs), 0
    dY = torch.randn(rows, Nw, device=TEST_DEVICE, dtype=torch.bfloat16)
    floor = dY.float() @ WEIGHTS[weights]["dequant"](B, Bs)
    _assert_every_admitted_config(
        dgrad_matmul_2d_kernel,
        lambda: dgrad_matmul_2d(dY, W, Ws, group, row_div, output_dtype=torch.float32, int_bits=int_bits),
        floor, "are wrong",
    )


# ── fused MoE, end to end ────────────────────────────────────────────────────────
# `moe_fused_*` chain the gate_up (fused GLU), the down and the routing-weighted reduce. These pin the gradients a
# fine-tune takes out of the experts, into the hidden states and through the routing weights into the router's,
# against autograd over an fp32 reference on the dequantized weights.

MOE_T, MOE_TOP_K, MOE_E, MOE_H, MOE_I = 16, 2, 4, 256, 256
MOE_FORMATS = [  # (weights, activation_format): None follows the weights
    ("fp8_128x128", None),
    ("mxfp8", None),
    ("nvfp4", None),
    ("mxfp4", None),  # W4A4
    ("mxfp4", "bf16"),  # W4A16, the QLoRA base
    ("int4", None),
    ("bf16", None),
]
MOE_FORWARDS = ["moe_fused_grouped", "moe_fused_batched"]


def _moe_glu(z, alpha=None, limit=None):
    """The fused epilogue's SwiGLU over interleaved gate|up columns, written independently."""
    g, u = z[..., 0::2], z[..., 1::2]
    if limit is not None:
        g, u = g.clamp(max=limit), u.clamp(-limit, limit)
    if alpha is not None:
        return (u + 1) * g * torch.sigmoid(alpha * g)
    return u * torch.nn.functional.silu(g)


def _moe_norm(y, norm, weight, eps=1e-6):
    if norm is None:
        return y
    factor = weight.float() if norm == "rms_norm" else 1.0 + weight.float()
    scaled = y * factor if norm == "input_scaled_rms_norm" else y
    return y * factor * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + eps)


def _moe_reference(x, top_k_index, top_k_weights, gate_up, down, alpha=None, limit=None, gate_up_bias=None,
                   down_bias=None, norm=None, norm_weight=None):
    """fp32 MoE over dequantized weights: the gradient target the quantized chain approximates."""
    kept = (top_k_index >= 0) & (top_k_index < gate_up.shape[0])  # EP sentinels (E, or -1) route nowhere
    idx = top_k_index.long().clamp(0, gate_up.shape[0] - 1)
    z = torch.einsum("th,tknh->tkn", x.float(), gate_up[idx])
    if gate_up_bias is not None:
        z = z + gate_up_bias.float()[idx]
    y = torch.einsum("tki,tkhi->tkh", _moe_glu(z, alpha, limit), down[idx])
    if down_bias is not None:
        y = y + down_bias.float()[idx]
    y = _moe_norm(y, norm, norm_weight)
    return (y * (top_k_weights.float() * kept)[..., None]).sum(1)


def _moe_problem(weights, activation_format, sentinels=0, sentinel_id=MOE_E, scale=1.0):
    torch.manual_seed(0)
    row = WEIGHTS[weights]
    gate_up, gate_up_s, gate_up_g = row["make"](2 * MOE_I, MOE_H, MOE_E)
    down, down_s, down_g = row["make"](MOE_H, MOE_I, MOE_E)
    x = scale * torch.randn(MOE_T, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    top_k_index = torch.randn(MOE_T, MOE_E, device=TEST_DEVICE).topk(MOE_TOP_K, dim=-1).indices.to(torch.int32)
    if sentinels:
        top_k_index.view(-1)[torch.randperm(top_k_index.numel(), device=TEST_DEVICE)[:sentinels]] = sentinel_id
    top_k_weights = torch.rand(MOE_T, MOE_TOP_K, device=TEST_DEVICE, dtype=torch.bfloat16)
    kwargs = dict(
        gate_up_proj=gate_up, down_proj=down, gate_up_proj_scale_inv=gate_up_s, down_proj_scale_inv=down_s,
        gate_up_proj_weight_global_scale=gate_up_g, down_proj_weight_global_scale=down_g,
        activation_format=activation_format,
    )
    dequant = (dequant_gate_up(row, gate_up, gate_up_s, gate_up_g), row["dequant"](down, down_s, down_g))
    if weights == "nvfp4" and activation_format is None:
        # the calibrated activation globals an NVFP4 checkpoint ships (`input_scale`): without them the test
        # weights saturate every NVFP4 activation and the comparison says nothing
        idx = top_k_index.long().clamp(0, MOE_E - 1)
        inter = _moe_glu(torch.einsum("th,tknh->tkn", x.float(), dequant[0][idx]))
        kwargs["gate_up_proj_input_global_scale"] = (x.float().abs().amax() / (6.0 * 448.0)).reshape(1)
        kwargs["down_proj_input_global_scale"] = (inter.abs().amax() / (6.0 * 448.0)).reshape(1)
    return x, top_k_index, top_k_weights, kwargs, dequant


def _moe_grads(forward, x, top_k_index, top_k_weights, grad_out, train_hidden=True, **kwargs):
    x = x.detach().clone().requires_grad_(train_hidden)
    w = top_k_weights.detach().clone().requires_grad_(True)
    out = getattr(fg, forward)(x, top_k_index, w, **kwargs)
    out.backward(grad_out)
    return out, x.grad, w.grad


def _reference_grads(x, top_k_index, top_k_weights, grad_out, dequant, **kwargs):
    x = x.detach().float().requires_grad_(True)
    w = top_k_weights.detach().float().requires_grad_(True)
    out = _moe_reference(x, top_k_index, w, *dequant, **kwargs)
    out.backward(grad_out.float())
    return out, x.grad, w.grad


def _assert_moe_grads(out, dx, dw, ref_out, ref_dx, ref_dw):
    """The forward's own quantization error bounds what the backward can reach."""
    floor = _rel(out, ref_out)
    bound = max(2 * floor, 5e-2)
    assert dw is not None and torch.isfinite(dw.float()).all()
    assert _rel(dw, ref_dw) < bound, f"dw {_rel(dw, ref_dw):.2e} (forward {floor:.2e})"
    if ref_dx is not None:
        assert dx is not None and dx.dtype == torch.bfloat16 and torch.isfinite(dx.float()).all()
        assert _rel(dx, ref_dx) < bound, f"dx {_rel(dx, ref_dx):.2e} (forward {floor:.2e})"


@pytest.mark.parametrize("forward", MOE_FORWARDS)
@pytest.mark.parametrize("weights,activation_format", MOE_FORMATS)
def test_moe_backward_matches_reference(weights, activation_format, forward):
    """Hidden-state and routing-weight gradients through the whole chain, EP sentinels included."""
    x, top_k_index, top_k_weights, kwargs, dequant = _moe_problem(weights, activation_format, sentinels=4)
    grad_out = torch.randn(MOE_T, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    _assert_moe_grads(
        *_moe_grads(forward, x, top_k_index, top_k_weights, grad_out, **kwargs),
        *_reference_grads(x, top_k_index, top_k_weights, grad_out, dequant),
    )


@pytest.mark.kernels_ci
def test_moe_fused_grouped_backward_fp8():
    """The cheap CI subset: block FP8 through the grouped (prefill) forward."""
    test_moe_backward_matches_reference("fp8_128x128", None, "moe_fused_grouped")


@pytest.mark.parametrize("sentinel_id", [MOE_E, -1])  # any id outside [0, E) is another rank's
@pytest.mark.parametrize("forward", MOE_FORWARDS)
def test_moe_backward_skips_ep_sentinels(forward, sentinel_id):
    """A sentinel slot routes to no local expert: its routing weight takes exactly zero gradient."""
    x, top_k_index, top_k_weights, kwargs, dequant = _moe_problem("mxfp8", None, sentinels=6, sentinel_id=sentinel_id)
    grad_out = torch.randn(MOE_T, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    out, dx, dw = _moe_grads(forward, x, top_k_index, top_k_weights, grad_out, **kwargs)
    sentinel = top_k_index == sentinel_id
    assert sentinel.any() and (dw[sentinel] == 0).all(), "a sentinel slot received a gradient"
    _assert_moe_grads(out, dx, dw, *_reference_grads(x, top_k_index, top_k_weights, grad_out, dequant))


@pytest.mark.parametrize("forward", MOE_FORWARDS)
def test_moe_backward_clamped_swiglu_with_bias(forward):
    """The GPT-OSS arm: alpha-scaled, limit-clamped SwiGLU, both projections biased (frozen). The hidden states are
    scaled so a few percent of the gates saturate, as in a trained model, so the clamp's zero gradient is exercised
    without the MXFP8 error deciding which ones saturate."""
    x, top_k_index, top_k_weights, kwargs, dequant = _moe_problem("mxfp8", None, scale=0.2)
    gate_up_bias = torch.randn(MOE_E, 2 * MOE_I, device=TEST_DEVICE, dtype=torch.bfloat16)
    down_bias = 0.1 * torch.randn(MOE_E, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    idx = top_k_index.long()
    z = torch.einsum("th,tknh->tkn", x.float(), dequant[0][idx]) + gate_up_bias.float()[idx]
    saturated = (z[..., 0::2] >= 7.0).float().mean()
    assert 0 < saturated < 0.1, f"{saturated:.1%} of the gates saturate: the clamp is untested or dominant"
    grad_out = torch.randn(MOE_T, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    _assert_moe_grads(
        *_moe_grads(forward, x, top_k_index, top_k_weights, grad_out, gate_up_proj_bias=gate_up_bias,
                    down_proj_bias=down_bias, swiglu_alpha=1.702, swiglu_limit=7.0, **kwargs),
        *_reference_grads(x, top_k_index, top_k_weights, grad_out, dequant, alpha=1.702, limit=7.0,
                          gate_up_bias=gate_up_bias, down_bias=down_bias),
    )


@pytest.mark.parametrize("norm", ["rms_norm", "centered_rms_norm", "input_scaled_rms_norm"])
@pytest.mark.parametrize("forward", MOE_FORWARDS)
def test_moe_backward_through_post_expert_norm(forward, norm):
    """A named post-expert norm rides inside the forward's reduce, its rows never materialized; the reduce's
    backward differentiates it, the per-row rsqrt included."""
    x, top_k_index, top_k_weights, kwargs, dequant = _moe_problem("mxfp8", None)
    norm_weight = 0.5 * torch.rand(MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    grad_out = torch.randn(MOE_T, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    _assert_moe_grads(
        *_moe_grads(forward, x, top_k_index, top_k_weights, grad_out, post_expert_norm=norm,
                    post_expert_norm_weight=norm_weight, **kwargs),
        *_reference_grads(x, top_k_index, top_k_weights, grad_out, dequant, norm=norm, norm_weight=norm_weight),
    )


def test_moe_backward_router_only():
    """Only the router trains: the hidden states take no gradient, so the chain keeps its quantized intermediate
    handoff and the routing weights' gradient still comes out of the reduce."""
    x, top_k_index, top_k_weights, kwargs, dequant = _moe_problem("fp8_128x128", None)
    grad_out = torch.randn(MOE_T, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    out, dx, dw = _moe_grads("moe_fused_grouped", x, top_k_index, top_k_weights, grad_out, train_hidden=False,
                             **kwargs)
    assert dx is None
    ref_out, _, ref_dw = _reference_grads(x, top_k_index, top_k_weights, grad_out, dequant)
    _assert_moe_grads(out, None, dw, ref_out, None, ref_dw)


def test_moe_backward_through_expanded_gather():
    """Prefill row counts trade the gate_up's in-kernel gather for one contiguous copy (``expand_regime``), laid out
    by sorted position: its gradient must sum back per token. The shared problem is too small to reach it."""
    num_tokens, top_k, num_experts = 2048, 2, 8
    assert internals.scheduling.expand_regime(num_tokens * top_k, num_experts)
    torch.manual_seed(0)
    gate_up = (0.05 * torch.randn(num_experts, 2 * MOE_I, MOE_H, device=TEST_DEVICE)).to(torch.bfloat16)
    down = (0.05 * torch.randn(num_experts, MOE_H, MOE_I, device=TEST_DEVICE)).to(torch.bfloat16)
    x = torch.randn(num_tokens, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    top_k_index = torch.randn(num_tokens, num_experts, device=TEST_DEVICE).topk(top_k, dim=-1).indices.to(torch.int32)
    top_k_weights = torch.rand(num_tokens, top_k, device=TEST_DEVICE, dtype=torch.bfloat16)
    grad_out = torch.randn(num_tokens, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16)
    out, dx, dw = _moe_grads("moe_fused_grouped", x, top_k_index, top_k_weights, grad_out, gate_up_proj=gate_up,
                             down_proj=down, gate_up_proj_scale_inv=None, down_proj_scale_inv=None)
    ref_out, ref_dx, ref_dw = _reference_grads(x, top_k_index, top_k_weights, grad_out, (gate_up.float(), down.float()))
    assert _rel(out, ref_out) < 2e-2
    assert _rel(dx, ref_dx) < 2e-2, f"dx {_rel(dx, ref_dx):.2e}"
    assert _rel(dw, ref_dw) < 2e-2, f"dw {_rel(dw, ref_dw):.2e}"


def test_moe_backward_refuses_a_trained_bias():
    """Biases stay frozen with the quantized base: one asking for a gradient raises instead of getting none."""
    x, top_k_index, top_k_weights, kwargs, _ = _moe_problem("mxfp8", None)
    bias = torch.zeros(MOE_E, MOE_H, device=TEST_DEVICE, dtype=torch.bfloat16, requires_grad=True)
    out = fg.moe_fused_grouped(x.requires_grad_(), top_k_index, top_k_weights, down_proj_bias=bias, **kwargs)
    with pytest.raises(NotImplementedError, match="bias"):
        out.sum().backward()
