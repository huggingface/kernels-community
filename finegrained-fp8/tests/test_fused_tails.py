"""The fused MX MoE kernels must stay inside ``hidden`` / ``intermediate`` extents.

Same defect class as ``test_grouped_tails.py``, one level up: the gate_up kernel tiles
``intermediate`` (N) and ``hidden`` (K), the down kernel tiles ``hidden`` (N) and
``intermediate`` (K), and both take their tiles from autotune. The intermediate is a
contiguous ``(S, I)`` buffer, so an unmasked N overhang in gate_up lands in the NEXT
token's activations, and an unmasked K tail in down multiplies the next token's
intermediate into valid output columns.

Exactness: hidden rows are powers of two, weights are ones, the activation is ``relu``.
Then gate = up = ``H * 2**t``, the intermediate is ``(H * 2**t)**2`` — exact in MXFP8 for
the ``H`` used here (``H**2`` has a 3-bit mantissa) — and the output is ``I * H**2 * 4**t``
in every column. Any deviation is a kernel leaving its extent.
"""

import pytest
import torch
import triton

from utils import MX_SCALE_GROUP_K, TEST_DEVICE  # type: ignore

import finegrained_fp8  # type: ignore
import finegrained_fp8.fused_grouped as fused  # type: ignore


EXPERTS = 2
TOKENS = 4  # tokens 0,1 -> expert 0; 2,3 -> expert 1 (top-k = 1)

# BN=64 / BK=128 pinned on both kernels: every non-multiple below produces a tail in the
# corresponding kernel/direction; (128, 128) is the aligned control.
TILES = {"COMPUTE_MODE": "dot_scaled", "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128}


def _pin(kernel, memory_mode, pre_hook):
    return triton.Config(
        {**TILES, "MEMORY_MODE": memory_mode}, num_warps=4, num_stages=2, pre_hook=pre_hook
    )


@pytest.fixture(params=["pointer", "host_descriptor"])
def memory_mode(request, monkeypatch):
    mode = request.param
    monkeypatch.setattr(
        fused.mxfp_dynamic_moe_grouped_gate_up_kernel,
        "configs",
        [_pin(fused.mxfp_dynamic_moe_grouped_gate_up_kernel, mode, fused._set_gate_up_descriptor)],
        raising=False,
    )
    monkeypatch.setattr(
        fused.mxfp_dynamic_moe_grouped_down_kernel,
        "configs",
        [_pin(fused.mxfp_dynamic_moe_grouped_down_kernel, mode, fused._set_down_descriptor)],
        raising=False,
    )
    return mode


def _inputs(H, I):
    hidden = (2.0 ** torch.arange(TOKENS, dtype=torch.float32)).unsqueeze(1).expand(TOKENS, H)
    hidden = hidden.contiguous().to(TEST_DEVICE, torch.bfloat16)
    top_k_index = torch.tensor([[0], [0], [1], [1]], device=TEST_DEVICE, dtype=torch.int32)
    top_k_weights = torch.ones(TOKENS, 1, device=TEST_DEVICE, dtype=torch.float32)
    gate_up = torch.ones(EXPERTS, 2 * I, H, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    down = torch.ones(EXPERTS, H, I, device=TEST_DEVICE).to(torch.float8_e4m3fn)
    gate_up_s = torch.ones(EXPERTS, 2 * I, H // MX_SCALE_GROUP_K, device=TEST_DEVICE)
    down_s = torch.ones(EXPERTS, H, I // MX_SCALE_GROUP_K, device=TEST_DEVICE)
    return (
        hidden,
        top_k_index,
        top_k_weights,
        gate_up,
        down,
        gate_up_s.to(torch.float8_e8m0fnu),
        down_s.to(torch.float8_e8m0fnu),
    )


@pytest.mark.parametrize(
    ("H", "I"),
    [
        (96, 96),  # N and K tails in both kernels
        (96, 160),
        (192, 96),
        (128, 96),  # gate_up: N tail only; down: K tail only
        (96, 128),  # gate_up: K tail only; down: N tail only
        (128, 128),  # aligned control
    ],
)
@pytest.mark.kernels_ci
def test_fused_mx_moe_stays_within_hidden_and_intermediate(H, I, memory_mode):
    args = _inputs(H, I)
    rows = 2.0 ** torch.arange(TOKENS, device=TEST_DEVICE, dtype=torch.float32)
    expected = (I * H * H * rows * rows).unsqueeze(1).expand(TOKENS, H)
    # Repeated: overhang and legitimate stores hit the same addresses; ordering is
    # unspecified, so one clean call proves nothing.
    for _ in range(3):
        out = finegrained_fp8.moe_fused_grouped(*args, block_size=None, act_fn="relu")
        torch.testing.assert_close(out.float(), expected, atol=0, rtol=0)
