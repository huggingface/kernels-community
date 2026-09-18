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

"""Row-wise RMS normalization, in the three forms models write it.

Every form is ``x * factor(h) * rsqrt(mean(y**2) + eps)`` over the row: the per-column
``factor`` is the norm weight (``"rms_norm"``) or ``1 + weight`` (the zero-centered
parameterization), and ``y`` is the raw row except under ``"input_scaled_rms_norm"``, which
scales BEFORE normalizing so the row's mean square is taken on the scaled values.

The MoE chain never materializes the normalized rows: ``rms_inv_rows`` computes the per-row
``rsqrt`` in one pass and ``weighted_reduce`` folds it — with the column factor — into the
reduce it already runs. ``rms_norm_rows`` is the standalone form for the paths that have no
reduce to fold into."""

import torch
import triton
import triton.language as tl

from .bayesian_autotuner import bayesian_autotune
from .compat import compile_time_only_triton_wrap, device_context

# the fused names, by how a model composes the weight and the normalization
RMS_NORMS = ("rms_norm", "centered_rms_norm", "input_scaled_rms_norm")


def get_supported_norms() -> tuple[str, ...]:
    """The norm names the kernels fuse (``post_expert_norm``), each ``x * factor * rsqrt(mean +
    eps)`` over the row: ``"rms_norm"`` scales the normalized row by ``weight``,
    ``"centered_rms_norm"`` by ``1 + weight`` (the zero-centered parameterization), and
    ``"input_scaled_rms_norm"`` scales by ``1 + weight`` BEFORE normalizing, so the mean square
    is taken on the scaled row. Anything else stays a host callable."""
    return RMS_NORMS


@triton.jit
def norm_column_factor(W, offs_h, mask, NORM: tl.constexpr):
    """The per-column multiplier of a fused norm: the weight, or ``1 + weight`` under the
    zero-centered parameterizations."""
    w = tl.load(W + offs_h, mask=mask, other=0.0).to(tl.float32)
    if NORM == "rms_norm":
        factor = w
    else:
        factor = 1.0 + w
    return factor


@bayesian_autotune(
    [
        triton.Config({"BLOCK_H": block_h}, num_warps=warps)
        for block_h in (256, 512, 1024, 2048)
        for warps in (4, 8)
    ],
    # one program per row, so the tile width trades the row's loop length against occupancy
    ["H", "NORM", "WRITE_ROWS"],
    n_trials=8,
)
@triton.jit
def _rms_norm_kernel(
    X,  # (S, H) rows to normalize
    W,  # (H,) norm weight
    Out,  # (S, H) normalized rows; None writes only Inv (the fused chain's one-pass arm)
    Inv,  # (S,) fp32 per-row rsqrt(mean + eps); None skips it
    H,
    eps,
    stride_x_m,
    stride_x_h,
    stride_o_m,
    stride_o_h,
    NORM: tl.constexpr,
    WRITE_ROWS: tl.constexpr,  # Out is written (else the pass stops at Inv)
    BLOCK_H: tl.constexpr,
):
    """One program per row: accumulate the row's mean square in fp32 over ``BLOCK_H`` tiles, then
    ``Inv[row] = rsqrt(mean + eps)`` and, under ``WRITE_ROWS``, a second pass writing
    ``x * factor * inv`` (the row is L2-resident by then). ``input_scaled_rms_norm`` squares the
    SCALED values, the other forms the raw ones — the one place the three differ."""
    row = tl.program_id(0)
    x_row = X + row * stride_x_m
    square_sum = tl.zeros((), tl.float32)
    for h in range(0, tl.cdiv(H, BLOCK_H)):
        offs_h = h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs_h < H
        x = tl.load(x_row + offs_h * stride_x_h, mask=mask, other=0.0).to(tl.float32)
        if NORM == "input_scaled_rms_norm":
            x = x * norm_column_factor(W, offs_h, mask, NORM)
        square_sum += tl.sum(x * x, 0)
    inv = tl.rsqrt(square_sum / H + eps)
    if Inv is not None:
        tl.store(Inv + row, inv)
    if WRITE_ROWS:
        out_row = Out + row * stride_o_m
        for h in range(0, tl.cdiv(H, BLOCK_H)):
            offs_h = h * BLOCK_H + tl.arange(0, BLOCK_H)
            mask = offs_h < H
            x = tl.load(x_row + offs_h * stride_x_h, mask=mask, other=0.0).to(tl.float32)
            out = x * norm_column_factor(W, offs_h, mask, NORM) * inv
            tl.store(out_row + offs_h * stride_o_h, out.to(Out.dtype.element_ty), mask=mask)


def _launch_rms_norm(x, weight, eps, norm, write_rows):
    """One launch of ``_rms_norm_kernel`` over ``x``'s rows, returning the normalized rows or the
    per-row ``rsqrt`` — whichever was asked for is the only one allocated."""
    assert norm in RMS_NORMS, f"norm must be one of {RMS_NORMS} or a callable, got {norm!r}"
    assert x.ndim == 2, f"rows must be 2D (rows, H), got ndim={x.ndim}"
    assert weight.ndim == 1 and weight.shape[0] == x.shape[1], (
        f"norm weight {tuple(weight.shape)} does not match the row width ({x.shape[1]},)"
    )
    S, H = x.shape
    out = torch.empty_like(x) if write_rows else None
    inv = None if write_rows else torch.empty(S, device=x.device, dtype=torch.float32)
    with device_context(x.device):
        compile_time_only_triton_wrap(_rms_norm_kernel)[(S,)](
            x,
            weight,
            out,
            inv,
            H,
            eps,
            x.stride(0),
            x.stride(1),
            out.stride(0) if write_rows else 0,
            out.stride(1) if write_rows else 0,
            NORM=norm,
            WRITE_ROWS=write_rows,
        )
    return out if write_rows else inv


def rms_norm_rows(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, norm: str = "rms_norm"
) -> torch.Tensor:
    """RMS-normalize ``(S, H)`` rows in one launch, in the ``get_supported_norms()`` form named by
    ``norm`` (fp32 accumulate, input dtype out). The MoE chain folds this into
    ``weighted_reduce`` instead; this is the standalone form."""
    return _launch_rms_norm(x, weight, eps, norm, write_rows=True)


def rms_inv_rows(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, norm: str = "rms_norm"
) -> torch.Tensor:
    """The ``(S,)`` fp32 ``rsqrt(mean + eps)`` of each row under ``norm`` — the half of
    ``rms_norm_rows`` a consumer that already reads the rows needs, so the rows are never
    written back."""
    return _launch_rms_norm(x, weight, eps, norm, write_rows=False)
