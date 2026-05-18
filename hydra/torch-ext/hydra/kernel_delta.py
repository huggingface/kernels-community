"""Single-pass Triton delta kernel.

Replaces the host-side ``(do.float() * o.float()).sum(-1)`` dance in
``hydra.kernel_bwd.launch_attn_bwd`` with a fused
GPU-resident kernel that reads ``o`` and ``do`` once (bf16, the same
loads the existing autograd chain already pays for) and writes a
single ``(B, Hq, T)`` fp32 ``delta`` tensor.

What this saves vs the host-side computation (Qwen3-8B-ish prefill,
B=1, H_q=32, T=8192, D=128):

  - ``do.float()`` 128 MB fp32 transient: gone
  - ``o.float()``  128 MB fp32 transient: gone
  - ``do.float() * o.float()`` 128 MB fp32 product: gone
  - net delta tensor still allocated: 1 MB fp32  (B*Hq*T*4)

For an at-rest measurement the peak transient eliminated is ~384 MB;
in the steady-state allocator the saving conservatively measured by
the prior agent's accounting is ~256 MB (after one of the upcasts is
reaped). Either way the dedicated kernel is the right move — the
``o``/``do`` bytes are already on-chip in the autograd path, we just
fold the multiply+reduce into one pass.

Math:

  delta[b, h, t] = sum_{d=0..D-1} o[b, h, t, d] * do[b, h, t, d]

bf16 inputs, fp32 accumulator, fp32 output. The reduction order is
Triton's parallel ``tl.sum`` tree across the D axis (per tile-row).
Compared to torch's left-to-right ``.sum(-1)`` the absolute error
bound is roughly ``D * eps_fp32 * max|o*do|`` which is ~1.5e-5 at
attention-typical scales — well within bf16's ~4e-3 quantum.

Layout:

  - ``o``, ``do``: ``(B, Hq, T, D)`` bf16, contiguous.
  - ``delta``:    ``(B, Hq, T)``   fp32, contiguous.
  - One program per ``(b*Hq + h, q_block_id)`` work item, computing
    ``BLOCK_SIZE`` rows of delta with full ``D`` in-tile reduction.
  - Grid: ``(B * Hq * num_q_blocks,)`` flattened.

Arbitrary T: the kernel masks the tail tile (``offs_tok < T``); the
launcher does not require ``T % BLOCK_SIZE == 0``.
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .kernel_fwd import BLOCK_SIZE, HEAD_DIM


_DELTA_NUM_WARPS = int(os.environ.get("HYDRA_DELTA_NUM_WARPS", "4"))
_DELTA_NUM_STAGES = int(os.environ.get("HYDRA_DELTA_NUM_STAGES", "1"))
_DISABLE_AUTOTUNE = int(os.environ.get("HYDRA_DISABLE_AUTOTUNE", "0"))


def _autotune_configs() -> list[triton.Config]:
    """Same warps/stages sweep used by kernel_fwd; D and BS are fixed."""
    configs: list[triton.Config] = []
    for num_warps in (1, 2, 4):
        for num_stages in (1, 2):
            configs.append(triton.Config({}, num_warps=num_warps, num_stages=num_stages))
    return configs


def _kernel_decorator(jit_kernel):
    if _DISABLE_AUTOTUNE:
        return jit_kernel
    return triton.autotune(
        configs=_autotune_configs(),
        key=["T_MAX", "D"],
    )(jit_kernel)


@triton.jit
def _compute_delta_jit(
    O,         # (B, Hq, T, D) bf16
    dO,        # (B, Hq, T, D) bf16
    Delta,     # (B, Hq, T)    fp32
    T_MAX: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
):
    """Compute ``Delta[b, h, t] = sum_d O[b, h, t, d] * dO[b, h, t, d]``.

    One program handles a single ``(b*Hq + h, q_block_id)`` tile of
    ``BS`` rows, reducing the full ``D``-axis in fp32 in-register.

    Grid: ``(batch_heads * num_q_blocks,)``.
    """
    # Per-head row strides match the (B, Hq, T, D) / (B, Hq, T) layout
    # the contiguous-tensor invariant gives us. Encoding them as
    # constexpr lets the compiler fold the address arithmetic.
    stride_h: tl.constexpr = T_MAX * D
    stride_t: tl.constexpr = D
    stride_lh: tl.constexpr = T_MAX
    NUM_Q_BLOCKS: tl.constexpr = (T_MAX + BS - 1) // BS

    pid = tl.program_id(0)
    bh_id = pid // NUM_Q_BLOCKS
    q_block_id = pid % NUM_Q_BLOCKS

    q_start = q_block_id * BS
    offs_tok = q_start + tl.arange(0, BS)
    offs_d = tl.arange(0, D)
    # Mask trailing tile when T is not a multiple of BS.
    row_mask = offs_tok < T_MAX

    # Load the (BS, D) o and do tiles in bf16. Out-of-bound rows are
    # zero-filled so their contribution to the (masked-out) sum is
    # exactly 0 and doesn't pollute neighbour rows under the tl.sum tree.
    O_ptr = O + bh_id * stride_h
    dO_ptr = dO + bh_id * stride_h
    addr = offs_tok[:, None] * stride_t + offs_d[None, :]
    o_bf16 = tl.load(O_ptr + addr, mask=row_mask[:, None], other=0.0)
    do_bf16 = tl.load(dO_ptr + addr, mask=row_mask[:, None], other=0.0)

    # bf16 -> fp32 -> elementwise mul -> reduce along D.
    # The cast-then-mul order matches the host-side path
    # ``do.float() * o.float()`` exactly (no intermediate bf16 product).
    prod = o_bf16.to(tl.float32) * do_bf16.to(tl.float32)
    di = tl.sum(prod, axis=-1)  # (BS,) fp32

    Delta_ptr = Delta + bh_id * stride_lh
    tl.store(Delta_ptr + offs_tok, di, mask=row_mask)


_compute_delta_kernel = _kernel_decorator(_compute_delta_jit)


def launch_compute_delta(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """Compute ``delta = sum_d o * do`` as a fused single-pass GPU kernel.

    Parameters
    ----------
    o, do : ``(B, Hq, T, D)`` bf16 tensors. Must be contiguous and on the
            same CUDA device. ``T`` is arbitrary (the tail tile is
            masked); ``D`` must equal ``HEAD_DIM``.

    Returns
    -------
    delta : ``(B, Hq, T)`` fp32 tensor, same device. Allocated fresh.

    Numerical equivalence to ``(do.float() * o.float()).sum(-1)``:

      Both compute ``sum_d (bf16_to_fp32(o[d]) * bf16_to_fp32(do[d]))``
      with ``D == 128`` summands per (b, h, t) row. Triton's ``tl.sum``
      uses a parallel reduction tree while torch's ``.sum(-1)`` is a
      left-to-right scan. The reordering error is bounded by
      ``D * eps_fp32 * max|o*do|`` ~ ``128 * 1.2e-7 * O(1)`` ~ ``1.5e-5``,
      well under bf16's ~4e-3 quantum.
    """
    if o.shape != do.shape:
        raise ValueError(f"o and do must have the same shape; got {o.shape} vs {do.shape}")
    if o.dim() != 4:
        raise ValueError(f"o must be 4D (B, Hq, T, D); got shape {o.shape}")
    if o.dtype != torch.bfloat16 or do.dtype != torch.bfloat16:
        raise ValueError(f"o, do must be bf16; got {o.dtype}, {do.dtype}")
    if o.device != do.device:
        raise ValueError(f"o and do must be on the same device; got {o.device} vs {do.device}")
    if not o.is_contiguous() or not do.is_contiguous():
        raise ValueError("o and do must be contiguous")

    B, Hq, T, D = o.shape
    if D != HEAD_DIM:
        raise ValueError(f"head_dim ({D}) must equal HEAD_DIM ({HEAD_DIM})")

    delta = torch.empty(B, Hq, T, dtype=torch.float32, device=o.device)

    batch_heads = B * Hq
    num_q_blocks = (T + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_heads * num_q_blocks,)

    kwargs = dict(
        T_MAX=T,
        BS=BLOCK_SIZE,
        D=D,
    )
    if _DISABLE_AUTOTUNE:
        kwargs["num_warps"] = _DELTA_NUM_WARPS
        kwargs["num_stages"] = _DELTA_NUM_STAGES

    _compute_delta_kernel[grid](o, do, delta, **kwargs)
    return delta
