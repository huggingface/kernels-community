"""Autograd ``Function`` wrapper for the Blackwell FA kernel (perf-tuned variant).

Changes vs. upstream ``hydra.function``:

- The transposed-CSR is fetched through ``build_csrT_cached`` (LRU keyed
  on pattern *content*, not ``data_ptr()``). The upstream cache misses
  every call today because ``api.hydra`` re-allocates
  ``row_ptr`` / ``col_idx`` per invocation; the new key fixes that.

- ``rp_T`` / ``ci_T`` are saved through ``ctx.save_for_backward`` so the
  backward never re-builds them. This matches the upstream behaviour but
  is now meaningfully cheap on cache hit (~O(num_q_blocks) device-side
  for the key check).
"""
from __future__ import annotations

import torch

from .kernel_fwd import BLOCK_SIZE, launch_attn_fwd
from .kernel_bwd import build_csrT_cached, launch_attn_bwd
from .kernel_decode import launch_attn_fwd_decode


class FlashAttnHydraFunction(torch.autograd.Function):
    """Autograd-aware wrapper. See ``api.hydra`` for the public surface."""

    @staticmethod
    def forward(ctx, q, k, v, row_ptr, col_idx, seq_lens, window):
        o, lse = launch_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens, window=window)
        num_q_blocks = q.shape[2] // BLOCK_SIZE
        rp_T, ci_T = build_csrT_cached(row_ptr, col_idx, num_q_blocks)
        ctx.save_for_backward(q, k, v, o, lse, row_ptr, col_idx, seq_lens, rp_T, ci_T)
        ctx.window = window
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, row_ptr, col_idx, seq_lens, rp_T, ci_T = ctx.saved_tensors
        # Both o and do must be contiguous for the delta kernel. The saved o
        # comes from launch_attn_fwd (torch.empty_like(q) — contiguous if q is)
        # but transformers can save a view of o into ctx, and `do` arriving
        # from upstream is often a transpose view from the attention output
        # reshape. Make both contiguous defensively.
        dq, dk, dv = launch_attn_bwd(
            q, k, v, o.contiguous(), do.contiguous(), lse,
            row_ptr, col_idx, seq_lens,
            row_ptr_T=rp_T, col_idx_T=ci_T,
            window=ctx.window,
        )
        return dq, dk, dv, None, None, None, None


class FlashAttnHydraDecodeFunction(torch.autograd.Function):
    """Forward-only autograd wrapper for the T_q==1 decode kernel.

    Generation runs under torch.no_grad so backward is intentionally
    unimplemented; calling .backward() raises with a clear message.
    """

    @staticmethod
    def forward(ctx, q, k, v, window):
        o, lse = launch_attn_fwd_decode(q, k, v, window=window)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.window = window
        ctx.set_materialize_grads(False)
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError(
            "FlashAttnHydraDecodeFunction has no backward. "
            "Decode-step (T_q == 1) is forward-only — wrap your call in "
            "torch.no_grad() (HF generation does this automatically). "
            "For training, use the prefill kernel with T_q == T_kv."
        )
