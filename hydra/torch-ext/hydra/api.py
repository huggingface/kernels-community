"""Public Python API (perf-tuned variant).

Same public signature as ``hydra.api.hydra``.

Internally:

- Caches the (row_ptr, col_idx, seq_lens) tuple keyed on
  ``(B, Hq, T, BLOCK_SIZE, window_arg, device, dtype_marker)``. The
  upstream api re-builds the CSR on every call, which (a) pays a Python
  cost per call and (b) defeats any data_ptr()-based downstream cache.

Cache invalidation rule
-----------------------
A new entry is created whenever any of the following change:
  - batch size B
  - num query heads Hq (CSR is broadcast across H so this is part of key)
  - sequence length T (kernel tile constraint: T % BLOCK_SIZE == 0)
  - BLOCK_SIZE (module-level constant; included for safety)
  - window_arg (the effective in-kernel window; 0 for dense)
  - device (per-CUDA-device pattern; CPU vs CUDA included)

The cache is bounded (default 32 entries). Reaching the cap evicts the
LRU entry. ``seq_lens`` is included in the cached tuple — by design
``seq_lens`` is a single ``torch.full`` produced by the CSR builders
and depends only on B / T, which are part of the key.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import torch

from .csr import build_dense_causal_csr, build_sliding_window_csr
from .kernel_fwd import BLOCK_SIZE, HEAD_DIM
from .function import FlashAttnHydraFunction, FlashAttnHydraDecodeFunction
from .policy import apply_runtime_policy


_CSR_CACHE_MAX = int(os.environ.get("HYDRA_CSR_CACHE_MAX", "32"))
_CSR_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()


def _get_csr(
    B: int,
    Hq: int,
    T: int,
    window_arg: int,
    device: torch.device,
    sliding_window: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        B,
        Hq,
        T,
        BLOCK_SIZE,
        window_arg,
        sliding_window,
        device.type,
        getattr(device, "index", None),
    )
    hit = _CSR_CACHE.get(key)
    if hit is not None:
        _CSR_CACHE.move_to_end(key)
        return hit

    if window_arg == 0:
        row_ptr, col_idx, seq_lens = build_dense_causal_csr(B, Hq, T, BLOCK_SIZE, device)
    else:
        row_ptr, col_idx, seq_lens = build_sliding_window_csr(
            window_arg, T, BLOCK_SIZE, B, Hq, device
        )
    _CSR_CACHE[key] = (row_ptr, col_idx, seq_lens)
    while len(_CSR_CACHE) > _CSR_CACHE_MAX:
        _CSR_CACHE.popitem(last=False)
    return row_ptr, col_idx, seq_lens


def hydra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = True,
    sliding_window: int | None = None,
    policy_layer_idx: int | None = None,
    precision: str = "high",
) -> torch.Tensor:
    """Blackwell-tuned causal FlashAttention with GQA and optional sliding window.

    Same semantics as ``hydra.hydra``.
    """
    if precision not in {"high", "fast"}:
        raise ValueError(f"precision must be 'high' or 'fast', got {precision!r}")
    if precision != "high":
        raise NotImplementedError("precision='fast' is not wired in this extracted Hydra API yet")
    if not is_causal:
        raise NotImplementedError("non-causal attention is not supported in this version")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError(f"q/k/v must be bf16; got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"q/k/v must be 4D (B, H, T, D); got {q.shape}, {k.shape}, {v.shape}")

    B, Hq, T, D = q.shape
    if D != HEAD_DIM:
        raise ValueError(f"head_dim={D} must equal HEAD_DIM={HEAD_DIM}")
    Hkv = k.shape[1]
    if Hq % Hkv != 0:
        raise ValueError(f"H_q={Hq} must be a multiple of H_kv={Hkv} (GQA constraint)")

    decision = apply_runtime_policy(
        q, k, v,
        sliding_window=sliding_window,
        block_size=BLOCK_SIZE,
        head_dim=HEAD_DIM,
        layer_idx=policy_layer_idx,
    )
    sliding_window = decision.effective_window

    # Decode-step specialization: T_q == 1 with arbitrary T_kv. The decode
    # kernel does not require T_kv % BLOCK_SIZE == 0 and has no Q-blocking,
    # so it services HF generation's per-token call without the eager fallback.
    if T == 1:
        if sliding_window is not None and sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {sliding_window}")
        window_arg = 0 if sliding_window is None else sliding_window
        return FlashAttnHydraDecodeFunction.apply(q, k, v, window_arg)

    if T % BLOCK_SIZE != 0:
        raise ValueError(f"T={T} must be a multiple of BLOCK_SIZE={BLOCK_SIZE}")

    if sliding_window is None:
        window_arg = 0
    else:
        if sliding_window <= 0:
            raise ValueError(f"sliding_window must be positive, got {sliding_window}")
        # When the window covers the full sequence, this degrades to dense causal;
        # skip the kernel-side masking work in that case.
        if sliding_window >= T:
            window_arg = 0
        else:
            window_arg = sliding_window

    row_ptr, col_idx, seq_lens = _get_csr(B, Hq, T, window_arg, q.device, sliding_window)

    return FlashAttnHydraFunction.apply(q, k, v, row_ptr, col_idx, seq_lens, window_arg)


def csr_cache_clear() -> None:
    """Drop all cached (row_ptr, col_idx, seq_lens) entries."""
    _CSR_CACHE.clear()


def csr_cache_info() -> dict:
    return {"size": len(_CSR_CACHE), "max": _CSR_CACHE_MAX}


# Compatibility aliases while the extraction is still being consolidated.
hydra_attention = hydra
flash_attn_blackwell = hydra
