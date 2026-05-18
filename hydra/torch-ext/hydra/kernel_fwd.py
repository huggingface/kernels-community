"""Cross-arch-tuned forward kernel.

This module contains the active forward Triton kernel used by the extracted
Hydra package. The JIT kernel body and launcher descend from the iter3
research kernel, with the cross-architecture tuning hooks kept explicit near
the top of the file.

Why
---
The iter3 autotune sweep is a 9-cell cartesian
``(num_warps ∈ {2,4,8}) × (num_stages ∈ {1,2,3})`` over a fixed
``BLOCK_SIZE = 64``. Tuning at import time per
``torch.cuda.get_device_capability()`` lets us:

1. Lower default ``BLOCK_SIZE`` from 64 → 32 on every measured arch
   (sm_86/89/120/121). Cross-arch sweep at T=4096 dense fwdbwd found
   BS=32 winning on every architecture we measured. The smaller K/V
   tile reduces shared-memory pressure and lets more CTAs run
   concurrently per SM, which dominates on Ampere/Ada cards
   (~100 KB smem/SM cap) and is neutral-to-positive on Blackwell.

2. Replace the 9-cell autotune sweep with a 1-3 cell list keyed on
   compute capability. For arches we've measured we ship the
   sweep-winning ``(num_warps, num_stages)`` plus one or two
   safe-fallback configs (so the autotuner can't get stuck if a
   particular load pattern at runtime regresses on the headline cell).
   For arches we have NOT measured (anything not in our table) we
   fall back to the original 9-cell sweep so the autotuner can find
   a good config on first use.

Both changes preserve the launcher API and the kernel signature.
No correctness change: ``BS`` is still a constexpr; the kernel just
gets specialised at a different value.

Per-arch defaults are provisional tuning seeds, not benchmark claims. They
control autotune order and can be overridden with the environment variables
below. Submission-facing benchmark claims must come from checked-in artifacts,
not this header.

Do not cite the defaults below as evidence of speedup or hardware coverage.

Override hooks
--------------
Three env vars short-circuit everything:
  - HYDRA_BLOCK_SIZE - wins over the per-arch default
  - HYDRA_DISABLE_AUTOTUNE=1 + HYDRA_NUM_WARPS
    + HYDRA_NUM_STAGES — wins over autotune
  - HYDRA_AUTOTUNE_MODE=full — forces the 9-cell sweep
    even on a known arch (useful for re-tuning new wheels of triton).
"""
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl


# ---------------- Per-arch defaults --------------------------------------

# Per-(major,minor) compute-capability -> (BLOCK_SIZE, [(num_warps, num_stages), ...])
#
# The autotune list is ordered with the SWEEP-WINNING config first so that
# the autotuner picks it on the first launch (which is also the only launch
# in many production deploys: the cache key includes T_MAX/D so a stable
# shape produces exactly one autotune cycle).
#
# Tail entries are conservative fallbacks: they exist so the autotuner has
# at least one safe config to fall back to if a specific runtime workload
# regresses on the headline cell (e.g. a different T that we did not sweep).
_PER_ARCH: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]] = {
    # GB10 (Blackwell sm_121).
    (12, 1): (32, [(2, 3), (4, 2)]),
    # Pro 6000 (Blackwell sm_120). NOT MEASURED in iter; default to the
    # sm_121 pick — they share most of the relevant resources. Adjust
    # after running the sweep on a free Pro 6000.
    (12, 0): (32, [(2, 3), (4, 2)]),
    # Ada (sm_89): 4080, 4070 Ti.
    (8, 9): (32, [(4, 1), (4, 2)]),
    # Ampere consumer (sm_86): 3090, 3080, 3060.
    (8, 6): (32, [(2, 1), (4, 2)]),
}

# The original 9-cell sweep — used when no per-arch entry exists, and also
# when the user opts in via HYDRA_AUTOTUNE_MODE=full.
_FULL_SWEEP: list[tuple[int, int]] = [
    (W, S) for W in (2, 4, 8) for S in (1, 2, 3)
]


def _device_capability_or_none() -> tuple[int, int] | None:
    """Return ``(major, minor)`` for cuda:0, or ``None`` if no CUDA."""
    try:
        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability(0))  # type: ignore[return-value]
    except Exception:
        pass
    return None


def _default_block_size_from_arch() -> int:
    """Pick BLOCK_SIZE based on the current CUDA device's compute capability.

    Fall back to 64 (iter3's value) if we don't have a measured entry, so
    no measured arch regresses below its iter3 baseline.
    """
    cc = _device_capability_or_none()
    if cc is not None and cc in _PER_ARCH:
        return _PER_ARCH[cc][0]
    return 64


def _arch_aware_autotune_configs() -> list[triton.Config]:
    """Per-arch shrunk autotune list, or the full 9-cell sweep if unknown."""
    if os.environ.get("HYDRA_AUTOTUNE_MODE", "").lower() == "full":
        ws_pairs = _FULL_SWEEP
    else:
        cc = _device_capability_or_none()
        if cc is not None and cc in _PER_ARCH:
            ws_pairs = _PER_ARCH[cc][1]
        else:
            ws_pairs = _FULL_SWEEP
    return [triton.Config({}, num_warps=W, num_stages=S) for (W, S) in ws_pairs]


# Env var override wins over per-arch default (preserves the iter3 escape hatch).
BLOCK_SIZE = int(os.environ.get("HYDRA_BLOCK_SIZE", str(_default_block_size_from_arch())))
HEAD_DIM = int(os.environ.get("HYDRA_HEAD_DIM", "128"))

_FWD_NUM_WARPS = int(os.environ.get("HYDRA_NUM_WARPS", "4"))
_FWD_NUM_STAGES = int(os.environ.get("HYDRA_NUM_STAGES", "2"))
_DISABLE_AUTOTUNE = int(os.environ.get("HYDRA_DISABLE_AUTOTUNE", "0"))


def _autotune_configs() -> list[triton.Config]:
    return _arch_aware_autotune_configs()


def _kernel_decorator(jit_kernel):
    if _DISABLE_AUTOTUNE:
        return jit_kernel
    return triton.autotune(
        configs=_autotune_configs(),
        key=["T_MAX", "D", "NUM_HEADS", "NUM_KV_HEADS", "ASSUME_FULL"],
    )(jit_kernel)


@triton.jit
def _hydra_fwd_jit(
    Q, K, V,
    O, LSE,
    RowPtr, ColIdx, SeqLens,
    stride_cih,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    CONTIGUOUS_OFFDIAG: tl.constexpr = 0,
    ASSUME_FULL: tl.constexpr = 0,
):
    stride_h: tl.constexpr = T_MAX * D
    stride_t: tl.constexpr = D
    stride_lh: tl.constexpr = T_MAX
    stride_rph: tl.constexpr = (T_MAX // BS) + 1
    NUM_Q_BLOCKS: tl.constexpr = T_MAX // BS

    pid = tl.program_id(0)
    bh_id = pid // NUM_Q_BLOCKS
    q_block_id = pid % NUM_Q_BLOCKS

    rep: tl.constexpr = NUM_HEADS // NUM_KV_HEADS
    b_id = bh_id // NUM_HEADS
    hq_id = bh_id % NUM_HEADS
    hkv_id = hq_id // rep
    kv_bh_id = b_id * NUM_KV_HEADS + hkv_id

    if ASSUME_FULL:
        seq_len = T_MAX  # constexpr-friendly: no SeqLens load, no early-return
    else:
        seq_len = tl.load(SeqLens + b_id)

    q_start = q_block_id * BS

    offs_tok = q_start + tl.arange(0, BS)
    offs_d = tl.arange(0, D)

    if ASSUME_FULL:
        pass  # no early-return path needed
    else:
        if q_start >= seq_len:
            O_ptr_e = O + bh_id * stride_h
            tl.store(O_ptr_e + offs_tok[:, None] * stride_t + offs_d[None, :],
                     tl.zeros([BS, D], dtype=tl.bfloat16))
            LSE_ptr_e = LSE + bh_id * stride_lh
            tl.store(LSE_ptr_e + offs_tok, tl.full([BS], float("-inf"), dtype=tl.float32))
            return

    LOG2E: tl.constexpr = 1.4426950408889634
    SCALE_2: tl.constexpr = SCALE * LOG2E
    Q_ptr = Q + bh_id * stride_h
    q_bf16 = (tl.load(Q_ptr + offs_tok[:, None] * stride_t + offs_d[None, :]).to(tl.float32) * SCALE_2).to(tl.bfloat16)

    m_i = tl.full([BS], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BS], dtype=tl.float32)
    acc = tl.zeros([BS, D], dtype=tl.float32)

    rp_base = RowPtr + bh_id * stride_rph + q_block_id
    ci_lo = tl.load(rp_base)
    ci_hi = tl.load(rp_base + 1)

    K_ptr = K + kv_bh_id * stride_h
    V_ptr = V + kv_bh_id * stride_h
    CI_ptr = ColIdx + bh_id * stride_cih

    offs_d_arange = tl.arange(0, BS)

    if ci_hi > ci_lo:
        k_start_d = q_block_id * BS
        offs_k_d = k_start_d + offs_d_arange

        if ASSUME_FULL:
            # No boundary branch — full unmasked diagonal load + causal mask.
            k_bf16_d = tl.load(K_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
            v_bf16_d = tl.load(V_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
            s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32)
            causal = offs_tok[:, None] >= offs_k_d[None, :]
            if WINDOW > 0:
                in_window = (offs_tok[:, None] - offs_k_d[None, :]) < WINDOW
                s_d = tl.where(causal & in_window, s_d, float("-inf"))
            else:
                s_d = tl.where(causal, s_d, float("-inf"))
        else:
            boundary = (q_start + BS) > seq_len
            if boundary:
                k_mask = offs_k_d < seq_len
                k_bf16_d = tl.load(K_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :],
                                   mask=k_mask[:, None], other=0.0)
                v_bf16_d = tl.load(V_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :],
                                   mask=k_mask[:, None], other=0.0)
                s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32)
                causal = offs_tok[:, None] >= offs_k_d[None, :]
                allowed = causal & k_mask[None, :]
                if WINDOW > 0:
                    in_window = (offs_tok[:, None] - offs_k_d[None, :]) < WINDOW
                    allowed = allowed & in_window
                s_d = tl.where(allowed, s_d, float("-inf"))
            else:
                k_bf16_d = tl.load(K_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
                v_bf16_d = tl.load(V_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
                s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32)
                causal = offs_tok[:, None] >= offs_k_d[None, :]
                if WINDOW > 0:
                    in_window = (offs_tok[:, None] - offs_k_d[None, :]) < WINDOW
                    s_d = tl.where(causal & in_window, s_d, float("-inf"))
                else:
                    s_d = tl.where(causal, s_d, float("-inf"))

        m_i = tl.max(s_d, axis=1)
        p_d = tl.exp2(s_d - m_i[:, None])
        l_i = tl.sum(p_d, axis=1)
        acc = tl.dot(p_d.to(tl.bfloat16), v_bf16_d, out_dtype=tl.float32)

    # Off-diag loop unchanged (no boundary semantics here in either mode).
    if CONTIGUOUS_OFFDIAG:
        k_block_id_start = tl.load(CI_ptr + ci_lo)
        for ci in range(ci_lo, ci_hi - 1):
            k_block_id = k_block_id_start + (ci - ci_lo)
            k_start = k_block_id * BS
            offs_k = k_start + offs_d_arange
            k_bf16 = tl.load(K_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            v_bf16 = tl.load(V_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            s = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32)
            if WINDOW > 0:
                in_window = (offs_tok[:, None] - offs_k[None, :]) < WINDOW
                s = tl.where(in_window, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = tl.dot(p.to(tl.bfloat16), v_bf16, acc=acc * alpha[:, None], out_dtype=tl.float32)
            m_i = m_new
    else:
        for ci in range(ci_lo, ci_hi - 1):
            k_block_id = tl.load(CI_ptr + ci)
            k_start = k_block_id * BS
            offs_k = k_start + offs_d_arange
            k_bf16 = tl.load(K_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            v_bf16 = tl.load(V_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            s = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32)
            if WINDOW > 0:
                in_window = (offs_tok[:, None] - offs_k[None, :]) < WINDOW
                s = tl.where(in_window, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = tl.dot(p.to(tl.bfloat16), v_bf16, acc=acc * alpha[:, None], out_dtype=tl.float32)
            m_i = m_new

    LN2: tl.constexpr = 0.6931471805599453
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    if ASSUME_FULL:
        o_tile = acc / l_safe[:, None]
        lse_out = (m_i + tl.log2(l_safe)) * LN2
    else:
        q_mask = offs_tok < seq_len
        o_tile = tl.where(q_mask[:, None], acc / l_safe[:, None], 0.0)
        lse_out = tl.where(q_mask, (m_i + tl.log2(l_safe)) * LN2, float("-inf"))

    O_ptr = O + bh_id * stride_h
    tl.store(O_ptr + offs_tok[:, None] * stride_t + offs_d[None, :], o_tile.to(tl.bfloat16))
    LSE_ptr = LSE + bh_id * stride_lh
    tl.store(LSE_ptr + offs_tok, lse_out)


_hydra_fwd_kernel = _kernel_decorator(_hydra_fwd_jit)


def launch_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    window: int = 0,
    contiguous_offdiag: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"q/k/v must be 4D; got {q.shape} {k.shape} {v.shape}")
    batch_size, num_heads, t_max, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if t_max % BLOCK_SIZE != 0:
        raise ValueError(f"T ({t_max}) must be multiple of BLOCK_SIZE ({BLOCK_SIZE})")
    if head_dim != HEAD_DIM:
        raise ValueError(f"head_dim ({head_dim}) must equal HEAD_DIM ({HEAD_DIM})")

    batch_heads = batch_size * num_heads
    num_q_blocks = t_max // BLOCK_SIZE
    o = torch.empty_like(q)
    lse_3d = torch.empty((batch_size, num_heads, t_max), device=q.device, dtype=torch.float32)
    lse_2d = lse_3d.view(batch_heads, t_max)
    cih = col_idx.shape[2] if col_idx.ndim == 3 else col_idx.shape[1]
    grid = (num_q_blocks * batch_heads,)

    # ASSUME_FULL is safe iff EVERY seq_len equals t_max. Cheap GPU-resident
    # check (no D2H sync if we just compare with a fused all-equal kernel,
    # but for now use a tiny .item() — it's a single int and seq_lens lives
    # in HBM near the launcher already).
    if seq_lens.numel() == 1:
        assume_full = int(seq_lens.item()) == t_max
    else:
        # Multi-batch: check min==max==t_max.
        sl_min = int(seq_lens.min().item())
        sl_max = int(seq_lens.max().item())
        assume_full = (sl_min == t_max) and (sl_max == t_max)

    common_kwargs = dict(
        T_MAX=t_max, NUM_HEADS=num_heads, NUM_KV_HEADS=num_kv_heads,
        SCALE=1.0 / math.sqrt(head_dim), BS=BLOCK_SIZE, D=head_dim,
        WINDOW=int(window),
        CONTIGUOUS_OFFDIAG=1 if contiguous_offdiag else 0,
        ASSUME_FULL=1 if assume_full else 0,
    )
    if _DISABLE_AUTOTUNE:
        common_kwargs["num_warps"] = _FWD_NUM_WARPS
        common_kwargs["num_stages"] = _FWD_NUM_STAGES

    _hydra_fwd_kernel[grid](
        q, k, v, o, lse_2d,
        row_ptr, col_idx, seq_lens, cih,
        **common_kwargs,
    )
    return o, lse_3d
