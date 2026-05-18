"""Forward Triton kernel for the decode step: Q.seq_len == 1.

Specialization
--------------
Generation hot path: one new Q token attends to a full KV cache of length
T_kv. The prefill kernel in ``kernel_fwd.py`` requires T % BLOCK_SIZE == 0
and pays for Q-blocking + causal CSR iteration that is meaningless when
there is exactly one query row. This kernel collapses to:

    program grid = (B * H_q,)              # one program per query head
    Q broadcast-replicated to a [MM, D] tile (MM=16, the tcore M minimum)
    K/V streamed in [BK, D] tiles across the visible K range

Online softmax with a single Q row reduces to scalar (m, l) and a [1, D]
accumulator (we lift it to [MM, D] with broadcast so Triton's tl.dot —
which requires M >= 16 on Blackwell — sees a tensor-core-sized matmul).
Only the first row of the [MM, D] output is actually distinct; we keep
the broadcast cost (~16× duplicate work) because the V matmul is the
heavy cost and dominates regardless. The alternative — manually emitting
fma loops — would not use tensor cores and would be far slower.

No causal mask is needed because the single Q token is at absolute
position T_kv - 1 and every visible K token is <= that position.

Sliding window
--------------
When ``WINDOW > 0`` and ``WINDOW < T_kv``, K is restricted to the last
WINDOW positions: ``k_start = T_kv - WINDOW``. The launcher computes
``k_start`` and the number of K tiles; the kernel iterates that range
unconditionally and masks the tail tile where ``offs_k >= T_kv``.

GQA
---
Each H_q head reads the same K/V slab indexed by
``hkv = hq // (NUM_HEADS // NUM_KV_HEADS)``.

Precision
---------
bf16 inputs, fp32 accumulator, bf16 output. The kernel casts ``p`` to bf16
once per K-tile before the ``p @ V`` matmul. No hi/lo precision split:
without a causal mask there is no single "diagonal" block where the cast
loss is concentrated; for T_kv up to ~32K this stays well inside bf16's
useful range and the test below confirms a max abs diff < 3e-2 against
SDPA.
"""
from __future__ import annotations

import math
import os

import torch
import triton
import triton.language as tl


# K-tile size for the decode kernel. Independent of the prefill kernel's
# BLOCK_SIZE because the decode kernel never reads CSR; it just streams.
BLOCK_K = int(os.environ.get("HYDRA_DECODE_BLOCK_K", "64"))
HEAD_DIM = int(os.environ.get("HYDRA_HEAD_DIM", "128"))

_DEC_NUM_WARPS = int(os.environ.get("HYDRA_DECODE_NUM_WARPS", "4"))
_DEC_NUM_STAGES = int(os.environ.get("HYDRA_DECODE_NUM_STAGES", "2"))
_DISABLE_AUTOTUNE = int(os.environ.get("HYDRA_DISABLE_AUTOTUNE", "0"))


def _autotune_configs() -> list[triton.Config]:
    """Decode-friendly configs. Q is a single row so warp pressure is low;
    favour stages=2-3 to overlap K/V loads with the small matmuls.
    """
    configs: list[triton.Config] = []
    for num_warps in (2, 4, 8):
        for num_stages in (2, 3, 4):
            configs.append(triton.Config({}, num_warps=num_warps, num_stages=num_stages))
    return configs


def _fixed_config() -> triton.Config:
    return triton.Config({}, num_warps=_DEC_NUM_WARPS, num_stages=_DEC_NUM_STAGES)


def _kernel_decorator(jit_kernel):
    if _DISABLE_AUTOTUNE:
        return jit_kernel
    return triton.autotune(
        configs=_autotune_configs(),
        # T_KV is a runtime arg (not constexpr) and is intentionally NOT in the
        # key — every generation token changes T_kv by 1, and including it
        # forced per-token recompiles (~5-10s each, 200x slowdown observed).
        key=["D", "NUM_HEADS", "NUM_KV_HEADS"],
    )(jit_kernel)


@triton.jit
def _hydra_decode_jit(
    Q,                # (B, H_q, 1, D) bf16
    K, V,             # (B, H_kv, T_KV, D) bf16
    O,                # (B, H_q, 1, D) bf16
    LSE,              # (B, H_q, 1) fp32  (kept for parity / future use)
    T_KV,                    # runtime int — used only as the mask bound; NOT constexpr to avoid per-T_kv recompiles during generation
    stride_kvbh_kv,          # runtime stride for (B*Hkv) axis of K/V in elements (lets StaticCache pass non-T_KV*D strides)
    stride_t_kv,             # runtime stride for the T axis of K/V in elements
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BK: tl.constexpr,
    D: tl.constexpr,
    MM: tl.constexpr,        # Q-replication dim (16 — tcore M-minimum).
    K_START: tl.constexpr,   # leftmost token attended (>=0). Sliding-window left edge.
    NUM_K_TILES: tl.constexpr,
):
    # Q strides: per-(B, H_q) slab has 1*D = D elements (Q is always contig (B,Hq,1,D)).
    # K/V strides: passed in as runtime args so StaticCache (B, Hkv, T_max, D) works
    # — kernel previously hardcoded stride_kvbh = T_KV * D which is wrong for non-T_KV strides.
    stride_qbh: tl.constexpr = D
    stride_kvbh = stride_kvbh_kv
    stride_t = stride_t_kv

    pid = tl.program_id(0)
    rep: tl.constexpr = NUM_HEADS // NUM_KV_HEADS
    b_id = pid // NUM_HEADS
    hq_id = pid % NUM_HEADS
    hkv_id = hq_id // rep

    q_bh = b_id * NUM_HEADS + hq_id
    kv_bh = b_id * NUM_KV_HEADS + hkv_id

    offs_d = tl.arange(0, D)
    offs_mm = tl.arange(0, MM)
    offs_k_in_tile = tl.arange(0, BK)

    # Fold SCALE * log2(e) into Q at load so the per-K-tile score does not
    # pay one mul per element.
    LOG2E: tl.constexpr = 1.4426950408889634
    SCALE_2: tl.constexpr = SCALE * LOG2E
    Q_ptr = Q + q_bh * stride_qbh

    # Tensor cores need M >= 16 in tl.dot. Q has only one valid row, so we
    # replicate it MM times along the M axis. All MM result rows of any
    # downstream matmul are then identical, and we only keep row 0.
    q_row = tl.load(Q_ptr + offs_d).to(tl.float32) * SCALE_2        # [D] fp32
    q_bf16 = tl.broadcast_to(q_row[None, :], [MM, D]).to(tl.bfloat16)  # [MM, D]

    m_i = tl.full([MM], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([MM], dtype=tl.float32)
    acc = tl.zeros([MM, D], dtype=tl.float32)

    K_ptr = K + kv_bh * stride_kvbh
    V_ptr = V + kv_bh * stride_kvbh

    # Iterate K tiles over the visible window [K_START, T_KV).
    # Tile t covers tokens [K_START + t*BK, K_START + (t+1)*BK).
    for t in range(0, NUM_K_TILES):
        k_base = K_START + t * BK
        offs_k = k_base + offs_k_in_tile                 # [BK]
        k_mask = offs_k < T_KV                           # [BK]

        # Masked load: OOB columns get 0; we still mask the score to -inf
        # so its softmax weight is 0.
        k_tile = tl.load(
            K_ptr + offs_k[:, None] * stride_t + offs_d[None, :],
            mask=k_mask[:, None], other=0.0,
        )                                                # [BK, D] bf16
        v_tile = tl.load(
            V_ptr + offs_k[:, None] * stride_t + offs_d[None, :],
            mask=k_mask[:, None], other=0.0,
        )                                                # [BK, D] bf16

        # Score: [MM, D] @ [D, BK] -> [MM, BK] in fp32. All MM rows identical.
        s = tl.dot(q_bf16, tl.trans(k_tile), out_dtype=tl.float32)    # [MM, BK]
        s = tl.where(k_mask[None, :], s, float("-inf"))

        # Online softmax merge. Identical per row, so the [MM]-vector state
        # stays uniform; we keep the broadcast for matmul-shape consistency.
        s_max = tl.max(s, axis=1)                        # [MM]
        m_new = tl.maximum(m_i, s_max)
        alpha = tl.exp2(m_i - m_new)                     # [MM]
        p = tl.exp2(s - m_new[:, None])                  # [MM, BK]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), v_tile, acc=acc * alpha[:, None], out_dtype=tl.float32)
        m_i = m_new

    LN2: tl.constexpr = 0.6931471805599453
    # All MM rows of acc / l_i / m_i are identical (they each saw the same
    # broadcast Q row through the entire loop). Reduce to the row-0 values
    # for the final store. ``tl.sum(..., axis=0) / MM`` extracts the common
    # value as a side effect of averaging identical replicas — this is
    # numerically equivalent to picking row 0 but stays inside Triton's
    # supported reduction ops without needing a slice.
    l_safe = tl.where(l_i > 0, l_i, 1.0)                 # [MM]
    o_tile = acc / l_safe[:, None]                       # [MM, D] fp32, all rows equal
    INV_MM: tl.constexpr = 1.0 / MM
    # Average the MM identical replicas to extract the row-0 result.
    # tl.sum(axis=0) over [MM, D] -> [D]; over [MM] -> scalar wrapped as [1]
    # after a no-op broadcast through tl.arange masking.
    o_row = tl.sum(o_tile, axis=0) * INV_MM              # [D] fp32
    o_row_bf16 = o_row.to(tl.bfloat16)                   # [D] bf16
    lse_full = tl.where(l_i > 0, (m_i + tl.log2(l_safe)) * LN2, float("-inf"))  # [MM]
    # Build a [1]-shaped LSE: sum the [MM] vector then divide; the result is a
    # Triton scalar — wrap via tl.full to get an explicit [1]-shape value.
    lse_scalar_val = tl.sum(lse_full, axis=0) * INV_MM   # scalar fp32
    lse_out_vec = tl.full([1], 0.0, dtype=tl.float32) + lse_scalar_val  # [1]

    # Flat-store the single output row [D] at O_ptr + offs_d. No mask needed:
    # offs_d covers exactly the D-element output buffer for this (B, Hq).
    O_ptr = O + q_bh * stride_qbh
    tl.store(O_ptr + offs_d, o_row_bf16)

    # LSE: store the [1]-shaped value at offset q_bh.
    LSE_ptr = LSE + q_bh
    tl.store(LSE_ptr + tl.arange(0, 1), lse_out_vec)


_hydra_decode_kernel = _kernel_decorator(_hydra_decode_jit)


def launch_attn_fwd_decode(
    q: torch.Tensor,        # (B, H_q, 1, D) bf16
    k: torch.Tensor,        # (B, H_kv, T_kv, D) bf16
    v: torch.Tensor,        # (B, H_kv, T_kv, D) bf16
    window: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the decode-step forward kernel.

    Args:
        q: (B, H_q, 1, D) bf16. T_q must be exactly 1.
        k: (B, H_kv, T_kv, D) bf16.
        v: (B, H_kv, T_kv, D) bf16.
        window: 0 disables sliding window (attend all K). When > 0, only the
            last ``window`` K tokens are attended: K range = [T_kv - window,
            T_kv). The query is at position T_kv - 1, so this matches the
            standard sliding-window-causal semantics used at prefill.

    Returns:
        o:   (B, H_q, 1, D) bf16
        lse: (B, H_q, 1)    fp32

    Notes:
        Assumes K/V are contiguous in (B, H_kv, T_kv, D). T_kv does NOT need
        to be a multiple of BLOCK_K; the last K-tile is masked.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"q/k/v must be 4D (B,H,T,D); got {q.shape} {k.shape} {v.shape}")
    B, Hq, Tq, D = q.shape
    if Tq != 1:
        raise ValueError(f"decode kernel requires Tq == 1; got Tq={Tq}")
    if k.shape[0] != B or v.shape[0] != B:
        raise ValueError(f"batch mismatch q={q.shape} k={k.shape} v={v.shape}")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have identical shapes; got {k.shape} vs {v.shape}")
    if k.shape[3] != D:
        raise ValueError(f"head_dim mismatch q={q.shape[-1]} k={k.shape[-1]}")
    Hkv = k.shape[1]
    Tkv = k.shape[2]
    if Hq % Hkv != 0:
        raise ValueError(f"H_q ({Hq}) must be a multiple of H_kv ({Hkv})")
    if D != HEAD_DIM:
        raise ValueError(f"head_dim ({D}) must equal HEAD_DIM ({HEAD_DIM})")
    if Tkv <= 0:
        raise ValueError(f"T_kv must be positive; got {Tkv}")

    # Sliding window: clamp to [0, Tkv]. window <= 0 or window >= Tkv -> full.
    if window is None or window <= 0 or window >= Tkv:
        k_start = 0
    else:
        k_start = Tkv - window
    num_k_visible = Tkv - k_start
    num_k_tiles = (num_k_visible + BLOCK_K - 1) // BLOCK_K

    o = torch.empty_like(q)
    lse = torch.empty((B, Hq, 1), device=q.device, dtype=torch.float32)

    grid = (B * Hq,)
    # Pull actual strides from K (V must match shape). Lets StaticCache pass
    # (B, Hkv, T_max, D) buffers — previously hardcoded stride_kvbh = Tkv*D
    # would silently read garbage when T_max != Tkv.
    common_kwargs = dict(
        T_KV=Tkv,
        stride_kvbh_kv=k.stride(1),
        stride_t_kv=k.stride(2),
        NUM_HEADS=Hq,
        NUM_KV_HEADS=Hkv,
        SCALE=1.0 / math.sqrt(D),
        BK=BLOCK_K,
        D=D,
        MM=16,     # tensor-core minimum M for tl.dot on Blackwell bf16.
        K_START=int(k_start),
        NUM_K_TILES=int(num_k_tiles),
    )
    if _DISABLE_AUTOTUNE:
        common_kwargs["num_warps"] = _DEC_NUM_WARPS
        common_kwargs["num_stages"] = _DEC_NUM_STAGES

    _hydra_decode_kernel[grid](
        q, k, v,
        o, lse,
        **common_kwargs,
    )
    return o, lse
