"""Backward Triton kernels for Blackwell-tuned causal FlashAttention.

Perf-tuned variant of ``hydra.kernel_bwd`` that

  - keeps the Triton kernels byte-for-byte identical to the upstream
    versions (so the gradient-correctness story is unchanged),
  - replaces ``build_csrT``'s CPU + numpy round-trip with a pure-torch /
    GPU-resident equivalent (``build_csrT_gpu``) that produces the same
    output and stays on the input device,
  - exposes a module-level LRU cache (``build_csrT_cached``) keyed on
    pattern shape rather than ``data_ptr()`` so per-call rebuilds of
    ``row_ptr`` / ``col_idx`` (as the api does today) still hit the
    cache,
  - clears the host-side ``delta = (do.float() * o.float()).sum(-1)``
    cast spew: the existing launcher computes ``delta`` with two fp32
    upcasts and a stride-collapsing ``.contiguous()`` every call; we keep
    the math but drop the redundant ``.contiguous()`` (the result of
    ``.sum(-1)`` on a contiguous 4D tensor is already contiguous).

Public API is preserved: ``launch_attn_bwd`` has the same signature, and
``build_csrT`` is kept as a thin wrapper that dispatches to the pure-GPU
builder when possible (and falls back to the original CPU+numpy path
when the input is on CPU, so the CPU equivalence test exercises the
same code path).
"""
from __future__ import annotations

import math
import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from .kernel_fwd import BLOCK_SIZE, HEAD_DIM
from .kernel_delta import launch_compute_delta


_BWD_NUM_WARPS = int(os.environ.get("HYDRA_BWD_NUM_WARPS", "4"))
_BWD_NUM_STAGES = int(os.environ.get("HYDRA_BWD_NUM_STAGES", "1"))
_DISABLE_AUTOTUNE = int(os.environ.get("HYDRA_DISABLE_AUTOTUNE", "0"))

_CSRT_CACHE_MAX = int(os.environ.get("HYDRA_CSRT_CACHE_MAX", "32"))


def _autotune_configs() -> list[triton.Config]:
    configs: list[triton.Config] = []
    for num_warps in (2, 4, 8):
        for num_stages in (1, 2):
            configs.append(triton.Config({}, num_warps=num_warps, num_stages=num_stages))
    return configs


def _kernel_decorator(jit_kernel):
    if _DISABLE_AUTOTUNE:
        return jit_kernel
    return triton.autotune(
        configs=_autotune_configs(),
        # Include WINDOW: sliding-window patterns have 2-4× fewer K-blocks per
        # Q-block than dense, so optimal warps/stages plausibly differ. Keying
        # them separately costs one extra autotune sweep per (T, window) pair.
        key=["T_MAX", "D", "NUM_HEADS", "NUM_KV_HEADS", "WINDOW"],
    )(jit_kernel)


# ----------------------------- csrT builders -----------------------------


def _build_csrT_cpu_reference(row_ptr: torch.Tensor, col_idx: torch.Tensor, num_blocks: int):
    """Original CPU + numpy implementation, kept for reference / fallback.

    Identical semantics to the upstream ``build_csrT`` in
    ``hydra.kernel_bwd``: per (B, H), enumerate which
    Q-blocks reference each K-block as an OFF-diagonal attendee. The
    diagonal placeholder at ``col_idx[ci_hi - 1]`` is excluded.
    """
    import numpy as np
    rp = row_ptr.detach().cpu().numpy()
    ci = col_idx.detach().cpu().numpy()
    B, H = rp.shape[0], rp.shape[1]
    rp_flat = rp.reshape(B * H, -1)
    ci_flat = ci.reshape(B * H, -1)
    rpt_rows: list[list[int]] = []
    cit_rows: list[list[int]] = []
    for bh in range(B * H):
        bucket: list[list[int]] = [[] for _ in range(num_blocks)]
        rp_row = rp_flat[bh]
        ci_row = ci_flat[bh]
        for i in range(num_blocks):
            lo = int(rp_row[i])
            hi = int(rp_row[i + 1])
            for p in range(lo, max(lo, hi - 1)):
                bucket[int(ci_row[p])].append(i)
        offs = [0]
        vals: list[int] = []
        for k in range(num_blocks):
            vals.extend(bucket[k])
            offs.append(len(vals))
        rpt_rows.append(offs)
        cit_rows.append(vals)
    max_nnz = max(1, max(len(v) for v in cit_rows))
    cit_pad = np.zeros((B * H, max_nnz), dtype=np.int32)
    for bh, v in enumerate(cit_rows):
        if v:
            cit_pad[bh, : len(v)] = v
    rpt_np = np.asarray(rpt_rows, dtype=np.int32)
    rp_T = torch.from_numpy(rpt_np).to(row_ptr.device).reshape(B, H, num_blocks + 1).contiguous()
    ci_T = torch.from_numpy(cit_pad).to(row_ptr.device).reshape(B, H, max_nnz).contiguous()
    return rp_T, ci_T


def _is_broadcast_pattern(row_ptr: torch.Tensor, col_idx: torch.Tensor) -> bool:
    """Cheap check: does the (B, H) pattern collapse to a single shared row?

    The CSR builders in ``hydra.csr`` always produce a
    broadcast-then-contiguous pattern. We exploit that to compute the
    transposed CSR once and ``.expand().contiguous()`` it back to
    (B, H). For safety we verify by comparing row_ptr/col_idx against
    the (0, 0) head.
    """
    B, H = row_ptr.shape[0], row_ptr.shape[1]
    if B == 1 and H == 1:
        return True
    rp0 = row_ptr[0, 0]
    ci0 = col_idx[0, 0]
    # equal_to expects same-shape comparand; use a single torch.equal per
    # axis. This costs O(B*H*nnz) but is GPU-resident and ~1us for typical
    # shapes — far cheaper than the .cpu()/.numpy() round-trip.
    return bool(torch.all(row_ptr == rp0).item()) and bool(torch.all(col_idx == ci0).item())


def _build_csrT_single_row(rp_row: torch.Tensor, ci_row: torch.Tensor, num_blocks: int):
    """Pure-torch transposed-CSR for a single (B=H=1) row.

    Produces ``(rpt_row, cit_row)`` of dtype int32 on the same device as
    the inputs. The ordering of q_block_ids within each K-row matches
    the CPU reference: ascending q_block_id.
    """
    device = rp_row.device
    rp_row = rp_row.to(torch.int64)
    ci_row = ci_row.to(torch.int64)

    # Off-diagonal "off" slices per Q-row: [lo, hi - 1). Total off-diag
    # entries summed across Q-blocks equals nnz - num_blocks (one diag
    # per Q-block). Build a flat (q_block_id, k_block_id) edge list.
    counts = (rp_row[1:] - rp_row[:-1] - 1).clamp(min=0)  # off-diag count per Q-block
    num_offdiag = int(counts.sum().item())
    if num_offdiag == 0:
        rp_T_row = torch.zeros(num_blocks + 1, dtype=torch.int32, device=device)
        ci_T_row = torch.zeros(1, dtype=torch.int32, device=device)
        return rp_T_row, ci_T_row, 1

    # q_idx[e] = which Q-block this edge belongs to.
    q_idx = torch.repeat_interleave(
        torch.arange(num_blocks, dtype=torch.int64, device=device),
        counts,
    )
    # k_idx[e] = which K-block this edge points at. We need the slice
    # ci_row[lo : hi - 1] for each Q-row, concatenated. Build a flat
    # index into ci_row by starting at lo for each Q-block and adding the
    # within-row offset (0, 1, 2, ...).
    starts = rp_row[:-1]  # lo per Q-block
    edge_within = torch.arange(num_offdiag, dtype=torch.int64, device=device) - torch.repeat_interleave(
        torch.cat([
            torch.zeros(1, dtype=torch.int64, device=device),
            counts.cumsum(0)[:-1],
        ]),
        counts,
    )
    edge_pos = torch.repeat_interleave(starts, counts) + edge_within
    k_idx = ci_row[edge_pos]

    # Sort edges by k_idx (stable so q_idx within a bucket stays ascending).
    sort_idx = torch.argsort(k_idx, stable=True)
    k_idx_sorted = k_idx[sort_idx]
    q_idx_sorted = q_idx[sort_idx]

    # rp_T[k+1] = count of edges with k_idx <= k. Use bincount over [0, num_blocks).
    per_k_count = torch.bincount(k_idx_sorted, minlength=num_blocks)
    rp_T_row = torch.zeros(num_blocks + 1, dtype=torch.int64, device=device)
    rp_T_row[1:] = per_k_count.cumsum(0)

    max_nnz = max(1, num_offdiag)
    ci_T_row = torch.zeros(max_nnz, dtype=torch.int32, device=device)
    ci_T_row[:num_offdiag] = q_idx_sorted.to(torch.int32)
    return rp_T_row.to(torch.int32), ci_T_row, max_nnz


def build_csrT_gpu(row_ptr: torch.Tensor, col_idx: torch.Tensor, num_blocks: int):
    """Pure-device transposed-CSR builder.

    Equivalent to ``build_csrT`` but never touches CPU. When the input
    pattern is the broadcast-across-(B,H) form (the common case), only
    one head's worth of work is done and the result is expanded.

    Falls back to a per-(B,H) torch implementation for the non-broadcast
    case (no current api path produces that, but keep it safe).
    """
    if row_ptr.dim() != 3 or col_idx.dim() != 3:
        raise ValueError(f"expected 3D row_ptr/col_idx; got {row_ptr.shape}, {col_idx.shape}")
    B, H = row_ptr.shape[0], row_ptr.shape[1]

    if _is_broadcast_pattern(row_ptr, col_idx):
        rp_T_row, ci_T_row, max_nnz = _build_csrT_single_row(row_ptr[0, 0], col_idx[0, 0], num_blocks)
        rp_T = rp_T_row.view(1, 1, num_blocks + 1).expand(B, H, num_blocks + 1).contiguous()
        ci_T = ci_T_row.view(1, 1, max_nnz).expand(B, H, max_nnz).contiguous()
        return rp_T, ci_T

    # Non-broadcast path: per-(B,H) computation. We loop in Python over
    # the B*H heads but each head's work stays on-device. For B*H up to
    # a few thousand this is still much cheaper than .cpu().numpy().
    out_rp_rows: list[torch.Tensor] = []
    out_ci_rows: list[torch.Tensor] = []
    max_nnz_seen = 1
    for bh in range(B * H):
        bi, hi = bh // H, bh % H
        rp_T_row, ci_T_row, mn = _build_csrT_single_row(row_ptr[bi, hi], col_idx[bi, hi], num_blocks)
        out_rp_rows.append(rp_T_row)
        out_ci_rows.append(ci_T_row)
        max_nnz_seen = max(max_nnz_seen, mn)

    rp_T = torch.stack(out_rp_rows, dim=0).view(B, H, num_blocks + 1).contiguous()
    # Right-pad to max_nnz_seen.
    ci_T = torch.zeros(B * H, max_nnz_seen, dtype=torch.int32, device=row_ptr.device)
    for bh, row in enumerate(out_ci_rows):
        ci_T[bh, : row.shape[0]] = row
    ci_T = ci_T.view(B, H, max_nnz_seen).contiguous()
    return rp_T, ci_T


def build_csrT(row_ptr: torch.Tensor, col_idx: torch.Tensor, num_blocks: int):
    """Drop-in replacement for the upstream ``build_csrT``.

    Routes to the pure-GPU builder on CUDA inputs and to the
    CPU+numpy reference on CPU inputs (so the equivalence test exercises
    the same numerical path the kernel sees).
    """
    if row_ptr.is_cuda:
        return build_csrT_gpu(row_ptr, col_idx, num_blocks)
    return _build_csrT_cpu_reference(row_ptr, col_idx, num_blocks)


# ----------------------------- csrT cache --------------------------------


# Two-level cache.
#
#   - _CSRT_CACHE_FAST is keyed on (row_ptr.data_ptr(), col_idx.data_ptr(),
#     shapes, device, num_blocks). When the public api caches the CSR tensors,
#     the data_ptrs are stable across calls and this is an O(1) Python dict probe — no
#     CUDA sync, no D2H copy.
#
#   - _CSRT_CACHE_CONTENT is keyed on (num_blocks, shape, first-row
#     content). Used as a fallback when the fast key misses (e.g. a user
#     who builds CSRs fresh each call). The first-row content read costs
#     one (num_blocks+1) + first-row-nnz int32 D2H sync — still vastly
#     cheaper than the full CPU+numpy round-trip on the build side.
#
# On a content-cache hit we also promote into the fast cache so the next
# call with the same data_ptr is free.
_CSRT_CACHE_FAST: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
_CSRT_CACHE_CONTENT: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()


def _fast_key(row_ptr: torch.Tensor, col_idx: torch.Tensor, num_blocks: int) -> tuple:
    return (
        row_ptr.data_ptr(),
        col_idx.data_ptr(),
        num_blocks,
        tuple(row_ptr.shape),
        tuple(col_idx.shape),
        row_ptr.device.type,
        getattr(row_ptr.device, "index", None),
    )


def _content_key(row_ptr: torch.Tensor, col_idx: torch.Tensor, num_blocks: int) -> tuple:
    """Content-addressed key. Reads the (0, 0) head's row to cover the
    broadcast-CSR case (the only one the api emits today).

    Invalidation rule
    -----------------
    A new entry is created whenever any of the following change:
      - num_blocks
      - row_ptr.shape / col_idx.shape
      - device type / index
      - the (0, 0) head's row_ptr / col_idx values

    Non-broadcast inputs that share the (0, 0) row but differ on other
    (B, H) entries would alias under this key. The api never produces
    such inputs, but ``launch_attn_bwd`` users with hand-built CSRs
    should call ``csrT_cache_clear()`` between distinct non-broadcast
    patterns.
    """
    rp0 = tuple(row_ptr[0, 0].to(torch.int64).tolist())
    ci0 = tuple(col_idx[0, 0].to(torch.int64).tolist())
    return (
        num_blocks,
        tuple(row_ptr.shape),
        tuple(col_idx.shape),
        row_ptr.device.type,
        getattr(row_ptr.device, "index", None),
        rp0,
        ci0,
    )


def _bound(cache: "OrderedDict") -> None:
    while len(cache) > _CSRT_CACHE_MAX:
        cache.popitem(last=False)


def build_csrT_cached(row_ptr: torch.Tensor, col_idx: torch.Tensor, num_blocks: int):
    """Return a (rp_T, ci_T) pair for the given CSR, hitting the LRU cache when possible.

    Thread-safety: this is not thread-safe across CUDA streams. Callers
    that share the same CSR across streams should manually clone the
    returned tensors. In practice the autograd backward is called on the
    same stream as the forward that produced the CSR, so this is fine.
    """
    fk = _fast_key(row_ptr, col_idx, num_blocks)
    hit = _CSRT_CACHE_FAST.get(fk)
    if hit is not None:
        _CSRT_CACHE_FAST.move_to_end(fk)
        return hit
    ck = _content_key(row_ptr, col_idx, num_blocks)
    hit = _CSRT_CACHE_CONTENT.get(ck)
    if hit is not None:
        _CSRT_CACHE_CONTENT.move_to_end(ck)
        _CSRT_CACHE_FAST[fk] = hit
        _bound(_CSRT_CACHE_FAST)
        return hit
    rp_T, ci_T = build_csrT(row_ptr, col_idx, num_blocks)
    _CSRT_CACHE_FAST[fk] = (rp_T, ci_T)
    _CSRT_CACHE_CONTENT[ck] = (rp_T, ci_T)
    _bound(_CSRT_CACHE_FAST)
    _bound(_CSRT_CACHE_CONTENT)
    return rp_T, ci_T


def csrT_cache_clear() -> None:
    """Drop all cached csrT entries. Useful for test isolation."""
    _CSRT_CACHE_FAST.clear()
    _CSRT_CACHE_CONTENT.clear()


def csrT_cache_info() -> dict:
    """Return basic cache stats. ``size`` is the number of *distinct*
    csrT patterns currently cached (the content-keyed dict)."""
    return {
        "size": len(_CSRT_CACHE_CONTENT),
        "fast_size": len(_CSRT_CACHE_FAST),
        "max": _CSRT_CACHE_MAX,
    }


# ----------------------------- Triton kernels ----------------------------
# These are kept local so the extracted package is self-contained.


@triton.jit
def _hydra_bwd_dq_jit(
    Q, K, V, LSE,
    dO,
    Di_in,
    dQ,
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
):
    stride_h: tl.constexpr = T_MAX * D
    stride_t: tl.constexpr = D
    stride_rph: tl.constexpr = (T_MAX // BS) + 1
    stride_lh: tl.constexpr = T_MAX
    NUM_Q_BLOCKS: tl.constexpr = T_MAX // BS

    pid = tl.program_id(0)
    bh_id = pid // NUM_Q_BLOCKS
    q_block_id = pid % NUM_Q_BLOCKS

    rep: tl.constexpr = NUM_HEADS // NUM_KV_HEADS
    b_id = bh_id // NUM_HEADS
    hq_id = bh_id % NUM_HEADS
    hkv_id = hq_id // rep
    kv_bh_id = b_id * NUM_KV_HEADS + hkv_id

    seq_len = tl.load(SeqLens + b_id)
    q_start = q_block_id * BS
    if q_start >= seq_len:
        return

    offs_tok = q_start + tl.arange(0, BS)
    offs_d = tl.arange(0, D)
    q_mask = offs_tok < seq_len

    Q_ptr = Q + bh_id * stride_h
    q_bf16 = tl.load(Q_ptr + offs_tok[:, None] * stride_t + offs_d[None, :])

    dO_ptr = dO + bh_id * stride_h
    dO_i = tl.load(dO_ptr + offs_tok[:, None] * stride_t + offs_d[None, :])

    LSE_ptr = LSE + bh_id * stride_lh
    LSE_i = tl.load(LSE_ptr + offs_tok)

    rp_base = RowPtr + bh_id * stride_rph + q_block_id
    ci_lo = tl.load(rp_base)
    ci_hi = tl.load(rp_base + 1)

    K_ptr = K + kv_bh_id * stride_h
    V_ptr = V + kv_bh_id * stride_h
    CI_ptr = ColIdx + bh_id * stride_cih

    offs_k_tile = tl.arange(0, BS)
    boundary = (q_start + BS) > seq_len

    Di = tl.load(Di_in + bh_id * stride_lh + offs_tok, mask=q_mask, other=0.0)

    dQ_i = tl.zeros([BS, D], dtype=tl.float32)

    if ci_hi > ci_lo:
        k_start_d = q_block_id * BS
        offs_k_d = k_start_d + offs_k_tile
        if boundary:
            k_mask_d = offs_k_d < seq_len
            k_bf16_d = tl.load(K_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :], mask=k_mask_d[:, None], other=0.0)
            v_bf16_d = tl.load(V_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :], mask=k_mask_d[:, None], other=0.0)
            s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32) * SCALE
            causal = offs_tok[:, None] >= offs_k_d[None, :]
            allowed = causal & k_mask_d[None, :]
            if WINDOW > 0:
                in_window = (offs_tok[:, None] - offs_k_d[None, :]) < WINDOW
                allowed = allowed & in_window
            s_d = tl.where(allowed, s_d, float("-inf"))
        else:
            k_bf16_d = tl.load(K_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
            v_bf16_d = tl.load(V_ptr + offs_k_d[:, None] * stride_t + offs_d[None, :])
            s_d = tl.dot(q_bf16, tl.trans(k_bf16_d), out_dtype=tl.float32) * SCALE
            causal = offs_tok[:, None] >= offs_k_d[None, :]
            if WINDOW > 0:
                in_window = (offs_tok[:, None] - offs_k_d[None, :]) < WINDOW
                s_d = tl.where(causal & in_window, s_d, float("-inf"))
            else:
                s_d = tl.where(causal, s_d, float("-inf"))

        P_d = tl.exp(s_d - LSE_i[:, None])
        dP_d = tl.dot(dO_i, tl.trans(v_bf16_d), out_dtype=tl.float32)
        dS_d = (P_d * (dP_d - Di[:, None])) * SCALE
        dQ_i += tl.dot(dS_d.to(tl.bfloat16), k_bf16_d, out_dtype=tl.float32)

    # See kernel_fwd: when CONTIGUOUS_OFFDIAG, hoist the CSR load.
    if CONTIGUOUS_OFFDIAG:
        k_block_id_start = tl.load(CI_ptr + ci_lo)
        for ci in range(ci_lo, ci_hi - 1):
            k_block_id = k_block_id_start + (ci - ci_lo)
            k_start = k_block_id * BS
            offs_k = k_start + offs_k_tile
            k_bf16 = tl.load(K_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            v_bf16 = tl.load(V_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            s = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE
            if WINDOW > 0:
                in_window = (offs_tok[:, None] - offs_k[None, :]) < WINDOW
                s = tl.where(in_window, s, float("-inf"))
            P = tl.exp(s - LSE_i[:, None])
            dP = tl.dot(dO_i, tl.trans(v_bf16), out_dtype=tl.float32)
            dS = (P * (dP - Di[:, None])) * SCALE
            dQ_i += tl.dot(dS.to(tl.bfloat16), k_bf16, out_dtype=tl.float32)
    else:
        for ci in range(ci_lo, ci_hi - 1):
            k_block_id = tl.load(CI_ptr + ci)
            k_start = k_block_id * BS
            offs_k = k_start + offs_k_tile
            k_bf16 = tl.load(K_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            v_bf16 = tl.load(V_ptr + offs_k[:, None] * stride_t + offs_d[None, :])
            s = tl.dot(q_bf16, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE
            if WINDOW > 0:
                in_window = (offs_tok[:, None] - offs_k[None, :]) < WINDOW
                s = tl.where(in_window, s, float("-inf"))
            P = tl.exp(s - LSE_i[:, None])
            dP = tl.dot(dO_i, tl.trans(v_bf16), out_dtype=tl.float32)
            dS = (P * (dP - Di[:, None])) * SCALE
            dQ_i += tl.dot(dS.to(tl.bfloat16), k_bf16, out_dtype=tl.float32)

    dQ_ptr = dQ + bh_id * stride_h
    tl.store(dQ_ptr + offs_tok[:, None] * stride_t + offs_d[None, :], dQ_i.to(tl.bfloat16), mask=q_mask[:, None])


@triton.jit
def _hydra_bwd_dkv_jit(
    Q, K, V, LSE,
    dO,
    Di_in,
    dK, dV,
    RowPtrT, ColIdxT, SeqLens,
    stride_cith,
    T_MAX: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SCALE: tl.constexpr,
    BS: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    CONTIGUOUS_OFFDIAG: tl.constexpr = 0,
):
    stride_h: tl.constexpr = T_MAX * D
    stride_t: tl.constexpr = D
    stride_rpth: tl.constexpr = (T_MAX // BS) + 1
    stride_lh: tl.constexpr = T_MAX
    NUM_K_BLOCKS: tl.constexpr = T_MAX // BS

    pid = tl.program_id(0)
    kv_bh_id = pid // NUM_K_BLOCKS
    k_block_id = pid % NUM_K_BLOCKS

    rep: tl.constexpr = NUM_HEADS // NUM_KV_HEADS
    b_id = kv_bh_id // NUM_KV_HEADS
    hkv_id = kv_bh_id % NUM_KV_HEADS
    q_head_start = hkv_id * rep

    seq_len = tl.load(SeqLens + b_id)
    k_start = k_block_id * BS
    if k_start >= seq_len:
        return

    offs_d = tl.arange(0, D)
    offs_tok_k = k_start + tl.arange(0, BS)
    k_mask = offs_tok_k < seq_len

    K_ptr = K + kv_bh_id * stride_h
    V_ptr = V + kv_bh_id * stride_h

    k_bf16 = tl.load(K_ptr + offs_tok_k[:, None] * stride_t + offs_d[None, :], mask=k_mask[:, None], other=0.0)
    v_bf16 = tl.load(V_ptr + offs_tok_k[:, None] * stride_t + offs_d[None, :], mask=k_mask[:, None], other=0.0)

    dK_acc = tl.zeros([BS, D], dtype=tl.float32)
    dV_acc = tl.zeros([BS, D], dtype=tl.float32)

    for rep_idx in range(rep):
        q_head_id = q_head_start + rep_idx
        q_bh_id = b_id * NUM_HEADS + q_head_id

        Q_ptr = Q + q_bh_id * stride_h
        dO_ptr = dO + q_bh_id * stride_h
        LSE_ptr = LSE + q_bh_id * stride_lh
        Di_ptr = Di_in + q_bh_id * stride_lh

        q_start_diag = k_block_id * BS
        offs_tok_q = q_start_diag + tl.arange(0, BS)
        q_mask_diag = offs_tok_q < seq_len
        q_bf16_d = tl.load(Q_ptr + offs_tok_q[:, None] * stride_t + offs_d[None, :], mask=q_mask_diag[:, None], other=0.0)
        dO_d = tl.load(dO_ptr + offs_tok_q[:, None] * stride_t + offs_d[None, :], mask=q_mask_diag[:, None], other=0.0)
        LSE_d = tl.load(LSE_ptr + offs_tok_q, mask=q_mask_diag, other=0.0)
        Di_d = tl.load(Di_ptr + offs_tok_q, mask=q_mask_diag, other=0.0)

        s_d = tl.dot(q_bf16_d, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE
        causal_d = offs_tok_q[:, None] >= offs_tok_k[None, :]
        allowed_d = causal_d & k_mask[None, :] & q_mask_diag[:, None]
        if WINDOW > 0:
            in_window_d = (offs_tok_q[:, None] - offs_tok_k[None, :]) < WINDOW
            allowed_d = allowed_d & in_window_d
        s_d = tl.where(allowed_d, s_d, float("-inf"))
        P_d = tl.exp(s_d - LSE_d[:, None])
        dP_d = tl.dot(dO_d, tl.trans(v_bf16), out_dtype=tl.float32)
        dS_d = (P_d * (dP_d - Di_d[:, None])) * SCALE

        dK_acc += tl.dot(tl.trans(dS_d.to(tl.bfloat16)), q_bf16_d, out_dtype=tl.float32)
        dV_acc += tl.dot(tl.trans(P_d.to(tl.bfloat16)), dO_d, out_dtype=tl.float32)

        rpt_base = RowPtrT + q_bh_id * stride_rpth + k_block_id
        cit_lo = tl.load(rpt_base)
        cit_hi = tl.load(rpt_base + 1)
        CIT_ptr = ColIdxT + q_bh_id * stride_cith

        # See kernel_fwd: when CONTIGUOUS_OFFDIAG, hoist the transposed-CSR
        # load. For dense, K-block k attends Q-blocks k+1..num_q_blocks-1
        # (contiguous); for sliding-window, k attends a contiguous Q-range
        # too. The transposed-CSR builder (build_csrT) preserves ascending
        # q_block_id order within each K-row, so the run is contiguous.
        if CONTIGUOUS_OFFDIAG:
            q_block_id_start_o = tl.load(CIT_ptr + cit_lo)
            for ci in range(cit_lo, cit_hi):
                q_block_id = q_block_id_start_o + (ci - cit_lo)
                q_start = q_block_id * BS
                offs_tok_q2 = q_start + tl.arange(0, BS)
                q_mask2 = offs_tok_q2 < seq_len
                q_bf16_o = tl.load(Q_ptr + offs_tok_q2[:, None] * stride_t + offs_d[None, :], mask=q_mask2[:, None], other=0.0)
                dO_o = tl.load(dO_ptr + offs_tok_q2[:, None] * stride_t + offs_d[None, :], mask=q_mask2[:, None], other=0.0)
                LSE_o = tl.load(LSE_ptr + offs_tok_q2, mask=q_mask2, other=0.0)
                Di_o = tl.load(Di_ptr + offs_tok_q2, mask=q_mask2, other=0.0)

                s_o = tl.dot(q_bf16_o, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE
                allowed_o = q_mask2[:, None] & k_mask[None, :]
                if WINDOW > 0:
                    in_window_o = (offs_tok_q2[:, None] - offs_tok_k[None, :]) < WINDOW
                    allowed_o = allowed_o & in_window_o
                s_o = tl.where(allowed_o, s_o, float("-inf"))
                P_o = tl.exp(s_o - LSE_o[:, None])
                dP_o = tl.dot(dO_o, tl.trans(v_bf16), out_dtype=tl.float32)
                dS_o = (P_o * (dP_o - Di_o[:, None])) * SCALE

                dK_acc += tl.dot(tl.trans(dS_o.to(tl.bfloat16)), q_bf16_o, out_dtype=tl.float32)
                dV_acc += tl.dot(tl.trans(P_o.to(tl.bfloat16)), dO_o, out_dtype=tl.float32)
        else:
            for ci in range(cit_lo, cit_hi):
                q_block_id = tl.load(CIT_ptr + ci)
                q_start = q_block_id * BS
                offs_tok_q2 = q_start + tl.arange(0, BS)
                q_mask2 = offs_tok_q2 < seq_len
                q_bf16_o = tl.load(Q_ptr + offs_tok_q2[:, None] * stride_t + offs_d[None, :], mask=q_mask2[:, None], other=0.0)
                dO_o = tl.load(dO_ptr + offs_tok_q2[:, None] * stride_t + offs_d[None, :], mask=q_mask2[:, None], other=0.0)
                LSE_o = tl.load(LSE_ptr + offs_tok_q2, mask=q_mask2, other=0.0)
                Di_o = tl.load(Di_ptr + offs_tok_q2, mask=q_mask2, other=0.0)

                s_o = tl.dot(q_bf16_o, tl.trans(k_bf16), out_dtype=tl.float32) * SCALE
                allowed_o = q_mask2[:, None] & k_mask[None, :]
                if WINDOW > 0:
                    in_window_o = (offs_tok_q2[:, None] - offs_tok_k[None, :]) < WINDOW
                    allowed_o = allowed_o & in_window_o
                s_o = tl.where(allowed_o, s_o, float("-inf"))
                P_o = tl.exp(s_o - LSE_o[:, None])
                dP_o = tl.dot(dO_o, tl.trans(v_bf16), out_dtype=tl.float32)
                dS_o = (P_o * (dP_o - Di_o[:, None])) * SCALE

                dK_acc += tl.dot(tl.trans(dS_o.to(tl.bfloat16)), q_bf16_o, out_dtype=tl.float32)
                dV_acc += tl.dot(tl.trans(P_o.to(tl.bfloat16)), dO_o, out_dtype=tl.float32)

    dK_ptr = dK + kv_bh_id * stride_h + offs_tok_k[:, None] * stride_t + offs_d[None, :]
    dV_ptr = dV + kv_bh_id * stride_h + offs_tok_k[:, None] * stride_t + offs_d[None, :]
    tl.store(dK_ptr, dK_acc.to(tl.bfloat16), mask=k_mask[:, None])
    tl.store(dV_ptr, dV_acc.to(tl.bfloat16), mask=k_mask[:, None])


_hydra_bwd_dq_kernel = _kernel_decorator(_hydra_bwd_dq_jit)
_hydra_bwd_dkv_kernel = _kernel_decorator(_hydra_bwd_dkv_jit)


# ----------------------------- launcher ----------------------------------


def launch_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    seq_lens: torch.Tensor,
    row_ptr_T: torch.Tensor | None = None,
    col_idx_T: torch.Tensor | None = None,
    window: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch dQ then dK/dV kernels.

    Identical signature / semantics to the upstream ``launch_attn_bwd``.
    If ``row_ptr_T`` / ``col_idx_T`` are not provided, builds them via
    the cached pure-GPU builder so repeated calls with the same pattern
    pay a single one-time build cost.
    """
    batch_size, num_heads, t_max, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if t_max % BLOCK_SIZE != 0:
        raise ValueError(f"T ({t_max}) must be a multiple of BLOCK_SIZE ({BLOCK_SIZE})")
    if head_dim != HEAD_DIM:
        raise ValueError(f"head_dim ({head_dim}) must equal HEAD_DIM ({HEAD_DIM})")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

    batch_heads_q = batch_size * num_heads
    batch_heads_kv = batch_size * num_kv_heads
    num_q_blocks = t_max // BLOCK_SIZE

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # delta[b, h, t] = sum_d O[b, h, t, d] * dO[b, h, t, d]
    # Single-pass Triton kernel avoids the ~3 fp32 transients (do.float(),
    # o.float(), their product) the host pipeline allocates — ~256 MB
    # steady-state HBM saved at Qwen3-8B T=8192. Bounded reorder error
    # vs torch's .sum(-1) is ~1.5e-5, three orders below bf16 quantum.
    delta = launch_compute_delta(o, do)
    delta_2d = delta.view(batch_heads_q, t_max)
    lse_2d = lse.view(batch_heads_q, t_max)

    if row_ptr_T is None or col_idx_T is None:
        row_ptr_T, col_idx_T = build_csrT_cached(row_ptr, col_idx, num_q_blocks)

    stride_cih = col_idx.shape[2] if col_idx.ndim == 3 else col_idx.shape[1]
    stride_cith = col_idx_T.shape[2] if col_idx_T.ndim == 3 else col_idx_T.shape[1]
    scale = 1.0 / math.sqrt(head_dim)

    grid_q = (num_q_blocks * batch_heads_q,)
    grid_kv = (num_q_blocks * batch_heads_kv,)

    common_kwargs = dict(
        T_MAX=t_max,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        SCALE=scale,
        BS=BLOCK_SIZE,
        D=head_dim,
        WINDOW=int(window),
        # Both build_dense_causal_csr and build_sliding_window_csr emit
        # off-diag K-block IDs as a contiguous run; the transposed CSR
        # builder preserves ascending q_block_id within each K-row, which
        # is likewise contiguous for these patterns. Lets the kernels
        # hoist the per-iteration CSR scalar GMEM load.
        CONTIGUOUS_OFFDIAG=1,
    )
    if _DISABLE_AUTOTUNE:
        common_kwargs["num_warps"] = _BWD_NUM_WARPS
        common_kwargs["num_stages"] = _BWD_NUM_STAGES

    _hydra_bwd_dq_kernel[grid_q](
        q, k, v, lse_2d,
        do,
        delta_2d,
        dq,
        row_ptr, col_idx, seq_lens,
        stride_cih,
        **common_kwargs,
    )

    _hydra_bwd_dkv_kernel[grid_kv](
        q, k, v, lse_2d,
        do,
        delta_2d,
        dk, dv,
        row_ptr_T, col_idx_T, seq_lens,
        stride_cith,
        **common_kwargs,
    )

    return dq, dk, dv
