#!/usr/bin/env python3
"""Validate the uploaded TokenSpeed attention Hub kernel on a CUDA/ROCm GPU.

Examples:
  python scripts/validate_hub.py
  python scripts/validate_hub.py --repo-id adarshxs/tokenspeed-attention --version 1
  python scripts/validate_hub.py --skip-bench
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch


DEFAULT_REPO_ID = "adarshxs/tokenspeed-attention"


@dataclass(frozen=True)
class CaseResult:
    name: str
    max_abs: float
    max_rel: float


def _make_cu_seqlens(seqlens: list[int], device: str) -> tuple[list[int], torch.Tensor]:
    cu_cpu = [0]
    for seqlen in seqlens:
        cu_cpu.append(cu_cpu[-1] + seqlen)
    cu_gpu = torch.tensor(cu_cpu, device=device, dtype=torch.int32)
    return cu_cpu, cu_gpu


def _make_page_table(
    lengths: torch.Tensor, page_size: int, device: str
) -> tuple[torch.Tensor, int]:
    pages_per_batch = (lengths + page_size - 1) // page_size
    total_pages = int(pages_per_batch.sum().item())
    max_pages = int(pages_per_batch.max().item())
    page_table = torch.zeros(
        lengths.shape[0], max_pages, device=device, dtype=torch.int32
    )
    next_page = 0
    for batch_idx, num_pages in enumerate(pages_per_batch.tolist()):
        page_table[batch_idx, :num_pages] = torch.arange(
            next_page,
            next_page + num_pages,
            device=device,
            dtype=torch.int32,
        )
        next_page += num_pages
    return page_table, total_pages


def _fill_paged_cache(
    num_pages: int,
    page_size: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    return torch.randn(
        num_pages, page_size, num_heads, head_dim, device=device, dtype=dtype
    )


def _gather_paged(
    cache: torch.Tensor, page_table: torch.Tensor, batch_idx: int, length: int
) -> torch.Tensor:
    page_size = cache.shape[1]
    rows = []
    for pos in range(length):
        page = int(page_table[batch_idx, pos // page_size].item())
        rows.append(cache[page, pos % page_size])
    return torch.stack(rows, dim=0)


def _apply_logit_cap(scores: torch.Tensor, logit_cap: float) -> torch.Tensor:
    if logit_cap > 0:
        return logit_cap * torch.tanh(scores / logit_cap)
    return scores


def _attention_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    causal: bool,
    window_left: int,
    logit_cap: float,
    sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    sq = q.shape[0]
    sk = k.shape[0]
    scores = (q.float() @ k.float().T) * scale
    scores = _apply_logit_cap(scores, logit_cap)

    q_positions = torch.arange(sq, device=q.device).view(-1, 1) + max(sk - sq, 0)
    k_positions = torch.arange(sk, device=q.device).view(1, -1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if causal:
        mask |= k_positions > q_positions
    if window_left >= 0:
        mask |= q_positions > k_positions + window_left
    scores = scores.masked_fill(mask, float("-inf"))

    row_max = scores.max(dim=-1).values
    if sink is not None:
        row_max = torch.maximum(row_max, sink.expand_as(row_max).float())
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(scores - row_max[:, None])
    denom = exp_scores.sum(dim=-1)
    if sink is not None:
        denom = denom + torch.exp(sink.float() - row_max)
    out = exp_scores @ v.float() / denom[:, None]
    lse = torch.log(denom) + row_max
    return out, lse


def _mha_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: list[int],
    cu_kv: list[int],
    *,
    scale: float,
    causal: bool,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((q.shape[0], q.shape[1], v.shape[-1]), device=q.device)
    lse = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
    kv_group = q.shape[1] // k.shape[1]
    for batch_idx in range(len(cu_q) - 1):
        q_start, q_end = cu_q[batch_idx], cu_q[batch_idx + 1]
        kv_start, kv_end = cu_kv[batch_idx], cu_kv[batch_idx + 1]
        for head_idx in range(q.shape[1]):
            kv_head = head_idx // kv_group
            head_out, head_lse = _attention_rows(
                q[q_start:q_end, head_idx],
                k[kv_start:kv_end, kv_head],
                v[kv_start:kv_end, kv_head],
                scale=scale,
                causal=causal,
                window_left=window_left,
                logit_cap=logit_cap,
                sink=sinks[head_idx] if sinks is not None else None,
            )
            out[q_start:q_end, head_idx] = head_out
            lse[q_start:q_end, head_idx] = head_lse
    return out.to(q.dtype), lse


def _mha_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    query_lens: list[int],
    *,
    scale: float,
    causal: bool,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((q.shape[0], q.shape[1], v_cache.shape[-1]), device=q.device)
    lse = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
    kv_group = q.shape[1] // k_cache.shape[2]
    q_offset = 0
    for batch_idx, query_len in enumerate(query_lens):
        cache_len = int(cache_seqlens[batch_idx].item())
        k_seq = _gather_paged(k_cache, page_table, batch_idx, cache_len)
        v_seq = _gather_paged(v_cache, page_table, batch_idx, cache_len)
        for head_idx in range(q.shape[1]):
            kv_head = head_idx // kv_group
            head_out, head_lse = _attention_rows(
                q[q_offset : q_offset + query_len, head_idx],
                k_seq[:, kv_head],
                v_seq[:, kv_head],
                scale=scale,
                causal=causal,
                window_left=window_left,
                logit_cap=logit_cap,
                sink=sinks[head_idx] if sinks is not None else None,
            )
            out[q_offset : q_offset + query_len, head_idx] = head_out
            lse[q_offset : q_offset + query_len, head_idx] = head_lse
        q_offset += query_len
    return out.to(q.dtype), lse


def _mla_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: list[int],
    cu_kv: list[int],
    *,
    scale: float,
    causal: bool,
    logit_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _mha_varlen_reference(
        q,
        k,
        v,
        cu_q,
        cu_kv,
        scale=scale,
        causal=causal,
        logit_cap=logit_cap,
    )


def _mla_decode_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    scale: float,
    logit_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(q.shape[:-1] + (kv_lora_rank,), device=q.device)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    for batch_idx in range(q.shape[0]):
        cache_len = int(cache_seqlens[batch_idx].item())
        kv = _gather_paged(kv_cache, page_table, batch_idx, cache_len)[:, 0].float()
        k_latent = kv[:, :kv_lora_rank]
        k_rope = kv[:, kv_lora_rank : kv_lora_rank + qk_rope_head_dim]
        for head_idx in range(q.shape[2]):
            q_head = q[batch_idx, 0, head_idx].float()
            scores = (
                k_latent @ q_head[:kv_lora_rank]
                + k_rope @ q_head[kv_lora_rank : kv_lora_rank + qk_rope_head_dim]
            ) * scale
            scores = _apply_logit_cap(scores, logit_cap)
            probs = torch.softmax(scores, dim=0)
            out[batch_idx, 0, head_idx] = probs @ k_latent
            lse[batch_idx, 0, head_idx] = torch.logsumexp(scores, dim=0)
    return out.to(q.dtype), lse


def _merge_state_reference(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    lse_scale_log2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    lse_a_log2 = lse_a.float() * lse_scale_log2
    lse_b_log2 = lse_b.float() * lse_scale_log2
    lse_max = torch.maximum(lse_a_log2, lse_b_log2)
    weight_a = torch.exp2(lse_a_log2 - lse_max)
    weight_b = torch.exp2(lse_b_log2 - lse_max)
    denom = weight_a + weight_b
    out = (
        out_a.float() * weight_a[..., None] + out_b.float() * weight_b[..., None]
    ) / denom[..., None]
    lse = (lse_max + torch.log2(denom)) / lse_scale_log2
    return out.to(out_a.dtype), lse.to(lse_a.dtype)


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f = actual.float()
    expected_f = expected.float()
    abs_err = (actual_f - expected_f).abs()
    rel_err = abs_err / expected_f.abs().clamp_min(1e-6)
    return float(abs_err.max().item()), float(rel_err.max().item())


def _check_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> CaseResult:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    max_abs, max_rel = _max_errors(actual, expected)
    return CaseResult(name, max_abs, max_rel)


def _time_cuda(fn: Callable[[], object], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _load_kernel(args: argparse.Namespace):
    from kernels import VariantAccepted, get_kernel, get_kernel_variants, has_kernel

    selector = {"revision": args.revision} if args.revision else {"version": args.version}
    print(f"repo_id={args.repo_id} selector={selector}")
    print(f"has_kernel={has_kernel(args.repo_id, **selector)}")
    for decision in get_kernel_variants(args.repo_id, **selector):
        variant = decision.variant.variant_str
        if isinstance(decision, VariantAccepted):
            print(f"variant compatible: {variant}")
        else:
            print(f"variant rejected:   {variant} ({decision.reason})")

    return get_kernel(
        args.repo_id,
        **selector,
        trust_remote_code=args.trust_remote_code,
    )


def run_correctness(kernel, *, device: str, dtype: torch.dtype) -> list[CaseResult]:
    results: list[CaseResult] = []
    torch.manual_seed(1234)
    scale = 1.0 / math.sqrt(64)

    # MHA prefill: causal + sliding window + sinks + LSE.
    seqlens = [17, 23, 11]
    cu_cpu, cu = _make_cu_seqlens(seqlens, device)
    total = cu_cpu[-1]
    q = torch.randn(total, 8, 64, device=device, dtype=dtype)
    k = torch.randn(total, 2, 64, device=device, dtype=dtype)
    v = torch.randn(total, 2, 64, device=device, dtype=dtype)
    sinks = torch.randn(8, device=device, dtype=dtype)
    out, lse = kernel.mha_prefill(
        q,
        k,
        v,
        cu,
        cu_cpu,
        max(seqlens),
        window_left=9,
        logit_cap=30.0,
        sinks=sinks,
        return_lse=True,
    )
    ref, ref_lse = _mha_varlen_reference(
        q,
        k,
        v,
        cu_cpu,
        cu_cpu,
        scale=scale,
        causal=True,
        window_left=9,
        logit_cap=30.0,
        sinks=sinks,
    )
    results.append(_check_close("mha_prefill/out", out, ref, atol=1.0e-1, rtol=1.0e-1))
    results.append(_check_close("mha_prefill/lse", lse, ref_lse, atol=1.0e-1, rtol=1.0e-1))

    # MHA extend with paged KV cache.
    query_lens = [3, 2, 4]
    cache_lens = torch.tensor([21, 7, 30], device=device, dtype=torch.int32)
    page_size = 16
    page_table, total_pages = _make_page_table(cache_lens, page_size, device)
    q_ext = torch.randn(sum(query_lens), 8, 64, device=device, dtype=dtype)
    k_cache = _fill_paged_cache(total_pages, page_size, 2, 64, dtype, device)
    v_cache = _fill_paged_cache(total_pages, page_size, 2, 64, dtype, device)
    cu_q_cpu, cu_q = _make_cu_seqlens(query_lens, device)
    cu_kv_cpu, cu_kv = _make_cu_seqlens(cache_lens.tolist(), device)
    out, lse = kernel.mha_extend_with_kvcache(
        q_ext,
        cu_q,
        cu_kv,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        max(query_lens),
        int(cache_lens.max().item()),
        return_lse=True,
    )
    ref, ref_lse = _mha_paged_reference(
        q_ext,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        query_lens,
        scale=scale,
        causal=False,
    )
    results.append(_check_close("mha_extend/out", out, ref, atol=1.0e-1, rtol=1.0e-1))
    results.append(_check_close("mha_extend/lse", lse, ref_lse, atol=1.0e-1, rtol=1.0e-1))

    # MHA decode with paged KV cache.
    cache_lens = torch.tensor([31, 16, 23, 9], device=device, dtype=torch.int32)
    page_table, total_pages = _make_page_table(cache_lens, page_size, device)
    q_dec = torch.randn(cache_lens.shape[0], 8, 64, device=device, dtype=dtype)
    k_cache = _fill_paged_cache(total_pages, page_size, 2, 64, dtype, device)
    v_cache = _fill_paged_cache(total_pages, page_size, 2, 64, dtype, device)
    out = kernel.mha_decode_with_kvcache(
        q_dec,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        int(cache_lens.max().item()),
    )
    ref, _ = _mha_paged_reference(
        q_dec,
        k_cache,
        v_cache,
        page_table,
        cache_lens,
        [1] * cache_lens.shape[0],
        scale=scale,
        causal=False,
    )
    results.append(_check_close("mha_decode/out", out, ref, atol=1.0e-1, rtol=1.0e-1))

    # MLA prefill.
    # Use the canonical DeepSeek MLA shape (kv_lora_rank=512, qk_rope_head_dim=64,
    # head_dim=576). This exercises the kernel's real production code path; a tiny
    # rank (<64) instead drives the value-block width below 64, which is not a
    # configuration any real MLA workload uses.
    rank = 512
    rope_dim = 64
    head_dim = rank + rope_dim
    seqlens_q = [8, 5]
    seqlens_kv = [10, 6]
    cu_q_cpu, cu_q = _make_cu_seqlens(seqlens_q, device)
    cu_kv_cpu, cu_kv = _make_cu_seqlens(seqlens_kv, device)
    q_mla = torch.randn(sum(seqlens_q), 4, head_dim, device=device, dtype=dtype)
    k_mla = torch.randn(sum(seqlens_kv), 1, head_dim, device=device, dtype=dtype)
    v_mla = torch.randn(sum(seqlens_kv), 1, rank, device=device, dtype=dtype)
    mla_scale = 1.0 / math.sqrt(head_dim)
    out, lse = kernel.mla_prefill(
        q_mla,
        k_mla,
        v_mla,
        cu_q,
        cu_kv,
        max(seqlens_q),
        max(seqlens_kv),
        mla_scale,
        is_causal=True,
        logit_cap=30.0,
        return_lse=True,
    )
    ref, ref_lse = _mla_prefill_reference(
        q_mla,
        k_mla,
        v_mla,
        cu_q_cpu,
        cu_kv_cpu,
        scale=mla_scale,
        causal=True,
        logit_cap=30.0,
    )
    results.append(_check_close("mla_prefill/out", out, ref, atol=1.0e-1, rtol=1.0e-1))
    results.append(_check_close("mla_prefill/lse", lse, ref_lse, atol=1.0e-1, rtol=1.0e-1))

    # MLA decode with paged KV cache.
    cache_lens = torch.tensor([7, 11], device=device, dtype=torch.int32)
    page_table, total_pages = _make_page_table(cache_lens, page_size, device)
    q_mla_dec = torch.randn(2, 1, 4, head_dim, device=device, dtype=dtype)
    kv_cache = _fill_paged_cache(total_pages, page_size, 1, head_dim, dtype, device)
    out, lse = kernel.mla_decode_with_kvcache(
        q_mla_dec,
        kv_cache,
        page_table,
        cache_lens,
        int(cache_lens.max().item()),
        rank,
        rank,
        rope_dim,
        mla_scale,
        logit_cap=30.0,
        return_lse=True,
    )
    ref, ref_lse = _mla_decode_reference(
        q_mla_dec,
        kv_cache,
        page_table,
        cache_lens,
        kv_lora_rank=rank,
        qk_rope_head_dim=rope_dim,
        scale=mla_scale,
        logit_cap=30.0,
    )
    results.append(_check_close("mla_decode/out", out, ref, atol=1.0e-1, rtol=1.0e-1))
    results.append(_check_close("mla_decode/lse", lse, ref_lse, atol=1.0e-1, rtol=1.0e-1))

    # Merge state.
    out_a = torch.randn(3, 5, 32, device=device, dtype=dtype)
    out_b = torch.randn(3, 5, 32, device=device, dtype=dtype)
    lse_a = torch.randn(3, 5, device=device, dtype=torch.float32)
    lse_b = torch.randn(3, 5, device=device, dtype=torch.float32)
    lse_scale = math.log2(math.e)
    out, lse = kernel.attn_merge_state(out_a, lse_a, out_b, lse_b, lse_scale)
    ref, ref_lse = _merge_state_reference(out_a, lse_a, out_b, lse_b, lse_scale)
    results.append(_check_close("attn_merge/out", out, ref, atol=2.0e-2, rtol=2.0e-2))
    results.append(_check_close("attn_merge/lse", lse, ref_lse, atol=2.0e-4, rtol=2.0e-4))

    return results


def run_benchmarks(
    kernel, *, device: str, dtype: torch.dtype, warmup: int, iters: int
) -> list[tuple[str, float]]:
    torch.manual_seed(4321)
    timings: list[tuple[str, float]] = []

    seqlens = [512, 768, 1024]
    cu_cpu, cu = _make_cu_seqlens(seqlens, device)
    total = cu_cpu[-1]
    q = torch.randn(total, 16, 128, device=device, dtype=dtype)
    k = torch.randn(total, 2, 128, device=device, dtype=dtype)
    v = torch.randn(total, 2, 128, device=device, dtype=dtype)
    timings.append(
        (
            "mha_prefill_2304tok",
            _time_cuda(
                lambda: kernel.mha_prefill(q, k, v, cu, cu_cpu, max(seqlens)),
                warmup=warmup,
                iters=iters,
            ),
        )
    )

    batch = 32
    page_size = 64
    cache_lens = torch.full((batch,), 2048, device=device, dtype=torch.int32)
    page_table, total_pages = _make_page_table(cache_lens, page_size, device)
    q_dec = torch.randn(batch, 16, 128, device=device, dtype=dtype)
    k_cache = _fill_paged_cache(total_pages, page_size, 2, 128, dtype, device)
    v_cache = _fill_paged_cache(total_pages, page_size, 2, 128, dtype, device)
    timings.append(
        (
            "mha_decode_b32_s2048",
            _time_cuda(
                lambda: kernel.mha_decode_with_kvcache(
                    q_dec,
                    k_cache,
                    v_cache,
                    page_table,
                    cache_lens,
                    int(cache_lens.max().item()),
                ),
                warmup=warmup,
                iters=iters,
            ),
        )
    )

    rank = 128
    rope_dim = 64
    head_dim = rank + rope_dim
    q_mla = torch.randn(batch, 1, 16, head_dim, device=device, dtype=dtype)
    kv_cache = _fill_paged_cache(total_pages, page_size, 1, head_dim, dtype, device)
    scale = 1.0 / math.sqrt(head_dim)
    timings.append(
        (
            "mla_decode_b32_s2048",
            _time_cuda(
                lambda: kernel.mla_decode_with_kvcache(
                    q_mla,
                    kv_cache,
                    page_table,
                    cache_lens,
                    int(cache_lens.max().item()),
                    rank,
                    rank,
                    rope_dim,
                    scale,
                ),
                warmup=warmup,
                iters=iters,
            ),
        )
    )

    return timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading a personal Hub kernel repo.",
    )
    parser.set_defaults(trust_remote_code=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is required")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"torch={torch.__version__} cuda={torch.version.cuda} hip={torch.version.hip}")
    print(f"device={props.name} capability={props.major}.{props.minor}")

    start = time.time()
    kernel = _load_kernel(args)
    print(f"loaded kernel in {time.time() - start:.2f}s")
    for name in (
        "mha_prefill",
        "mha_extend_with_kvcache",
        "mha_decode_with_kvcache",
        "mla_prefill",
        "mla_decode_with_kvcache",
        "attn_merge_state",
    ):
        if not hasattr(kernel, name):
            raise AssertionError(f"loaded kernel is missing {name}")

    print("\ncorrectness:")
    for result in run_correctness(kernel, device=args.device, dtype=dtype):
        print(
            f"  PASS {result.name:<22} "
            f"max_abs={result.max_abs:.4e} max_rel={result.max_rel:.4e}"
        )

    if not args.skip_bench:
        print("\nbenchmarks:")
        for name, ms in run_benchmarks(
            kernel,
            device=args.device,
            dtype=dtype,
            warmup=args.bench_warmup,
            iters=args.bench_iters,
        ):
            print(f"  {name:<24} {ms:.3f} ms")

    print("\nall checks passed")


if __name__ == "__main__":
    main()

