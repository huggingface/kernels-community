#!/usr/bin/env python3

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent
TORCH_EXT = ROOT / "torch-ext"
if str(TORCH_EXT) not in sys.path:
    sys.path.insert(0, str(TORCH_EXT))

import torch
import triton

import finegrained_fp8
import finegrained_fp8.fp4 as finegrained_fp4


FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
FP8_DTYPE = torch.float8_e4m3fn
FP4_BS_DTYPE = torch.float8_e8m0fnu
FP4_VALUES_PER_BYTE = 2
FP4_SCALE_GROUP_K = 32
FP4_E2M1_DECODE_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@dataclass(frozen=True)
class MoeProblem:
    name: str
    S: int
    E: int
    N: int
    K: int
    top_k: int
    block_n: int
    block_k: int

    @property
    def block_size(self) -> list[int]:
        return [self.block_n, self.block_k]

    @property
    def num_tokens(self) -> int:
        return self.S // self.top_k


@dataclass(frozen=True)
class MatmulProblem:
    name: str
    M: int
    N: int
    K: int
    block_n: int
    block_k: int

    @property
    def block_size(self) -> list[int]:
        return [self.block_n, self.block_k]


@dataclass(frozen=True)
class BenchStats:
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


MOE_PRESETS = {
    "deepseek_v4_gate_up_decode": MoeProblem(
        name="deepseek_v4_gate_up_decode", S=192, E=256, N=4096, K=4096, top_k=6, block_n=128, block_k=128
    ),
    "deepseek_v4_down_decode": MoeProblem(
        name="deepseek_v4_down_decode", S=192, E=256, N=4096, K=2048, top_k=6, block_n=128, block_k=128
    ),
    "deepseek_v4_gate_up_prefill": MoeProblem(
        name="deepseek_v4_gate_up_prefill", S=1536, E=256, N=4096, K=4096, top_k=6, block_n=128, block_k=128
    ),
    "deepseek_v4_down_prefill": MoeProblem(
        name="deepseek_v4_down_prefill", S=1536, E=256, N=4096, K=2048, top_k=6, block_n=128, block_k=128
    ),
    "qwen3_gate_up_decode": MoeProblem(
        name="qwen3_gate_up_decode", S=256, E=128, N=1536, K=2048, top_k=8, block_n=128, block_k=128
    ),
    "qwen3_down_decode": MoeProblem(
        name="qwen3_down_decode", S=256, E=128, N=2048, K=768, top_k=8, block_n=128, block_k=128
    ),
    "qwen3_gate_up_prefill": MoeProblem(
        name="qwen3_gate_up_prefill", S=1024, E=128, N=1536, K=2048, top_k=8, block_n=128, block_k=128
    ),
    "qwen3_down_prefill": MoeProblem(
        name="qwen3_down_prefill", S=1024, E=128, N=2048, K=768, top_k=8, block_n=128, block_k=128
    ),
}

MATMUL_PRESETS = {
    "deepseek_v4_gate_up_decode": MatmulProblem(
        name="deepseek_v4_gate_up_decode", M=32, N=4096, K=4096, block_n=128, block_k=128
    ),
    "deepseek_v4_down_decode": MatmulProblem(
        name="deepseek_v4_down_decode", M=32, N=4096, K=2048, block_n=128, block_k=128
    ),
    "deepseek_v4_gate_up_prefill": MatmulProblem(
        name="deepseek_v4_gate_up_prefill", M=256, N=4096, K=4096, block_n=128, block_k=128
    ),
    "deepseek_v4_down_prefill": MatmulProblem(
        name="deepseek_v4_down_prefill", M=256, N=4096, K=2048, block_n=128, block_k=128
    ),
    "decode_matmul": MatmulProblem(
        name="decode_matmul", M=1, N=1536, K=2048, block_n=128, block_k=128
    ),
    "prefill_matmul": MatmulProblem(
        name="prefill_matmul", M=1024, N=1536, K=2048, block_n=128, block_k=128
    ),
}


def _get_device(device: Optional[str]) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return device
    if device == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("XPU requested but not available")
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    raise RuntimeError("No supported accelerator found; expected CUDA or XPU")


def _accelerator_module(device: str):
    if device == "cuda":
        return torch.cuda
    if device == "xpu":
        return torch.xpu
    raise ValueError(f"Unsupported device: {device}")


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def _sync(device: str):
    _accelerator_module(device).synchronize()


def _maybe_empty_cache(device: str):
    _accelerator_module(device).empty_cache()


def _quantize_weights_2d(weight: torch.Tensor, block_n: int, block_k: int):
    n, k = weight.shape
    if n % block_n != 0 or k % block_k != 0:
        raise AssertionError(
            f"matmul benchmark expects aligned weights, got N={n}, K={k}, block=({block_n}, {block_k})"
        )
    rt, ct = n // block_n, k // block_k
    reshaped = weight.reshape(rt, block_n, ct, block_k)
    max_abs = reshaped.abs().amax(dim=(1, 3))
    safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scale = FP8_MAX / safe
    weight_q = (reshaped * scale[:, None, :, None]).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    return weight_q.reshape(n, k).contiguous(), (1.0 / scale).to(torch.float32)


def _make_experts_weights(num_experts: int, out_features: int, in_features: int, block_size: list[int], device: str):
    block_n, block_k = block_size
    weight = torch.randn(num_experts, out_features, in_features, dtype=torch.float32, device=device)
    e, n, k = weight.shape
    if n % block_n != 0 or k % block_k != 0:
        raise AssertionError(
            f"MoE benchmark expects aligned weights, got N={n}, K={k}, block=({block_n}, {block_k})"
        )
    rt, ct = n // block_n, k // block_k
    reshaped = weight.reshape(e, rt, block_n, ct, block_k)
    max_abs = reshaped.abs().amax(dim=(-3, -1))
    safe = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scale = FP8_MAX / safe
    weight_q = (reshaped * scale.unsqueeze(-1).unsqueeze(-3)).clamp(FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    weight_q = weight_q.reshape(e, n, k).contiguous()
    scales = (1.0 / scale).to(torch.float32)
    return weight_q, scales


def _make_experts_weights_fp4(num_experts: int, out_features: int, in_features: int, device: str):
    if in_features % FP4_SCALE_GROUP_K != 0:
        raise AssertionError(
            f"FP4 benchmark expects K divisible by {FP4_SCALE_GROUP_K}, got K={in_features}"
        )
    packed_k = in_features // FP4_VALUES_PER_BYTE
    weights = torch.randint(-8, 8, (num_experts, out_features, packed_k), dtype=torch.int8, device=device)
    scales = torch.ones((num_experts, out_features, in_features // FP4_SCALE_GROUP_K), dtype=FP4_BS_DTYPE, device=device)
    return weights.contiguous(), scales.contiguous()


def _make_weights_fp4_2d(out_features: int, in_features: int, device: str):
    if in_features % FP4_SCALE_GROUP_K != 0:
        raise AssertionError(
            f"FP4 benchmark expects K divisible by {FP4_SCALE_GROUP_K}, got K={in_features}"
        )
    packed_k = in_features // FP4_VALUES_PER_BYTE
    weights = torch.randint(-8, 8, (out_features, packed_k), dtype=torch.int8, device=device)
    scales = torch.ones((out_features, in_features // FP4_SCALE_GROUP_K), dtype=FP4_BS_DTYPE, device=device)
    return weights.contiguous(), scales.contiguous()


def _make_routed_inputs(problem: MoeProblem, dtype: torch.dtype, device: str):
    if problem.S % problem.top_k != 0:
        raise AssertionError(f"S ({problem.S}) must be divisible by top_k ({problem.top_k})")
    num_tokens = problem.num_tokens
    hidden_states = torch.randn(num_tokens, problem.K, dtype=dtype, device=device)
    top_k_index = torch.randint(0, problem.E, (num_tokens, problem.top_k), device=device)
    token_idx = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, problem.top_k).reshape(-1)
    expert_ids = top_k_index.reshape(-1).to(torch.int32)
    selected_hidden_states = hidden_states[token_idx].contiguous()
    return selected_hidden_states, expert_ids


def _prepare_grouped(a: torch.Tensor, expert_ids: torch.Tensor, num_experts: int):
    perm = torch.argsort(expert_ids)
    a_sorted = a[perm].contiguous()
    expert_ids_sorted = expert_ids[perm]
    tokens_per_expert = torch.histc(
        expert_ids_sorted.float(), bins=num_experts, min=0, max=num_experts - 1
    ).to(torch.int32)
    offsets = torch.cumsum(tokens_per_expert, dim=0).to(torch.int32)
    return a_sorted, expert_ids_sorted, offsets, tokens_per_expert


def _measure(fn: Callable[[], object], repeats: int) -> BenchStats:
    runs_ms = [triton.testing.do_bench(fn) for _ in range(repeats)]
    return BenchStats(
        median_ms=statistics.median(runs_ms),
        mean_ms=statistics.mean(runs_ms),
        min_ms=min(runs_ms),
        max_ms=max(runs_ms),
    )


def _effective_tops(m: int, n: int, k: int, median_ms: float) -> float:
    return (2.0 * m * n * k) / (median_ms * 1e-3) / 1e12


def _expand_activation_scales(as_: torch.Tensor, block_k: int, k: int) -> torch.Tensor:
    if as_.shape[-1] == k:
        return as_
    return as_.repeat_interleave(block_k, dim=-1)


def _decode_fp4_packed(weights: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(FP4_E2M1_DECODE_TABLE, dtype=torch.float32, device=weights.device)
    packed = weights.to(torch.uint8)
    low_nibble = torch.bitwise_and(packed, 0x0F).to(torch.long)
    high_nibble = torch.bitwise_right_shift(packed, 4).to(torch.long)

    decoded = torch.empty(
        (*weights.shape[:-1], weights.shape[-1] * FP4_VALUES_PER_BYTE),
        dtype=torch.float32,
        device=weights.device,
    )
    decoded[..., 0::2] = lut[low_nibble]
    decoded[..., 1::2] = lut[high_nibble]
    return decoded * scales.float().repeat_interleave(FP4_SCALE_GROUP_K, dim=-1)


def _dequantize_fp8_activations(a_fp8: torch.Tensor, scales: torch.Tensor, block_k: int, k: int) -> torch.Tensor:
    if scales.shape[-1] == k:
        return a_fp8.float() * scales.float()

    m = a_fp8.shape[0]
    return (a_fp8.float().reshape(m, k // block_k, block_k) * scales.float().unsqueeze(-1)).reshape(m, k)


def _dequantize_fp8_weights(b_fp8: torch.Tensor, scales: torch.Tensor, block_n: int, block_k: int) -> torch.Tensor:
    n, k = b_fp8.shape
    return (
        b_fp8.float().reshape(n // block_n, block_n, k // block_k, block_k) * scales.float().unsqueeze(1).unsqueeze(-1)
    ).reshape(n, k)


def _ref_matmul(a_fp8, b_fp8, as_, bs, block_n: int, block_k: int):
    _, k = b_fp8.shape
    a_deq = _dequantize_fp8_activations(a_fp8, as_, block_k, k)
    b_deq = _dequantize_fp8_weights(b_fp8, bs, block_n, block_k)
    return a_deq @ b_deq.T


def _bench_moe_op(
    op: str,
    problem: MoeProblem,
    dtype: torch.dtype,
    device: str,
    repeats: int,
    compare_eager: bool,
    check: bool,
    weight_format: str,
):
    _maybe_empty_cache(device)
    torch.manual_seed(0)
    a, expert_ids = _make_routed_inputs(problem, dtype=dtype, device=device)
    if weight_format == "fp4":
        weights, scales = _make_experts_weights_fp4(problem.E, problem.N, problem.K, device)
    else:
        weights, scales = _make_experts_weights(problem.E, problem.N, problem.K, problem.block_size, device)

    if op == "batched":
        args = (a, weights, scales, expert_ids, problem.block_size)
        if weight_format == "fp4":
            kernel_name = "w4a8_fp8_matmul_batched"
            run = lambda: finegrained_fp4.w4a8_fp8_matmul_batched(*args)
        else:
            kernel_name = "w8a8_fp8_matmul_batched"
            run = lambda: finegrained_fp8.w8a8_fp8_matmul_batched(*args)
        eager_a = a
        eager_expert_ids = expert_ids
    elif op == "grouped":
        a_sorted, eager_expert_ids, offsets, tokens_per_expert = _prepare_grouped(a, expert_ids, problem.E)
        args = (a_sorted, weights, scales, offsets, tokens_per_expert, problem.block_size)
        if weight_format == "fp4":
            kernel_name = "w4a8_fp8_matmul_grouped"
            run = lambda: finegrained_fp4.w4a8_fp8_matmul_grouped(*args)
        else:
            kernel_name = "w8a8_fp8_matmul_grouped"
            run = lambda: finegrained_fp8.w8a8_fp8_matmul_grouped(*args)
        eager_a = a_sorted
    else:
        raise ValueError(f"Unsupported MoE op: {op}")

    stats = _measure(run, repeats=repeats)
    result = {
        "op": op,
        "kernel": kernel_name,
        "problem": asdict(problem),
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "median_ms": stats.median_ms,
        "mean_ms": stats.mean_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "effective_tops": _effective_tops(problem.S, problem.N, problem.K, stats.median_ms),
    }

    if compare_eager or check:
        eager_expert_ids_host = eager_expert_ids.to("cpu", torch.int64).tolist()

        def eager_quant_only():
            quantized = []
            for idx in range(problem.S):
                quantized.append(finegrained_fp8.fp8_act_quant(eager_a[idx : idx + 1], problem.block_k))
            return quantized

        def fp4_row_ref_run():
            out = torch.empty(problem.S, problem.N, dtype=torch.float32, device=device)
            quantized = eager_quant_only()
            for idx, expert in enumerate(eager_expert_ids_host):
                q_a_i, s_a_i = quantized[idx]
                if weight_format == "fp4":
                    out[idx] = finegrained_fp4.w4a8_block_fp8_matmul(
                        q_a_i,
                        s_a_i,
                        weights[expert],
                        scales[expert],
                        problem.block_size,
                        torch.float32,
                    )
                else:
                    out[idx] = finegrained_fp8.w8a8_fp8_matmul(
                        q_a_i,
                        weights[expert],
                        s_a_i,
                        scales[expert],
                        problem.block_size,
                    )
            return out

        def torch_fp8_run():
            quantized = eager_quant_only()
            out = torch.empty(problem.S, problem.N, dtype=torch.float32, device=device)
            rows_by_expert = {}
            for idx, expert in enumerate(eager_expert_ids_host):
                rows_by_expert.setdefault(expert, []).append(idx)
            for expert, row_indices in rows_by_expert.items():
                q_rows = torch.cat([quantized[row_idx][0] for row_idx in row_indices], dim=0)
                s_rows = torch.cat([quantized[row_idx][1] for row_idx in row_indices], dim=0)
                a_rows = _dequantize_fp8_activations(q_rows, s_rows, problem.block_k, problem.K)
                b_rows = _dequantize_fp8_weights(weights[expert], scales[expert], problem.block_n, problem.block_k)
                out[row_indices] = a_rows @ b_rows.T
            return out

        def torch_fp4_run():
            quantized = eager_quant_only()
            out = torch.empty(problem.S, problem.N, dtype=torch.float32, device=device)
            rows_by_expert = {}
            for idx, expert in enumerate(eager_expert_ids_host):
                rows_by_expert.setdefault(expert, []).append(idx)
            for expert, row_indices in rows_by_expert.items():
                decoded_weight = _decode_fp4_packed(weights[expert], scales[expert])
                a_rows = []
                for row_idx in row_indices:
                    q_a_i, s_a_i = quantized[row_idx]
                    a_rows.append(q_a_i.float() * _expand_activation_scales(s_a_i, problem.block_k, problem.K))
                out[row_indices] = torch.cat(a_rows, dim=0) @ decoded_weight.T
            return out

        if compare_eager:
            baselines = []
            if weight_format == "fp4":
                torch_fp4_stats = _measure(torch_fp4_run, repeats=repeats)
                baselines.append(
                    {
                        "name": "torch_fp4_baseline",
                        "median_ms": torch_fp4_stats.median_ms,
                        "mean_ms": torch_fp4_stats.mean_ms,
                        "min_ms": torch_fp4_stats.min_ms,
                        "max_ms": torch_fp4_stats.max_ms,
                        "speedup_vs_baseline": torch_fp4_stats.median_ms / stats.median_ms,
                    }
                )
            else:
                torch_fp8_stats = _measure(torch_fp8_run, repeats=repeats)
                baselines.append(
                    {
                        "name": "torch_fp8_baseline",
                        "median_ms": torch_fp8_stats.median_ms,
                        "mean_ms": torch_fp8_stats.mean_ms,
                        "min_ms": torch_fp8_stats.min_ms,
                        "max_ms": torch_fp8_stats.max_ms,
                        "speedup_vs_baseline": torch_fp8_stats.median_ms / stats.median_ms,
                    }
                )
            result["baselines"] = baselines

        if check:
            out = run()
            ref = fp4_row_ref_run() if weight_format == "fp4" else torch_fp8_run()
            _sync(device)
            torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    return result


def _bench_matmul(
    problem: MatmulProblem,
    dtype: torch.dtype,
    device: str,
    repeats: int,
    compare_eager: bool,
    include_act_quant: bool,
    check: bool,
    weight_format: str,
):
    _maybe_empty_cache(device)
    torch.manual_seed(0)
    a = torch.randn(problem.M, problem.K, dtype=dtype, device=device)
    if weight_format == "fp4":
        weights, scales = _make_weights_fp4_2d(problem.N, problem.K, device)
        run = lambda: finegrained_fp4.w4a8_fp8_matmul(a, weights, scales, problem.block_size)
        kernel_name = "w4a8_fp8_matmul"
    else:
        w = torch.randn(problem.N, problem.K, dtype=torch.float32, device=device)
        weights, scales = _quantize_weights_2d(w, problem.block_n, problem.block_k)
        q_a, s_a = finegrained_fp8.fp8_act_quant(a, problem.block_k)

        if include_act_quant:
            def run():
                q_a_local, s_a_local = finegrained_fp8.fp8_act_quant(a, problem.block_k)
                return finegrained_fp8.w8a8_fp8_matmul(q_a_local, weights, s_a_local, scales, problem.block_size)

            kernel_name = "fp8_act_quant + w8a8_fp8_matmul"
        else:
            run = lambda: finegrained_fp8.w8a8_fp8_matmul(q_a, weights, s_a, scales, problem.block_size)
            kernel_name = "w8a8_fp8_matmul"

    stats = _measure(run, repeats=repeats)
    result = {
        "op": "matmul",
        "kernel": kernel_name,
        "problem": asdict(problem),
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "median_ms": stats.median_ms,
        "mean_ms": stats.mean_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "effective_tops": _effective_tops(problem.M, problem.N, problem.K, stats.median_ms),
    }

    if compare_eager:
        baselines = []
        if weight_format == "fp4":

            def torch_fp4_matmul_run():
                q_a_local, s_a_local = finegrained_fp8.fp8_act_quant(a, problem.block_k)
                decoded_weight = _decode_fp4_packed(weights, scales)
                a_deq = q_a_local.float() * _expand_activation_scales(s_a_local, problem.block_k, problem.K)
                return a_deq @ decoded_weight.T

            torch_fp4_stats = _measure(torch_fp4_matmul_run, repeats=repeats)
            baselines.append(
                {
                    "name": "torch_fp4_baseline",
                    "median_ms": torch_fp4_stats.median_ms,
                    "mean_ms": torch_fp4_stats.mean_ms,
                    "min_ms": torch_fp4_stats.min_ms,
                    "max_ms": torch_fp4_stats.max_ms,
                    "speedup_vs_baseline": torch_fp4_stats.median_ms / stats.median_ms,
                }
            )
        else:

            def torch_fp8_matmul_run():
                if include_act_quant:
                    q_a_local, s_a_local = finegrained_fp8.fp8_act_quant(a, problem.block_k)
                    return _ref_matmul(q_a_local, weights, s_a_local, scales, problem.block_n, problem.block_k)
                return _ref_matmul(q_a, weights, s_a, scales, problem.block_n, problem.block_k)

            torch_fp8_stats = _measure(torch_fp8_matmul_run, repeats=repeats)
            baselines.append(
                {
                    "name": "torch_fp8_baseline",
                    "median_ms": torch_fp8_stats.median_ms,
                    "mean_ms": torch_fp8_stats.mean_ms,
                    "min_ms": torch_fp8_stats.min_ms,
                    "max_ms": torch_fp8_stats.max_ms,
                    "speedup_vs_baseline": torch_fp8_stats.median_ms / stats.median_ms,
                }
            )
        result["baselines"] = baselines

    if check:
        if weight_format == "fp4":
            out = finegrained_fp4.w4a8_fp8_matmul(a, weights, scales, problem.block_size)
            ref_rows = []
            for idx in range(problem.M):
                ref_rows.append(
                    finegrained_fp4.w4a8_fp8_matmul(
                        a[idx : idx + 1],
                        weights,
                        scales,
                        problem.block_size,
                    )
                )
            ref = torch.cat(ref_rows, dim=0)
        else:
            out = finegrained_fp8.w8a8_fp8_matmul(q_a, weights, s_a, scales, problem.block_size)
            ref = _ref_matmul(q_a, weights, s_a, scales, problem.block_n, problem.block_k)
        _sync(device)
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    return result


def _print_result(result: dict):
    op = result["op"]
    problem = result["problem"]
    shape_keys = ["M", "S", "E", "N", "K", "top_k", "block_n", "block_k"]
    shape_summary = " ".join(f"{key}={problem[key]}" for key in shape_keys if key in problem)
    print(
        f"[{op}] {problem['name']} | {shape_summary} | "
        f"median={result['median_ms']:.4f}ms mean={result['mean_ms']:.4f}ms "
        f"min={result['min_ms']:.4f}ms max={result['max_ms']:.4f}ms "
        f"effective={result['effective_tops']:.3f} TOPS"
    )
    baselines = result.get("baselines")
    if baselines is None:
        baseline = result.get("baseline")
        baselines = [] if baseline is None else [baseline]
    for baseline in baselines:
        print(
            f"  baseline={baseline['name']} median={baseline['median_ms']:.4f}ms "
            f"speedup={baseline['speedup_vs_baseline']:.3f}x"
        )
        breakdown = baseline.get("breakdown")
        if breakdown is not None:
            print(
                "  baseline_breakdown="
                f"quant_only {breakdown['quant_only']['median_ms']:.4f}ms, "
                f"matmul_only {breakdown['matmul_only']['median_ms']:.4f}ms"
            )


def _write_json(path: Path, payload: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _resolve_moe_problem(args) -> MoeProblem:
    preset = MOE_PRESETS[args.preset]
    return MoeProblem(
        name=preset.name,
        S=args.S if args.S is not None else preset.S,
        E=args.E if args.E is not None else preset.E,
        N=args.N if args.N is not None else preset.N,
        K=args.K if args.K is not None else preset.K,
        top_k=args.top_k if args.top_k is not None else preset.top_k,
        block_n=args.block_n if args.block_n is not None else preset.block_n,
        block_k=args.block_k if args.block_k is not None else preset.block_k,
    )


def _resolve_matmul_problem(args) -> MatmulProblem:
    preset = MATMUL_PRESETS[args.matmul_preset]
    return MatmulProblem(
        name=preset.name,
        M=args.M if args.M is not None else preset.M,
        N=args.N if args.N is not None else preset.N,
        K=args.K if args.K is not None else preset.K,
        block_n=args.block_n if args.block_n is not None else preset.block_n,
        block_k=args.block_k if args.block_k is not None else preset.block_k,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark finegrained-fp8 kernels on CUDA or XPU")
    parser.add_argument("--op", choices=["matmul", "batched", "grouped", "all"], default="all")
    parser.add_argument("--device", choices=["cuda", "xpu"], default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--weight-format", choices=["fp8", "fp4"], default="fp8")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--preset", choices=sorted(MOE_PRESETS), default="deepseek_v4_gate_up_prefill")
    parser.add_argument(
        "--matmul-preset", choices=sorted(MATMUL_PRESETS), default="deepseek_v4_gate_up_prefill"
    )
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--S", type=int, default=None)
    parser.add_argument("--E", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--block-n", type=int, default=None)
    parser.add_argument("--block-k", type=int, default=None)
    parser.add_argument("--compare-eager", action="store_true")
    parser.add_argument("--include-act-quant", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def main():
    args = parse_args()
    device = _get_device(args.device)
    dtype = _dtype_from_name(args.dtype)
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")

    results = []
    if args.op in {"matmul", "all"}:
        matmul_problem = _resolve_matmul_problem(args)
        results.append(
            _bench_matmul(
                matmul_problem,
                dtype=dtype,
                device=device,
                repeats=args.repeats,
                compare_eager=args.compare_eager,
                include_act_quant=args.include_act_quant,
                check=args.check,
                weight_format=args.weight_format,
            )
        )

    if args.op in {"batched", "grouped", "all"}:
        moe_problem = _resolve_moe_problem(args)
        ops = [args.op] if args.op in {"batched", "grouped"} else ["batched", "grouped"]
        for op in ops:
            results.append(
                _bench_moe_op(
                    op,
                    moe_problem,
                    dtype=dtype,
                    device=device,
                    repeats=args.repeats,
                    compare_eager=args.compare_eager,
                    check=args.check,
                    weight_format=args.weight_format,
                )
            )

    for result in results:
        _print_result(result)

    if args.json is not None:
        _write_json(args.json, results)
        print(f"wrote JSON results to {args.json}")


if __name__ == "__main__":
    main()
