"""Compare published Triton MHA with local FlyDSL, including FP8 producer cost.

Run with LOCAL_KERNELS pointing at the new package's kernel-builder output.
Results are written after every case so an interrupted long run remains useful.
"""

import argparse
import gc
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import kernels
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO = "kernels-community/aiter-flash-attn-flydsl"
BASELINE = "amd/aiter-flash-attn"


def sdpa(q, k, v, causal=False):
    return (
        F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
        )
        .transpose(1, 2)
        .contiguous()
    )


def fp32_reference(q, k, v, causal, chunk=256):
    """Chunk queries to avoid allocating a full quadratic FP32 score matrix."""
    kh = k.float().transpose(1, 2)
    vh = v.float().transpose(1, 2)
    result = torch.empty_like(q, dtype=torch.float32)
    columns = torch.arange(k.shape[1], device=q.device)
    for start in range(0, q.shape[1], chunk):
        stop = min(start + chunk, q.shape[1])
        scores = (
            q[:, start:stop].float().transpose(1, 2) @ kh.transpose(-1, -2)
        ) * q.shape[-1] ** -0.5
        if causal:
            rows = torch.arange(start, stop, device=q.device)
            scores.masked_fill_(columns[None, :] > rows[:, None], -float("inf"))
        result[:, start:stop] = (scores.softmax(-1) @ vh).transpose(1, 2)
    return result


def measure(fn, repeats, warmup, timing="events"):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - start) * 1000
    del out
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        fn()
        end_event.record()
        end_event.synchronize()
        samples.append(start_event.elapsed_time(end_event))
    result = {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "first_call_ms": first_ms,
        "peak_extra_allocated_bytes": max(
            0, torch.cuda.max_memory_allocated() - before
        ),
    }
    if timing == "profiler":
        # AITER run_perftest counts GPU durations, excluding Python dispatch gaps.
        # Reuse inputs: one rotation slot, rather than AITER's automatic rotation.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(repeats):
                fn()
            torch.cuda.synchronize()
        events = [
            e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        if not events:
            raise RuntimeError("Profiler recorded no GPU activity")
        # Native FlyDSL launches do not always carry CPU scope correlation.
        # Sum the whole steady-state trace, including every quantization kernel.
        result["mean_kernel_ms"] = (
            sum(e.device_time_total for e in events) / repeats / 1000
        )
        result["profile_gpu_events"] = len(events)
    return result


def accuracy(out, ref):
    if not torch.isfinite(out).all():
        raise RuntimeError("non-finite attention output")
    cos = F.cosine_similarity(out.float().flatten(0, 2), ref.flatten(0, 2), dim=-1)
    return {
        "relative_l2": ((out.float() - ref).norm() / ref.norm()).item(),
        "min_row_cosine": cos.min().item(),
        "mean_row_cosine": cos.mean().item(),
    }


def latency(provider):
    return provider.get("mean_kernel_ms", provider.get("median_ms"))


def write_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# gfx1201 attention benchmark",
        "",
        "Baseline: published `amd/aiter-flash-attn` v2 at `"
        + results["environment"]["baseline_revision"]
        + "`.",
        "FP8 total includes Q/K/V quantization with Hadamard Q/K rotation. Timing: "
        + results["environment"]["timing"]
        + "; warmup excluded; inputs reused.",
        "",
        "| dtype | Sq / Sk | H / D | causal | provider | latency ms | vs Hub Triton | relative L2 | min row cosine |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for case in results["cases"]:
        baseline_ms = latency(case["providers"].get("hub_triton", {}))
        for name, provider in case["providers"].items():
            if "error" in provider:
                lines.append(
                    f"| {case['dtype']} | {case['sq']} / {case['sk']} | {case['heads']} / {case['dim']} | {case['causal']} | {name} | unsupported/error | — | — | — |"
                )
                continue
            speedup = f"{baseline_ms / latency(provider):.2f}x" if baseline_ms else "—"
            err = provider.get("relative_l2")
            cosine = provider.get("min_row_cosine")
            lines.append(
                f"| {case['dtype']} | {case['sq']} / {case['sk']} | {case['heads']} / {case['dim']} | {case['causal']} | {name} | {latency(provider):.3f} | {speedup} | "
                + (f"{err:.5f}" if err is not None else "—")
                + " | "
                + (f"{cosine:.5f}" if cosine is not None else "—")
                + " |"
            )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument(
        "--seq", type=int, nargs="+", default=[1024, 4096, 16384, 32768]
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--extra-cases",
        action="store_true",
        help="Also run causal self-attention and cross-attention.",
    )
    parser.add_argument("--timing", choices=["events", "profiler"], default="events")
    parser.add_argument(
        "--aiter-shapes",
        action="store_true",
        help="Use upstream Flux/Wan/SD3.5 BF16 shapes.",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":")[0] != "gfx1201":
        raise RuntimeError(f"expected gfx1201, got {props.gcnArchName}")
    baseline_revision = (
        HfApi().repo_info(BASELINE, repo_type="kernel", revision="v2").sha
    )
    baseline = kernels.get_kernel(BASELINE, revision=baseline_revision)
    candidate = kernels.get_kernel(REPO, version=1)
    results = {
        "environment": {
            "gpu": props.name,
            "arch": props.gcnArchName,
            "gpu_memory_bytes": props.total_memory,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "rocm": torch.version.rocm,
            "timing": args.timing,
            "reported_statistic": "mean GPU duration"
            if args.timing == "profiler"
            else "median CUDA event duration",
            "input_rotation_slots": 1,
            "flydsl": "kernels-sdk/flydsl (kernel dependency)",
            "kernels": importlib.metadata.version("kernels"),
            "triton": __import__("triton").__version__,
            "baseline_revision": baseline_revision,
            "candidate_aiter_revision": "c1096fa40a907cf83cf95d6d87c4fad5823e7054",
            "repeats": args.repeats,
            "warmup": args.warmup,
            "seed": 42,
            "tf32": False,
            "candidate_daz": True,
        },
        "cases": [],
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    shapes = [(s, s, args.heads, d, False) for s in args.seq for d in args.dims]
    if args.extra_cases:
        shapes.extend(
            [
                (4096, 4096, args.heads, 128, True),
                (4096, 512, args.heads, 128, False),
                (4093, 4093, args.heads, 128, False),
            ]
        )
    if args.aiter_shapes:
        shapes = [
            (1536, 1536, 24, 128, False),
            (4096, 4096, 24, 128, False),
            (8192, 8192, 12, 128, False),
            (32768, 32768, 12, 128, False),
            (2048, 2048, 38, 64, False),
        ]
    with torch.inference_mode():
        for dtype in (
            (torch.bfloat16,) if args.aiter_shapes else (torch.bfloat16, torch.float16)
        ):
            for sq, sk, heads, dim, causal in shapes:
                torch.manual_seed(42)
                q = torch.randn(1, sq, heads, dim, dtype=dtype, device="cuda")
                k = torch.randn(1, sk, heads, dim, dtype=dtype, device="cuda")
                v = torch.randn_like(k)
                ref = fp32_reference(q, k, v, causal)
                case = {
                    "dtype": str(dtype).split(".")[-1],
                    "sq": sq,
                    "sk": sk,
                    "heads": heads,
                    "dim": dim,
                    "causal": causal,
                    "providers": {},
                }
                providers = {
                    "hub_triton": lambda: baseline.flash_attn_func(
                        q, k, v, causal=causal
                    ),
                    "flydsl": lambda: candidate.flash_attn_func(q, k, v, causal=causal),
                    "sdpa": lambda: sdpa(q, k, v, causal),
                }
                qb, kb, vb = (x.transpose(1, 2).contiguous() for x in (q, k, v))

                def sdpa_flash():
                    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                        return F.scaled_dot_product_attention(
                            qb, kb, vb, is_causal=causal
                        )

                providers["sdpa_flash"] = sdpa_flash
                if dtype == torch.bfloat16:
                    case["providers"]["fp8_quant"] = measure(
                        lambda: candidate.fp8_quant(q, k, v, rotation=True),
                        args.repeats,
                        args.warmup,
                        args.timing,
                    )
                    quant = candidate.fp8_quant(q, k, v, rotation=True)
                    q8, k8, v8, qs, ks, vs = quant

                    def fp8_attention():
                        return candidate.flash_attn_func(
                            q8,
                            k8,
                            v8,
                            causal=causal,
                            q_descale=qs,
                            k_descale=ks,
                            v_descale=vs,
                        )

                    def fp8_total():
                        aq, ak, av, aqs, aks, avs = candidate.fp8_quant(
                            q, k, v, rotation=True
                        )
                        return candidate.flash_attn_func(
                            aq,
                            ak,
                            av,
                            causal=causal,
                            q_descale=aqs,
                            k_descale=aks,
                            v_descale=avs,
                        )

                    providers.update(
                        {"fp8_attention": fp8_attention, "fp8_total": fp8_total}
                    )
                for name, fn in providers.items():
                    try:
                        result = measure(fn, args.repeats, args.warmup, args.timing)
                        output = fn()
                        if name == "sdpa_flash":
                            output = output.transpose(1, 2)
                        result.update(accuracy(output, ref))
                        del output
                        case["providers"][name] = result
                    except (ValueError, NotImplementedError) as error:
                        case["providers"][name] = {
                            "error": f"{type(error).__name__}: {error}"
                        }
                    print(
                        case["dtype"],
                        sq,
                        sk,
                        dim,
                        causal,
                        name,
                        case["providers"][name],
                        flush=True,
                    )
                results["cases"].append(case)
                write_results(args.output, results)
                providers.clear()
                if dtype == torch.bfloat16:
                    quant = q8 = k8 = v8 = qs = ks = vs = None
                q = k = v = qb = kb = vb = ref = None
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
