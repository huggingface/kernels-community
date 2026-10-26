# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "apache-tvm-ffi==0.1.14.post1",
#     "kernels==0.17.2",
#     "torch==2.14.0",
# ]
# ///
import argparse
import gc
import hashlib
import json
import math
import os
import random
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ID = "kernels-community/flash-attn2-rs"
CASES = {
    # name: operation, B, Sq, Sk, Hq, Hkv, D, dtype, dropout, causal
    "forward-short": ("forward", 1, 17, 23, 4, 2, 64, "float16", 0.0, True),
    "forward-dense": ("forward", 2, 1024, 1024, 8, 8, 128, "float16", 0.0, False),
    "forward-causal": ("forward", 2, 1024, 1024, 8, 8, 128, "float16", 0.0, True),
    "forward-gqa": ("forward", 2, 1024, 1024, 8, 2, 128, "float16", 0.0, True),
    "forward-bf16": ("forward", 2, 1024, 1024, 8, 2, 128, "bfloat16", 0.0, True),
    "dropout-short": ("forward", 1, 17, 23, 4, 2, 64, "float16", 0.17, True),
    "dropout-long": ("forward", 2, 1024, 1024, 8, 2, 128, "float16", 0.17, True),
    "backward": ("backward", 2, 1024, 1024, 8, 2, 128, "float16", 0.0, False),
    "backward-dropout": ("backward", 2, 1024, 1024, 8, 2, 128, "float16", 0.17, True),
    "packed": ("packed", 2, 128, 256, 8, 2, 64, "float16", 0.17, True),
    "packed-backward": (
        "packed-backward",
        2,
        128,
        256,
        8,
        2,
        64,
        "float16",
        0.17,
        True,
    ),
    "decode": ("decode", 2, 1, 2048, 8, 2, 128, "float16", 0.0, False),
    "paged": ("paged", 2, 32, 1024, 8, 2, 64, "float16", 0.0, True),
    # Extended cases are opt-in, or selected individually with --cases.
    "decode-append": ("decode-append", 2, 1, 2048, 8, 2, 128, "float16", 0.0, False),
    "forward-d192": ("forward", 2, 1024, 1024, 8, 8, 192, "float16", 0.0, False),
    "cross-d32": ("forward", 1, 64, 2048, 4, 2, 32, "float16", 0.0, True),
    "forward-4096": ("forward", 2, 4096, 4096, 8, 8, 128, "float16", 0.0, True),
}
QUICK = list(CASES)[:13]
FIELDS = [
    "operation",
    "batch",
    "seqlen_q",
    "seqlen_k",
    "heads",
    "kv_heads",
    "head_dim",
    "dtype",
    "dropout",
    "causal",
]


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_info(path):
    """Fingerprint actual bytes, not merely a git revision or mutable symlink."""
    path = Path(path).resolve(strict=True)
    files = {}
    for entry in sorted(path.rglob("*")):
        if (
            not entry.is_file()
            or "__pycache__" in entry.parts
            or entry.suffix == ".pyc"
        ):
            continue
        files[entry.relative_to(path).as_posix()] = digest_file(entry)
    if not files:
        raise ValueError(f"empty artifact: {path}")
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    metadata = {
        p.relative_to(path).as_posix(): json.loads(p.read_text())
        for p in path.rglob("metadata.json")
    }
    return {
        "path": str(path),
        "sha256": fingerprint,
        "files": files,
        "metadata": metadata,
    }


def freeze(source, destination):
    source = Path(source).resolve(strict=True)
    destination = destination.absolute()
    if (
        destination.exists()
        or destination.is_symlink()
        or destination.with_name(destination.name + ".manifest.json").exists()
    ):
        raise ValueError(f"baseline destination already exists: {destination}")
    if destination.is_relative_to(source):
        raise ValueError("baseline destination must be outside the source artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        snapshot = Path(tmp) / "artifact"
        shutil.copytree(
            source, snapshot, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        info = artifact_info(snapshot)
        assert info["sha256"] == artifact_info(source)["sha256"]
        # A sibling manifest avoids changing the kernel's own signed file set.
        snapshot.chmod(0o755)
        snapshot.rename(destination)
    info["path"] = str(destination)
    destination.with_name(destination.name + ".manifest.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    print(f"Frozen {destination}\nSHA256 {info['sha256']}")


def telemetry():
    fields = "uuid,name,driver_version,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used"
    try:
        return subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_activity(gpu_uuid, worker_pids):
    """Snapshot other GPU processes at sample boundaries, outside timing."""
    if not gpu_uuid:
        return {"available": False, "reason": "device UUID unavailable"}
    # Torch exposes the UUID without NVIDIA's GPU- prefix on some versions.
    if not gpu_uuid.startswith(("GPU-", "MIG-")):
        gpu_uuid = "GPU-" + gpu_uuid
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "pmon", "-i", gpu_uuid, "-c", "1", "-s", "u"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": str(error)}
    active = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        pid = int(fields[1])
        sm, memory = (int(x) if x.isdigit() else 0 for x in fields[3:5])
        if pid not in worker_pids and (sm > 0 or memory > 0):
            active.append({"pid": pid, "sm_percent": sm, "memory_percent": memory})
    return {"available": True, "active_other_processes": active, "raw": raw.strip()}


def fixture(case, seed):
    """One CPU fixture shared by both workers."""
    import torch

    op, b, sq, sk, h, hk, d, dtype, _, _ = case
    generator = torch.Generator().manual_seed(seed)

    def rand(shape):
        return torch.randn(shape, generator=generator, dtype=getattr(torch, dtype))

    q, k, v = rand((b, sq, h, d)), rand((b, sk, hk, d)), rand((b, sk, hk, d))
    data = {"q": q, "k": k, "v": v, "dout": rand(q.shape)}
    if op.startswith("packed") or op == "paged":
        # Unequal lengths exercise packed offsets.
        qlens, klens = [sq - i for i in range(b)], [sk - i for i in range(b)]
        data.update(
            q=torch.cat([q[i, :n] for i, n in enumerate(qlens)]),
            dout=torch.cat([data["dout"][i, :n] for i, n in enumerate(qlens)]),
            cuq=torch.tensor([0, *accumulate(qlens)], dtype=torch.int32),
            cuk=torch.tensor([0, *accumulate(klens)], dtype=torch.int32),
        )
        if op == "paged":
            data.update(
                k=k.reshape(-1, 256, hk, d),
                v=v.reshape(-1, 256, hk, d),
                table=torch.arange(b * sk // 256, dtype=torch.int32).reshape(b, -1),
            )
        else:
            data.update(
                k=torch.cat([k[i, :n] for i, n in enumerate(klens)]),
                v=torch.cat([v[i, :n] for i, n in enumerate(klens)]),
            )
    if op in ("decode", "decode-append"):
        data["lengths"] = torch.full(
            (b,), sk - sq if op == "decode-append" else sk, dtype=torch.int32
        )
        if op == "decode-append":
            data.update(new_k=rand((b, sq, hk, d)), new_v=rand((b, sq, hk, d)))
    return data


def check_cache_append(data, original):
    """Check complete cache contents outside timing, including untouched prefixes."""
    import torch

    for name in ("k", "v"):
        expected = original[name].clone()
        new = original[f"new_{name}"]
        for batch, length in enumerate(original["lengths"].tolist()):
            expected[batch, length : length + new.shape[1]].copy_(new[batch])
        assert torch.equal(data[name].cpu(), expected), (
            f"incorrect {name} cache mutation"
        )
    assert torch.equal(data["lengths"].cpu(), original["lengths"]), (
        "cache lengths changed"
    )


def accumulate(values):
    total = 0
    for value in values:
        total += value
        yield total


def cpu_result(result):
    return [x.detach().cpu() for x in result]


def check_results(actual, expected, dtype, operation):
    import torch

    if len(actual) != len(expected):
        raise AssertionError("different result counts")
    atol, rtol = (
        (0.06, 0.08)
        if dtype == "bfloat16"
        else ((0.008, 0.04) if "backward" in operation else (0.005, 0.03))
    )
    errors = []
    for x, y in zip(actual, expected):
        exact = not x.is_floating_point()
        torch.testing.assert_close(
            x, y, atol=0 if exact else atol, rtol=0 if exact else rtol, equal_nan=False
        )
        delta = (x.float() - y.float()).abs()
        finite = delta[torch.isfinite(delta)]
        errors.append(float(finite.max()) if finite.numel() else 0.0)
    return {"atol": atol, "rtol": rtol, "max_abs_errors": errors}


def make_call(module, case, data, saved, deterministic):
    """Both artifacts must return the same defined outputs."""
    op, _, sq, sk, _, _, _, _, p, causal = case
    q, k, v = (data[key] for key in ("q", "k", "v"))
    kwargs = {"p_dropout": p, "is_causal": causal}
    if op.startswith("packed") or op == "paged":
        kwargs.update(max_seqlen_q=sq, max_seqlen_k=sk, zero_tensors=True)
        if op == "paged":
            kwargs["block_table"] = data["table"]
        forward = lambda: module.varlen_fwd(q, k, v, data["cuq"], data["cuk"], **kwargs)
    else:
        forward = lambda: module.fwd(q, k, v, **kwargs)
    if "backward" in op:
        if saved is None:
            saved = forward()
        args = (data["dout"], q, k, v, saved[0], saved[1])
        if op == "packed-backward":
            call = lambda: module.varlen_bwd(
                *args,
                data["cuq"],
                data["cuk"],
                rng_state=saved[3] if p else None,
                deterministic=deterministic,
                **kwargs,
            )[:3]
        else:
            call = lambda: module.bwd(
                *args,
                rng_state=saved[3] if p else None,
                deterministic=deterministic,
                **kwargs,
            )[:3]
        # Reuse forward inputs so forward diffs cannot mask backward errors.
        return call, saved
    if op == "decode":
        return lambda: module.fwd_kvcache(q, k, v, seqlens_k=data["lengths"]), None
    if op == "decode-append":
        # v2 keeps lengths, so repeated calls overwrite the same slot.
        return lambda: module.fwd_kvcache(
            q, k, v, k=data["new_k"], v=data["new_v"], seqlens_k=data["lengths"]
        ), None

    def call():
        outputs = forward()
        return [*outputs[:2], outputs[3]] if p else outputs[:2]

    return call, None


def worker(args):
    import kernels
    import torch
    import tvm_ffi

    torch.set_num_threads(1)
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    gc.disable()
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    start = time.perf_counter()
    # The parent maps REPO_ID to this worker's artifact with LOCAL_KERNELS.
    module = kernels.get_kernel(REPO_ID, version=1)
    load_ms = (time.perf_counter() - start) * 1000
    props = torch.cuda.get_device_properties(args.device)
    context = {}

    def reply(value):
        print(json.dumps(value), flush=True)

    reply(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": sys.version,
            "kernels": kernels.__version__,
            "tvm_ffi": tvm_ffi.__version__,
            "gpu": props.name,
            "gpu_uuid": str(props.uuid) if hasattr(props, "uuid") else None,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "torch_threads": torch.get_num_threads(),
            "gc_enabled": gc.isenabled(),
            "capability": [props.major, props.minor],
            "total_memory": props.total_memory,
            "load_ms": load_ms,
            "module_file": module.__file__,
            "cuda_environment": {
                k: v for k, v in os.environ.items() if k.startswith("CUDA_")
            },
            "telemetry": telemetry(),
        }
    )
    for line in sys.stdin:
        request = json.loads(line)
        action = request["action"]
        if action == "prepare":
            case = request["case"]
            data = torch.load(request["fixture"], weights_only=True)
            data = {key: value.to(args.device) for key, value in data.items()}
            saved_path = Path(request["saved"])
            saved = (
                torch.load(saved_path, weights_only=True)
                if saved_path.exists()
                else None
            )
            if saved is not None:
                saved = [x.to(args.device) for x in saved]
            torch.cuda.manual_seed(args.seed)
            call, saved = make_call(module, case, data, saved, args.deterministic)
            if saved is not None and not saved_path.exists():
                state = (
                    saved[3]
                    if case[8]
                    else torch.zeros(2, device=args.device, dtype=torch.int64)
                )
                torch.save(
                    cpu_result(
                        [saved[0], saved[1], torch.empty(0, device=args.device), state]
                    ),
                    saved_path,
                )
            torch.cuda.synchronize()
            torch.cuda.manual_seed(args.seed)
            before = torch.cuda.get_rng_state()
            start = time.perf_counter()
            outputs = call()
            torch.cuda.synchronize()
            first_ms = (time.perf_counter() - start) * 1000
            torch.save(
                {"outputs": cpu_result(outputs), "rng": torch.cuda.get_rng_state()},
                request["result"],
            )
            deadline = time.perf_counter() + args.warmup_ms / 1000
            warmed = 0
            while warmed < args.warmup or time.perf_counter() < deadline:
                call()
                warmed += 1
            torch.cuda.synchronize()
            graph = None
            if args.mode in ("both", "graph"):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(args.graph_calls):
                        graph_outputs = call()
                # Validate graph replay and RNG advancement too.
                torch.cuda.set_rng_state(before)
                for _ in range(args.graph_calls):
                    expected = call()
                expected = cpu_result(expected)
                after = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(before)
                graph.replay()
                check_results(cpu_result(graph_outputs), expected, case[7], case[0])
                assert torch.equal(after, torch.cuda.get_rng_state()), (
                    "graph RNG mismatch"
                )
            cache_append = case[0] == "decode-append"
            if cache_append:
                check_cache_append(
                    data, torch.load(request["fixture"], weights_only=True)
                )
            context = {"call": call, "graph": graph, "data": data, "saved": saved}
            reply(
                {
                    "first_call_ms": first_ms,
                    "graph_correctness": graph is not None,
                    "cache_mutation_correctness": True if cache_append else None,
                }
            )
        elif action == "measure":
            mode, count = request["mode"], request["count"]
            torch.cuda.manual_seed(args.seed)
            torch.cuda.synchronize()
            # Diagnostics only: they never filter samples or alter verdicts.
            usage_before = resource.getrusage(resource.RUSAGE_THREAD)
            cpu_start = time.thread_time_ns()
            host_start = time.perf_counter_ns()
            enqueue_end = None
            if mode == "eager":
                for _ in range(count):
                    context["call"]()
                enqueue_end = time.perf_counter_ns()
                torch.cuda.synchronize()
                host_end = time.perf_counter_ns()
                us = (host_end - host_start) / (1000 * count)
            else:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(count):
                    context["graph"].replay()
                end.record()
                end.synchronize()
                us = start.elapsed_time(end) * 1000 / (count * args.graph_calls)
                host_end = time.perf_counter_ns()
            cpu_end = time.thread_time_ns()
            usage_after = resource.getrusage(resource.RUSAGE_THREAD)
            diagnostics = {
                "host_wall_ms": (host_end - host_start) / 1e6,
                "thread_cpu_ms": (cpu_end - cpu_start) / 1e6,
                "voluntary_context_switches": usage_after.ru_nvcsw
                - usage_before.ru_nvcsw,
                "involuntary_context_switches": usage_after.ru_nivcsw
                - usage_before.ru_nivcsw,
                "minor_page_faults": usage_after.ru_minflt - usage_before.ru_minflt,
                "major_page_faults": usage_after.ru_majflt - usage_before.ru_majflt,
            }
            if enqueue_end is not None:
                diagnostics["enqueue_wall_ms"] = (enqueue_end - host_start) / 1e6
                diagnostics["synchronize_wall_ms"] = (host_end - enqueue_end) / 1e6
            reply({"us": us, "host_diagnostics": diagnostics})
        elif action == "profile":
            # Profiling runs after measurement, never as a sample.
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profile:
                context["call"]()
                torch.cuda.synchronize()
            profile.export_chrome_trace(request["path"])
            reply({"trace": request["path"]})
        elif action == "finish":
            context.clear()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            reply({"telemetry": telemetry()})
        else:
            raise ValueError(action)


class Worker:
    def __init__(
        self, args, artifact, python, directory, label, clear_pythonpath=False
    ):
        env = dict(os.environ)
        env["LOCAL_KERNELS"] = f"{REPO_ID}={Path(artifact).resolve()}"
        if clear_pythonpath:
            env.pop("PYTHONPATH", None)
        if args.cold:
            env["CUDA_CACHE_PATH"] = str(directory / (label + "-cuda-cache"))
        command = [
            python,
            str(Path(__file__).resolve()),
            "--worker",
            "--device",
            args.device,
            "--mode",
            args.mode,
            "--seed",
            str(args.seed),
            "--warmup",
            str(args.warmup),
            "--warmup-ms",
            str(args.warmup_ms),
            "--graph-calls",
            str(args.graph_calls),
        ]
        if args.cpu is not None:
            command += ["--cpu", str(args.cpu)]
        if args.deterministic:
            command += ["--deterministic"]
        self.process = subprocess.Popen(
            command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        self.metadata = self.read()

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("benchmark worker exited; see its stderr above")
        return json.loads(line)

    def ask(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lo = int(position)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (
        position - lo
    )


def summarize(samples, threshold, max_noise, identical, comparable):
    if len(samples[0]) != len(samples[1]) or len(samples[0]) < 5:
        raise ValueError("need at least five paired samples")
    if any(not math.isfinite(x) or x <= 0 for group in samples for x in group):
        raise ValueError("timing samples must be positive and finite")
    ratios = [candidate / baseline for baseline, candidate in zip(*samples)]
    ratio = statistics.median(ratios)
    randomizer = random.Random(0)
    bootstraps = [
        statistics.median(randomizer.choices(ratios, k=len(ratios)))
        for _ in range(1000)
    ]
    ci = [percentile(bootstraps, 0.025), percentile(bootstraps, 0.975)]
    spread = (percentile(ratios, 0.9) - percentile(ratios, 0.1)) / ratio
    status = "inconclusive"
    if not comparable:
        status = "environment-mismatch"
    elif spread > max_noise or identical and abs(ratio - 1) > max_noise:
        status = "unstable"
    elif identical:
        status = "aa-pass"
    elif ci[0] > 1 + threshold:
        status = "regression"
    elif ci[1] < 1 - threshold:
        status = "improvement"
    return {
        "baseline_us": statistics.median(samples[0]),
        "candidate_us": statistics.median(samples[1]),
        "paired_ratio": ratio,
        "ratio_ci95": ci,
        "ratio_p10_p90_spread": spread,
        "status": status,
        "samples_us": {"baseline": samples[0], "candidate": samples[1]},
    }


def sample_count(costs, calls, target_ms):
    """Bound a batch using the faster worker's measured per-call cost."""
    return max(5, min(10000, math.ceil(target_ms * 1000 / (min(costs) * calls))))


def run(args):
    import torch

    torch.set_num_threads(1)
    names = args.cases or (list(CASES) if args.suite == "full" else QUICK)
    unknown = set(names) - CASES.keys()
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}; use --list")
    artifacts = [artifact_info(args.baseline), artifact_info(args.candidate)]
    for info in artifacts:
        path = Path(info["path"])
        manifest = path.with_name(path.name + ".manifest.json")
        if (
            manifest.exists()
            and json.loads(manifest.read_text())["sha256"] != info["sha256"]
        ):
            raise ValueError(f"frozen artifact changed: {path}")
    identical = artifacts[0]["sha256"] == artifacts[1]["sha256"]
    report = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": {"baseline": artifacts[0], "candidate": artifacts[1]},
        "aa": identical,
        "runner_sha256": digest_file(Path(__file__)),
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "timing_scope": {
            "eager": "synchronized public-API wall time, including validation, allocation and RNG",
            "graph": "CUDA events around replay; Python validation/allocation excluded",
            "backward": f"public backward, deterministic={args.deterministic}, preprocessing included; forward excluded",
            "host": "one Torch CPU thread; cyclic GC disabled; recorded CPU affinity",
            "inputs": "fixed resident inputs, same CPU fixture; no append mutation",
            "cold": "fresh workers/cache directory per case only with --cold; first-call timings include lazy work, backward forward-setup excluded",
            "ratio": "candidate / baseline latency; lower is better",
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    workers = []
    try:
        with tempfile.TemporaryDirectory(prefix="oxide-benchmark-") as tmp:
            directory = Path(tmp)
            for index, name in enumerate(names):
                report["active_case"] = name
                if not workers:
                    for info, python, label in zip(
                        artifacts,
                        (args.baseline_python or sys.executable, sys.executable),
                        ("baseline", "candidate"),
                    ):
                        workers.append(
                            Worker(
                                args,
                                info["path"],
                                python,
                                directory,
                                f"{label}-{index}",
                                clear_pythonpath=(
                                    label == "baseline"
                                    and args.baseline_python is not None
                                ),
                            )
                        )
                case = CASES[name]
                path = directory / f"{name}.pt"
                torch.save(fixture(case, args.seed), path)
                row = {
                    "name": name,
                    "config": dict(zip(FIELDS, case)),
                    "workers": [w.metadata for w in workers],
                }
                comparable = all(
                    workers[0].metadata[k] == workers[1].metadata[k]
                    for k in (
                        "torch",
                        "cuda",
                        "python",
                        "kernels",
                        "tvm_ffi",
                        "cpu_affinity",
                        "torch_threads",
                        "gpu",
                        "gpu_uuid",
                        "capability",
                        "cuda_environment",
                    )
                )
                # Fresh cache paths differ per worker in cold mode.
                if args.cold:
                    left = {
                        **workers[0].metadata,
                        "cuda_environment": {
                            k: v
                            for k, v in workers[0].metadata["cuda_environment"].items()
                            if k != "CUDA_CACHE_PATH"
                        },
                    }
                    right = {
                        **workers[1].metadata,
                        "cuda_environment": {
                            k: v
                            for k, v in workers[1].metadata["cuda_environment"].items()
                            if k != "CUDA_CACHE_PATH"
                        },
                    }
                    comparable = all(
                        left[k] == right[k]
                        for k in (
                            "torch",
                            "cuda",
                            "python",
                            "kernels",
                            "tvm_ffi",
                            "cpu_affinity",
                            "torch_threads",
                            "gpu",
                            "gpu_uuid",
                            "capability",
                            "cuda_environment",
                        )
                    )
                results, preparation = [], []
                for i, w in enumerate(workers):
                    result = directory / f"{name}-{i}.pt"
                    preparation.append(
                        w.ask(
                            action="prepare",
                            case=case,
                            fixture=str(path),
                            saved=str(directory / f"{name}-saved.pt"),
                            result=str(result),
                        )
                    )
                    results.append(torch.load(result, weights_only=True))
                row["correctness"] = check_results(
                    results[1]["outputs"], results[0]["outputs"], case[7], case[0]
                )
                assert torch.equal(results[0]["rng"], results[1]["rng"]), (
                    "generator advancement differs"
                )
                row["preparation"] = preparation
                row["measurements"] = {}
                for mode in ("eager", "graph") if args.mode == "both" else (args.mode,):
                    worker_pids = {w.process.pid for w in workers}
                    gpu_uuid = workers[0].metadata["gpu_uuid"]
                    activity_before = gpu_activity(gpu_uuid, worker_pids)
                    calibration = [
                        w.ask(action="measure", mode=mode, count=5)["us"]
                        for w in workers
                    ]
                    calls = args.graph_calls if mode == "graph" else 1

                    # Recalibrate warm; the first calls include startup.
                    calibration_count = sample_count(calibration, calls, args.target_ms)
                    warm_calibration = [
                        w.ask(action="measure", mode=mode, count=calibration_count)[
                            "us"
                        ]
                        for w in workers
                    ]
                    count = sample_count(warm_calibration, calls, args.target_ms)
                    samples = [[], []]
                    diagnostics = [[], []]
                    for repetition in range(args.repeats):
                        order = (0, 1) if (repetition + index) % 2 == 0 else (1, 0)
                        for i in order:
                            sample = workers[i].ask(
                                action="measure", mode=mode, count=count
                            )
                            samples[i].append(sample["us"])
                            diagnostics[i].append(sample["host_diagnostics"])
                    measurement = summarize(
                        samples, args.threshold, args.max_noise, identical, comparable
                    )
                    activity_after = gpu_activity(gpu_uuid, worker_pids)
                    measurement["gpu_activity"] = {
                        "before": activity_before,
                        "after": activity_after,
                    }
                    if any(
                        x.get("active_other_processes")
                        for x in (activity_before, activity_after)
                    ):
                        measurement["statistical_status"] = measurement["status"]
                        measurement["status"] = "interference"
                    measurement["calibration"] = {
                        "initial_calls": 5,
                        "initial_us": calibration,
                        "warm_calls": calibration_count,
                        "warm_us": warm_calibration,
                    }
                    measurement["host_diagnostics"] = dict(
                        zip(("baseline", "candidate"), diagnostics)
                    )
                    measurement["iterations_per_sample"] = count
                    row["measurements"][mode] = measurement
                    print(
                        f"{name:20} {mode:5} {measurement['baseline_us']:9.2f} -> {measurement['candidate_us']:9.2f} us  {measurement['paired_ratio']:.3f}x  {measurement['status']}",
                        flush=True,
                    )
                if args.profile_dir is not None:
                    args.profile_dir.mkdir(parents=True, exist_ok=True)
                    row["profiles"] = [
                        w.ask(
                            action="profile",
                            path=str(
                                (args.profile_dir / f"{name}-{label}.json").resolve()
                            ),
                        )
                        for label, w in zip(("baseline", "candidate"), workers)
                    ]
                row["after"] = [w.ask(action="finish") for w in workers]
                report["cases"].append(row)
                save()
                if args.cold:
                    for w in workers:
                        w.close()
                    workers.clear()
    except Exception as error:
        report["error"] = str(error)
        save()
        raise
    finally:
        for w in workers:
            w.close()
    statuses = [
        m["status"] for row in report["cases"] for m in row["measurements"].values()
    ]
    report["summary"] = {
        status: statuses.count(status) for status in sorted(set(statuses))
    }
    save()
    print(f"Results: {args.output}\n{report['summary']}")
    if args.fail_on_regression and any(
        x in statuses
        for x in ("regression", "unstable", "environment-mismatch", "interference")
    ):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Paired, correctness-gated benchmarks of FlashAttention v2 artifacts."
    )
    parser.add_argument("--candidate", type=Path, default=Path("result"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--baseline-python",
        default=None,
        help="alternate baseline Python; inherited PYTHONPATH is cleared",
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        help="copy candidate to a fixed baseline directory and exit",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="export eager CPU/CUDA traces after timing; use with a selected case",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=948)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--warmup-ms", type=float, default=50)
    parser.add_argument(
        "--cpu",
        type=int,
        help="worker CPU affinity; defaults to first allowed CPU on Linux",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="benchmark deterministic backward instead of the default path",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--target-ms", type=float, default=10)
    parser.add_argument("--graph-calls", type=int, default=10)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="minimum meaningful latency change",
    )
    parser.add_argument(
        "--max-noise",
        type=float,
        default=0.05,
        help="maximum relative paired p10–p90 spread",
    )
    parser.add_argument(
        "--cold",
        action="store_true",
        help="fresh workers and CUDA cache directory per case",
    )
    parser.add_argument("--fail-on-regression", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list:
        for name, case in CASES.items():
            print(name, json.dumps(dict(zip(FIELDS, case))))
        return
    if args.freeze:
        freeze(args.candidate, args.freeze)
        return
    if (
        args.warmup < 1
        or args.repeats < 5
        or args.graph_calls < 1
        or not 0 < args.target_ms <= 1000
    ):
        parser.error(
            "require warmup/graph-calls >= 1, repeats >= 5, and target-ms in (0, 1000]"
        )
    if not 0 < args.threshold < 1 or not 0 < args.max_noise < 1:
        parser.error("threshold and max-noise must be in (0, 1)")
    if args.cpu is None and hasattr(os, "sched_getaffinity"):
        args.cpu = min(os.sched_getaffinity(0))
    if args.warmup_ms < 0 or not math.isfinite(args.warmup_ms):
        parser.error("warmup-ms must be finite and nonnegative")
    if args.worker:
        worker(args)
    elif args.baseline is None or args.output is None:
        parser.error("comparison requires --baseline and --output; use --freeze first")
    else:
        run(args)


if __name__ == "__main__":
    main()
