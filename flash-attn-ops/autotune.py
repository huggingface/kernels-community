"""Generate Triton configs for LayerNorm/RMSNorm forward and backward workloads."""

import argparse
import json
import math
import os
import shutil
import tempfile
import tomllib
from pathlib import Path


def collect(cache_dir: Path, output: Path):
    paths = sorted(cache_dir.glob("*/*.autotune.json"))
    if not paths:
        raise ValueError("Autotuning produced no configs")
    if output.exists():
        shutil.rmtree(output)
    for path in paths:
        data = json.loads(path.read_text())
        config, timing = min(data["configs_timings"], key=lambda item: item[1])
        if not all(math.isfinite(value) for value in timing):
            raise ValueError(f"No successful config in {path}")
        # Keep only the winner and omit timing noise, so an unchanged selection
        # produces no diff. Triton still reads this as its native cache format.
        data["configs_timings"] = [[config, [0.0, 0.0, 0.0]]]
        target = output / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def tune_workloads(kernel, workloads):
    """Exercise each tuning key; Triton's autotuner benchmarks the candidates."""
    import torch

    def run(hidden, dtype, norm, has_bias, eps, has_residual=False):
        x = torch.randn(
            workloads["rows"], hidden, device="cuda", dtype=dtype, requires_grad=True
        )
        weight = torch.randn(hidden, device="cuda", dtype=dtype, requires_grad=True)
        bias = torch.randn_like(weight, requires_grad=True) if has_bias else None
        kwargs = {}
        if has_residual:
            kwargs = {
                "residual": torch.randn_like(x, requires_grad=True),
                "prenorm": True,
            }
        outputs = norm(x, weight, bias, eps=eps, **kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # Supply gradients for both prenorm outputs to exercise HAS_DRESIDUAL.
        torch.autograd.backward(outputs, tuple(torch.randn_like(y) for y in outputs))

    for dtype_name in workloads["dtypes"]:
        dtype = getattr(torch, dtype_name)
        for hidden in workloads["hidden_sizes"]:
            for has_bias in (True, False):
                run(hidden, dtype, kernel.layer_norm_fn, has_bias, 1e-5)
            run(hidden, dtype, kernel.rms_norm_fn, False, 1e-6)
        for hidden in workloads["residual_hidden_sizes"]:
            run(hidden, dtype, kernel.layer_norm_fn, True, 1e-5, has_residual=True)
            run(hidden, dtype, kernel.rms_norm_fn, False, 1e-6, has_residual=True)
    torch.cuda.synchronize()


def tune(gpu: str):
    # Set up a fresh cache before importing the kernel so existing shipped
    # configs cannot seed this tuning run. The published variant stays read-only.
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["TRITON_CACHE_DIR"] = tmp
        os.environ["TRITON_CACHE_AUTOTUNING"] = "1"
        os.environ["FLASH_ATTN_OPS_AUTOTUNE"] = "1"

        import torch
        from kernels import get_kernel

        if torch.cuda.get_device_name() != gpu:
            raise ValueError(f"This workload must be tuned on {gpu}")
        general = tomllib.loads(Path("build.toml").read_text())["general"]
        repo_id = general["hub"]["repo-id"]
        branch = general["hub"].get("branch", f"v{general['version']}")
        # CI pins the published variant through LOCAL_KERNELS. Standalone runs
        # can also use a local build, or the published version branch.
        kernel = get_kernel(repo_id, revision=branch, backend="cuda")
        workloads = tomllib.loads(Path("autotune.toml").read_text())["workloads"]
        tune_workloads(kernel, workloads)
        output = Path(
            os.environ.get("AUTOTUNE_CONFIGS", "torch-ext/flash_attn_ops/configs")
        )
        collect(Path(tmp), output / gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default=os.environ.get("AUTOTUNE_GPU", "NVIDIA L4"))
    tune(parser.parse_args().gpu)
