"""Tune the existing LayerNorm/RMSNorm problem space and verify cache reuse."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
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


def tune(gpu: str):
    import torch
    from kernels import get_kernel

    if torch.cuda.get_device_name() != gpu:
        raise ValueError(f"This workload must be tuned on {gpu}")
    general = tomllib.loads(Path("build.toml").read_text())["general"]
    repo_id = general["hub"]["repo-id"]
    branch = general["hub"].get("branch", f"v{general['version']}")
    # The workflow pins a downloaded variant through LOCAL_KERNELS. Standalone
    # runs can also use an existing local build, or the published version branch.
    module = get_kernel(repo_id, revision=branch, backend="cuda")
    variant = Path(module.__file__).parent
    package = Path(
        os.environ.get("AUTOTUNE_CONFIGS", "torch-ext/flash_attn_ops/configs")
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        env = dict(
            os.environ,
            LOCAL_KERNELS=f"{repo_id}={variant}",
            TRITON_CACHE_DIR=str(tmp / "tune"),
            TRITON_CACHE_AUTOTUNING="1",
            FLASH_ATTN_OPS_AUTOTUNE="1",
        )
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_layer_norm.py", "-qx"],
            env=env,
            check=True,
        )
        collect(tmp / "tune", package / gpu)

        # Verify only the serialized configs in a fresh process. Packaging and
        # loading configs from the distributed kernel are release-time work.
        shutil.copytree(package / gpu, tmp / "verify")
        env["TRITON_CACHE_DIR"] = str(tmp / "verify")
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from unittest.mock import patch; import pytest; "
                "p = patch('triton.runtime.autotuner.Autotuner._bench', side_effect=AssertionError('autotune cache miss')); "
                "p.start(); raise SystemExit(pytest.main(['tests/test_layer_norm.py', '-qx']))",
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default=os.environ.get("AUTOTUNE_GPU", "NVIDIA L4"))
    tune(parser.parse_args().gpu)
