"""
Plan and run minimum-Torch tests of the current CUDA stable-ABI kernel.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
from urllib.parse import unquote
from urllib.request import urlopen


PYTHON_VERSION = "3.12"


def read_config(kernel):
    config = kernel / "src" / "build.toml"
    if not config.is_file():
        config = kernel / "build.toml"
    with config.open("rb") as f:
        return config.parent, tomllib.load(f)


def minimum_version(config, backends_filter):
    if "cuda" not in config["general"]["backends"]:
        return None
    if backends_filter and "cuda" not in backends_filter:
        return None
    stable_abi = config.get("torch", {}).get("stable-abi", {})
    version = stable_abi if isinstance(stable_abi, str) else stable_abi.get("cuda")
    if version is not None and (
        not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+", version)
    ):
        raise ValueError(f"Invalid CUDA stable ABI version: {version!r}")
    return version


def wheel_available(version, compute):
    # Probe the official index instead of maintaining a second Torch/CUDA
    # support table. Test the first patch release: a later patch is not the
    # minimum promised by an x.y stable-ABI declaration.
    with urlopen(f"https://download.pytorch.org/whl/{compute}/torch/", timeout=30) as f:
        index = unquote(f.read().decode())
    tag = "cp" + PYTHON_VERSION.replace(".", "")
    return (
        re.search(
            rf"torch-{re.escape(version)}\.0\+{re.escape(compute)}-{tag}-{tag}-"
            r"(?:manylinux[0-9_]*|linux)_x86_64\.whl",
            index,
        )
        is not None
    )


def select_variant(variants, version):
    pattern = re.compile(
        rf"torch-stable-abi{version.replace('.', '')}-(cu[0-9]+)-x86_64-linux"
    )
    candidates = []
    for variant in variants:
        if match := pattern.fullmatch(variant):
            compute = match[1]
            candidates.append((int(compute[2:]), variant, compute))
    # Nix has already applied the kernel's CUDA bounds and flake overrides.
    # Prefer the lowest remaining toolkit with a wheel for the minimum Torch.
    for _, variant, compute in sorted(candidates):
        if wheel_available(version, compute):
            return variant, compute
    raise ValueError(
        f"No redistributable CUDA variant has a Python {PYTHON_VERSION} "
        f"Linux x86_64 wheel for Torch {version}.0. Candidates: {candidates}"
    )


def plan(kernel, backends_filter):
    _, config = read_config(kernel)
    version = minimum_version(config, backends_filter)
    result = None
    message = "Minimum-Torch test skipped: no CUDA stable ABI selected."
    if version is not None:
        variants = json.loads(
            subprocess.check_output(
                [
                    "nix",
                    "eval",
                    ".#redistributable",
                    "--apply",
                    "builtins.attrNames",
                    "--json",
                    "--system",
                    "x86_64-linux",
                ],
                cwd=kernel,
                text=True,
            )
        )
        variant, compute = select_variant(variants, version)
        result = {
            "torch": f"{version}.0+{compute}",
            "python": PYTHON_VERSION,
            "index": f"https://download.pytorch.org/whl/{compute}",
            "variant": variant,
        }
        message = f"Minimum-Torch test: {variant} on Torch {result['torch']}."
    print(message, file=sys.stderr)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(message + "\n\n")
    return result


def run_tests(kernel, build, expected_torch):
    source, config = read_config(kernel)
    build = build.resolve()
    repo_id = config["general"]["hub"]["repo-id"]
    # Set this before importing kernels: the loader reads it at import time.
    os.environ["LOCAL_KERNELS"] = f"{repo_id}={build}"
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch

    if torch.__version__ != expected_torch:
        raise RuntimeError(f"Expected Torch {expected_torch}, got {torch.__version__}")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("Minimum-Torch CUDA test requires a working CUDA device")

    import kernels
    import pytest

    # Verify the loader uses the local CI artifact before running tests.
    module = kernels.get_kernel(repo_id, version=config["general"]["version"])
    if not Path(module.__file__).resolve().is_relative_to(build):
        raise RuntimeError(
            f"Loaded {module.__file__} instead of the CI build in {build}"
        )
    print(
        f"Testing {repo_id} from {module.__file__} with Torch {torch.__version__}",
        flush=True,
    )
    result = pytest.main(
        [str(source / "tests"), "-m", "kernels_ci", "-p", "no:cacheprovider"]
    )
    # Match ci-test's handling of suites with no kernels_ci tests.
    return 0 if result == pytest.ExitCode.NO_TESTS_COLLECTED else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("kernel", type=Path)
    planner.add_argument("--backends", default="")
    test = commands.add_parser("test")
    test.add_argument("kernel", type=Path)
    test.add_argument("build", type=Path)
    test.add_argument("--torch", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        backends_filter = {b.strip() for b in args.backends.split(",") if b.strip()}
        result = plan(args.kernel, backends_filter)
        print(json.dumps(result))
        if output := os.environ.get("GITHUB_OUTPUT"):
            with open(output, "a") as f:
                f.write(f"enabled={str(result is not None).lower()}\n")
                for key, value in (result or {}).items():
                    f.write(f"{key}={value}\n")
        return 0
    return run_tests(args.kernel, args.build, args.torch)


if __name__ == "__main__":
    sys.exit(main())
