"""
Plan the ROCm GPU test of a kernel on the AMD MI300 runner.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

# GPU architecture of the AMD runner (MI300X).
RUNNER_ARCH = "gfx942"

SYSTEM = "x86_64-linux"

_ROCM_VARIANT = re.compile(r"torch(\d)(\d+)-rocm(\d+)-x86_64-linux")


def read_config(kernel):
    config = kernel / "src" / "build.toml"
    if not config.is_file():
        config = kernel / "build.toml"
    with config.open("rb") as f:
        return tomllib.load(f)


def runs_on_runner(config):
    """False when every ROCm kernel pins archs that exclude the runner's GPU."""
    rocm_kernels = [
        k for k in config.get("kernel", {}).values() if k.get("language") == "cpp-rocm"
    ]
    # Triton/Python-only kernels compile at runtime for whatever GPU is present.
    return not rocm_kernels or any(
        "rocm-archs" not in k or RUNNER_ARCH in k["rocm-archs"] for k in rocm_kernels
    )


def _rocm_version(flat):
    # kernel-builder flattens major.minor: 7.0 -> 70, 7.14 -> 714, 10.0 -> 100.
    split = 2 if flat.startswith("1") and len(flat) >= 3 else 1
    return int(flat[:split]), int(flat[split:] or 0)


def select_variant(ci_tests):
    """Fallback for kernel-builder pins without `backendCiTests`.

    Mirrors kernel-builder's preference (newest Torch, then oldest ROCm),
    but has to recover the versions from variant names.
    """
    candidates = []
    for name in ci_tests:
        if match := _ROCM_VARIANT.fullmatch(name):
            torch = (int(match[1]), int(match[2]))
            candidates.append(((-torch[0], -torch[1], _rocm_version(match[3])), name))
    return min(candidates)[1] if candidates else None


def choose(config, probe, backends_filter):
    """Return (flake attr or None, message)."""
    if backends_filter and "rocm" not in backends_filter:
        return None, "ROCm test skipped: rocm is not in the backends filter."
    if not probe["rocm"]:
        return None, "ROCm test skipped: the kernel has no ROCm build."
    if not runs_on_runner(config):
        return None, f"ROCm test skipped: no ROCm kernel is built for the runner's {RUNNER_ARCH}."
    if probe["backendCiTest"]:
        return "backendCiTests.rocm", "ROCm test: backendCiTests.rocm."
    if probe["ciTests"] is None:
        return None, (
            "ROCm test skipped: kernel-builder in flake.lock has no `ciTests`, "
            "run `python3 scripts/update_flakes.py <kernel>`."
        )
    variant = select_variant(probe["ciTests"])
    if variant is None:
        return None, "ROCm test skipped: no ROCm variant in `ciTests`."
    return f"ciTests.{variant}", f"ROCm test: {variant} (selected by name, update kernel-builder for `backendCiTests`)."


def attr_names(kernel, attr):
    return json.loads(
        subprocess.check_output(
            [
                "nix",
                "eval",
                f".#{attr}",
                "--apply",
                "builtins.attrNames",
                "--json",
                "--system",
                SYSTEM,
            ],
            cwd=kernel,
            text=True,
        )
    )


def probe_flake(kernel):
    # Check which outputs the pinned kernel-builder has instead of treating any
    # eval failure as "missing", so a broken flake fails the plan loudly.
    # `backendCiTests` is the exact test paired with `backendCi.rocm`; older
    # pins only have the per-variant `ciTests`.
    outputs = set(attr_names(kernel, f"packages.{SYSTEM}"))
    return {
        "rocm": "backendCi" in outputs and "rocm" in attr_names(kernel, "backendCi"),
        "backendCiTest": "backendCiTests" in outputs
        and "rocm" in attr_names(kernel, "backendCiTests"),
        "ciTests": attr_names(kernel, "ciTests") if "ciTests" in outputs else None,
    }


def plan(kernel, backends_filter):
    attr, message = choose(read_config(kernel), probe_flake(kernel), backends_filter)
    print(message, file=sys.stderr)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(message + "\n\n")
    return attr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("kernel", type=Path)
    planner.add_argument("--backends", default="")
    args = parser.parse_args()
    backends_filter = {b.strip() for b in args.backends.split(",") if b.strip()}
    attr = plan(args.kernel, backends_filter)
    print(json.dumps(attr))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as f:
            f.write(f"enabled={str(attr is not None).lower()}\n")
            f.write(f"attr={attr or ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
