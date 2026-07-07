"""Generate a GitHub Actions build matrix from per-architecture backend lists.

Usage:
    KERNEL=<kernel> python3 generate-build-matrix.py <x86-backends-json> <arm-backends-json>

<kernel> is the kernel directory name (e.g. 'flash-attn2'), passed via the
KERNEL environment variable. Each JSON argument is the array produced by:
    nix eval .#backendCi --apply builtins.attrNames --json --system <arch>

Prints a single-line JSON object suitable for use as a GitHub Actions matrix,
with one entry per (backend, arch) combination. Each entry includes max_jobs
and cores, falling back to DEFAULT_MAX_JOBS and DEFAULT_CORES respectively,
with per-kernel/backend overrides read from build-concurrency.json at the
repo root.

The optional BACKENDS_FILTER environment variable (a comma-separated list)
restricts the matrix to those backends. When empty or unset, every backend
nix exposes is built. Filter names not exposed by nix are simply ignored.
"""

import json
import os
import sys
from pathlib import Path

VALID_BACKENDS = frozenset({"cuda", "cpu", "rocm", "metal", "xpu"})

ARCHES = [
    ("x86_64-linux", "aws-highmemory-32-plus-nix"),
    ("aarch64-linux", "aws-r8g-8xl-plus-nix"),
]

DEFAULT_MAX_JOBS = 2
DEFAULT_CORES = 12


def load_concurrency_overrides() -> dict:
    overrides_path = Path(__file__).parent.parent.parent / "build-concurrency.json"
    if overrides_path.exists():
        with open(overrides_path) as f:
            return json.load(f)
    return {}


def validate_backend(backend) -> str:
    if backend not in VALID_BACKENDS:
        print(
            f"Invalid backend name: {backend!r} "
            f"(expected one of {sorted(VALID_BACKENDS)})",
            file=sys.stderr,
        )
        sys.exit(1)
    return backend


def validate_int(value, name: str) -> int:
    # bool is a subclass of int, but we don't want to accept it either.
    if isinstance(value, bool) or not isinstance(value, int):
        print(f"Invalid {name} value: {value!r} (expected integer)", file=sys.stderr)
        sys.exit(1)
    return value


def main():
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} <x86-backends-json> <arm-backends-json>",
            file=sys.stderr,
        )
        sys.exit(1)

    kernel = os.environ.get("KERNEL")
    if not kernel:
        print("KERNEL environment variable is not set or empty", file=sys.stderr)
        sys.exit(1)

    backends_filter = {
        b.strip() for b in os.environ.get("BACKENDS_FILTER", "").split(",") if b.strip()
    }

    backends_by_arch = {
        "x86_64-linux": json.loads(sys.argv[1]),
        "aarch64-linux": json.loads(sys.argv[2]),
    }

    kernel_overrides = load_concurrency_overrides().get(kernel, {})

    include = [
        {
            "backend": validate_backend(backend),
            "arch": arch,
            "runner": runner,
            "max_jobs": validate_int(
                kernel_overrides.get(backend, {}).get("max-jobs", DEFAULT_MAX_JOBS),
                "max-jobs",
            ),
            "cores": validate_int(
                kernel_overrides.get(backend, {}).get("cores", DEFAULT_CORES),
                "cores",
            ),
        }
        for arch, runner in ARCHES
        for backend in backends_by_arch[arch]
        if not backends_filter or backend in backends_filter
    ]

    if backends_filter:
        dropped = {
            backend
            for backends in backends_by_arch.values()
            for backend in backends
            if backend not in backends_filter
        }
        if dropped:
            print(
                f"BACKENDS_FILTER={sorted(backends_filter)} excluded backends "
                f"{sorted(dropped)} for kernel {kernel!r}",
                file=sys.stderr,
            )

    print(json.dumps({"include": include}))


if __name__ == "__main__":
    main()
