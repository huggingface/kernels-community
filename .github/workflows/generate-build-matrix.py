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
repo root. A backend entry may also hold per-arch overrides keyed by the
arch (e.g. 'aarch64-linux'), which take precedence over the backend's own
values.

The optional BACKENDS_FILTER environment variable (comma-separated) restricts
the matrix to those backends; empty or unset builds every backend nix exposes.
"""

import json
import os
import sys
from pathlib import Path

VALID_BACKENDS = frozenset({"cuda", "cpu", "rocm", "metal", "xpu", "tpu"})

ARCHES = [
    ("x86_64-linux", "aws-highmemory-32-plus-nix"),
    ("aarch64-linux", "aws-r8g-8xl-plus-nix"),
]

VALID_ARCHES = frozenset(arch for arch, _ in ARCHES)
CONCURRENCY_KEYS = frozenset({"max-jobs", "cores"})

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


def validate_concurrency(settings, allowed: frozenset, where: str) -> dict:
    if not isinstance(settings, dict):
        print(f"Invalid {where} value: {settings!r} (expected object)", file=sys.stderr)
        sys.exit(1)
    for key, value in settings.items():
        if key not in allowed:
            print(
                f"Invalid key {key!r} in {where} (expected one of {sorted(allowed)})",
                file=sys.stderr,
            )
            sys.exit(1)
        if key in VALID_ARCHES:
            validate_concurrency(value, CONCURRENCY_KEYS, f"{where}.{key}")
        else:
            validate_int(value, f"{where}.{key}")
    return settings


def validate_kernel_overrides(kernel: str, overrides) -> dict:
    if not isinstance(overrides, dict):
        print(
            f"Invalid {kernel} value: {overrides!r} (expected object)", file=sys.stderr
        )
        sys.exit(1)
    for backend, settings in overrides.items():
        validate_backend(backend)
        validate_concurrency(
            settings, CONCURRENCY_KEYS | VALID_ARCHES, f"{kernel}.{backend}"
        )
    return overrides


def resolve_concurrency(kernel_overrides: dict, backend: str, arch: str) -> dict:
    backend_overrides = kernel_overrides.get(backend, {})
    arch_overrides = backend_overrides.get(arch, {})
    return {
        "max_jobs": arch_overrides.get(
            "max-jobs", backend_overrides.get("max-jobs", DEFAULT_MAX_JOBS)
        ),
        "cores": arch_overrides.get(
            "cores", backend_overrides.get("cores", DEFAULT_CORES)
        ),
    }


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

    kernel_overrides = validate_kernel_overrides(
        kernel, load_concurrency_overrides().get(kernel, {})
    )

    include = [
        {
            "backend": validate_backend(backend),
            "arch": arch,
            "runner": runner,
            **resolve_concurrency(kernel_overrides, backend, arch),
        }
        for arch, runner in ARCHES
        for backend in backends_by_arch[arch]
        if not backends_filter or backend in backends_filter
    ]

    print(json.dumps({"include": include}))


if __name__ == "__main__":
    main()
