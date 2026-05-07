"""Generate a GitHub Actions build matrix from per-architecture backend lists.

Usage:
    python3 generate-build-matrix.py <x86-backends-json> <arm-backends-json>

Each argument is the JSON array produced by:
    nix eval .#backendCi --apply builtins.attrNames --json --system <arch>

Prints a single-line JSON object suitable for use as a GitHub Actions matrix,
with one entry per (backend, arch) combination.
"""

import json
import sys

ARCHES = [
    ("x86_64-linux", "aws-highmemory-32-plus-nix"),
    ("aarch64-linux", "aws-r8g-8xl-plus-nix"),
]


def main():
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} <x86-backends-json> <arm-backends-json>",
            file=sys.stderr,
        )
        sys.exit(1)

    backends_by_arch = {
        "x86_64-linux": json.loads(sys.argv[1]),
        "aarch64-linux": json.loads(sys.argv[2]),
    }

    include = [
        {"backend": backend, "arch": arch, "runner": runner}
        for arch, runner in ARCHES
        for backend in backends_by_arch[arch]
    ]

    print(json.dumps({"include": include}))


if __name__ == "__main__":
    main()
