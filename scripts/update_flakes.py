#!/usr/bin/env python3
"""Update flake.lock in all kernel directories."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_kernels() -> list[str]:
    return sorted(
        p.parent.name for p in REPO_ROOT.glob("*/flake.nix")
        if p.parent != REPO_ROOT
    )


def update_kernel(name: str) -> bool:
    result = subprocess.run(
        ["nix", "flake", "update"],
        cwd=REPO_ROOT / name,
        capture_output=True,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernels", nargs="*", help="Kernel dirs to update (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="List kernels without updating")
    args = parser.parse_args()

    kernels = args.kernels or find_kernels()

    if args.dry_run:
        print("\n".join(kernels))
        return

    failed = []
    for kernel in kernels:
        if not (REPO_ROOT / kernel / "flake.nix").exists():
            print(f"SKIP {kernel}")
            continue
        ok = update_kernel(kernel)
        print(f"{'  OK' if ok else 'FAIL'} {kernel}")
        if not ok:
            failed.append(kernel)

    if failed:
        print(f"\n{len(failed)} failed: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
