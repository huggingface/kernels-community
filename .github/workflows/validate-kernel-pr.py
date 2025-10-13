#!/usr/bin/env python3
"""
Validate kernel build request and directory structure.
"""

import sys
import re
import argparse
from pathlib import Path


def validate_kernel_name(kernel_name: str) -> bool:
    """
    Validate that the kernel name is safe and follows expected patterns.
    Only allow alphanumeric characters, hyphens, and underscores.
    """
    if not kernel_name:
        return False

    # Only allow alphanumeric characters, hyphens, and underscores
    if not re.match(r"^[a-zA-Z0-9_-]+$", kernel_name):
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate kernel build request and directory structure",
        epilog="Returns exit code 0 if valid, 1 if invalid, and outputs kernel name to stdout"
    )
    parser.add_argument(
        "build_type",
        choices=["pr", "release"],
        help="Type of build to validate for"
    )
    parser.add_argument(
        "pr_title",
        help="Title of the pull request containing kernel name"
    )
    
    args = parser.parse_args()
    
    build_type = args.build_type
    pr_title = args.pr_title

    if ":" not in pr_title:
        print("No colon found in PR title, skipping build", file=sys.stderr)
        sys.exit(1)

    kernel_name = pr_title.split(":", 1)[0].strip()

    if not validate_kernel_name(kernel_name):
        print(f"Invalid kernel name '{kernel_name}', skipping build", file=sys.stderr)
        sys.exit(1)

    kernel_path = Path(kernel_name)
    if not kernel_path.exists() or not kernel_path.is_dir():
        print(f"Kernel '{kernel_name}' does not exist, skipping build", file=sys.stderr)
        sys.exit(1)

    # Check for build-type specific skip-ci file
    skip_ci_file = kernel_path / f".skip-{build_type}-ci"
    if skip_ci_file.exists():
        print(f"Kernel '{kernel_name}' has .skip-{build_type}-ci file, skipping build", file=sys.stderr)
        sys.exit(1)

    flake_nix = kernel_path / "flake.nix"
    build_toml = kernel_path / "build.toml"

    if not flake_nix.exists() or not build_toml.exists():
        print(
            f"Kernel '{kernel_name}' missing flake.nix or build.toml, skipping build",
            file=sys.stderr,
        )
        sys.exit(1)

    print(kernel_name)
    sys.exit(0)


if __name__ == "__main__":
    main()
