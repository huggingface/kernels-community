# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "tomli",
# ]
# ///
"""Check license information for all kernels in the repository."""

import sys
from pathlib import Path

import tomli


def find_kernel_dirs(root: Path) -> list[Path]:
    """Find all directories containing build.toml, excluding backup directories."""
    kernel_dirs = []
    for build_toml in root.rglob("build.toml"):
        # Skip backup/result directories
        if "result/" in str(build_toml) or ".bak" in str(build_toml):
            continue
        kernel_dirs.append(build_toml.parent)
    return sorted(kernel_dirs)


def check_license_in_toml(build_toml: Path) -> str | None:
    """Check if build.toml has a license field in [general] section."""
    with open(build_toml, "rb") as f:
        data = tomli.load(f)
    return data.get("general", {}).get("license")


def find_license_file(kernel_dir: Path) -> Path | None:
    """Check if a LICENSE file exists in the kernel directory."""
    for pattern in ["LICENSE", "LICENSE.*", "LICENCE", "LICENCE.*"]:
        matches = list(kernel_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def main():
    root = Path(__file__).parent
    kernel_dirs = find_kernel_dirs(root)

    print(f"Found {len(kernel_dirs)} kernels with build.toml\n")

    missing_license = []
    has_license = []

    for kernel_dir in kernel_dirs:
        build_toml = kernel_dir / "build.toml"
        kernel_name = kernel_dir.name

        license_in_toml = check_license_in_toml(build_toml)
        license_file = find_license_file(kernel_dir)

        if license_in_toml:
            has_license.append((kernel_name, f"build.toml: {license_in_toml}"))
        elif license_file:
            has_license.append((kernel_name, f"file: {license_file.name}"))
        else:
            missing_license.append(kernel_name)

    # Report results
    if has_license:
        print("Kernels with license information:")
        for name, source in has_license:
            print(f"  {name}: {source}")
        print()

    if missing_license:
        print("Kernels MISSING license information:")
        for name in missing_license:
            print(f"  {name}")
        print()
        print(f"Total: {len(missing_license)} kernels missing license info")
        sys.exit(1)
    else:
        print("All kernels have license information!")
        sys.exit(0)


if __name__ == "__main__":
    main()
