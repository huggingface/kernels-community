import sys
from pathlib import Path

from check_kernel_freshness import KERNEL_SOURCE_MAPPING


def discover_kernel_dirs(root_path: Path) -> list[str]:
    excluded_dirs = {".github", "scripts", ".git", "relu"}

    kernel_dirs = []
    for item in root_path.iterdir():
        if item.is_dir() and item.name not in excluded_dirs:
            kernel_dirs.append(item.name)

    return sorted(kernel_dirs)


def test_kernel_mapping_completeness(root_path: Path) -> bool:
    all_passed = True

    kernel_dirs = discover_kernel_dirs(root_path)
    print(f"Found {len(kernel_dirs)} kernel directories: {kernel_dirs}")

    mapped_dirs = set(KERNEL_SOURCE_MAPPING.keys())
    print(f"Found {len(mapped_dirs)} entries in KERNEL_SOURCE_MAPPING")

    unmapped_dirs = set(kernel_dirs) - mapped_dirs
    if unmapped_dirs:
        print(f"\n❌ ERROR: The following kernel directories are missing from KERNEL_SOURCE_MAPPING:")
        for dir_name in sorted(unmapped_dirs):
            print(f"  - {dir_name}")
        all_passed = False
    else:
        print("\n✅ All kernel directories are present in KERNEL_SOURCE_MAPPING")

    skipped_entries = []
    invalid_entries = []
    for kernel_dir, source_url in KERNEL_SOURCE_MAPPING.items():
        if source_url == "":
            skipped_entries.append(kernel_dir)
        elif not source_url:
            invalid_entries.append(kernel_dir)

    if skipped_entries:
        print(f"\nℹ️  INFO: The following entries have empty source URLs (intentionally skipped):")
        for dir_name in sorted(skipped_entries):
            print(f"  - {dir_name}")

    if invalid_entries:
        print(f"\n❌ ERROR: The following entries have None or invalid source URLs:")
        for dir_name in sorted(invalid_entries):
            print(f"  - {dir_name}: {KERNEL_SOURCE_MAPPING[dir_name]!r}")
        all_passed = False
    else:
        print("✅ All entries in KERNEL_SOURCE_MAPPING have valid or intentionally empty source URLs")

    extra_dirs = mapped_dirs - set(kernel_dirs)
    if extra_dirs:
        print(f"\n❌ ERROR: The following entries in KERNEL_SOURCE_MAPPING don't have corresponding directories:")
        for dir_name in sorted(extra_dirs):
            print(f"  - {dir_name}")
        all_passed = False
    else:
        print("✅ All entries in KERNEL_SOURCE_MAPPING have corresponding directories")

    return all_passed


def main() -> int:
    root_path = Path(__file__).parent.parent.resolve()
    print(f"Testing kernel mapping in: {root_path}\n")

    if test_kernel_mapping_completeness(root_path):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
