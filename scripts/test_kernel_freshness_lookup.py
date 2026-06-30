import sys
from datetime import datetime
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


def test_freshness_report_includes_kernel_dir() -> bool:
    """Verify that _format_freshness_report includes kernel_dir in each line.

    Regression test for: multiple kernels sharing the same source_url would
    produce duplicate, unactionable alert lines with no local directory name.
    """
    from datetime import timezone
    from check_kernel_freshness import _format_freshness_report

    shared_url = "https://github.com/Dao-AILab/flash-attention"
    now = datetime.now(tz=timezone.utc)
    results = [
        {"kernel_dir": "flash-attn2", "source_url": shared_url, "days_behind": 12,
         "upstream_date": now, "local_date": now},
        {"kernel_dir": "flash-attn3", "source_url": shared_url, "days_behind": 5,
         "upstream_date": now, "local_date": now},
    ]

    report = _format_freshness_report(results, skipped_kernels=[])

    all_passed = True
    for result in results:
        if result["kernel_dir"] not in report:
            print(f"❌ ERROR: '{result['kernel_dir']}' not found in freshness report")
            all_passed = False

    if all_passed:
        print("✅ _format_freshness_report correctly includes kernel_dir in each report line")

    return all_passed


def main() -> int:
    root_path = Path(__file__).parent.parent.resolve()
    print(f"Testing kernel mapping in: {root_path}\n")

    results = [
        test_kernel_mapping_completeness(root_path),
        test_freshness_report_includes_kernel_dir(),
    ]

    if all(results):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
