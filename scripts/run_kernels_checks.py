import argparse
import logging
import os
import subprocess

ORG = "kernels-community"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run `kernels check` for every top-level directory in the current repository.")
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Local path whose immediate subdirectories will be checked. Defaults to the current directory.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[".github", "scripts"],
        help="Directory name to skip. Can be passed multiple times. Defaults to .github.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print commands instead of executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def discover_kernel_dirs(root: str, excludes: list[str]) -> list[str]:
    filtered = {exclude.strip() for exclude in excludes}
    try:
        entries = os.listdir(root)
    except OSError as err:
        raise RuntimeError(f"Unable to list directories under {root}: {err}") from err

    directories = []
    for entry in sorted(entries):
        if entry.startswith(".") or entry in filtered:
            continue
        path = os.path.join(root, entry)
        if os.path.isdir(path):
            directories.append(entry)
    return directories


def run_kernels_checks(directories: list[str], dry_run: bool) -> list[str]:
    failures = []
    for directory in directories:
        target = f"{ORG}/{directory}"
        command = f"kernels check {target}".split()
        logging.info("🧪 Running %s", " ".join(command))
        if dry_run:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            failures.append(directory)
    return failures


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        format="%(levelname)s: %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    root_path = os.path.abspath(args.root)
    logging.debug(f"Using root path {root_path}")

    try:
        directories = discover_kernel_dirs(root_path, args.exclude)
    except RuntimeError as err:
        logging.error(err)
        raise

    if not directories:
        logging.error(f"⛔️ No kernel directories found in {root_path}.")
        raise

    logging.info(f"🧪 Checking {len(directories)} kernel directories: {directories=}.")

    failures = run_kernels_checks(directories, args.dry_run)
    if failures:
        logging.error(
            "❌ kernels check failed for %d directories: %s",
            len(failures),
            ", ".join(sorted(failures)),
        )
        raise

    logging.info("✅ All kernels checks completed successfully.")


if __name__ == "__main__":
    main()
