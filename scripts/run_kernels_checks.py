import argparse
import logging
import subprocess
from pathlib import Path

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


def discover_kernel_dirs(root: Path, excludes: list[str]) -> list[str]:
    filtered = {exclude.strip() for exclude in excludes}
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as err:
        raise RuntimeError(f"Unable to list directories under {root}: {err}") from err

    directories = []
    for entry in entries:
        name = entry.name
        if name in filtered:
            continue
        elif entry.is_dir() and not (entry / "build.toml").exists():
            logging.debug(f"Skipping {name} because it doesn't contain a `build.toml`.")
            continue
        if entry.is_dir():
            directories.append(name)
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

    root_path = Path(args.root).resolve()
    logging.debug(f"Using root path {root_path}")

    try:
        directories = discover_kernel_dirs(root_path, args.exclude)
    except RuntimeError as err:
        logging.error(err)
        return 1

    if not directories:
        logging.error(f"⛔️ No kernel directories found in {root_path}.")
        return 1

    logging.info(f"🧪 Checking {len(directories)} kernel directories: {directories=}.")

    failures = run_kernels_checks(directories, args.dry_run)
    if failures:
        logging.error(
            "❌ kernels check failed for %d directories: %s",
            len(failures),
            ", ".join(sorted(failures)),
        )
        return 1

    logging.info("✅ All kernels checks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
