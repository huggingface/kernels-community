#!/usr/bin/env python3
import filecmp
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Extract source arguments; kernel-port validates them.
SOURCE_RE = re.compile(
    r"^source\b(?P<body>(?:[^\n\\]|\\\s*\n)*)", re.MULTILINE
)
ARG_RE = re.compile(r'(\w+)="([^"]*)"')

DEFAULT_RUNNER = "kernel-port"

CHECKER_FILES = {"scripts/check_ports.py", ".github/workflows/check-ports.yaml"}


def runner_cmd() -> list:
    return shlex.split(os.environ.get("KERNEL_PORT", DEFAULT_RUNNER))


def find_ports(root: Path) -> list:
    return sorted(p.parent.parent.name for p in root.glob("*/port/port.kdl"))


def changed_ports(root: Path, ref: str) -> list:
    """Select changed ports, or all ports when the checker or workflow changes."""
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    touched = set()
    for line in out.splitlines():
        if line in CHECKER_FILES:
            return find_ports(root)
        parts = line.split("/")
        if len(parts) >= 3 and parts[1] in ("port", "src"):
            touched.add(parts[0])
    return sorted(k for k in find_ports(root) if k in touched)


def parse_source(recipe: Path) -> tuple:
    text = recipe.read_text()
    m = SOURCE_RE.search(text)
    if not m:
        raise SystemExit(f"{recipe}: no `source` op found")
    args = dict(ARG_RE.findall(m.group("body").replace("\\\n", " ")))
    try:
        return args["repo"], args["commit"]
    except KeyError as exc:
        raise SystemExit(f"{recipe}: source is missing {exc}") from exc


def checkout(repo: str, commit: str, dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", repo, str(dest)],
        check=True,
    )
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", commit], check=True)


def tree_files(root: Path) -> set:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def compare(generated: Path, committed: Path) -> list:
    gen, com = tree_files(generated), tree_files(committed)
    problems = []
    for rel in sorted(com - gen):
        problems.append(f"committed but not generated: {rel}")
    for rel in sorted(gen - com):
        problems.append(f"generated but not committed: {rel}")
    for rel in sorted(gen & com):
        if not filecmp.cmp(generated / rel, committed / rel, shallow=False):
            problems.append(f"differs: {rel}")
    return problems


def check(kernel: str, root: Path, workdir: Path) -> list:
    recipe = root / kernel / "port" / "port.kdl"
    repo, commit = parse_source(recipe)
    print(f"  {kernel}: {repo} @ {commit[:12]}")

    # Reuse checkouts for kernels sharing an upstream commit.
    src = workdir / f"upstream-{commit[:12]}"
    if not src.exists():
        checkout(repo, commit, src)

    out = workdir / f"out-{kernel}"
    shutil.rmtree(out, ignore_errors=True)
    proc = subprocess.run(
        [*runner_cmd(), str(recipe), "--dir", str(src), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-8:]
        return [f"kernel-port failed:\n    " + "\n    ".join(tail)]

    return compare(out, root / kernel / "src")


def main() -> int:
    root = Path.cwd()
    args = sys.argv[1:]
    if args[:1] == ["--changed-since"]:
        if len(args) != 2:
            raise SystemExit("usage: check_ports.py --changed-since <ref>")
        kernels = changed_ports(root, args[1])
        if not kernels:
            print(f"No ported kernels changed since {args[1]}.")
            return 0
    else:
        kernels = args or find_ports(root)
    if not kernels:
        print("No ported kernels found (no */port/port.kdl).")
        return 0

    print(f"Checking {len(kernels)} port(s) reproduce their committed tree:\n")
    failed = {}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for kernel in kernels:
            if not (root / kernel / "port" / "port.kdl").is_file():
                failed[kernel] = [f"no port/port.kdl under {kernel}/"]
                continue
            problems = check(kernel, root, workdir)
            if problems:
                failed[kernel] = problems
            else:
                print("     ok - regenerates byte for byte")

    if failed:
        print("\nERROR: committed tree does not match the recipe.\n", file=sys.stderr)
        for kernel, problems in failed.items():
            print(f"  {kernel}:", file=sys.stderr)
            for p in problems:
                print(f"    - {p}", file=sys.stderr)
        print(
            "\nRegenerate with:\n"
            "  kernel-port <kernel>/port/port.kdl --dir <upstream> --out <kernel>/src\n"
            "If an overlay file went stale, refresh it from <kernel>/src first.\n"
            "If only port-provenance.json differs, the recipe changed and the\n"
            "port was not rerun.",
            file=sys.stderr,
        )
        return 1

    print("\nAll ports reproduce their committed tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
