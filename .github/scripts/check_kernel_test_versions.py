#!/usr/bin/env python3
"""Check that kernel tests load the version declared in build.toml."""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Kernel:
    root: Path
    repo_id: str
    version: int


@dataclass(frozen=True)
class Problem:
    path: Path
    line: int
    message: str

    def render(self, repository_root: Path) -> str:
        try:
            path = self.path.relative_to(repository_root)
        except ValueError:
            path = self.path
        return f"{path}:{self.line}: {self.message}"


def discover_kernels(repository_root: Path) -> list[Kernel]:
    """Return kernels that have both a manifest and a tests directory."""
    kernels = []
    for root in sorted(path for path in repository_root.iterdir() if path.is_dir()):
        manifest = root / "build.toml"
        if not manifest.is_file():
            # Ported kernels keep generated sources and build.toml under src/.
            manifest = root / "src" / "build.toml"
        if not manifest.is_file() or not (root / "tests").is_dir():
            continue

        with manifest.open("rb") as handle:
            config = tomllib.load(handle)
        general = config.get("general", {})
        repo_id = general.get("hub", {}).get("repo-id")
        version = general.get("version")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError(f"{manifest}: [general.hub].repo-id must be a string")
        if type(version) is not int:
            raise ValueError(f"{manifest}: [general].version must be an integer")
        kernels.append(Kernel(root=root, repo_id=repo_id, version=version))
    return kernels


def _literal(node: ast.AST | None) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def check_test_file(path: Path, kernel: Kernel) -> tuple[list[Problem], int]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as error:
        line = getattr(error, "lineno", None) or 1
        return [Problem(path, line, f"could not parse test file: {error}")], 0

    problems = []
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function = node.func
        function_name = (
            function.id
            if isinstance(function, ast.Name)
            else function.attr if isinstance(function, ast.Attribute) else None
        )
        if function_name != "get_kernel" or not node.args:
            continue

        # A test suite may load another kernel as a reference or dependency.
        # Only calls targeting this kernel are governed by its build.toml.
        if _literal(node.args[0]) != kernel.repo_id:
            continue

        checked += 1
        expected = f'get_kernel("{kernel.repo_id}", version={kernel.version})'
        if expected not in (ast.get_source_segment(source, node) or ""):
            problems.append(
                Problem(
                    path,
                    node.lineno,
                    f"expected {expected}",
                )
            )
    return problems, checked


def check_repository(repository_root: Path) -> tuple[list[Problem], int, int]:
    problems = []
    checked_calls = 0
    kernels_with_calls = 0
    for kernel in discover_kernels(repository_root):
        kernel_calls = 0
        for path in sorted((kernel.root / "tests").rglob("*.py")):
            file_problems, file_calls = check_test_file(path, kernel)
            problems.extend(file_problems)
            kernel_calls += file_calls
        checked_calls += kernel_calls
        kernels_with_calls += kernel_calls > 0
    return problems, checked_calls, kernels_with_calls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (default: inferred from this script).",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        problems, checked_calls, kernels_with_calls = check_repository(root)
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for problem in problems:
        print(problem.render(root), file=sys.stderr)
    if problems:
        print(
            f"Found {len(problems)} problem(s) in {checked_calls} get_kernel call(s).",
            file=sys.stderr,
        )
        return 1

    print(
        f"Checked {checked_calls} get_kernel call(s) across "
        f"{kernels_with_calls} kernel test suite(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
