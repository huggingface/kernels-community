import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Private by name but still part of the public contract.
PUBLIC_DUNDERS = {"__init__", "__call__"}
DEF_RE = re.compile(r"\.def\(\s*\"((?:[^\"\\]|\\.)*)\"")


def public(name: str) -> bool:
    return not name.startswith("_")


def signature(node) -> str:
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"({args}){returns}"


def class_api(node: ast.ClassDef, rel: str) -> dict:
    out = {f"{rel} {node.name}(bases)": ", ".join(ast.unparse(b) for b in node.bases)}
    attrs = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if public(item.name) or item.name in PUBLIC_DUNDERS:
                out[f"{rel} {node.name}.{item.name}"] = signature(item)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if public(item.target.id):
                attrs.append(item.target.id)
        elif isinstance(item, ast.Assign):
            attrs += [
                t.id for t in item.targets if isinstance(t, ast.Name) and public(t.id)
            ]
    if attrs:
        out[f"{rel} {node.name}(attrs)"] = ", ".join(sorted(set(attrs)))
    return out


def python_api(path: Path, rel: str) -> dict:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if public(node.name):
                out[f"{rel} {node.name}"] = signature(node)
        elif isinstance(node, ast.ClassDef) and public(node.name):
            out.update(class_api(node, rel))
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                try:
                    out[f"{rel} __all__"] = repr(sorted(ast.literal_eval(node.value)))
                except (ValueError, TypeError):
                    pass
    return out


def extract_api(kernel_root: Path) -> dict:
    ext = kernel_root / "torch-ext"
    if not ext.is_dir():
        return {}
    api = {}
    for src in sorted([*ext.rglob("*.cpp"), *ext.rglob("*.cc")]):
        for match in DEF_RE.finditer(src.read_text(errors="replace")):
            schema = " ".join(match.group(1).split())
            if schema:
                api[f"op {schema.split('(', 1)[0].strip()}"] = schema
    for py in sorted(ext.rglob("*.py")):
        # Skip private modules (e.g. _ops.py) but keep dunders (__init__.py).
        if py.stem.startswith("_") and not py.stem.startswith("__"):
            continue
        api.update(python_api(py, py.relative_to(ext).as_posix()))
    return api


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def changed_kernels(ref: str) -> list:
    found = set()
    for line in git("diff", "--name-only", ref).splitlines():
        top = line.split("/", 1)[0]
        if "/" in line and (Path(top) / "build.toml").is_file():
            found.add(top)
    return sorted(found)


def api_at_ref(kernel: str, ref: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="api-base-") as tmp:
        try:
            git("worktree", "add", "--detach", "--quiet", tmp, ref)
            return extract_api(Path(tmp) / kernel)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", tmp],
                capture_output=True,
                text=True,
            )


def report(kernel: str, base: dict, head: dict) -> bool:
    removed = {k: v for k, v in base.items() if k not in head}
    added = {k: v for k, v in head.items() if k not in base}
    changed = {k: (base[k], head[k]) for k in base if k in head and base[k] != head[k]}

    if not (removed or changed or added):
        print(f"✅ {kernel}: public API unchanged.")
        return False

    print(f"\n🔎 {kernel}: public API changes detected.")
    for key, val in sorted(removed.items()):
        print(f"  ❌ removed: {key}  =  {val}")
    for key, (before, after) in sorted(changed.items()):
        print(f"  ❌ changed: {key}\n       before: {before}\n       after:  {after}")
    for key, val in sorted(added.items()):
        print(f"  ➕ added:   {key}  =  {val}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "kernels",
        nargs="*",
        help="Kernels to check (default: those changed vs --base-ref).",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Target branch to compare against (default: origin/main).",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Print the working-tree API for each kernel and exit.",
    )
    args = parser.parse_args()

    if args.dump:
        if not args.kernels:
            parser.error("--dump requires explicit kernel name(s).")
        for kernel in args.kernels:
            print(f"# {kernel}")
            print(json.dumps(extract_api(Path(kernel)), indent=2, sort_keys=True))
        return 0

    # Compare against where the branch diverged, so changes others made on the
    # target branch aren't mistaken for changes made here.
    try:
        baseline = git("merge-base", args.base_ref, "HEAD")
    except subprocess.CalledProcessError:
        baseline = args.base_ref

    kernels = args.kernels or changed_kernels(baseline)
    if not kernels:
        print("No kernel sources changed; nothing to check.")
        return 0

    changed = False
    for kernel in kernels:
        if not Path(kernel).is_dir():
            print(f"⚠️  {kernel}: directory not found, skipping.", file=sys.stderr)
            continue
        if report(kernel, api_at_ref(kernel, baseline), extract_api(Path(kernel))):
            changed = True

    if changed:
        print(
            "\n💥 Public API changed. If intentional, note it in the PR "
            "description so reviewers can approve the change.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
