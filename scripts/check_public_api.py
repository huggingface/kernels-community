#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "libclang",
#     "numpy",
#     "pygit2",
#     "torch",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu" }]
# ///
import argparse
import ast
import json
import sys
import tomllib
from pathlib import Path

import pygit2

# Private by name but still part of the public contract.
PUBLIC_DUNDERS = {"__init__", "__call__"}
CPP_EXT = (".cpp", ".cc", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".hh")
# Python package lives under a build-variant dir, e.g. torch-ext or tvm-ffi-ext.
EXT_SUFFIX = "-ext"


# Kernel files (working copy or git tree) keyed by relative path.
class Source:
    def __init__(self, files: dict, reader):
        self.files = files
        self._read = reader

    def is_file(self, rel: str) -> bool:
        return rel in self.files

    def read(self, rel: str) -> str:
        return self._read(self.files[rel])

    @staticmethod
    def from_disk(root: Path) -> "Source":
        if not root.is_dir():
            return Source({}, None)
        files = {
            p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()
        }
        return Source(files, lambda p: p.read_text(errors="replace"))

    @staticmethod
    def from_tree(repo, root_tree, kernel: str) -> "Source":
        try:
            obj = repo[root_tree[kernel].id]
        except KeyError:
            return Source({}, None)
        if not isinstance(obj, pygit2.Tree):
            return Source({}, None)
        files = {}

        def walk(tree, prefix):
            for entry in tree:
                name = f"{prefix}{entry.name}"
                if entry.type_str == "tree":
                    walk(repo[entry.id], f"{name}/")
                elif entry.type_str == "blob":
                    files[name] = entry.id
                # skip submodule gitlinks (type_str == "commit")

        walk(obj, "")
        return Source(files, lambda oid: repo[oid].data.decode("utf-8", "replace"))


# Public API extraction from Python AST and C++ binding sources.


def public(name: str) -> bool:
    return not name.startswith("_")


def signature(node) -> str:
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"({args}){returns}"


def class_api(node: ast.ClassDef, access: str) -> dict:
    out = {f"{access}(bases)": ", ".join(ast.unparse(b) for b in node.bases)}
    attrs = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if public(item.name) or item.name in PUBLIC_DUNDERS:
                out[f"{access}.{item.name}"] = signature(item)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if public(item.target.id):
                attrs.append(item.target.id)
        elif isinstance(item, ast.Assign):
            attrs += [
                t.id for t in item.targets if isinstance(t, ast.Name) and public(t.id)
            ]
    if attrs:
        out[f"{access}(attrs)"] = ", ".join(sorted(set(attrs)))
    return out


# C++ binding sources listed under the given build.toml sections.
def _section_sources(src: Source, sections: tuple) -> list:
    if not src.is_file("build.toml"):
        return []
    data = tomllib.loads(src.read("build.toml"))
    listed = []
    for section in sections:
        listed += data.get(section, {}).get("src", [])
    return sorted({s for s in listed if s.endswith(CPP_EXT) and src.is_file(s)})


def _unquote(token: str) -> str:
    inner = token[token.index('"') + 1 : token.rindex('"')]
    return inner.replace('\\"', '"').replace("\\\\", "\\")


# Schema strings from `.def("...")` calls
def _def_schemas(index, cindex, content: str) -> list:
    try:
        tu = index.parse(
            "u.cpp",
            args=["-x", "c++", "-std=c++17"],
            unsaved_files=[("u.cpp", content)],
        )
    except cindex.TranslationUnitLoadError:
        return []
    toks = list(tu.get_tokens(extent=tu.cursor.extent))
    sp = [t.spelling for t in toks]  # def is not a C++ keyword; spelling match suffices
    out, i, n = [], 0, len(toks)
    while i < n:
        if sp[i] == "def" and i and sp[i - 1] == "." and i + 1 < n and sp[i + 1] == "(":
            j, parts = i + 2, []
            while (
                j < n
                and toks[j].kind == cindex.TokenKind.LITERAL
                and sp[j].lstrip("uUL8R").startswith('"')
            ):
                parts.append(_unquote(sp[j]))
                j += 1
            if parts:
                out.append("".join(parts))
            i = j
        else:
            i += 1
    return out


# (exported op, bound function) pairs from TVM_FFI_DLL_EXPORT_TYPED_FUNC(op, fn).
def _tvm_exports(index, cindex, content: str) -> list:
    try:
        tu = index.parse(
            "u.cpp", args=["-x", "c++", "-std=c++17"], unsaved_files=[("u.cpp", content)]
        )
    except cindex.TranslationUnitLoadError:
        return []
    sp = [t.spelling for t in tu.get_tokens(extent=tu.cursor.extent)]
    out, n = [], len(sp)
    for i in range(n):
        if sp[i] == "TVM_FFI_DLL_EXPORT_TYPED_FUNC" and i + 1 < n and sp[i + 1] == "(":
            args = []
            j = i + 2
            while j < n and sp[j] != ")":
                if sp[j] != ",":
                    args.append(sp[j])
                j += 1
            if args:
                out.append((args[0], args[1] if len(args) > 1 else args[0]))
    return out


# Operator schemas: PyTorch (.def, normalized via torch) and tvm-ffi exports.
def op_schemas(src: Source) -> dict:
    torch_srcs = _section_sources(src, ("torch", "torch-noarch"))
    tvm_srcs = _section_sources(src, ("tvm-ffi",))
    if not (torch_srcs or tvm_srcs):
        return {}
    from clang import cindex

    index = cindex.Index.create()
    out = {}
    if torch_srcs:
        import torch

        for rel in torch_srcs:
            for raw in _def_schemas(index, cindex, src.read(rel)):
                try:
                    schema = torch._C.parse_schema(raw)
                    out[f"op {schema.name}"] = str(schema)
                except RuntimeError:  # not a full schema (bare op name)
                    norm = " ".join(raw.split())
                    out[f"op {norm.split('(', 1)[0].strip()}"] = norm
    for rel in tvm_srcs:
        for name, bound in _tvm_exports(index, cindex, src.read(rel)):
            out[f"op {name}"] = bound
    return out


def _parse(src: Source, rel: str, cache: dict) -> ast.Module:
    if rel not in cache:
        cache[rel] = ast.parse(src.read(rel))
    return cache[rel]


def _module_file(src: Source, ext: str, comps: tuple) -> str:
    base = "/".join((ext, *comps))
    for rel in (f"{base}/__init__.py", f"{base}.py"):
        if src.is_file(rel):
            return rel
    return None


# Parse ast and collect definitions, submodules, re-exports, assignments, and __all__.
def _module_maps(src: Source, ext: str, comps: tuple, cache: dict):
    rel = _module_file(src, ext, comps)
    tree = _parse(src, rel, cache)
    pkg = comps if rel.endswith("/__init__.py") else comps[:-1]
    defs, submods, named, assigns, all_decl = {}, {}, {}, set(), None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if public(node.target.id):
                assigns.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if not isinstance(t, ast.Name):
                    continue
                if t.id == "__all__":
                    try:
                        all_decl = sorted(ast.literal_eval(node.value))
                    except (ValueError, TypeError):
                        pass
                elif public(t.id):
                    assigns.add(t.id)
        elif isinstance(node, ast.ImportFrom) and node.level:
            up = node.level - 1
            if up > len(pkg):
                continue  # import escapes the package
            base = pkg[: len(pkg) - up] if up else pkg
            mod = base + tuple(node.module.split(".")) if node.module else base
            for alias in node.names:
                local = alias.asname or alias.name
                if alias.name == "*":
                    named["*"] = (mod, "*")
                elif _module_file(src, ext, mod + (alias.name,)):
                    submods[local] = mod + (alias.name,)  # re-exported submodule
                else:
                    named[local] = (mod, alias.name)
    return defs, submods, named, assigns, all_decl


# Capture the public python api from the source tree, including re-exports and __all__.
def python_api(src: Source) -> dict:
    out, cache, seen = {}, {}, set()

    def emit(ext, comps, name, access, depth=0):
        if depth > 25 or not _module_file(src, ext, comps):
            out[f"py {access}"] = "<external>"  # generated/absent module, e.g. _ops
            return
        defs, submods, named, _, _ = _module_maps(src, ext, comps, cache)
        if name in defs:
            node = defs[name]
            if isinstance(node, ast.ClassDef):
                out.update(class_api(node, f"py {access}"))
            else:
                out[f"py {access}"] = signature(node)
        elif name in submods:
            collect(ext, submods[name], access)
        elif name in named:
            mod, orig = named[name]
            emit(ext, mod, orig, access, depth + 1)
        else:
            out[f"py {access}"] = "<value>"  # assignment or unresolvable re-export

    def collect(ext, comps, prefix):
        if (ext, comps, prefix) in seen:
            return
        seen.add((ext, comps, prefix))
        defs, submods, named, assigns, all_decl = _module_maps(src, ext, comps, cache)
        if "*" in named:  # star re-export
            collect(ext, named["*"][0], prefix)
        if all_decl is not None:
            exported = set(all_decl)
            out[f"py {prefix} __all__"] = repr(sorted(all_decl))
        else:
            exported = (
                {n for n in defs if public(n)}
                | {n for n in submods if public(n)}
                | {n for n in named if n != "*" and public(n)}
                | assigns
            )
        for name in sorted(exported):
            emit(ext, comps, name, f"{prefix}.{name}")

    for rel in sorted(src.files):
        parts = rel.split("/")
        if len(parts) == 3 and parts[0].endswith(EXT_SUFFIX) and parts[2] == "__init__.py":
            collect(parts[0], (parts[1],), parts[1])
    return out


def extract_api(src: Source) -> dict:
    api = {}
    api.update(op_schemas(src))
    api.update(python_api(src))
    return api


# Resolve a branch/tag/SHA via libgit2.
def resolve_ref(repo, ref: str):
    return repo.revparse_single(ref).peel(pygit2.Commit)


def changed_kernels(base_tree, head_tree) -> list:
    found = set()
    for delta in base_tree.diff_to_tree(head_tree).deltas:
        for path in (delta.old_file.path, delta.new_file.path):
            top = path.split("/", 1)[0]
            if "/" in path and (Path(top) / "build.toml").is_file():
                found.add(top)
    return sorted(found)


def _short(text: str, width: int = 60) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


# Each item is (text, [detail lines]); details render under it with the
# correct tree continuation (| while siblings remain, blank on the last).
def _tree(items: list) -> None:
    for i, (text, details) in enumerate(items):
        last = i == len(items) - 1
        print(f"     {'`-' if last else '|-'} {text}")
        for d in details:
            print(f"     {'  ' if last else '| '} {d}")


def report(
    kernel: str, base: dict, head: dict, limit: int = 20, preview: int = 6
) -> bool:
    removed = sorted(k for k in base if k not in head)
    changed = sorted(k for k in base if k in head and base[k] != head[k])
    added = sorted(k for k in head if k not in base)

    if not (removed or changed or added):
        keys = sorted(head)
        print(f"  [ok] {kernel}: {len(keys)} symbols")
        items = [(f"{k} = {_short(head[k])}", []) for k in keys[:preview]]
        if len(keys) > preview:
            items.append((f"... and {len(keys) - preview} more", []))
        _tree(items)
        return False

    counts = ", ".join(
        f"{n} {label}"
        for n, label in (
            (len(removed), "removed"),
            (len(changed), "changed"),
            (len(added), "added"),
        )
        if n
    )
    print(f"  [changed] {kernel}: {counts}")
    items = (
        [(f"removed {k} = {_short(base[k])}", []) for k in removed]
        + [(f"changed {k}", [f"- {base[k]}", f"+ {head[k]}"]) for k in changed]
        + [(f"added   {k} = {_short(head[k])}", []) for k in added]
    )
    if len(items) > limit:
        items = items[:limit] + [(f"... and {len(items) - limit} more", [])]
    _tree(items)
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
            src = Source.from_disk(Path(kernel))
            print(json.dumps(extract_api(src), indent=2, sort_keys=True))
        return 0

    repo = pygit2.Repository(pygit2.discover_repository("."))
    try:
        base = resolve_ref(repo, args.base_ref)
    except (KeyError, pygit2.GitError):
        print(f"error: base ref {args.base_ref!r} could not be resolved.", file=sys.stderr)
        return 1

    head = resolve_ref(repo, "HEAD")
    # Diff from the merge-base so target-branch changes aren't attributed here.
    mb = repo.merge_base(base.id, head.id)
    baseline = repo[mb] if mb else base
    base_tree, head_tree = baseline.tree, head.tree

    kernels = args.kernels or changed_kernels(base_tree, head_tree)

    print(
        f"Public API: {str(baseline.id)[:8]}..{str(head.id)[:8]} ({args.base_ref}..HEAD)\n"
    )
    if not kernels:
        print("  (no kernel sources changed)")
        return 0

    changed = False
    for kernel in kernels:
        if not Path(kernel).is_dir():
            print(f"  [skip] {kernel}: directory not found")
            continue
        base_api = extract_api(Source.from_tree(repo, base_tree, kernel))
        head_api = extract_api(Source.from_disk(Path(kernel)))
        if report(kernel, base_api, head_api):
            changed = True

    if changed:
        print(
            "\nERROR: public API changed - bump the kernel version if intentional.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
