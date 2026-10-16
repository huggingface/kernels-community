#!/usr/bin/env python3
"""Regenerate DeepEP from clean pinned checkouts, optionally checking committed output."""
import argparse
import subprocess
import tempfile
from pathlib import Path


def files(root):
    ignored = {"port", "__pycache__", ".pytest_cache", ".ruff_cache"}
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in ignored for part in path.relative_to(root).parts)
        and path.suffix != ".pyc"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-ep", type=Path, required=True)
    parser.add_argument("--deepjit", type=Path, required=True)
    parser.add_argument("--nccl", type=Path, required=True)
    parser.add_argument("--kernel-port", default="kernel-port")
    parser.add_argument("--output", type=Path, help="Generated output outside the kernel directory")
    parser.add_argument("--check", action="store_true", help="Compare generated output with this kernel")
    args = parser.parse_args()
    if not args.check and args.output is None:
        parser.error("provide --check or --output to keep the result")
    port = Path(__file__).resolve().parent
    kernel = port.parent
    with tempfile.TemporaryDirectory(prefix="deep-ep-port-") as temporary:
        output = (args.output or Path(temporary) / "deep-ep").resolve()
        inputs = [kernel, args.deep_ep.resolve(), args.deepjit.resolve(),
                  args.nccl.resolve()]
        if any(output == source or output in source.parents or source in output.parents
               for source in inputs):
            parser.error("--output must not overlap this kernel or any source checkout")
        command = [args.kernel_port, str(port / "port.kdl"), "--dir", str(args.deep_ep),
                   "--vendor", f"python={args.deep_ep}", "--vendor", f"deepjit={args.deepjit}",
                   "--vendor", f"nccl={args.nccl}",
                   "--out", str(output)]
        subprocess.run(command, check=True)
        if args.check:
            expected, actual = files(output), files(kernel)
            changed = sorted(path for path in expected.keys() | actual.keys()
                             if expected.get(path) != actual.get(path))
            if changed:
                print("Generated output differs:")
                print("\n".join(str(path) for path in changed))
                return 1
            print("All generated files match the recipe, including provenance.")
        elif args.output:
            print(f"Generated output: {output}")
        else:
            parser.error("provide --check or --output to keep the result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
