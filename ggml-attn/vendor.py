"""Vendor the llama.cpp files this package needs into `vendor/`, so it is self-contained (a hub
`kernels` build cannot clone at build time).

A curated list rather than whole trees. ggml's Metal backend ships one file per operation, so the
files below are simply the ones this package dispatches. If a pin bump makes upstream reach for
something new, the build fails loudly on a missing header, which is the signal to add it here.

Usage: python vendor.py [--src /path/to/llama.cpp] [--rev <git rev>]
"""

import argparse
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")

FILES = [
    # the dispatch's own include, for the pipeline constants
    "src/ggml-metal/ggml-metal-impl.h",
    "src/ggml-common.h",
    # the flash-attention kernels this package dispatches
    "src/ggml-metal/kernels/fa.metal",
    "src/ggml-metal/kernels/common.h",
    "src/ggml-metal/kernels/dequantize.h",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("LLAMA_CPP_SRC", os.path.join(HERE, "llama.cpp")))
    ap.add_argument("--rev", default=None, help="git rev to check out before copying")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        subprocess.run(
            ["git", "clone", "https://github.com/ggml-org/llama.cpp.git", args.src], check=True
        )
    if args.rev:
        subprocess.run(["git", "-C", args.src, "checkout", args.rev], check=True)

    rev = subprocess.run(
        ["git", "-C", args.src, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    if os.path.isdir(VENDOR):
        shutil.rmtree(VENDOR)
    for rel in FILES:
        dst = os.path.join(VENDOR, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(args.src, "ggml", rel), dst)

    with open(os.path.join(VENDOR, "UPSTREAM"), "w") as f:
        f.write(f"https://github.com/ggml-org/llama.cpp\n{rev}\n")
    shutil.copy2(os.path.join(args.src, "LICENSE"), os.path.join(VENDOR, "LICENSE"))

    print(f"vendored {sum(len(f) for _, _, f in os.walk(VENDOR))} files from llama.cpp @ {rev[:12]} into {VENDOR}")


if __name__ == "__main__":
    main()
