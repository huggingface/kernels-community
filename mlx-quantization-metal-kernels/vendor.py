"""Vendor the MLX files this package needs into `vendor/`, so it is self-contained (a hub `kernels`
build cannot clone at build time).

A curated list rather than whole trees: `quantized.metal` and the headers it includes, kept at their
upstream paths so its `#include "mlx/backend/metal/kernels/..."` lines resolve unchanged. If a pin
bump makes upstream reach for something new, the build fails loudly on a missing header, which is
the signal to add it here.

`quantized.cpp` is MLX's host-side dispatch. It is not compiled -- it is written against MLX's own
array and device types -- but `mlx_metal/mlx_dispatch.mm` transcribes its choices, and
`tests/test_vendor_drift.py` checks the transcription against this copy on every pin bump.

Usage: python vendor.py [--src /path/to/mlx] [--rev <git rev>]
"""

import argparse
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")

KERNELS = "mlx/backend/metal/kernels"
FILES = [
    # the kernels this package dispatches: qmv / qmv_fast / qmv_quad / qmv_wide, qmm_t, qmm_t_splitk
    f"{KERNELS}/quantized.metal",
    f"{KERNELS}/quantized.h",
    f"{KERNELS}/quantized_utils.h",
    # what those include
    f"{KERNELS}/utils.h",
    f"{KERNELS}/bf16.h",
    f"{KERNELS}/bf16_math.h",
    f"{KERNELS}/complex.h",
    f"{KERNELS}/defines.h",
    f"{KERNELS}/logging.h",
    f"{KERNELS}/steel/defines.h",
    f"{KERNELS}/steel/utils.h",
    f"{KERNELS}/steel/utils/integral_constant.h",
    f"{KERNELS}/steel/utils/type_traits.h",
    f"{KERNELS}/steel/gemm/gemm.h",
    f"{KERNELS}/steel/gemm/loader.h",
    f"{KERNELS}/steel/gemm/mma.h",
    f"{KERNELS}/steel/gemm/params.h",
    f"{KERNELS}/steel/gemm/transforms.h",
    # the host dispatch the .mm transcribes; reference only, never compiled
    "mlx/backend/metal/quantized.cpp",
]


# kernel-builder compiles every `.metal` it is given with its own flags, and only hands the build the
# files listed in build.toml. `mlx_metal/mlx_quantized.metal` compiles upstream's kernels under MLX's
# own math mode by including this file, so it is shipped with a header suffix: listed, but not
# compiled a second time. The contents are upstream's, byte for byte.
RENAMED = {f"{KERNELS}/quantized.metal": f"{KERNELS}/quantized.metal.h"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.environ.get("MLX_SRC", os.path.join(HERE, "mlx")))
    ap.add_argument("--rev", default=None, help="git rev to check out before copying")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        subprocess.run(["git", "clone", "https://github.com/ml-explore/mlx.git", args.src], check=True)
    if args.rev:
        subprocess.run(["git", "-C", args.src, "checkout", args.rev], check=True)

    rev = subprocess.run(
        ["git", "-C", args.src, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    if os.path.isdir(VENDOR):
        shutil.rmtree(VENDOR)
    for rel in FILES:
        dst = os.path.join(VENDOR, RENAMED.get(rel, rel))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(args.src, rel), dst)

    with open(os.path.join(VENDOR, "UPSTREAM"), "w") as f:
        f.write(f"https://github.com/ml-explore/mlx\n{rev}\n")
    shutil.copy2(os.path.join(args.src, "LICENSE"), os.path.join(VENDOR, "LICENSE"))

    print(f"vendored {sum(len(f) for _, _, f in os.walk(VENDOR))} files from mlx @ {rev[:12]} into {VENDOR}")


if __name__ == "__main__":
    main()
