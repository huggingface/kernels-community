# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "triton",
#   "kernels",
# ]
# ///

import os
from pathlib import Path
import sys

import torch
from kernels import get_kernel, get_local_kernel


def load_hydra_kernel():
    if os.environ.get("HYDRA_USE_HUB") == "1":
        return get_kernel("kernels-community/hydra")

    root = Path(__file__).resolve().parent
    for variant in (root / "build", root):
        if (variant / "metadata.json").exists():
            return get_local_kernel(variant)

    sys.path.insert(0, str(root / "torch-ext"))
    import hydra

    return hydra


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("Hydra requires CUDA for this example")

    kernel = load_hydra_kernel()
    q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 8, 8192, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 8, 8192, 128, device="cuda", dtype=torch.bfloat16)

    out = kernel.hydra(q, k, v)
    print(f"Hydra decode: {tuple(q.shape)} x {tuple(k.shape)} -> {tuple(out.shape)}")


if __name__ == "__main__":
    main()
