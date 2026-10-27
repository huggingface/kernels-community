"""Seed Triton's cache with configurations shipped inside this package."""

import os
from pathlib import Path

import torch
from triton.runtime.cache import FileCacheManager


def seed_cache():
    if os.environ.get("FLASH_ATTN_OPS_AUTOTUNE") == "1":
        return
    if not torch.cuda.is_available() or torch.version.hip:
        return
    # Triton's disk key distinguishes architectures, but not GPU models. Avoid
    # seeding a shared cache on machines with different models of GPU.
    names = {torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())}
    if len(names) != 1:
        return
    gpu = names.pop()
    root = Path(__file__).parent / "configs" / gpu
    for path in root.glob("*/*.autotune.json"):
        # These directory names are already encoded by Triton's cache manager.
        cache = FileCacheManager(path.parent.name)
        if not cache.get_file(path.name):
            cache.put(path.read_text(), path.name, binary=False)
