# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tuned autotuner configurations that SHIP with the kernel.

``bayesian_autotune`` already persists each crown to Triton's cache, but that cache is keyed on
``fn.cache_key`` (the kernel SOURCE hash), the Triton version, the invalidating env and the full
config grid — so it is machine-local and any edit to the kernel throws it away. It cannot be
shipped.

This is the durable layer: plain JSON under ``configs/``, keyed on what a deployment actually
shares — the device and the tuner's own key — and read before the search runs. A hit skips the
whole minutes-long tune; a miss falls through to tuning exactly as before, so a partial or absent
config set is never a correctness or availability problem. Written by a tuning run
(``FINEGRAINED_AUTOTUNE_EXPORT``) and collated by ``scripts/collect_tuned_configs.py``.

Layout mirrors the kernel-builder convention (one file per kernel and GPU, device in the name so
configs never cross devices)::

    configs/w8a8_block_dynamic_fp8_matmul_grouped_kernel,device_name=NVIDIA_B200.json

Each file maps a serialized tuner key to the winning config's ``all_kwargs()``::

    {"[4096, 2048, \\"weights\\", ...]": {"BLOCK_SIZE_M": 128, ..., "num_warps": 8}}

The stored value is ``Config.all_kwargs()`` — serializable, pre_hook-free, and the exact shape
the loader re-matches against the live (already pruned) ``Config`` objects, so the pre_hook comes
from code and a config that this launch's pruners reject is never replayed.
"""

import functools
import json
import os

import torch


CONFIGS_DIRNAME = "configs"


def device_name(device=None) -> str:
    """The GPU's name as it appears in a config filename (spaces to underscores), e.g.
    ``NVIDIA_B200``. Configs never cross devices, so a file tuned for a bigger GPU is inert
    rather than harmful on a smaller one."""
    return torch.cuda.get_device_name(device).replace(" ", "_").replace("/", "_")


def config_file_name(fn_name: str, device: str | None = None) -> str:
    return f"{fn_name},device_name={device or device_name()}.json"


def _configs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIGS_DIRNAME)


def serialize_key(key) -> str:
    """The tuner key as a stable string. The key holds only ints, bools, strings and ``None``
    (dtypes are already folded into ``name:dtype`` tags by the autotuner), so JSON round-trips
    it exactly and the text is diffable in review."""
    return json.dumps(list(key))


@functools.cache
def _load(fn_name: str, device: str) -> dict:
    """The shipped table for one (kernel, device), or ``{}``. Read at most once per process —
    the miss path is on the tuning path, never the per-launch path."""
    path = os.path.join(_configs_dir(), config_file_name(fn_name, device))
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}  # a corrupt shipped file must degrade to tuning, never raise


def lookup(fn_name: str, key) -> dict | None:
    """The shipped ``all_kwargs()`` for this kernel/device/key, or ``None`` to tune."""
    if os.environ.get("FINEGRAINED_AUTOTUNE_NO_SHIPPED"):
        return None
    try:
        table = _load(fn_name, device_name())
    except RuntimeError:  # no CUDA context (CPU-only import): nothing to look up
        return None
    return table.get(serialize_key(key))


def record(fn_name: str, key, config_kwargs: dict) -> None:
    """Append one crown to the export log named by ``FINEGRAINED_AUTOTUNE_EXPORT`` (no-op when
    unset). One JSON object per line, opened per call in append mode: a tuning sweep fans out
    over several processes (one per GPU), and short ``O_APPEND`` writes do not interleave.
    ``scripts/collect_tuned_configs.py`` collates the log into ``configs/``."""
    path = os.environ.get("FINEGRAINED_AUTOTUNE_EXPORT")
    if not path:
        return
    record = {
        "fn": fn_name,
        "device": device_name(),
        "key": list(key),
        "config": {k: v for k, v in config_kwargs.items() if v is not None},
    }
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # exporting is best-effort; never fail a tune over the log
