"""Look up shipped launch configurations, falling back to Triton autotuning."""

import functools
import json
import os
from pathlib import Path

import torch
import triton


# Include the row count and every optional execution path. A config for plain
# normalization must not silently select the configuration for dropout, etc.
FORWARD_KEY = [
    "M",
    "N",
    "HAS_RESIDUAL",
    "STORE_RESIDUAL_OUT",
    "IS_RMS_NORM",
    "HAS_BIAS",
    "HAS_X1",
    "HAS_W1",
    "HAS_B1",
    "HAS_DROPOUT",
    "STORE_DROPOUT_MASK",
    "HAS_ROWSCALE",
    "zero_centered_weight",
]
BACKWARD_KEY = [
    "M",
    "N",
    "HAS_DRESIDUAL",
    "STORE_DRESIDUAL",
    "IS_RMS_NORM",
    "HAS_BIAS",
    "HAS_DROPOUT",
    "HAS_ROWSCALE",
    "HAS_DY1",
    "HAS_DX1",
    "HAS_B1",
    "RECOMPUTE_OUTPUT",
    "zero_centered_weight",
]


def config_file_name(op, hidden, dtype, gpu):
    dtype = str(dtype).removeprefix("torch.")
    return f"op={op},N={hidden},dtype={dtype},device_name={gpu.replace(' ', '_')}.json"


@functools.cache
def _load_configs(filename):
    path = Path(__file__).parent / "configs" / filename
    return json.loads(path.read_text()) if path.exists() else {}


def tuned_configs(op, keys):
    """Select a single candidate for an exact shipped match, avoiding benchmarks."""

    def prune(configs, named_args, **kwargs):
        if os.environ.get("FLASH_ATTN_OPS_AUTOTUNE") == "1":
            return configs
        args = {**named_args, **kwargs}
        x = args["X"]
        if x.device.type != "cuda" or torch.version.hip:
            return configs
        filename = config_file_name(
            op, args["N"], x.dtype, torch.cuda.get_device_name(x.device)
        )
        cases = _load_configs(filename).get(str(args["M"]), [])
        key = {name: args[name] for name in keys if name not in {"M", "N"}}
        # Triton appends tensor dtypes in argument order to its tuning key.
        dtypes = [str(arg.dtype) for arg in args.values() if hasattr(arg, "dtype")]
        for case in cases:
            if case["key"] == key and case["tensor_dtypes"] == dtypes:
                return [triton.Config({}, **case["config"])]
        return configs

    return prune
