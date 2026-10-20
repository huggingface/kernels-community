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

"""``FINEGRAINED_AUTOTUNE_CONFIG``: the one runtime control over every tuned kernel.

Its value is a mode name (``fast``), inline JSON, or a path to a JSON file. JSON keys are the
tuner's own options; ``"mode"`` there applies a preset whose keys explicit ones override.

The tuner reads this when each kernel is decorated, i.e. when the package is imported.
"""

import json
import os
from pathlib import Path

ENV_VAR = "FINEGRAINED_AUTOTUNE_CONFIG"

# Tuner kwargs by name, plus the debugging outputs (``key_log_path``, ``call_count_path``,
# ``tracebacks``) and the cross-rank tune lock wait.
KEYS = {
    "n_trials": int, "strict": bool, "runoff": bool, "basin_probes": bool,
    "max_refine_iters": int, "log_path": str, "rotate_weights": bool,
    "key_log_path": str, "call_count_path": str, "tracebacks": bool, "lock_timeout": float,
}

MODES = {
    # one measured config, no search phases: correctness runs, not performance
    "fast": {"n_trials": 1, "runoff": False, "basin_probes": False, "max_refine_iters": 0},
}


def _read(value: str) -> dict:
    """The raw config, a mode name kept as ``{"mode": ...}`` so explicit keys compose with it."""
    if not value:
        return {}
    if value.lstrip().startswith("{"):
        return json.loads(value)
    if os.path.isfile(value):
        return json.loads(Path(value).read_text())
    return {"mode": value}


def load_autotune_config() -> dict:
    """Validated overrides from the environment: the mode's preset, then explicit keys."""
    config = _read(os.environ.get(ENV_VAR, ""))
    mode = config.pop("mode", None)
    if mode is not None:
        if mode not in MODES:
            raise ValueError(
                f"{ENV_VAR} mode {mode!r} is not one of {sorted(MODES)} (or a JSON object / JSON file path)"
            )
        config = {**MODES[mode], **config}
    unknown = set(config) - set(KEYS)
    if unknown:
        raise ValueError(
            f"{ENV_VAR} has unknown keys {sorted(unknown)}; expected a subset of {sorted(KEYS)} or \"mode\""
        )
    return {key: KEYS[key](value) for key, value in config.items()}

