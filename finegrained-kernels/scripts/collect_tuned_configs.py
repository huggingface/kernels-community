#!/usr/bin/env python3
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

"""Collate tuning-run export logs into the shipped ``configs/`` tables.

A tuning run writes one JSON object per crowned key::

    FINEGRAINED_AUTOTUNE_EXPORT=/tmp/tune.jsonl FINEGRAINED_AUTOTUNE_TRIALS=200 \\
        python bench/bench_moe.py

then this collates the log(s) into ``torch-ext/finegrained_kernels/configs/``, one file per
(kernel, device). Commit those files and they ship with the build — see
``torch-ext/finegrained_kernels/tuned_configs.py`` for how they are read back.

    python scripts/collect_tuned_configs.py /tmp/tune.jsonl [more.jsonl ...]

Later records win, so re-running a sweep at a higher trial budget refreshes the entries it
covers and leaves the rest intact. ``--prune`` drops entries whose kernel no longer exists.
"""

import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(ROOT, "torch-ext", "finegrained_kernels", "configs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", help="export logs written by FINEGRAINED_AUTOTUNE_EXPORT")
    ap.add_argument("--configs-dir", default=CONFIGS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = ap.parse_args()

    # (fn, device) -> {serialized key: config}
    tables: dict[tuple[str, str], dict] = {}
    seen = skipped = 0
    for log in args.logs:
        with open(log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = json.dumps(rec["key"])
                    tables.setdefault((rec["fn"], rec["device"]), {})[key] = rec["config"]
                    seen += 1
                except (ValueError, KeyError):
                    skipped += 1  # a torn last line from an interrupted run

    if skipped:
        print(f"skipped {skipped} unparseable record(s)")
    if not tables:
        print("no records found — did the run set FINEGRAINED_AUTOTUNE_EXPORT?")
        return 1

    os.makedirs(args.configs_dir, exist_ok=True)
    total_new = total_changed = 0
    for (fn, device), table in sorted(tables.items()):
        path = os.path.join(args.configs_dir, f"{fn},device_name={device}.json")
        existing = {}
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
        new = {k: v for k, v in table.items() if k not in existing}
        changed = {k: v for k, v in table.items() if k in existing and existing[k] != v}
        merged = {**existing, **table}
        total_new += len(new)
        total_changed += len(changed)
        print(
            f"{os.path.basename(path)}: {len(merged)} entries "
            f"(+{len(new)} new, {len(changed)} updated)"
        )
        if not args.dry_run:
            with open(path, "w") as f:
                json.dump(dict(sorted(merged.items())), f, indent=1, sort_keys=True)
                f.write("\n")

    print(
        f"{seen} record(s) -> {len(tables)} table(s): {total_new} new, {total_changed} updated"
        + (" [dry run]" if args.dry_run else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
