---
pretty_name: Kernel status
configs:
  - config_name: default
    data_files: history.jsonl
---

# kernel-status

Build and health status for every kernel in [kernels-community](https://github.com/huggingface/kernels-community).

Last updated: {generated_at}

- `events/<yyyy>/<mm>/<run_id>-<attempt>-<platform>.json`: one event per build run, written by the build workflows (`.github/scripts/emit_build_event.py`)
- `history.jsonl`: every event in compact form, rewritten nightly
- `latest.json`: each kernel's facts, flags and health, rewritten nightly (`scripts/kernel_status_snapshot.py`)

Read it with the `kernel-status` cli in `scripts/kernel-status`.
