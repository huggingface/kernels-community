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

## `history.jsonl` columns

One row per build run.

| column | meaning |
| --- | --- |
| `path` | event file this row was compacted from, relative to the root of this dataset (see [`events/`](https://huggingface.co/datasets/kernels-community/kernel-status/tree/main/events)) |
| `kernel` | kernel directory name |
| `platform` | build workflow: `linux`, `mac` or `windows` |
| `mode` | `pr` (ci only), `stage` (build, upload and test) or `release` (build and upload) |
| `result` | outcome of the build and test jobs: `failure` if any failed, else `cancelled` if any was cancelled, else `success` |
| `finished_at` | when the run finished |
| `run_id` | GitHub Actions run id |
| `pr` | kernels-community PR the build ran for, empty for releases |
| `by` | GitHub login that requested the build |
| `shipped` | `true` for a `release` that uploaded to the `kernels-community` org |
| `builder_rev` | `kernel-builder` rev pinned in the kernel's `flake.lock` at the built commit |
| `tree_sha` | git tree sha of the kernel directory at the built commit |
| `failed` | every job in the run that failed, with its `backend`, `arch` and failing `step` |

The full event at `path` also has the run url, per job durations and the Hub
uploads of the run.

`result` only looks at the build and test jobs (`setup`, `build-kernel`,
`build-ci-test` and `test-kernel-gpu` on linux, `build-kernel` on mac and
windows), while `failed` lists any failed job. So a `success` row can still
have entries in `failed`, e.g. when `report-status` could not update the PR
comment after a good build.

### Missing values

The build workflows have written events since 2026-10-29. Runs before that
(back to 2026-05) were backfilled from past runs, and some fields could not be
recovered for them:

- `by` is empty, since the build workflows did not record the requester yet
- `builder_rev` and `tree_sha` are only set for `release` builds
- `pr` is empty for some `pr` builds

`pr` is also always empty for `release` builds, since they do not build a PR.

## `latest.json`

The dataset viewer only shows `history.jsonl`. `latest.json` is a single
object keyed by kernel (Hub variants, pinned builder, flags and health), so it
does not render as a table. The cli reads it for `kernel-status ls` and
`kernel-status report`, while `kernel-status builds` reads `history.jsonl` plus
any events written since. To get the raw file:

```bash
hf download kernels-community/kernel-status latest.json --repo-type dataset
```

Read it with the `kernel-status` cli in [`scripts/kernel-status`](https://github.com/huggingface/kernels-community/tree/main/scripts/kernel-status).
