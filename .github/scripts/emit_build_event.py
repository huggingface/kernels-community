#!/usr/bin/env python3
# Record one build workflow run in the kernel-status dataset.
#
# Input (environment, set by the `emit-status-event` job of the build workflows):
#   KERNEL, MODE, PR_NUMBER, DISPATCH_KEY, TARGET_BRANCH, REPO_PREFIX, UPLOAD,
#   REQUESTED_BY, NEEDS_JSON (`toJSON(needs)`), DATA_REF (git ref holding the
#   built sources), GH_TOKEN (reads the run's jobs), HF_TOKEN (dataset write).
#
# Output:
#   The event JSON on stdout, uploaded to
#   events/<yyyy>/<mm>/<run_id>-<attempt>-<platform>.json when HF_TOKEN is set.
#
# Example:
#   python3 .github/scripts/emit_build_event.py linux --uploads hub-uploads
import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATASET_REPO = os.getenv("KERNEL_STATUS_REPO") or "kernels-community/kernel-status"
KERNEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
BACKEND_RE = re.compile(r"\b(cuda|cpu|rocm|metal|xpu|tpu)\b")
ARCH_RE = re.compile(r"\b(x86_64|aarch64)")
DEFAULT_JOB = {"mac": ("metal", "aarch64"), "windows": (None, "x86_64"), "linux": (None, None)}
OWN_JOB = "emit-status-event"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def read_sources(kernel: str, ref: str) -> dict:
    config = None
    for candidate in (f"{kernel}/src/build.toml", f"{kernel}/build.toml"):
        try:
            config = tomllib.loads(git("show", f"{ref}:{candidate}"))
            break
        except subprocess.CalledProcessError:
            continue
    builder = {}
    try:
        lock = json.loads(git("show", f"{ref}:{kernel}/flake.lock"))
        builder = lock["nodes"]["kernel-builder"]["locked"]
    except (subprocess.CalledProcessError, KeyError):
        pass
    general = (config or {}).get("general", {})
    return {
        "source": {
            "sha": git("rev-parse", ref),
            "kernel_tree_sha": git("rev-parse", f"{ref}:{kernel}"),
            "builder_rev": builder.get("rev"),
            "builder_date": builder.get("lastModified")
            and datetime.fromtimestamp(builder["lastModified"], tz=timezone.utc).isoformat(),
        },
        "config": {
            "version": general.get("version"),
            "repo_id": general.get("hub", {}).get("repo-id"),
            "backends": general.get("backends", []),
        },
    }


def fetch_jobs(repository: str, run_id: str, attempt: str, token: str) -> list[dict]:
    jobs, page = [], 1
    while True:
        url = (
            f"https://api.github.com/repos/{repository}/actions/runs/{run_id}"
            f"/attempts/{attempt}/jobs?per_page=100&page={page}"
        )
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            batch = json.load(response)["jobs"]
        jobs.extend(batch)
        if len(batch) < 100:
            return jobs
        page += 1


def summarize_job(job: dict, platform: str) -> dict:
    name = job["name"]
    backend, arch = DEFAULT_JOB[platform]
    if m := BACKEND_RE.search(name):
        backend = m.group(1)
    if m := ARCH_RE.search(name):
        arch = m.group(1)
    failed = next((s["name"] for s in job.get("steps", []) if s.get("conclusion") == "failure"), None)
    duration = None
    if job.get("started_at") and job.get("completed_at"):
        duration = int((_parse(job["completed_at"]) - _parse(job["started_at"])).total_seconds())
    return {
        "id": job.get("id"),
        "name": name,
        "backend": backend,
        "arch": arch,
        "conclusion": job.get("conclusion"),
        "failed_step": failed,
        "duration_s": duration,
    }


def overall_result(needs: dict) -> str:
    results = {n["result"] for n in needs.values()}
    for result in ("failure", "cancelled"):
        if result in results:
            return result
    return "success"


def load_uploads(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(directory.rglob("*.json"))]


def build_event(platform: str, env: dict, jobs: list[dict], sources: dict, uploads: list[dict]) -> dict:
    needs = json.loads(env.get("NEEDS_JSON") or "{}")
    jobs = [j for j in jobs if not j["name"].startswith(OWN_JOB)]
    started = min((j["started_at"] for j in jobs if j.get("started_at")), default=None)
    repository, run_id = env["GITHUB_REPOSITORY"], env["GITHUB_RUN_ID"]
    return {
        "schema": 1,
        "kernel": env["KERNEL"],
        "workflow": platform,
        "mode": env.get("MODE") or "release",
        "run_id": int(run_id),
        "run_attempt": int(env.get("GITHUB_RUN_ATTEMPT") or 1),
        "run_url": f"https://github.com/{repository}/actions/runs/{run_id}",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "trigger": {
            "requested_by": env.get("REQUESTED_BY") or None,
            "pr_number": int(env["PR_NUMBER"]) if env.get("PR_NUMBER") else None,
            "dispatch_key": env.get("DISPATCH_KEY") or None,
            "target_branch": env.get("TARGET_BRANCH") or None,
            "repo_prefix": env.get("REPO_PREFIX") or "kernels-community",
            "upload": env.get("UPLOAD", "true") != "false",
        },
        **sources,
        "result": {
            "overall": overall_result(needs),
            "needs": {name: n["result"] for name, n in needs.items()},
            "gpu_tests": needs.get("test-kernel-gpu", {}).get("result"),
            "jobs": [summarize_job(j, platform) for j in jobs],
        },
        "uploads": uploads,
    }


def upload_event(event: dict, token: str) -> str:
    from huggingface_hub import HfApi

    finished = _parse(event["finished_at"])
    path = (
        f"events/{finished:%Y}/{finished:%m}/"
        f"{event['run_id']}-{event['run_attempt']}-{event['workflow']}.json"
    )
    HfApi(token=token).upload_file(
        path_or_fileobj=json.dumps(event, indent=2).encode(),
        path_in_repo=path,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message=f"{event['kernel']}: {event['workflow']} {event['mode']} {event['result']['overall']}",
    )
    return path


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("platform", choices=sorted(DEFAULT_JOB))
    parser.add_argument("--uploads", default="hub-uploads", help="Directory of kernel-builder upload JSONs.")
    args = parser.parse_args()

    env = os.environ
    if not KERNEL_RE.match(env.get("KERNEL", "")):
        print(f"Invalid kernel name: {env.get('KERNEL')!r}", file=sys.stderr)
        return 1
    sources = read_sources(env["KERNEL"], env.get("DATA_REF") or "HEAD")
    jobs = fetch_jobs(env["GITHUB_REPOSITORY"], env["GITHUB_RUN_ID"], env.get("GITHUB_RUN_ATTEMPT") or "1", env["GH_TOKEN"])
    event = build_event(args.platform, env, jobs, sources, load_uploads(Path(args.uploads)))
    print(json.dumps(event, indent=2))

    if not env.get("HF_TOKEN"):
        print("HF_TOKEN is not set; not uploading the event.", file=sys.stderr)
        return 0
    print(f"Uploaded {upload_event(event, env['HF_TOKEN'])} to {DATASET_REPO}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
