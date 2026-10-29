import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

from check_kernel_freshness import _github_api_request
from run_kernels_checks import discover_kernel_dirs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".github" / "scripts"))
from dispatch import WINDOWS_SKIP_BACKENDS, WORKFLOW_TO_BACKENDS, select_workflows  # noqa: E402

DATASET_REPO = os.getenv("KERNEL_STATUS_REPO") or "kernels-community/kernel-status"
WORKFLOW_PLATFORM = {"build.yaml": "linux", "build-mac.yaml": "mac", "build-windows.yaml": "windows"}
TORCH_RE = re.compile(r"^torch(\d)(\d+)-")
STABLE_ABI_RE = re.compile(r"^torch-stable-abi(\d)(\d+)-")
EVENT_MONTHS = 6
BUILDER_DAYS = 14
SEVERITIES = ("error", "warn", "info")
FLAGS = {
    "release-failed": ("error", "last release build failed on at least one platform"),
    "abi-fail": ("error", "nightly `kernel-builder check-abi` failed"),
    "missing-torch": ("error", "Hub has no variant for one of the current Torch versions"),
    "pending-merge": ("warn", "current Torch variants only exist in an open Hub PR on the vendor repo"),
    "needs-rebuild": ("warn", "published build used a different builder than main's flake.lock"),
    "unreleased": ("warn", "kernel sources on main changed since the last successful release"),
    "stale-builder": ("warn", f"flake.lock builder is >{BUILDER_DAYS}d behind the latest kernel-builder"),
}


# Build history


class Dataset:
    """The kernel-status dataset on the Hub, or a local copy with the same layout."""

    def __init__(self, local: str | None = None):
        self.local = Path(local) if local else None

    def read(self, path: str) -> bytes | None:
        if self.local:
            return (self.local / path).read_bytes() if (self.local / path).is_file() else None
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return Path(hf_hub_download(DATASET_REPO, path, repo_type="dataset")).read_bytes()
        except EntryNotFoundError:
            return None

    def list(self, directory: str) -> list[str]:
        if self.local:
            return sorted(str(p.relative_to(self.local)) for p in (self.local / directory).glob("*.json"))
        from huggingface_hub import HfApi
        from huggingface_hub.errors import EntryNotFoundError

        try:
            entries = HfApi().list_repo_tree(DATASET_REPO, path_in_repo=directory, repo_type="dataset")
            return sorted(e.path for e in entries if e.path.endswith(".json"))
        except EntryNotFoundError:
            return []


def history_row(path: str, e: dict) -> dict:
    """The compact form of an event kept in history.jsonl."""
    t, source = e["trigger"], e["source"]
    return {
        "path": path,
        "kernel": e["kernel"],
        "platform": e["workflow"],
        "mode": e["mode"],
        "result": e["result"]["overall"],
        "finished_at": e["finished_at"],
        "run_id": e["run_id"],
        "pr": t.get("pr_number"),
        "by": t.get("requested_by"),
        "shipped": e["mode"] == "release" and t["upload"] and t["repo_prefix"] == "kernels-community",
        "builder_rev": source["builder_rev"],
        "tree_sha": source["kernel_tree_sha"],
        "failed": [
            {"job": j["name"], "job_id": j.get("id"), "backend": j["backend"], "arch": j["arch"], "step": j["failed_step"]}
            for j in e["result"]["jobs"] if j["conclusion"] == "failure"
        ],
    }


def load_history(ds: Dataset, now: datetime) -> list[dict]:
    """history.jsonl plus events written since; only months at or after its newest row are listed."""
    raw = ds.read("history.jsonl")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line] if raw else []
    known = {r["path"] for r in rows}
    start = max((_parse(r["finished_at"]) for r in rows), default=now - timedelta(days=31 * EVENT_MONTHS))
    month = start.replace(day=1)
    while month <= now:
        for path in ds.list(f"events/{month:%Y/%m}"):
            if path not in known:
                rows.append(history_row(path, json.loads(ds.read(path))))
        month = (month + timedelta(days=32)).replace(day=1)
    return sorted(rows, key=lambda r: r["finished_at"])


# Kernel facts


def kernel_facts(root: Path, name: str) -> dict:
    config_path = next(p for p in (root / name / "src" / "build.toml", root / name / "build.toml") if p.is_file())
    with open(config_path, "rb") as f:
        general = tomllib.load(f)["general"]
    with open(root / name / "flake.lock") as f:
        locked = json.load(f)["nodes"]["kernel-builder"]["locked"]
    tree_sha = subprocess.run(
        ["git", "rev-parse", f"HEAD:{name}"], capture_output=True, text=True, cwd=root, check=True
    ).stdout.strip()
    ci_tests = sum(
        test.read_text(errors="ignore").count("pytest.mark.kernels_ci") for test in (root / name).rglob("test*.py")
    )
    hub = general.get("hub", {})
    backends = general.get("backends", [])
    return {
        "version": general.get("version", 1),
        "repo_id": hub.get("repo-id", f"kernels-community/{name}"),
        "hub_branch": hub.get("branch", f"v{general.get('version', 1)}"),
        "backends": backends,
        "platforms": current_platforms(name, backends),
        "builder": {"rev": locked["rev"], "date": _iso(locked["lastModified"])},
        "tree_sha": tree_sha,
        "ci_tests": ci_tests,
    }


def current_platforms(name: str, backends: list[str]) -> list[str]:
    """Platforms a release of this kernel builds on today, mirroring dispatch.py."""
    workflows = select_workflows(name, backends, notes=[])
    windows = WORKFLOW_TO_BACKENDS["build-windows.yaml"] - WINDOWS_SKIP_BACKENDS.get(name, set())
    if not set(backends) & windows:
        workflows.discard("build-windows.yaml")
    return [p for wf, p in WORKFLOW_PLATFORM.items() if wf in workflows]


def hub_state(api, repo_id: str, branch: str) -> dict:
    try:
        variants = build_variants(api, repo_id, branch)
    except Exception as err:
        return {"branch": branch, "error": str(err).splitlines()[0]}
    state = {"branch": branch, "variants": variants, **variant_coverage(variants)}
    # Releases to repos outside the community org only open a Hub PR (see
    # .github/scripts/hub_pr_upload_args.py), so builds can sit there unmerged.
    if not repo_id.startswith("kernels-community/"):
        state["pending"] = pending_prs(api, repo_id, branch)
    return state


def build_variants(api, repo_id: str, revision: str) -> list[str]:
    entries = api.list_repo_tree(repo_id, path_in_repo="build", revision=revision, repo_type="kernel")
    return sorted(Path(e.path).name for e in entries)


def variant_coverage(variants: list[str]) -> dict:
    torch = sorted({f"{m[1]}.{m[2]}" for v in variants if (m := TORCH_RE.match(v))}, key=_version)
    stable = sorted({f"{m[1]}.{m[2]}" for v in variants if (m := STABLE_ABI_RE.match(v))}, key=_version)
    return {
        "torch_versions": torch,
        "stable_abi": stable[0] if stable else None,
        # Noarch variants carry no Torch version (`torch-cuda`, `torch-universal`, ...).
        "noarch": any(not TORCH_RE.match(v) and not STABLE_ABI_RE.match(v) for v in variants),
    }


def pending_prs(api, repo_id: str, branch: str) -> list[dict]:
    """Open Hub PRs into `branch` with the Torch coverage of their build, newest first."""
    # huggingface_hub's discussion helpers do not accept repo_type="kernel" yet.
    from huggingface_hub.utils import build_hf_headers, get_session

    def get(path: str) -> dict:
        r = get_session().get(f"{api.endpoint}/api/kernels/{repo_id}/{path}", headers=build_hf_headers())
        r.raise_for_status()
        return r.json()

    pending = []
    try:
        for d in get("discussions?status=open&type=pull_request")["discussions"]:
            if get(f"discussions/{d['num']}").get("changes", {}).get("base") != f"refs/heads/{branch}":
                continue
            pending.append({
                "num": d["num"],
                "created_at": d["createdAt"],
                **variant_coverage(build_variants(api, repo_id, f"refs/pr/{d['num']}")),
            })
    except Exception as err:
        print(f"{repo_id}: could not list Hub PRs: {str(err).splitlines()[0]}", file=sys.stderr)
    return sorted(pending, key=lambda p: p["num"], reverse=True)


def latest_builder(token: str | None) -> dict:
    commit = _github_api_request("https://api.github.com/repos/huggingface/kernels/commits/main", token)
    return {"rev": commit["sha"], "date": commit["commit"]["committer"]["date"]}


# Health


def releases(name: str, k: dict, history: list[dict]) -> tuple[dict, dict | None]:
    """The last shipped release per current platform, and the last successful one."""
    shipped = [r for r in history if r["kernel"] == name and r["shipped"] and r["platform"] in k["platforms"]]
    latest = {r["platform"]: r for r in shipped}
    ok = [r for r in shipped if r["result"] == "success" and r["builder_rev"]]
    return latest, ok[-1] if ok else None


def flag_details(k: dict, torch_versions: list[str], latest: dict, released: dict | None) -> dict[str, str]:
    """Each raised flag with a short explanation of why."""
    failed = [r for r in latest.values() if r["result"] == "failure"]
    absent = [v for v in torch_versions if not torch_covered(k["hub"], v)]
    # Versions an open Hub PR would add are waiting on the repo owner, not on us.
    prs = {v: next((p for p in k["hub"].get("pending", []) if torch_covered(p, v)), None) for v in absent}
    missing = [v for v in absent if not prs[v]]
    pending = [v for v in absent if prs[v]]
    lag = k["builder"]["behind_days"]
    checks = {
        "release-failed": (failed, lambda: ", ".join(f"{r['platform']} on {r['finished_at'][:10]}" for r in failed)),
        "abi-fail": ((k.get("abi_check") or {}).get("result") == "fail", lambda: f"checked {k['abi_check']['checked_at'][:10]}"),
        "missing-torch": (missing, lambda: k["hub"].get("error") or f"no {', '.join(missing)} on {k['hub']['branch']}"),
        "pending-merge": (pending, lambda: pending_detail(k["repo_id"], {v: prs[v] for v in pending})),
        "needs-rebuild": (released and released["builder_rev"] != k["builder"]["rev"],
                          lambda: f"released with {released['builder_rev'][:8]}, main pins {k['builder']['rev'][:8]}"),
        "unreleased": (released and released["tree_sha"] != k["tree_sha"],
                       lambda: f"main changed since the release on {released['finished_at'][:10]}"),
        "stale-builder": (lag > BUILDER_DAYS, lambda: f"{lag}d behind latest"),
    }
    return {name: detail() for name, (hit, detail) in checks.items() if hit}


def pending_detail(repo_id: str, prs: dict[str, dict]) -> str:
    by_pr = {}
    for version, pr in prs.items():
        by_pr.setdefault(pr["num"], (pr, []))[1].append(version)
    return "; ".join(
        f"{', '.join(versions)} in https://huggingface.co/kernels/{repo_id}/discussions/{num} ({pr['created_at'][:10]})"
        for num, (pr, versions) in by_pr.items()
    )


def torch_covered(hub: dict, version: str) -> bool:
    if "error" in hub:
        return False
    stable = hub.get("stable_abi")
    return hub["noarch"] or version in hub["torch_versions"] or (stable is not None and _version(stable) <= _version(version))


def add_health(k: dict, name: str, torch_versions: list[str], history: list[dict]) -> dict:
    latest, released = releases(name, k, history)
    flags = flag_details(k, torch_versions, latest, released)
    return {
        **k,
        "health": next((s for s in SEVERITIES if any(FLAGS[f][0] == s for f in flags)), "ok"),
        "flags": flags,
        "release": {p: {"result": r["result"], "finished_at": r["finished_at"], "run_id": r["run_id"]} for p, r in latest.items()},
        "released": released and {"finished_at": released["finished_at"], "builder_rev": released["builder_rev"]},
    }


# Snapshot


def build_snapshot(root: Path, abi_results: dict, history: list[dict], token: str | None) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    builder = latest_builder(token)
    kernels = {}
    for name in discover_kernel_dirs(root, [".github", "scripts"]):
        facts = kernel_facts(root, name)
        facts["builder"]["behind_days"] = max(0, (_parse(builder["date"]) - _parse(facts["builder"]["date"])).days)
        kernels[name] = {
            **facts,
            "hub": hub_state(api, facts["repo_id"], facts["hub_branch"]),
            "abi_check": abi_results.get(name),
        }
    # The builder only builds the two newest Torch versions, so the newest seen are the current ones.
    seen = {v for k in kernels.values() for v in k["hub"].get("torch_versions", [])}
    torch_versions = sorted(seen, key=_version)[-2:]
    return {
        "schema": 2,
        "generated_at": now.isoformat(),
        "latest_builder": builder,
        "torch_versions": torch_versions,
        "flags": [{"name": name, "severity": s, "description": d} for name, (s, d) in FLAGS.items()],
        "kernels": {name: add_health(k, name, torch_versions, history) for name, k in kernels.items()},
    }


def _parse(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _version(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the kernel-status snapshot.")
    parser.add_argument("--root", default=".", help="Root of the kernels-community checkout.")
    parser.add_argument("--abi-results", help="JSON from `report_kernel_failures.py --results-json`.")
    parser.add_argument("--out", help="Also write latest.json and history.jsonl to this directory.")
    parser.add_argument("--upload", action="store_true", help=f"Upload latest.json and history.jsonl to {DATASET_REPO}.")
    parser.add_argument("--github-token", default=os.getenv("GITHUB_TOKEN"))
    args = parser.parse_args()

    abi_results = {}
    if args.abi_results:
        with open(args.abi_results) as f:
            abi_results = json.load(f)
    # Fold the events written since the last snapshot into the compact history.
    history = load_history(Dataset(), datetime.now(timezone.utc))
    snapshot = build_snapshot(Path(args.root).resolve(), abi_results, history, args.github_token)
    day = snapshot["generated_at"][:10]
    files = {
        "latest.json": json.dumps(snapshot, indent=2),
        "history.jsonl": "".join(json.dumps(r) + "\n" for r in history),
    }

    if args.out:
        for path, content in files.items():
            Path(args.out, path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out, path).write_text(content)
    if args.upload:
        from huggingface_hub import CommitOperationAdd, HfApi

        HfApi().create_commit(
            DATASET_REPO,
            [CommitOperationAdd(path, content.encode()) for path, content in files.items()],
            repo_type="dataset",
            commit_message=f"Snapshot {day}: {len(snapshot['kernels'])} kernels, {len(history)} builds",
        )
    print(f"Snapshot of {len(snapshot['kernels'])} kernels, torch {', '.join(snapshot['torch_versions'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
