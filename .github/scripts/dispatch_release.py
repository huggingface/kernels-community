#!/usr/bin/env python3
"""
Dispatch release workflows for a kernel.

Three entrypoints call this script:
  1. The PR-merge dummy workflow (via CLI)
  2. The comment bot (via import)
  3. Local CLI invocation
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path


RELEASE_WORKFLOWS = [
    "build-release.yaml",
    "build-release-mac.yaml",
    "build-release-windows.yaml",
]

KERNEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class ReleaseDispatchResult:
    kernel_name: str
    dispatched: list[tuple[str, str]] = field(default_factory=list)  # (workflow, dispatch_key)
    failed: list[tuple[str, int]] = field(default_factory=list)  # (workflow, http_code)
    skipped: list[str] = field(default_factory=list)  # workflow filenames


def github_api_request(
    url: str, token: str, method: str = "GET", data: dict | None = None
):
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.read().decode("utf-8")


def run_local(
    workflow: str,
    kernel_name: str,
    *,
    skip_build: bool = False,
    pr_number: str = "",
    target_branch: str = "",
    upload: bool = True,
) -> bool:
    """Run a release workflow locally via act."""
    cmd = [
        "act", "workflow_dispatch",
        "--container-options", "--privileged",
        "-W", f".github/workflows/{workflow}",
        "--input", f"kernel_name={kernel_name}",
    ]
    if skip_build:
        cmd.extend(["--input", "skip_build=true"])
    if pr_number:
        cmd.extend(["--input", f"pr_number={pr_number}"])
    if target_branch:
        cmd.extend(["--input", f"target_branch={target_branch}"])
    if not upload:
        cmd.extend(["--input", "upload=false"])
    print(f"Running locally: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def get_token() -> str | None:
    """Resolve GitHub token: env var first, then ``gh auth token`` fallback."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def get_repo() -> str | None:
    """Resolve repository: GITHUB_REPOSITORY env var, or parse from git remote."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    if repo:
        return repo
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        url = result.stdout.strip()
        match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
        if match:
            return match.group(1)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return None


BACKEND_TO_WORKFLOWS = {
    "cuda": {"build-release.yaml", "build-release-windows.yaml"},
    "cpu": {"build-release.yaml"},
    "rocm": {"build-release.yaml"},
    "metal": {"build-release-mac.yaml"},
    "xpu": {"build-release.yaml", "build-release-windows.yaml"},
}


def read_backends(kernel_name: str) -> list[str] | None:
    """Read the backends list from a kernel's build.toml. Returns None if not found."""
    build_toml = Path(kernel_name) / "build.toml"
    if not build_toml.exists():
        return None
    with open(build_toml, "rb") as f:
        config = tomllib.load(f)
    backends = config.get("general", {}).get("backends")
    if backends is None:
        backends = config.get("backends")
    if isinstance(backends, list):
        return backends
    return None


def select_workflows(kernel_name: str) -> list[str]:
    """
    Determine which release workflows to dispatch based on the kernel's
    backends declared in build.toml.

    Mapping:
      cuda, cpu, rocm -> build-release.yaml (Linux)
      metal           -> build-release-mac.yaml (macOS)
      xpu             -> build-release-windows.yaml (Windows)

    Falls back to all workflows if build.toml can't be read.
    """
    backends = read_backends(kernel_name)
    if backends is None:
        print(f"Could not read backends for {kernel_name}, dispatching all workflows")
        return set(RELEASE_WORKFLOWS)

    workflows = set()
    for b in backends:
        workflows.update(BACKEND_TO_WORKFLOWS.get(b, set()))

    if not workflows:
        print(f"No known backends found for {kernel_name}: {backends}, dispatching all workflows")
        return set(RELEASE_WORKFLOWS)

    return workflows


def dispatch_release(
    kernel_name: str,
    *,
    token: str,
    repo: str,
    ref: str = "main",
    dispatch_key_prefix: str = "",
    local: bool = False,
    skip_build: bool = False,
    pr_number: str = "",
    target_branch: str = "",
    upload: bool = True,
) -> ReleaseDispatchResult:
    """
    Dispatch the appropriate release workflows for a kernel.

    Args:
        kernel_name: Name of the kernel directory.
        token: GitHub API token.
        repo: GitHub repository in "owner/repo" format.
        ref: Git ref to dispatch against (default "main").
        dispatch_key_prefix: Optional prefix for dispatch keys (e.g. "pr42-").
        local: Run locally via act instead of remote dispatch.
        skip_build: Skip build and upload steps.
        pr_number: Optional PR number to checkout before building.
        target_branch: Target branch for upload.
        upload: Whether to upload after build.

    Returns:
        ReleaseDispatchResult with dispatched/failed/skipped lists.
    """
    if not KERNEL_NAME_RE.match(kernel_name):
        print(f"Invalid kernel name: {kernel_name!r}", file=sys.stderr)
        result = ReleaseDispatchResult(kernel_name=kernel_name)
        for wf in RELEASE_WORKFLOWS:
            result.failed.append((wf, 0))
        return result

    result = ReleaseDispatchResult(kernel_name=kernel_name)

    workflows = select_workflows(kernel_name)
    skipped_workflows = set(RELEASE_WORKFLOWS) - workflows
    result.skipped = sorted(skipped_workflows)

    api_base = f"https://api.github.com/repos/{repo}"
    for workflow in workflows:
        dispatch_key = (
            f"{dispatch_key_prefix}{kernel_name}-{workflow}-{uuid.uuid4().hex[:12]}"
        )
        if local:
            if run_local(
                workflow, kernel_name,
                skip_build=skip_build,
                pr_number=pr_number,
                target_branch=target_branch,
                upload=upload,
            ):
                result.dispatched.append((workflow, dispatch_key))
            else:
                result.failed.append((workflow, 0))
        else:
            dispatch_url = f"{api_base}/actions/workflows/{workflow}/dispatches"
            inputs = {
                "kernel_name": kernel_name,
                "dispatch_key": dispatch_key,
            }
            if skip_build:
                inputs["skip_build"] = "true"
            if pr_number:
                inputs["pr_number"] = pr_number
            if target_branch:
                inputs["target_branch"] = target_branch
            if not upload:
                inputs["upload"] = "false"
            dispatch_body = {
                "ref": ref,
                "inputs": inputs,
            }
            try:
                print(f"Dispatching {workflow} for kernel `{kernel_name}` on ref `{ref}`")
                github_api_request(dispatch_url, token, method="POST", data=dispatch_body)
                result.dispatched.append((workflow, dispatch_key))
            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="replace")
                print(f"Failed to dispatch {workflow} (HTTP {e.code}): {err_text}", file=sys.stderr)
                result.failed.append((workflow, e.code))

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dispatch release workflows for a kernel"
    )
    parser.add_argument("kernel_name", help="Kernel directory name")
    parser.add_argument(
        "--ref", default="main", help="Git ref to dispatch on (default: main)"
    )
    parser.add_argument(
        "--repo", default=None, help="GitHub repo in owner/repo format (default: auto-detect)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run release workflows locally via act instead of dispatching remotely",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip build and upload steps (for testing workflow plumbing)",
    )
    parser.add_argument(
        "--pr-number", default="",
        help="PR number to checkout before building",
    )
    parser.add_argument(
        "--target-branch", default="",
        help="Target branch for upload",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Build only, do not upload",
    )
    args = parser.parse_args()

    common = dict(
        skip_build=args.skip_build,
        pr_number=args.pr_number,
        target_branch=args.target_branch,
        upload=not args.no_upload,
    )

    if args.local:
        result = dispatch_release(
            args.kernel_name,
            token="",
            repo="",
            ref=args.ref,
            local=True,
            **common,
        )
    else:
        token = get_token()
        if not token:
            print(
                "Error: No GitHub token found. Set GITHUB_TOKEN or run `gh auth login`.",
                file=sys.stderr,
            )
            return 1

        repo = args.repo or get_repo()
        if not repo:
            print(
                "Error: Cannot determine repository. Set GITHUB_REPOSITORY or use --repo.",
                file=sys.stderr,
            )
            return 1

        result = dispatch_release(
            args.kernel_name,
            token=token,
            repo=repo,
            ref=args.ref,
            **common,
        )

    if result.dispatched:
        print(f"\nDispatched ({len(result.dispatched)}):")
        for wf, dk in result.dispatched:
            print(f"  - {wf} (key: {dk})")
    if result.skipped:
        print(f"\nSkipped ({len(result.skipped)}):")
        for wf in result.skipped:
            print(f"  - {wf}")
    if result.failed:
        print(f"\nFailed ({len(result.failed)}):")
        for wf, code in result.failed:
            print(f"  - {wf} (HTTP {code})")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
