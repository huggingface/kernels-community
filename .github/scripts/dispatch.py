#!/usr/bin/env python3
"""
Dispatch build workflows for a kernel.

Four entrypoints call this script:
  1. The PR-merge dispatch workflow (via CLI)
  2. The PR-open dispatch workflow (via CLI)
  3. The comment bot (via import)
  4. Local CLI invocation
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
    "build.yaml",
    "build-mac.yaml",
    "build-windows.yaml",
]

# Dispatched alongside a build when run_security is set (see dispatch_release).
SECURITY_WORKFLOW = "security-audit.yml"

KERNEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class ReleaseDispatchResult:
    kernel_name: str
    dispatched: list[tuple[str, str]] = field(default_factory=list)  # (workflow, dispatch_key)
    failed: list[tuple[str, int]] = field(default_factory=list)  # (workflow, http_code)
    skipped: list[str] = field(default_factory=list)  # workflow filenames
    # Set when run_security dispatches the security audit (see dispatch_release).
    security_dispatch_key: str | None = None  # dispatch_key of the audit run
    security_failed_code: int | None = None  # http_code if the audit dispatch failed


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


# Maps workflow filename to the `name:` field in the YAML, used for commit
# status context strings that must match between pending and final statuses.
WORKFLOW_DISPLAY_NAMES = {
    "build.yaml": "Build",
    "build-mac.yaml": "Build (macOS)",
    "build-windows.yaml": "Build (Windows)",
}

BACKEND_TO_WORKFLOWS = {
    "cuda": {"build.yaml", "build-windows.yaml"},
    "cpu": {"build.yaml"},
    "rocm": {"build.yaml"},
    "metal": {"build-mac.yaml"},
    "xpu": {"build.yaml", "build-windows.yaml"},
}

# Only these kernels are known to build successfully on Windows.
# Add new entries here as Windows support is validated for a kernel.
WINDOWS_KERNELS = {
    "relu",
    "activation",
    # "flash-attn2",
}

# Backends to skip on Windows for specific kernels (e.g. due to toolchain issues).
WINDOWS_SKIP_BACKENDS: dict[str, set[str]] = {
    "flash-attn2": {"xpu"},  # CUTLASS XPU headers fail on Windows (ushort undefined)
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


def select_workflows(kernel_name: str) -> set[str]:
    """
    Determine which build workflows to dispatch based on the kernel's
    backends declared in build.toml.

    Mapping:
      cuda, cpu, rocm -> build.yaml (Linux)
      metal           -> build-mac.yaml (macOS)
      cuda, xpu       -> build-windows.yaml (Windows, allowlisted kernels only)

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

    # Only dispatch Windows builds for kernels known to build there.
    if "build-windows.yaml" in workflows and kernel_name not in WINDOWS_KERNELS:
        workflows.discard("build-windows.yaml")
        print(f"Skipping Windows build for {kernel_name} (not in WINDOWS_KERNELS allowlist)")

    return workflows


def dispatch_security_audit(
    *,
    token: str,
    repo: str,
    ref: str,
    pr_number: str,
    head_sha: str = "",
    dispatch_key_prefix: str = "",
    dry_run: bool = False,
) -> str:
    """
    Dispatch the security-audit workflow for a PR and return its dispatch_key.

    The workflow definition is always read from ``ref`` (the default branch),
    while the audit itself checks out the PR head for analysis. Raises
    ``urllib.error.HTTPError`` if the dispatch request fails.
    """
    dispatch_key = f"{dispatch_key_prefix}security-{uuid.uuid4().hex[:12]}"
    dispatch_body = {
        "ref": ref,
        "inputs": {
            "pr_number": pr_number,
            "dispatch_key": dispatch_key,
            "head_sha": head_sha,
        },
    }
    if dry_run:
        print(f"\n[dry-run] {SECURITY_WORKFLOW}:")
        print(json.dumps(dispatch_body, indent=2))
        return dispatch_key

    dispatch_url = (
        f"https://api.github.com/repos/{repo}"
        f"/actions/workflows/{SECURITY_WORKFLOW}/dispatches"
    )
    print(f"Dispatching {SECURITY_WORKFLOW} for PR #{pr_number} on ref `{ref}`")
    github_api_request(dispatch_url, token, method="POST", data=dispatch_body)
    return dispatch_key


def dispatch_release(
    kernel_name: str,
    *,
    token: str,
    repo: str,
    ref: str = "main",
    mode: str = "release",
    repo_prefix: str = "kernels-community",
    dispatch_key_prefix: str = "",
    dry_run: bool = False,
    skip_build: bool = False,
    pr_number: str = "",
    head_sha: str = "",
    target_branch: str = "",
    upload: bool = True,
    run_security: bool = False,
) -> ReleaseDispatchResult:
    """
    Dispatch the appropriate build workflows for a kernel.

    Args:
        kernel_name: Name of the kernel directory.
        token: GitHub API token.
        repo: GitHub repository in "owner/repo" format.
        ref: Git ref to dispatch against (default "main").
        mode: Build mode - "pr" for CI builds, "release" for full builds.
        repo_prefix: Hub org prefix for uploads (default "kernels-community").
        dispatch_key_prefix: Optional prefix for dispatch keys (e.g. "pr42-").
        dry_run: Print what would be dispatched without actually dispatching.
        skip_build: Skip build and upload steps.
        pr_number: Optional PR number to checkout before building.
        head_sha: Optional PR head SHA for commit status reporting.
        target_branch: Target branch for upload.
        upload: Whether to upload after build.
        run_security: Also dispatch the security-audit workflow for this PR
            (requires pr_number). The audit runs concurrently with the build;
            its key/failure are reported via the result's security_* fields.

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

    backends = read_backends(kernel_name) or []
    workflows = select_workflows(kernel_name)

    # Invert BACKEND_TO_WORKFLOWS so we can scope backends per workflow.
    workflow_to_backends: dict[str, set[str]] = {}
    for backend, wfs in BACKEND_TO_WORKFLOWS.items():
        for wf in wfs:
            workflow_to_backends.setdefault(wf, set()).add(backend)

    skipped_workflows = set(RELEASE_WORKFLOWS) - workflows
    result.skipped = sorted(skipped_workflows)

    api_base = f"https://api.github.com/repos/{repo}"
    for workflow in workflows:
        # Only pass backends that this workflow can actually build.
        scoped = sorted(b for b in backends if b in workflow_to_backends.get(workflow, set()))

        # Drop backends known to fail on Windows for this kernel.
        if workflow == "build-windows.yaml":
            skip = WINDOWS_SKIP_BACKENDS.get(kernel_name, set())
            if skip:
                before = set(scoped)
                scoped = [b for b in scoped if b not in skip]
                skipped_backends = before - set(scoped)
                if skipped_backends:
                    print(f"Skipping backends {skipped_backends} on Windows for {kernel_name}")
            if not scoped:
                result.skipped.append(workflow)
                print(f"Skipping {workflow} for {kernel_name} (no backends remaining after filtering)")
                continue

        backends_csv = ",".join(scoped)

        dispatch_key = (
            f"{dispatch_key_prefix}{kernel_name}-{workflow}-{uuid.uuid4().hex[:12]}"
        )
        if dry_run:
            inputs = {
                "kernel_name": kernel_name,
                "dispatch_key": dispatch_key,
                "mode": mode,
                "backends": backends_csv,
                "repo_prefix": repo_prefix,
            }
            if skip_build:
                inputs["skip_build"] = "true"
            if pr_number:
                inputs["pr_number"] = pr_number
            if head_sha:
                inputs["head_sha"] = head_sha
            if target_branch:
                inputs["target_branch"] = target_branch
            if not upload:
                inputs["upload"] = "false"
            dispatch_body = {"ref": ref, "inputs": inputs}
            print(f"\n[dry-run] {workflow}:")
            print(json.dumps(dispatch_body, indent=2))
            result.dispatched.append((workflow, dispatch_key))
            continue
        dispatch_url = f"{api_base}/actions/workflows/{workflow}/dispatches"
        inputs = {
            "kernel_name": kernel_name,
            "dispatch_key": dispatch_key,
            "mode": mode,
            "backends": backends_csv,
            "repo_prefix": repo_prefix,
        }
        if skip_build:
            inputs["skip_build"] = "true"
        if pr_number:
            inputs["pr_number"] = pr_number
        if head_sha:
            inputs["head_sha"] = head_sha
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

            # Post a pending commit status so the PR shows the build is in progress.
            if head_sha:
                display_name = WORKFLOW_DISPLAY_NAMES.get(workflow, workflow)
                context = f"{display_name} / {kernel_name}"
                status_url = f"{api_base}/statuses/{head_sha}"
                try:
                    github_api_request(
                        status_url,
                        token,
                        method="POST",
                        data={
                            "state": "pending",
                            "description": "Build queued",
                            "context": context,
                        },
                    )
                except urllib.error.HTTPError as e:
                    # Non-fatal: the build was dispatched, status is best-effort.
                    print(f"Warning: failed to set pending status for {context}: {e}", file=sys.stderr)

        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            print(f"Failed to dispatch {workflow} (HTTP {e.code}): {err_text}", file=sys.stderr)
            result.failed.append((workflow, e.code))

    # Optionally fire the security audit for this PR, concurrently with the build.
    if run_security:
        if not pr_number:
            print(
                "run_security set but no pr_number provided; skipping security audit.",
                file=sys.stderr,
            )
        else:
            try:
                result.security_dispatch_key = dispatch_security_audit(
                    token=token,
                    repo=repo,
                    ref=ref,
                    pr_number=pr_number,
                    head_sha=head_sha,
                    dispatch_key_prefix=dispatch_key_prefix,
                    dry_run=dry_run,
                )
            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="replace")
                print(
                    f"Failed to dispatch {SECURITY_WORKFLOW} (HTTP {e.code}): {err_text}",
                    file=sys.stderr,
                )
                result.security_failed_code = e.code

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
        "--mode", default="release", choices=["pr", "release"],
        help="Build mode: pr (CI only) or release (build + upload) (default: release)",
    )
    parser.add_argument(
        "--repo", default=None, help="GitHub repo in owner/repo format (default: auto-detect)"
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
        "--head-sha", default="",
        help="PR head SHA for commit status reporting",
    )
    parser.add_argument(
        "--target-branch", default="",
        help="Target branch for upload",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Build only, do not upload",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the dispatch payloads without actually dispatching",
    )
    parser.add_argument(
        "--repo-prefix", default="kernels-community",
        help="Hub org prefix for uploads (default: kernels-community)",
    )
    parser.add_argument(
        "--security", action="store_true",
        help="Also dispatch the security-audit workflow for the PR (requires --pr-number)",
    )
    args = parser.parse_args()

    common = dict(
        mode=args.mode,
        repo_prefix=args.repo_prefix,
        dry_run=args.dry_run,
        skip_build=args.skip_build,
        pr_number=args.pr_number,
        head_sha=args.head_sha,
        target_branch=args.target_branch,
        upload=not args.no_upload,
        run_security=args.security,
    )

    if args.dry_run:
        result = dispatch_release(
            args.kernel_name,
            token="",
            repo=args.repo or "",
            ref=args.ref,
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
    if result.security_dispatch_key:
        print(f"\nSecurity audit: {SECURITY_WORKFLOW} (key: {result.security_dispatch_key})")
    if result.security_failed_code is not None:
        print(f"\nSecurity audit failed: {SECURITY_WORKFLOW} (HTTP {result.security_failed_code})")
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
