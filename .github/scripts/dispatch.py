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


# Configuration

WORKFLOWS = {
    "build": [
        "build.yaml",
        "build-mac.yaml",
        "build-windows.yaml",
    ],
    "security": [
        "security-audit.yml",
    ],
}

KERNEL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Workflow file -> its YAML `name:`; must match the context the build reports.
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

WORKFLOW_TO_BACKENDS = {
    workflow: {b for b, wfs in BACKEND_TO_WORKFLOWS.items() if workflow in wfs}
    for workflow in WORKFLOWS["build"]
}

WINDOWS_KERNELS = {
    "relu",
    "activation",
    # "flash-attn2",
}

WINDOWS_SKIP_BACKENDS: dict[str, set[str]] = {
    "flash-attn2": {"xpu"},  # CUTLASS XPU headers fail on Windows (ushort undefined)
}


# Data types


@dataclass
class DispatchResult:
    kernel_name: str
    dispatched: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, int]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    security_dispatched: list[tuple[str, str]] = field(default_factory=list)
    security_failed: list[tuple[str, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dry_run_payloads: list[tuple[str, dict]] = field(default_factory=list)


@dataclass
class PlannedDispatch:
    kind: str  # "build" or "security"
    workflow: str
    dispatch_key: str
    body: dict
    description: str
    status_context: str | None = None  # build only


@dataclass
class DispatchPlan:
    kernel_name: str
    head_sha: str = ""
    actions: list[PlannedDispatch] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# Reading build.toml and selecting workflows


def read_backends(kernel_name: str) -> list[str] | None:
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


def select_workflows(kernel_name: str, *, notes: list[str]) -> set[str]:
    backends = read_backends(kernel_name)
    if backends is None:
        notes.append(
            f"Could not read backends for {kernel_name}, dispatching all workflows"
        )
        return set(WORKFLOWS["build"])

    workflows = set()
    for b in backends:
        workflows.update(BACKEND_TO_WORKFLOWS.get(b, set()))

    if not workflows:
        notes.append(
            f"No known backends found for {kernel_name}: {backends}, dispatching all workflows"
        )
        return set(WORKFLOWS["build"])

    if "build-windows.yaml" in workflows and kernel_name not in WINDOWS_KERNELS:
        workflows.discard("build-windows.yaml")
        notes.append(
            f"Skipping Windows build for {kernel_name} (not in WINDOWS_KERNELS allowlist)"
        )

    return workflows


# Planning


def _build_inputs(
    kernel_name: str,
    dispatch_key: str,
    mode: str,
    backends_csv: str,
    repo_prefix: str,
    *,
    skip_build: bool,
    pr_number: str,
    head_sha: str,
    target_branch: str,
    upload: bool,
    comment_pr_number: str,
) -> dict:
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
    # PR to report Hub upload pull-request links back to. Distinct from
    # pr_number (which selects the branch to check out): release/merge uploads
    # build from the default branch but still comment on the originating PR.
    if comment_pr_number:
        inputs["comment_pr_number"] = comment_pr_number
    return inputs


def _plan_security_actions(
    plan: DispatchPlan,
    *,
    ref: str,
    pr_number: str,
    head_sha: str,
    dispatch_key_prefix: str,
) -> None:
    for workflow in WORKFLOWS["security"]:
        dispatch_key = (
            f"{dispatch_key_prefix}security-{workflow}-{uuid.uuid4().hex[:12]}"
        )
        body = {
            "ref": ref,
            "inputs": {
                "pr_number": pr_number,
                "dispatch_key": dispatch_key,
                "head_sha": head_sha,
            },
        }
        plan.actions.append(
            PlannedDispatch(
                kind="security",
                workflow=workflow,
                dispatch_key=dispatch_key,
                body=body,
                description=f"for PR #{pr_number} on ref `{ref}`",
            )
        )


def _windows_scoped_backends(
    scoped: list[str], kernel_name: str, plan: DispatchPlan
) -> list[str] | None:
    skip = WINDOWS_SKIP_BACKENDS.get(kernel_name, set())
    kept = [b for b in scoped if b not in skip]
    if dropped := set(scoped) - set(kept):
        plan.notes.append(f"Skipping backends {dropped} on Windows for {kernel_name}")
    if not kept:
        plan.skipped.append("build-windows.yaml")
        plan.notes.append(
            f"Skipping build-windows.yaml for {kernel_name} (no backends remaining after filtering)"
        )
        return None
    return kept


def _plan_build_actions(
    plan: DispatchPlan,
    kernel_name: str,
    *,
    ref: str,
    mode: str,
    repo_prefix: str,
    dispatch_key_prefix: str,
    skip_build: bool,
    pr_number: str,
    head_sha: str,
    target_branch: str,
    upload: bool,
    comment_pr_number: str,
) -> None:
    backends = read_backends(kernel_name) or []
    workflows = select_workflows(kernel_name, notes=plan.notes)
    plan.skipped = sorted(set(WORKFLOWS["build"]) - workflows)

    for workflow in sorted(workflows):
        scoped = sorted(
            b for b in backends if b in WORKFLOW_TO_BACKENDS.get(workflow, set())
        )
        if workflow == "build-windows.yaml":
            scoped = _windows_scoped_backends(scoped, kernel_name, plan)
            if scoped is None:
                continue

        dispatch_key = (
            f"{dispatch_key_prefix}{kernel_name}-{workflow}-{uuid.uuid4().hex[:12]}"
        )
        status_context = (
            f"{WORKFLOW_DISPLAY_NAMES.get(workflow, workflow)} / {kernel_name}"
            if head_sha
            else None
        )
        plan.actions.append(
            PlannedDispatch(
                kind="build",
                workflow=workflow,
                dispatch_key=dispatch_key,
                body={
                    "ref": ref,
                    "inputs": _build_inputs(
                        kernel_name,
                        dispatch_key,
                        mode,
                        ",".join(scoped),
                        repo_prefix,
                        skip_build=skip_build,
                        pr_number=pr_number,
                        head_sha=head_sha,
                        target_branch=target_branch,
                        upload=upload,
                        comment_pr_number=comment_pr_number,
                    ),
                },
                description=f"for kernel `{kernel_name}` on ref `{ref}`",
                status_context=status_context,
            )
        )


def plan_dispatch(
    kernel_name: str = "",
    *,
    ref: str = "main",
    mode: str = "release",
    repo_prefix: str = "kernels-community",
    dispatch_key_prefix: str = "",
    skip_build: bool = False,
    pr_number: str = "",
    head_sha: str = "",
    target_branch: str = "",
    upload: bool = True,
    comment_pr_number: str = "",
    run_security: bool = False,
    security_only: bool = False,
) -> DispatchPlan:
    want_security = run_security or security_only
    plan = DispatchPlan(kernel_name=kernel_name, head_sha=head_sha)

    if not security_only:
        _plan_build_actions(
            plan,
            kernel_name,
            ref=ref,
            mode=mode,
            repo_prefix=repo_prefix,
            dispatch_key_prefix=dispatch_key_prefix,
            skip_build=skip_build,
            pr_number=pr_number,
            head_sha=head_sha,
            target_branch=target_branch,
            upload=upload,
            comment_pr_number=comment_pr_number,
        )

    if want_security:
        if pr_number:
            _plan_security_actions(
                plan,
                ref=ref,
                pr_number=pr_number,
                head_sha=head_sha,
                dispatch_key_prefix=dispatch_key_prefix,
            )
        elif security_only:
            plan.notes.append(
                "security_only set but no pr_number provided; nothing to do."
            )
        else:
            plan.notes.append(
                "run_security set but no pr_number provided; skipping security audit."
            )

    return plan


# Execution (performs the GitHub API I/O)


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


def _blank_result(plan: DispatchPlan) -> DispatchResult:
    return DispatchResult(
        kernel_name=plan.kernel_name,
        skipped=list(plan.skipped),
        notes=list(plan.notes),
    )


def _set_pending_status(
    action: PlannedDispatch,
    result: DispatchResult,
    *,
    token: str,
    api_base: str,
    head_sha: str,
) -> None:
    if not (action.status_context and head_sha):
        return
    status_url = f"{api_base}/statuses/{head_sha}"
    try:
        github_api_request(
            status_url,
            token,
            method="POST",
            data={
                "state": "pending",
                "description": "Build queued",
                "context": action.status_context,
            },
        )
    except urllib.error.HTTPError as e:
        # Non-fatal: the build was dispatched, status is best-effort.
        result.notes.append(
            f"Warning: failed to set pending status for {action.status_context}: {e}"
        )


def execute_plan(plan: DispatchPlan, *, token: str, repo: str) -> DispatchResult:
    result = _blank_result(plan)
    api_base = f"https://api.github.com/repos/{repo}"
    for action in plan.actions:
        url = f"{api_base}/actions/workflows/{action.workflow}/dispatches"
        try:
            result.notes.append(f"Dispatching {action.workflow} {action.description}")
            github_api_request(url, token, method="POST", data=action.body)
            if action.kind == "security":
                result.security_dispatched.append(
                    (action.workflow, action.dispatch_key)
                )
            else:
                result.dispatched.append((action.workflow, action.dispatch_key))
                _set_pending_status(
                    action,
                    result,
                    token=token,
                    api_base=api_base,
                    head_sha=plan.head_sha,
                )
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            result.notes.append(
                f"Failed to dispatch {action.workflow} (HTTP {e.code}): {err_text}"
            )
            if action.kind == "security":
                result.security_failed.append((action.workflow, e.code))
            else:
                result.failed.append((action.workflow, e.code))
    return result


def _result_from_plan(plan: DispatchPlan) -> DispatchResult:
    result = _blank_result(plan)
    for action in plan.actions:
        if action.kind == "security":
            result.security_dispatched.append((action.workflow, action.dispatch_key))
        else:
            result.dispatched.append((action.workflow, action.dispatch_key))
        result.dry_run_payloads.append((action.workflow, action.body))
    return result


# Orchestration


def dispatch(
    kernel_name: str = "",
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
    comment_pr_number: str = "",
    run_security: bool = False,
    security_only: bool = False,
) -> DispatchResult:
    if not security_only and (not kernel_name or not KERNEL_NAME_RE.match(kernel_name)):
        result = DispatchResult(kernel_name=kernel_name)
        result.notes.append(f"Invalid kernel name: {kernel_name!r}")
        for wf in WORKFLOWS["build"]:
            result.failed.append((wf, 0))
        return result

    plan = plan_dispatch(
        kernel_name,
        ref=ref,
        mode=mode,
        repo_prefix=repo_prefix,
        dispatch_key_prefix=dispatch_key_prefix,
        skip_build=skip_build,
        pr_number=pr_number,
        head_sha=head_sha,
        target_branch=target_branch,
        upload=upload,
        comment_pr_number=comment_pr_number,
        run_security=run_security,
        security_only=security_only,
    )
    if dry_run:
        return _result_from_plan(plan)
    return execute_plan(plan, token=token, repo=repo)


# Command-line interface


def get_token() -> str | None:
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


def format_dry_run_payloads(result: DispatchResult) -> str:
    lines = []
    for workflow, body in result.dry_run_payloads:
        lines.append(f"\n[dry-run] {workflow}:")
        lines.append(json.dumps(body, indent=2))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dispatch release workflows for a kernel"
    )
    parser.add_argument(
        "kernel_name",
        nargs="?",
        default="",
        help="Kernel directory name (not required with --security-only)",
    )
    parser.add_argument(
        "--ref", default="main", help="Git ref to dispatch on (default: main)"
    )
    parser.add_argument(
        "--mode",
        default="release",
        choices=["pr", "release"],
        help="Build mode: pr (CI only) or release (build + upload) (default: release)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo in owner/repo format (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip build and upload steps (for testing workflow plumbing)",
    )
    parser.add_argument(
        "--pr-number",
        default="",
        help="PR number to checkout before building",
    )
    parser.add_argument(
        "--head-sha",
        default="",
        help="PR head SHA for commit status reporting",
    )
    parser.add_argument(
        "--target-branch",
        default="",
        help="Target branch for upload",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Build only, do not upload",
    )
    parser.add_argument(
        "--comment-pr-number",
        default="",
        help="PR to report Hub upload pull-request links back to (for --create-pr uploads)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the dispatch payloads without actually dispatching",
    )
    parser.add_argument(
        "--repo-prefix",
        default="kernels-community",
        help="Hub org prefix for uploads (default: kernels-community)",
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help="Also dispatch the security-audit workflow for the PR (requires --pr-number)",
    )
    parser.add_argument(
        "--security-only",
        action="store_true",
        help="Only dispatch the security-audit workflow, skip all builds (requires --pr-number)",
    )
    parser.add_argument(
        "--dispatch-key-prefix",
        default="",
        help="Prefix for dispatch keys (the bot uses 'pr<N>-')",
    )
    args = parser.parse_args()

    if args.security_only:
        if not args.pr_number:
            print("Error: --security-only requires --pr-number.", file=sys.stderr)
            return 1

    if not args.security_only and not args.kernel_name:
        print(
            "Error: kernel_name is required (unless using --security-only).",
            file=sys.stderr,
        )
        return 1

    common = dict(
        mode=args.mode,
        repo_prefix=args.repo_prefix,
        dry_run=args.dry_run,
        skip_build=args.skip_build,
        pr_number=args.pr_number,
        head_sha=args.head_sha,
        target_branch=args.target_branch,
        upload=not args.no_upload,
        comment_pr_number=args.comment_pr_number,
        run_security=args.security,
        security_only=args.security_only,
        dispatch_key_prefix=args.dispatch_key_prefix,
    )

    if args.dry_run:
        result = dispatch(
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

        result = dispatch(
            args.kernel_name,
            token=token,
            repo=repo,
            ref=args.ref,
            **common,
        )

    # Diagnostics to stderr; results (payloads + summary) to stdout.
    for note in result.notes:
        print(note, file=sys.stderr)
    if args.dry_run and result.dry_run_payloads:
        print(format_dry_run_payloads(result))

    def section(title: str, lines: list[str]) -> None:
        if lines:
            print(f"\n{title} ({len(lines)}):")
            for line in lines:
                print(f"  - {line}")

    section("Dispatched", [f"{wf} (key: {dk})" for wf, dk in result.dispatched])
    section("Security", [f"{wf} (key: {dk})" for wf, dk in result.security_dispatched])
    section("Security failed", [f"{wf} (HTTP {c})" for wf, c in result.security_failed])
    section("Skipped", result.skipped)
    section("Failed", [f"{wf} (HTTP {c})" for wf, c in result.failed])

    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
