from dataclasses import dataclass, field
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
import uuid

from dispatch import (
    BACKEND_TO_WORKFLOWS,
    WORKFLOWS,
    dispatch,
    format_dry_run_payloads,
    parse_kernel_arg,
)

KERNEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
# Kernels may be scoped to backends, e.g. `flash-attn2[xpu,cpu]`, so the gate
# additionally allows `[`, `]`, and `,`.
COMMENT_CHARS_RE = re.compile(r"^/kernel-bot[ A-Za-z0-9_./,\[\]-]*$")
# Backends a kernel may be scoped to (typos are rejected up front).
KNOWN_BACKENDS = set(BACKEND_TO_WORKFLOWS)
COMMAND_PERMISSIONS = {
    "build": {"admin", "write"},
    "security": {"admin", "write"},
    "security-and-build": {"admin", "write"},
    "build-and-stage": {"admin", "write"},
    "merge-and-upload": {"admin", "write"},
    "release": {"admin"},
}
# Commands that operate per-PR and do not require kernel names.
KERNELLESS_COMMANDS = {"security"}
MAX_COMMENT_LENGTH = 1024
RUN_LOOKUP_ATTEMPTS = 10
RUN_LOOKUP_SLEEP_SECONDS = 2
RUN_LOOKUP_PAGE_SIZE = 100
COMMAND_USAGE = (
    "Invalid command. Use `/kernel-bot <build|security|security-and-build|build-and-stage|merge-and-upload|release> "
    "<kernel1> [kernel2 ...] [--branch <target_branch>]`.\n"
    "A kernel may be scoped to a subset of backends, e.g. `flash-attn2[xpu,cpu]`.\n"
    "The `security` command does not require kernel names."
)


@dataclass
class ParsedCommand:
    command: str | None = None
    kernels: list[str] = field(default_factory=list)
    # Kernel name -> requested backend subset (absent means "all backends").
    backends: dict[str, list[str]] = field(default_factory=dict)
    branch: str | None = None
    error: str | None = None


@dataclass
class DispatchResult:
    kernel_name: str
    dispatch_key: str
    action_url: str | None = None


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


def create_issue_comment(api_base: str, token: str, issue_number: int, message: str):
    url = f"{api_base}/issues/{issue_number}/comments"
    _, body = github_api_request(url, token, method="POST", data={"body": message})
    if not body:
        return {}
    return json.loads(body)


def update_issue_comment(api_base: str, token: str, comment_id: int, message: str):
    url = f"{api_base}/issues/comments/{comment_id}"
    github_api_request(url, token, method="PATCH", data={"body": message})


def post_issue_comment_reaction(
    api_base: str, token: str, comment_id: int, reaction: str
):
    url = f"{api_base}/issues/comments/{comment_id}/reactions"
    github_api_request(url, token, method="POST", data={"content": reaction})


def try_post_issue_comment(api_base: str, token: str, issue_number: int, message: str):
    try:
        create_issue_comment(api_base, token, issue_number, message)
        return True
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        print(f"Failed to post PR comment (HTTP {e.code}).", file=sys.stderr)
        print(err_text, file=sys.stderr)
        return False


def try_create_issue_comment(
    api_base: str, token: str, issue_number: int, message: str
):
    try:
        return create_issue_comment(api_base, token, issue_number, message)
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        print(f"Failed to post PR comment (HTTP {e.code}).", file=sys.stderr)
        print(err_text, file=sys.stderr)
        return None


def try_update_issue_comment(api_base: str, token: str, comment_id: int, message: str):
    try:
        update_issue_comment(api_base, token, comment_id, message)
        return True
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        print(
            f"Failed to update PR comment {comment_id} (HTTP {e.code}).",
            file=sys.stderr,
        )
        print(err_text, file=sys.stderr)
        return False


def try_post_issue_comment_reaction(
    api_base: str, token: str, comment_id: int, reaction: str
):
    try:
        post_issue_comment_reaction(api_base, token, comment_id, reaction)
        return True
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        print(
            f"Failed to add reaction `{reaction}` to comment {comment_id} (HTTP {e.code}).",
            file=sys.stderr,
        )
        print(err_text, file=sys.stderr)
        return False


def get_user_permission(api_base: str, token: str, username: str):
    url = f"{api_base}/collaborators/{username}/permission"
    try:
        _, body = github_api_request(url, token, method="GET")
        parsed = json.loads(body)
        return parsed.get("permission")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def get_pull_request(api_base: str, token: str, issue_number: int):
    url = f"{api_base}/pulls/{issue_number}"
    _, body = github_api_request(url, token, method="GET")
    return json.loads(body)


def merge_pull_request(api_base: str, token: str, issue_number: int):
    url = f"{api_base}/pulls/{issue_number}/merge"
    _, body = github_api_request(url, token, method="PUT", data={})
    return json.loads(body)


def list_workflow_runs(
    api_base: str,
    token: str,
    workflow_filename: str,
    *,
    branch: str | None = None,
    event: str | None = None,
    per_page: int = RUN_LOOKUP_PAGE_SIZE,
):
    query = {"per_page": str(per_page)}
    if branch is not None:
        query["branch"] = branch
    if event is not None:
        query["event"] = event

    encoded = urllib.parse.urlencode(query)
    url = f"{api_base}/actions/workflows/{workflow_filename}/runs?{encoded}"
    _, body = github_api_request(url, token, method="GET")
    parsed = json.loads(body)
    return parsed.get("workflow_runs", [])


def emit_dispatch_diagnostics(result, *, dry_run: bool):
    for note in result.notes:
        print(note, file=sys.stderr)
    if dry_run:
        payloads = format_dry_run_payloads(result)
        if payloads:
            print(payloads)


def make_dispatch_key(issue_number: int, kernel_name: str):
    return f"pr{issue_number}-{kernel_name}-{uuid.uuid4().hex[:12]}"


def workflow_run_matches_dispatch(run: dict, dispatch_key: str):
    for field in ("display_title", "name"):
        value = run.get(field)
        if isinstance(value, str) and dispatch_key in value:
            return True
    return False


def workflow_run_url(repository: str, run: dict):
    html_url = run.get("html_url")
    if isinstance(html_url, str) and html_url:
        return html_url

    run_id = run.get("id")
    if run_id is None:
        return None
    return f"https://github.com/{repository}/actions/runs/{run_id}"


def resolve_dispatch_run_urls(
    api_base: str,
    token: str,
    repository: str,
    default_branch: str,
    dispatches: list[DispatchResult],
    *,
    workflows: list[str] | None = None,
):
    pending = {dispatch.dispatch_key: dispatch for dispatch in dispatches}
    if not pending:
        return

    if workflows is None:
        workflows = WORKFLOWS["build"]

    for attempt in range(RUN_LOOKUP_ATTEMPTS):
        for workflow in workflows:
            try:
                workflow_runs = list_workflow_runs(
                    api_base,
                    token,
                    workflow,
                    branch=default_branch,
                    event="workflow_dispatch",
                )
            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="replace")
                print(
                    f"Failed to list workflow runs for `{workflow}` (HTTP {e.code}).",
                    file=sys.stderr,
                )
                print(err_text, file=sys.stderr)
                continue

            for run in workflow_runs:
                matched_key = next(
                    (
                        dispatch_key
                        for dispatch_key in pending
                        if workflow_run_matches_dispatch(run, dispatch_key)
                    ),
                    None,
                )
                if matched_key is None:
                    continue

                pending[matched_key].action_url = workflow_run_url(repository, run)

        pending = {
            dispatch_key: dispatch
            for dispatch_key, dispatch in pending.items()
            if dispatch.action_url is None
        }
        if not pending:
            return

        if attempt < RUN_LOOKUP_ATTEMPTS - 1:
            time.sleep(RUN_LOOKUP_SLEEP_SECONDS)


def format_dispatched_lines(dispatches: list[DispatchResult]):
    lines = ["", f"Dispatched ({len(dispatches)}):"]
    for dispatch in dispatches:
        if dispatch.action_url:
            lines.append(f"- `{dispatch.kernel_name}`: {dispatch.action_url}")
        else:
            lines.append(
                f"- `{dispatch.kernel_name}`: dispatched, but run URL is not available yet"
            )
    return lines


def comment_base_lines(
    title: str,
    command_summary: str,
    mode_text: str,
    target_branch: str,
    pr_head_sha: str | None,
):
    lines = [
        title,
        "",
        f"Command: `{command_summary}`",
        f"Mode: `{mode_text}`",
        f"Target branch: `{target_branch}`",
    ]
    if pr_head_sha:
        lines.append(f"PR head SHA: `{pr_head_sha}`")
    lines.append(f"Workflows: `{', '.join(WORKFLOWS['build'])}`")
    return lines


def format_pending_comment(
    command_summary: str,
    mode_text: str,
    target_branch: str,
    pr_head_sha: str | None,
    *,
    include_security: bool = False,
):
    lines = comment_base_lines(
        "Build request received.",
        command_summary,
        mode_text,
        target_branch,
        pr_head_sha,
    )
    if include_security:
        wf_names = ", ".join(WORKFLOWS["security"])
        lines.append(f"Security audit: `{wf_names}` (running concurrently)")
    lines.extend(["", "Status: `processing`"])
    return "\n".join(lines)


def format_result_comment(
    command_summary: str,
    mode_text: str,
    target_branch: str,
    pr_head_sha: str | None,
    *,
    merge_result_message: str | None = None,
    dispatches: list[DispatchResult] | None = None,
    failed: list[tuple[str, int]] | None = None,
    failure_message: str | None = None,
    security_dispatches: list[DispatchResult] | None = None,
    security_failed: list[tuple[str, int]] | None = None,
):
    dispatches = dispatches or []
    failed = failed or []
    security_dispatches = security_dispatches or []
    security_failed = security_failed or []
    if failure_message is not None or (failed and not dispatches):
        title = "Build request failed."
    else:
        title = "Build request processed."

    lines = comment_base_lines(
        title,
        command_summary,
        mode_text,
        target_branch,
        pr_head_sha,
    )
    if failure_message:
        lines.extend(["", f"Failure: {failure_message}"])
    if merge_result_message:
        lines.extend(["", f"Merge result: {merge_result_message}"])
    if dispatches:
        lines.extend(format_dispatched_lines(dispatches))
    if security_dispatches:
        lines.extend(format_dispatched_lines(security_dispatches))
    if security_failed:
        failed_text = ", ".join(f"{wf} (HTTP {code})" for wf, code in security_failed)
        lines.extend(["", f"Security audit failed: `{failed_text}`"])
    if failed:
        failed_text = ", ".join(f"{kernel} (HTTP {code})" for kernel, code in failed)
        lines.extend(["", f"Failed ({len(failed)}): `{failed_text}`"])
    return "\n".join(lines)


def parse_command(comment: str) -> ParsedCommand:
    tokens = comment.strip().split()
    if len(tokens) < 2:
        return ParsedCommand(error=COMMAND_USAGE)

    command = tokens[1]
    if tokens[0] != "/kernel-bot" or command not in COMMAND_PERMISSIONS:
        return ParsedCommand(error=COMMAND_USAGE)

    branch = None
    args = tokens[2:]

    if "--branch" in args:
        branch_idx = args.index("--branch")
        if branch_idx != len(args) - 2:
            return ParsedCommand(
                error="Invalid `--branch` usage. Put it at the end: `--branch <target_branch>`.",
            )
        branch = args[branch_idx + 1]
        args = args[:branch_idx]

    if not args and command not in KERNELLESS_COMMANDS:
        return ParsedCommand(
            error="No kernels provided. Use `/kernel-bot <build|security-and-build|build-and-stage|merge-and-upload> <kernel1> [kernel2 ...]`.",
        )

    kernels = []
    backends: dict[str, list[str]] = {}
    seen = set()
    for token in args:
        name, requested = parse_kernel_arg(token)
        if name is None:
            return ParsedCommand(error=f"Invalid kernel name `{token}`.")
        if requested is not None:
            unknown = [b for b in requested if b not in KNOWN_BACKENDS]
            if unknown:
                known = ", ".join(sorted(KNOWN_BACKENDS))
                return ParsedCommand(
                    error=(
                        f"Unknown backend(s) `{', '.join(unknown)}` in `{token}`. "
                        f"Known backends: {known}."
                    )
                )
        if name not in seen:
            kernels.append(name)
            seen.add(name)
        if requested is not None:
            scoped = backends.setdefault(name, [])
            for b in requested:
                if b not in scoped:
                    scoped.append(b)

    if branch is not None and not BRANCH_RE.match(branch):
        return ParsedCommand(error=f"Invalid target branch `{branch}`.")

    return ParsedCommand(
        command=command, kernels=kernels, backends=backends, branch=branch
    )


def parse_numeric_id(raw_value: str | None):
    if raw_value is None:
        return None
    raw = raw_value.strip()
    if not raw.isdigit():
        return None
    return int(raw)


def comment_id_from_response(comment: dict | None):
    if not isinstance(comment, dict):
        return None
    raw_comment_id = comment.get("id")
    if isinstance(raw_comment_id, int):
        return raw_comment_id
    if isinstance(raw_comment_id, str):
        return parse_numeric_id(raw_comment_id)
    return None


def try_send_issue_comment(
    api_base: str,
    token: str,
    issue_number: int,
    message: str,
    *,
    comment_id: int | None = None,
):
    if comment_id is not None and try_update_issue_comment(
        api_base, token, comment_id, message
    ):
        return True
    return try_post_issue_comment(api_base, token, issue_number, message)


def comment_has_only_supported_characters(comment: str):
    return bool(COMMENT_CHARS_RE.fullmatch(comment))


def _resolve_context_from_env():
    token = os.environ.get("GITHUB_TOKEN", "")
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if not token or not repository:
        print("Missing required environment variables.", file=sys.stderr)
        return None

    comment = os.environ.get("COMMENT_BODY")
    comment_id = parse_numeric_id(os.environ.get("COMMENT_ID"))
    issue_number = parse_numeric_id(os.environ.get("COMMENT_ISSUE_NUMBER"))
    commenter = os.environ.get("COMMENT_AUTHOR")
    sender_type = os.environ.get("COMMENT_SENDER_TYPE")
    default_branch = os.environ.get("COMMENT_DEFAULT_BRANCH")

    if (
        comment is None
        or issue_number is None
        or not commenter
        or sender_type is None
        or not default_branch
    ):
        print(
            "Missing required comment context environment variables.", file=sys.stderr
        )
        return None

    if sender_type == "Bot":
        print("Ignoring bot comment.")
        return None
    if len(comment) > MAX_COMMENT_LENGTH:
        print("Ignoring oversized comment payload.", file=sys.stderr)
        return None
    if not comment.strip().startswith("/kernel-bot"):
        print("Ignoring non /kernel-bot comment.")
        return None
    if not comment_has_only_supported_characters(comment.strip()):
        print("Ignoring /kernel-bot comment with unsupported characters.")
        return None

    return dict(
        token=token,
        repository=repository,
        comment=comment,
        comment_id=comment_id,
        issue_number=issue_number,
        default_branch=default_branch,
        commenter=commenter,
    )


def _resolve_context_from_cli():
    import argparse
    import subprocess as _sp

    parser = argparse.ArgumentParser(
        description="Simulate what the comment bot would dispatch for a PR comment"
    )
    parser.add_argument(
        "comment",
        help='The comment to simulate, e.g. "/kernel-bot security-and-build activation"',
    )
    parser.add_argument("--pr-number", required=True, help="PR number")
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo in owner/repo format (default: auto-detect from git remote)",
    )
    parser.add_argument(
        "--ref",
        default="main",
        help="Default branch (default: main)",
    )
    parser.add_argument(
        "--head-sha",
        default=None,
        help="PR head SHA (default: fetched via `gh pr view`)",
    )
    args = parser.parse_args()

    repository = args.repo
    if not repository:
        try:
            result = _sp.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", result.stdout.strip())
            if match:
                repository = match.group(1)
        except (FileNotFoundError, _sp.CalledProcessError, _sp.TimeoutExpired):
            pass
    if not repository:
        print("Error: Cannot determine repository. Use --repo.", file=sys.stderr)
        return None

    pr_head_sha = args.head_sha or ""
    if not pr_head_sha:
        try:
            result = _sp.run(
                [
                    "gh",
                    "pr",
                    "view",
                    args.pr_number,
                    "--json",
                    "headRefOid",
                    "-q",
                    ".headRefOid",
                ],
                stdin=_sp.DEVNULL,
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            pr_head_sha = result.stdout.strip()
        except (FileNotFoundError, _sp.CalledProcessError, _sp.TimeoutExpired) as e:
            print(f"Warning: could not fetch PR head SHA: {e}", file=sys.stderr)

    return dict(
        token="",
        repository=repository,
        comment=args.comment,
        comment_id=None,
        issue_number=int(args.pr_number),
        default_branch=args.ref,
        commenter=None,
        pr_head_sha=pr_head_sha,
    )


def main(*, dry_run: bool = False):
    # Resolve inputs from env (CI) or CLI args (dry-run).
    ctx = _resolve_context_from_cli() if dry_run else _resolve_context_from_env()
    if ctx is None:
        return 1 if not dry_run else 0

    token = ctx["token"]
    repository = ctx["repository"]
    comment = ctx["comment"]
    comment_id = ctx["comment_id"]
    issue_number = ctx["issue_number"]
    default_branch = ctx["default_branch"]

    api_base = f"https://api.github.com/repos/{repository}"
    if not dry_run and comment_id is not None:
        try_post_issue_comment_reaction(api_base, token, comment_id, "+1")

    parsed_command = parse_command(comment)
    if parsed_command.error:
        if not dry_run:
            try_post_issue_comment(api_base, token, issue_number, parsed_command.error)
        else:
            print(f"Parse error: {parsed_command.error}", file=sys.stderr)
        return 0

    command = parsed_command.command
    kernels = parsed_command.kernels
    kernel_backends = parsed_command.backends
    requested_branch = parsed_command.branch
    if command is None:
        print("Internal error: command parsing returned no command.", file=sys.stderr)
        return 1

    # Permission check — skip in dry-run.
    if not dry_run:
        commenter = ctx["commenter"]
        permission = get_user_permission(api_base, token, commenter)
        allowed_permissions = COMMAND_PERMISSIONS[command]
        if permission not in allowed_permissions:
            if "write" in allowed_permissions:
                permission_error = (
                    f"I can only run `/kernel-bot {command}` for users with `write` or `admin` "
                    "repository permission."
                )
            else:
                permission_error = (
                    f"I can only run `/kernel-bot {command}` for users with `admin` "
                    "repository permission."
                )
            try_post_issue_comment(api_base, token, issue_number, permission_error)
            return 0

    # Fetch PR metadata — API in CI, already resolved in dry-run.
    if dry_run:
        pr_head_sha = ctx.get("pr_head_sha", "")
    else:
        try:
            pull_request = get_pull_request(api_base, token, issue_number)
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            print(
                f"Failed to fetch PR metadata for #{issue_number} (HTTP {e.code}).",
                file=sys.stderr,
            )
            print(err_text, file=sys.stderr)
            try_post_issue_comment(
                api_base,
                token,
                issue_number,
                f"Could not verify PR metadata, so `/kernel-bot {command}` was not run.",
            )
            return 1
        pr_head_sha = pull_request.get("head", {}).get("sha")

    if command == "security":
        # Security-only: dispatch the audit and return early.
        command_summary = f"/kernel-bot {command}"
        mode_text = "security audit only"
        target_branch = f"pr-{issue_number}"
        status_comment_id = None
        if not dry_run:
            status_comment_id = comment_id_from_response(
                try_create_issue_comment(
                    api_base,
                    token,
                    issue_number,
                    format_pending_comment(
                        command_summary,
                        mode_text,
                        target_branch,
                        pr_head_sha,
                        include_security=True,
                    ),
                )
            )
        release_result = dispatch(
            token=token,
            repo=repository,
            ref=default_branch,
            pr_number=str(issue_number),
            head_sha=pr_head_sha or "",
            dispatch_key_prefix=f"pr{issue_number}-",
            security_only=True,
            dry_run=dry_run,
        )
        emit_dispatch_diagnostics(release_result, dry_run=dry_run)
        security_dispatches = [
            DispatchResult(kernel_name=f"security ({wf})", dispatch_key=dk)
            for wf, dk in release_result.security_dispatched
        ]

        if not dry_run and security_dispatches:
            resolve_dispatch_run_urls(
                api_base,
                token,
                repository,
                default_branch,
                security_dispatches,
                workflows=WORKFLOWS["security"],
            )

        if not dry_run:
            try_send_issue_comment(
                api_base,
                token,
                issue_number,
                format_result_comment(
                    command_summary,
                    mode_text,
                    target_branch,
                    pr_head_sha,
                    security_dispatches=security_dispatches,
                    security_failed=release_result.security_failed,
                ),
                comment_id=status_comment_id,
            )
        return 1 if release_result.security_failed else 0

    if command in ("build", "security-and-build"):
        target_branch = requested_branch or f"pr-{issue_number}"
        dispatch_pr_number = str(issue_number)
        dispatch_upload = False
        dispatch_repo_prefix = "kernels-community"
    elif command == "build-and-stage":
        target_branch = requested_branch or f"pr-{issue_number}"
        dispatch_pr_number = str(issue_number)
        dispatch_upload = True
        dispatch_repo_prefix = "kernels-staging"
    elif command == "release":
        target_branch = requested_branch or ""
        dispatch_pr_number = ""
        dispatch_upload = True
        dispatch_repo_prefix = "kernels-community"
    else:  # merge-and-upload
        target_branch = requested_branch or ""
        dispatch_pr_number = ""
        dispatch_upload = True
        dispatch_repo_prefix = "kernels-community"

    mode_text = {
        "build": "build only",
        "security-and-build": "security audit + build",
        "build-and-stage": "build and stage",
        "merge-and-upload": "merge, build and upload",
        "release": "release (linux + mac + windows)",
    }[command]
    kernel_tokens = [
        f"{k}[{','.join(kernel_backends[k])}]" if kernel_backends.get(k) else k
        for k in kernels
    ]
    command_summary = f"/kernel-bot {command} {' '.join(kernel_tokens)}"
    if requested_branch is not None:
        command_summary += f" --branch {requested_branch}"
    # `/kernel-bot security-and-build` runs the security audit concurrently with the build.
    run_security = command == "security-and-build"
    status_comment_id = None
    if not dry_run:
        status_comment_id = comment_id_from_response(
            try_create_issue_comment(
                api_base,
                token,
                issue_number,
                format_pending_comment(
                    command_summary,
                    mode_text,
                    target_branch,
                    pr_head_sha,
                    include_security=run_security,
                ),
            )
        )

    merge_result_message = None
    if command == "merge-and-upload":
        if dry_run:
            print("[dry-run] Would merge PR before dispatching.\n")
        elif pull_request.get("merged"):
            merge_result_message = "PR is already merged. Continuing with build/upload."
        elif pull_request.get("state") != "open":
            try_send_issue_comment(
                api_base,
                token,
                issue_number,
                format_result_comment(
                    command_summary,
                    mode_text,
                    target_branch,
                    pr_head_sha,
                    failure_message="PR is not open and cannot be merged via `/kernel-bot merge-and-upload`.",
                ),
                comment_id=status_comment_id,
            )
            return 0
        else:
            try:
                merge_response = merge_pull_request(api_base, token, issue_number)
            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="replace")
                print(
                    f"Failed to merge PR #{issue_number} (HTTP {e.code}).",
                    file=sys.stderr,
                )
                print(err_text, file=sys.stderr)
                try_send_issue_comment(
                    api_base,
                    token,
                    issue_number,
                    format_result_comment(
                        command_summary,
                        mode_text,
                        target_branch,
                        pr_head_sha,
                        failure_message="Failed to merge PR before build/upload. Check mergeability and required checks.",
                    ),
                    comment_id=status_comment_id,
                )
                return 1

            if not merge_response.get("merged"):
                merge_message = merge_response.get(
                    "message", "PR merge failed for an unknown reason."
                )
                try_send_issue_comment(
                    api_base,
                    token,
                    issue_number,
                    format_result_comment(
                        command_summary,
                        mode_text,
                        target_branch,
                        pr_head_sha,
                        failure_message=f"PR merge failed: {merge_message}",
                    ),
                    comment_id=status_comment_id,
                )
                return 1

            merge_result_message = merge_response.get(
                "message", "PR merged successfully."
            )
    dispatches = []
    failed = []
    security_dispatches = []
    security_failed = []

    for index, kernel_name in enumerate(kernels):
        release_result = dispatch(
            kernel_name,
            token=token,
            repo=repository,
            ref=default_branch,
            mode="release",
            repo_prefix=dispatch_repo_prefix,
            dispatch_key_prefix=f"pr{issue_number}-",
            pr_number=dispatch_pr_number,
            head_sha=pr_head_sha or "",
            target_branch=target_branch,
            upload=dispatch_upload,
            requested_backends=kernel_backends.get(kernel_name),
            # The audit is per-PR, so request it only once (on the first kernel).
            run_security=run_security and index == 0,
            dry_run=dry_run,
        )
        emit_dispatch_diagnostics(release_result, dry_run=dry_run)
        for wf, dk in release_result.dispatched:
            dispatches.append(
                DispatchResult(kernel_name=f"{kernel_name} ({wf})", dispatch_key=dk)
            )
        for wf, code in release_result.failed:
            failed.append((f"{kernel_name} ({wf})", code))
        for wf, dk in release_result.security_dispatched:
            security_dispatches.append(
                DispatchResult(kernel_name=f"security ({wf})", dispatch_key=dk)
            )
        security_failed.extend(release_result.security_failed)

    if not dry_run:
        resolve_dispatch_run_urls(
            api_base,
            token,
            repository,
            default_branch,
            [*dispatches, *security_dispatches],
            workflows=(
                [*WORKFLOWS["build"], *WORKFLOWS["security"]]
                if run_security
                else WORKFLOWS["build"]
            ),
        )

        comment_written = try_send_issue_comment(
            api_base,
            token,
            issue_number,
            format_result_comment(
                command_summary,
                mode_text,
                target_branch,
                pr_head_sha,
                merge_result_message=merge_result_message,
                dispatches=dispatches,
                failed=failed,
                security_dispatches=security_dispatches,
                security_failed=security_failed,
            ),
            comment_id=status_comment_id,
        )
        if not comment_written:
            print(
                "Bot response could not be posted. Ensure workflow token has issues/pull-requests write permission.",
                file=sys.stderr,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        sys.argv.remove("--dry-run")
        raise SystemExit(main(dry_run=True))
    raise SystemExit(main())
