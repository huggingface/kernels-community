#!/usr/bin/env python3
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


KERNEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
COMMENT_CHARS_RE = re.compile(r"^/kernel-bot[ A-Za-z0-9_./-]*$")
COMMAND_PERMISSIONS = {
    "build": {"admin", "write"},
    "build-and-upload": {"admin"},
    "merge-and-upload": {"admin"},
}
FORK_BLOCKED_COMMANDS = {"build", "build-and-upload"}
MAX_COMMENT_LENGTH = 1024
DISPATCH_WORKFLOW = "manual-build-upload.yaml"
RUN_LOOKUP_ATTEMPTS = 10
RUN_LOOKUP_SLEEP_SECONDS = 2
RUN_LOOKUP_PAGE_SIZE = 100
COMMAND_USAGE = (
    "Invalid command. Use `/kernel-bot <build|build-and-upload|merge-and-upload> "
    "<kernel1> [kernel2 ...] [--branch <target_branch>]`."
)


@dataclass
class ParsedCommand:
    command: str | None = None
    kernels: list[str] = field(default_factory=list)
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


def try_create_issue_comment(api_base: str, token: str, issue_number: int, message: str):
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
        print(f"Failed to update PR comment {comment_id} (HTTP {e.code}).", file=sys.stderr)
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
):
    pending = {dispatch.dispatch_key: dispatch for dispatch in dispatches}
    if not pending:
        return

    for attempt in range(RUN_LOOKUP_ATTEMPTS):
        try:
            workflow_runs = list_workflow_runs(
                api_base,
                token,
                DISPATCH_WORKFLOW,
                branch=default_branch,
                event="workflow_dispatch",
            )
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            print(
                f"Failed to list workflow runs for `{DISPATCH_WORKFLOW}` (HTTP {e.code}).",
                file=sys.stderr,
            )
            print(err_text, file=sys.stderr)
            return

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
            lines.append(f"- `{dispatch.kernel_name}`: dispatched, but run URL is not available yet")
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
    lines.append(f"Workflow: `{DISPATCH_WORKFLOW}`")
    return lines


def format_pending_comment(
    command_summary: str,
    mode_text: str,
    target_branch: str,
    pr_head_sha: str | None,
):
    lines = comment_base_lines(
        "Build request received.",
        command_summary,
        mode_text,
        target_branch,
        pr_head_sha,
    )
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
):
    dispatches = dispatches or []
    failed = failed or []
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
    if failed:
        failed_text = ", ".join(f"{kernel} (HTTP {code})" for kernel, code in failed)
        lines.extend(["", f"Failed ({len(failed)}): `{failed_text}`"])
    return "\n".join(lines)


def parse_command(comment: str) -> ParsedCommand:
    tokens = comment.strip().split()
    if len(tokens) < 3:
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

    if not args:
        return ParsedCommand(
            error="No kernels provided. Use `/kernel-bot <build|build-and-upload|merge-and-upload> <kernel1> [kernel2 ...]`.",
        )

    kernels = []
    seen = set()
    for kernel in args:
        if not KERNEL_RE.match(kernel):
            return ParsedCommand(error=f"Invalid kernel name `{kernel}`.")
        if kernel not in seen:
            kernels.append(kernel)
            seen.add(kernel)

    if branch is not None and not BRANCH_RE.match(branch):
        return ParsedCommand(error=f"Invalid target branch `{branch}`.")

    return ParsedCommand(command=command, kernels=kernels, branch=branch)


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


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")

    if not token or not repository:
        print("Missing required environment variables.", file=sys.stderr)
        return 1

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
        print("Missing required comment context environment variables.", file=sys.stderr)
        return 1

    if sender_type == "Bot":
        print("Ignoring bot comment.")
        return 0

    if len(comment) > MAX_COMMENT_LENGTH:
        print("Ignoring oversized comment payload.", file=sys.stderr)
        return 0
    if not comment.strip().startswith("/kernel-bot"):
        print("Ignoring non /kernel-bot comment.")
        return 0
    if not comment_has_only_supported_characters(comment.strip()):
        print("Ignoring /kernel-bot comment with unsupported characters.")
        return 0

    api_base = f"https://api.github.com/repos/{repository}"
    if comment_id is not None:
        try_post_issue_comment_reaction(api_base, token, comment_id, "+1")

    parsed_command = parse_command(comment)
    if parsed_command.error:
        try_post_issue_comment(
            api_base,
            token,
            issue_number,
            parsed_command.error,
        )
        return 0

    command = parsed_command.command
    kernels = parsed_command.kernels
    requested_branch = parsed_command.branch
    if command is None:
        print("Internal error: command parsing returned no command.", file=sys.stderr)
        return 1

    permission = get_user_permission(api_base, token, commenter)
    allowed_permissions = COMMAND_PERMISSIONS[command]
    if permission not in allowed_permissions:
        if command == "build":
            permission_error = (
                "I can only run `/kernel-bot build` for users with `write` or `admin` "
                "repository permission."
            )
        else:
            permission_error = (
                f"I can only run `/kernel-bot {command}` for users with `admin` "
                "repository permission."
            )
        try_post_issue_comment(
            api_base,
            token,
            issue_number,
            permission_error,
        )
        return 0

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

    if command in FORK_BLOCKED_COMMANDS:
        head_repo = pull_request.get("head", {}).get("repo", {}) or {}
        head_full_name = head_repo.get("full_name")
        if head_full_name != repository:
            try_post_issue_comment(
                api_base,
                token,
                issue_number,
                f"Fork PRs are blocked for `/kernel-bot {command}`. Use a branch in this repository.",
            )
            return 0

    dispatch_url = f"{api_base}/actions/workflows/{DISPATCH_WORKFLOW}/dispatches"
    if command == "build":
        target_branch = requested_branch or f"pr-{issue_number}"
        dispatch_pr_number = str(issue_number)
        upload_flag = "false"
        allow_main_dispatch = "false"
    elif command == "build-and-upload":
        target_branch = requested_branch or f"pr-{issue_number}"
        dispatch_pr_number = str(issue_number)
        upload_flag = "true"
        allow_main_dispatch = "false"
    else:
        target_branch = requested_branch or "main"
        dispatch_pr_number = ""
        upload_flag = "true"
        allow_main_dispatch = "true"

    mode_text = {
        "build": "build only",
        "build-and-upload": "build and upload",
        "merge-and-upload": "merge, build and upload",
    }[command]
    command_summary = f"/kernel-bot {command} {' '.join(kernels)}"
    if requested_branch is not None:
        command_summary += f" --branch {requested_branch}"
    pr_head_sha = pull_request.get("head", {}).get("sha")
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
            ),
        )
    )

    merge_result_message = None
    if command == "merge-and-upload":
        if pull_request.get("merged"):
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

    for kernel_name in kernels:
        dispatch_key = make_dispatch_key(issue_number, kernel_name)
        dispatch_body = {
            "ref": default_branch,
            "inputs": {
                "kernel_name": kernel_name,
                "pr_number": dispatch_pr_number,
                "target_branch": target_branch,
                "upload": upload_flag,
                "allow_main_dispatch": allow_main_dispatch,
                "dispatch_key": dispatch_key,
            },
        }
        try:
            print(
                f"Dispatching workflow for command `{command}`, kernel `{kernel_name}`, branch `{target_branch}`"
            )
            github_api_request(dispatch_url, token, method="POST", data=dispatch_body)
            dispatches.append(
                DispatchResult(kernel_name=kernel_name, dispatch_key=dispatch_key)
            )
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            print(err_text, file=sys.stderr)
            failed.append((kernel_name, e.code))

    resolve_dispatch_run_urls(
        api_base,
        token,
        repository,
        default_branch,
        dispatches,
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
    raise SystemExit(main())
