#!/usr/bin/env python3
from dataclasses import dataclass, field
import json
import os
import re
import sys
import urllib.error
import urllib.request


KERNEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
COMMAND_PERMISSIONS = {
    "build": {"admin", "write"},
    "build-and-upload": {"admin"},
    "merge-and-upload": {"admin"},
}
FORK_BLOCKED_COMMANDS = {"build", "build-and-upload"}
MAX_COMMENT_LENGTH = 1024
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


def post_issue_comment(api_base: str, token: str, issue_number: int, message: str):
    url = f"{api_base}/issues/{issue_number}/comments"
    github_api_request(url, token, method="POST", data={"body": message})


def try_post_issue_comment(api_base: str, token: str, issue_number: int, message: str):
    try:
        post_issue_comment(api_base, token, issue_number, message)
        return True
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        print(f"Failed to post PR comment (HTTP {e.code}).", file=sys.stderr)
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


def parse_issue_number(raw_value: str | None):
    if raw_value is None:
        return None
    raw = raw_value.strip()
    if not raw.isdigit():
        return None
    return int(raw)


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repository = os.environ.get("GITHUB_REPOSITORY")

    if not token or not repository:
        print("Missing required environment variables.", file=sys.stderr)
        return 1

    comment = os.environ.get("COMMENT_BODY")
    issue_number = parse_issue_number(os.environ.get("COMMENT_ISSUE_NUMBER"))
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

    api_base = f"https://api.github.com/repos/{repository}"

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

    merge_result_message = None
    if command == "merge-and-upload":
        if pull_request.get("merged"):
            merge_result_message = "PR is already merged. Continuing with build/upload."
        elif pull_request.get("state") != "open":
            try_post_issue_comment(
                api_base,
                token,
                issue_number,
                "PR is not open and cannot be merged via `/kernel-bot merge-and-upload`.",
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
                try_post_issue_comment(
                    api_base,
                    token,
                    issue_number,
                    "Failed to merge PR before build/upload. Check mergeability and required checks.",
                )
                return 1

            if not merge_response.get("merged"):
                merge_message = merge_response.get(
                    "message", "PR merge failed for an unknown reason."
                )
                try_post_issue_comment(
                    api_base,
                    token,
                    issue_number,
                    f"PR merge failed: {merge_message}",
                )
                return 1

            merge_result_message = merge_response.get(
                "message", "PR merged successfully."
            )

    dispatch_url = f"{api_base}/actions/workflows/manual-build-upload.yaml/dispatches"
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

    command_summary = f"/kernel-bot {command} {' '.join(kernels)}"
    if requested_branch is not None:
        command_summary += f" --branch {requested_branch}"
    succeeded = []
    failed = []

    for kernel_name in kernels:
        dispatch_body = {
            "ref": default_branch,
            "inputs": {
                "kernel_name": kernel_name,
                "pr_number": dispatch_pr_number,
                "target_branch": target_branch,
                "upload": upload_flag,
                "allow_main_dispatch": allow_main_dispatch,
            },
        }
        try:
            print(
                f"Dispatching workflow for command `{command}`, kernel `{kernel_name}`, branch `{target_branch}`"
            )
            github_api_request(dispatch_url, token, method="POST", data=dispatch_body)
            succeeded.append(kernel_name)
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            print(err_text, file=sys.stderr)
            failed.append((kernel_name, e.code))

    mode_text = {
        "build": "build only",
        "build-and-upload": "build and upload",
        "merge-and-upload": "merge, build and upload",
    }[command]

    lines = [
        "Build request processed.",
        "",
        f"Command: `{command_summary}`",
        f"Mode: `{mode_text}`",
        f"Target branch: `{target_branch}`",
        "Triggered workflow: `manual-build-upload.yaml`",
    ]
    if merge_result_message:
        lines.extend(["", f"Merge result: {merge_result_message}"])
    if succeeded:
        lines.extend(["", f"Dispatched ({len(succeeded)}): `{', '.join(succeeded)}`"])
    if failed:
        failed_text = ", ".join(f"{kernel} (HTTP {code})" for kernel, code in failed)
        lines.extend(["", f"Failed ({len(failed)}): `{failed_text}`"])

    comment_posted = try_post_issue_comment(api_base, token, issue_number, "\n".join(lines))
    if not comment_posted:
        print(
            "Bot response could not be posted. Ensure workflow token has issues/pull-requests write permission.",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
