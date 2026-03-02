#!/usr/bin/env python3
import json
import os
import re
import sys
import urllib.error
import urllib.request


KERNEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
ALLOWED_PERMISSIONS = {"admin", "write"}
MAX_COMMENT_LENGTH = 1024


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


def parse_command(comment: str):
    tokens = comment.strip().split()
    if len(tokens) < 3:
        return (
            None,
            None,
            "Invalid command. Use `/kernel-bot build <kernel1> [kernel2 ...] [--branch <target_branch>]`.",
        )

    if tokens[0] != "/kernel-bot" or tokens[1] != "build":
        return (
            None,
            None,
            "Invalid command. Use `/kernel-bot build <kernel1> [kernel2 ...] [--branch <target_branch>]`.",
        )

    branch = None
    args = tokens[2:]

    if "--branch" in args:
        branch_idx = args.index("--branch")
        if branch_idx != len(args) - 2:
            return (
                None,
                None,
                "Invalid `--branch` usage. Put it at the end: `--branch <target_branch>`.",
            )
        branch = args[branch_idx + 1]
        args = args[:branch_idx]

    if not args:
        return (
            None,
            None,
            "No kernels provided. Use `/kernel-bot build <kernel1> [kernel2 ...]`.",
        )

    kernels = []
    seen = set()
    for kernel in args:
        if not KERNEL_RE.match(kernel):
            return None, None, f"Invalid kernel name `{kernel}`."
        if kernel not in seen:
            kernels.append(kernel)
            seen.add(kernel)

    if branch is not None and not BRANCH_RE.match(branch):
        return None, None, f"Invalid target branch `{branch}`."

    return kernels, branch, None


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

    permission = get_user_permission(api_base, token, commenter)
    if permission not in ALLOWED_PERMISSIONS:
        try_post_issue_comment(
            api_base,
            token,
            issue_number,
            "I can only run builds for users with `write` or `admin` repository permission.",
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
            "Could not verify PR source repository, so `/kernel-bot build` was not run.",
        )
        return 1

    head_repo = pull_request.get("head", {}).get("repo", {}) or {}
    head_full_name = head_repo.get("full_name")
    if head_full_name != repository:
        try_post_issue_comment(
            api_base,
            token,
            issue_number,
            "Fork PRs are blocked for `/kernel-bot build`. Use a branch in this repository.",
        )
        return 0

    kernels, target_branch, parse_error = parse_command(comment)
    if parse_error:
        try_post_issue_comment(
            api_base,
            token,
            issue_number,
            parse_error,
        )
        return 0

    dispatch_url = f"{api_base}/actions/workflows/manual-build-upload.yaml/dispatches"
    target_branch = target_branch or f"pr-{issue_number}"
    command_summary = f"/kernel-bot build {' '.join(kernels)}"
    if target_branch and target_branch != f"pr-{issue_number}":
        command_summary += f" --branch {target_branch}"
    succeeded = []
    failed = []

    for kernel_name in kernels:
        dispatch_body = {
            "ref": default_branch,
            "inputs": {
                "kernel_name": kernel_name,
                "pr_number": str(issue_number),
                "target_branch": target_branch,
            },
        }
        try:
            print(
                f"Dispatching workflow for kernel `{kernel_name}` to branch `{target_branch}`"
            )
            github_api_request(dispatch_url, token, method="POST", data=dispatch_body)
            succeeded.append(kernel_name)
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            print(err_text, file=sys.stderr)
            failed.append((kernel_name, e.code))

    lines = [
        "Build request processed.",
        "",
        f"Command: `{command_summary}`",
        f"Target branch: `{target_branch}`",
        "Triggered workflow: `manual-build-upload.yaml`",
    ]
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
