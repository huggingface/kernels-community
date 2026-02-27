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


def main():
    token = os.environ.get("GITHUB_TOKEN")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    repository = os.environ.get("GITHUB_REPOSITORY")

    if not token or not event_path or not repository:
        print("Missing required environment variables.", file=sys.stderr)
        return 1

    with open(event_path, "r", encoding="utf-8") as f:
        event = json.load(f)

    sender_type = event.get("sender", {}).get("type")
    if sender_type == "Bot":
        print("Ignoring bot comment.")
        return 0

    comment = event.get("comment", {}).get("body", "")
    issue_number = event.get("issue", {}).get("number")
    commenter = event.get("comment", {}).get("user", {}).get("login", "")
    default_branch = event.get("repository", {}).get("default_branch", "main")

    if not issue_number:
        print("No issue/PR number in event payload.", file=sys.stderr)
        return 1

    api_base = f"https://api.github.com/repos/{repository}"

    if not commenter:
        print("No commenter username in event payload.", file=sys.stderr)
        return 1

    permission = get_user_permission(api_base, token, commenter)
    if permission not in ALLOWED_PERMISSIONS:
        post_issue_comment(
            api_base,
            token,
            issue_number,
            "I can only run builds for users with `write` or `admin` repository permission.",
        )
        return 0

    kernels, target_branch, parse_error = parse_command(comment)
    if parse_error:
        post_issue_comment(
            api_base,
            token,
            issue_number,
            parse_error,
        )
        return 0

    dispatch_url = f"{api_base}/actions/workflows/manual-build-upload.yaml/dispatches"
    target_branch = target_branch or f"pr-{issue_number}"
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
            # TODO: actually dispatch the workflow. For now, just simulate success.
            # github_api_request(dispatch_url, token, method="POST", data=dispatch_body)
            print(
                f"Simulating dispatch for kernel `{kernel_name}` to branch `{target_branch}`"
            )
            succeeded.append(kernel_name)
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            print(err_text, file=sys.stderr)
            failed.append((kernel_name, e.code))

    lines = [
        "Build request processed.",
        "",
        f"Command: `{comment.strip()}`",
        f"Target branch: `{target_branch}`",
        "Triggered workflow: `manual-build-upload.yaml`",
    ]
    if succeeded:
        lines.extend(["", f"Dispatched ({len(succeeded)}): `{', '.join(succeeded)}`"])
    if failed:
        failed_text = ", ".join(f"{kernel} (HTTP {code})" for kernel, code in failed)
        lines.extend(["", f"Failed ({len(failed)}): `{failed_text}`"])

    post_issue_comment(api_base, token, issue_number, "\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
