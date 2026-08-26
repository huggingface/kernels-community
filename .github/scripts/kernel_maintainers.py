#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_ROOT = "https://api.github.com"
API_TIMEOUT = int(os.environ.get("GITHUB_API_TIMEOUT", "30"))
SLACK_TIMEOUT = 10

DEFAULT_REGISTRY = ".github/kernel-maintainers.json"
WEBHOOK_ENV = "SLACK_WEBHOOK_URL_MAINTAINERS"
NOTIFIED_LABEL = "notified:maintainers"
MAX_TITLE_CHARS = 200

KERNEL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
SLACK_USER_ID_RE = re.compile(r"^U[A-Z0-9]{6,}$")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# --------------------------------------------------------------------------- #
# Kernel discovery
# --------------------------------------------------------------------------- #
def is_kernel(root: Path, name: str) -> bool:
    """A top-level directory holding a build.toml is a kernel."""
    if not KERNEL_NAME_RE.match(name):
        return False
    return (root / name / "build.toml").is_file()


def discover_kernels(root: Path) -> list[str]:
    try:
        entries = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    except OSError as err:
        raise RuntimeError(f"unable to list directories under {root}: {err}") from err
    return [name for name in entries if is_kernel(root, name)]


def kernels_for_paths(root: Path, paths: list[str]) -> list[str]:
    """Kernels touched by repo-relative paths: the first segment, if it is a kernel
    *in this checkout*. See the module docstring on why "in this checkout" matters.
    """
    found = set()
    for path in paths:
        head = path.split("/", 1)[0]
        if is_kernel(root, head):
            found.add(head)
    return sorted(found)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def load_registry(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        "kernels": data.get("kernels") or {},
        "unowned": data.get("unowned") or [],
    }


def maintainers_of(registry: dict, kernel: str) -> list[str]:
    """The maintainers of a kernel; empty when it has none."""
    return list(registry["kernels"].get(kernel) or [])


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
MISSING_OWNER_HELP = """kernel {name!r} has no maintainers.
    add it to {registry} under either:
      "kernels": {{ "{name}": ["U01ABCDEF"] }}   (someone maintains it)
      "unowned": [ "{name}" ]                  (deliberately bare)"""


def check(
    root: Path, registry: dict, registry_path: str
) -> tuple[list[str], list[str]]:
    """Validate the registry against the kernels on disk.

    Problems fail CI; warnings only print -- a leftover entry for a removed
    kernel is untidy but never mis-pings anyone.
    """
    problems: list[str] = []
    warnings: list[str] = []

    kernels = discover_kernels(root)
    unowned = registry["unowned"]
    unowned_set = set(unowned)

    if len(unowned) != len(unowned_set):
        duplicates = sorted({name for name in unowned if unowned.count(name) > 1})
        problems.append(f'duplicate entries in "unowned": {", ".join(duplicates)}')

    # Every kernel on disk, including one this PR adds, is in exactly one list.
    for name in kernels:
        owned = bool(maintainers_of(registry, name))
        if owned and name in unowned_set:
            problems.append(
                f'kernel {name!r} is listed in both "kernels" and "unowned"; pick one'
            )
        elif not owned and name not in unowned_set:
            problems.append(
                MISSING_OWNER_HELP.format(name=name, registry=registry_path)
            )

    for name, entry in sorted(registry["kernels"].items()):
        if not entry:
            problems.append(
                f"kernel {name!r} has an empty maintainer list; name a maintainer or "
                f'move it to "unowned"'
            )
        elif not isinstance(entry, list):
            problems.append(f"kernel {name!r} must map to a list of maintainers")
        elif len(set(entry)) != len(entry):
            problems.append(f"kernel {name!r} lists the same maintainer twice")

    known = set(kernels)
    for name in sorted(set(registry["kernels"]) | unowned_set):
        if name not in known:
            warnings.append(
                f"registry lists {name!r}, which is not a kernel directory (renamed or removed?)"
            )

    return problems, warnings


def cmd_check(args: argparse.Namespace) -> int:
    root = Path(args.root)
    registry = load_registry(args.registry)
    problems, warnings = check(root, registry, args.registry)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if problems:
        print(f"\n{len(problems)} problem(s) in {args.registry}:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    kernels = discover_kernels(root)
    owned = sum(1 for name in kernels if maintainers_of(registry, name))
    print(
        f"OK: {len(kernels)} kernels, {owned} maintained, {len(kernels) - owned} unowned."
    )
    return 0


# --------------------------------------------------------------------------- #
# GitHub API helpers (stdlib only)
# --------------------------------------------------------------------------- #
def get_token() -> str | None:
    """Resolve GitHub token: env var first, then ``gh auth token`` fallback."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def github_request(url: str, token: str, method: str = "GET", body: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("User-Agent", "kernels-community-maintainer-ping")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
        payload = response.read()
        return json.loads(payload) if payload else None


def github_paginate(url: str, token: str) -> list[dict]:
    items: list[dict] = []
    page = 1
    while True:
        paged = f"{url}?{urllib.parse.urlencode({'per_page': 100, 'page': page})}"
        batch = github_request(paged, token)
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


# --------------------------------------------------------------------------- #
# Slack
# --------------------------------------------------------------------------- #
def slack_escape(text: str) -> str:
    """Make untrusted text safe to interpolate into a Slack message.

    Escaping the mrkdwn metacharacters is what stops a PR title forging a
    mention: ``<!channel>`` in a title renders as literal text.
    """
    text = CONTROL_CHARS_RE.sub("", text or "")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_maintainer(entry: str) -> str:
    """A member ID pings the person; anything else prints as plain text, so a
    placeholder such as a GitHub handle stays readable.
    """
    if SLACK_USER_ID_RE.match(entry):
        return f"<@{entry}>"
    return f"`{slack_escape(entry)}`"


def truncate(text: str, limit: int = MAX_TITLE_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def post_to_slack(webhook_url: str, message: str) -> None:
    data = json.dumps({"text": message}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=SLACK_TIMEOUT) as response:
        response.read()


def format_message(registry: dict, kernels: list[str], pr: dict) -> str:
    """One message for the whole PR, a line per maintained kernel it touches."""
    title = truncate(slack_escape(pr.get("title") or ""))
    author = slack_escape((pr.get("user") or {}).get("login") or "unknown")

    lines = [
        f":eyes: PR #{pr['number']} touches maintained kernel(s):",
        f"*{title}*  —  by `{author}`",
    ]
    for kernel in kernels:
        who = " ".join(
            render_maintainer(entry) for entry in maintainers_of(registry, kernel)
        )
        lines.append(f"• `{slack_escape(kernel)}` — {who}")
    lines.append(pr.get("html_url", ""))
    return "\n".join(line for line in lines if line)


# --------------------------------------------------------------------------- #
# notify
# --------------------------------------------------------------------------- #
def cmd_notify(args: argparse.Namespace) -> int:
    root = Path(args.root)
    registry = load_registry(args.registry)

    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        print("GITHUB_REPOSITORY is not set", file=sys.stderr)
        return 1
    token = get_token()
    if not token:
        print("no GitHub token (set GITHUB_TOKEN)", file=sys.stderr)
        return 1

    pr = github_request(f"{API_ROOT}/repos/{repo}/pulls/{args.pr}", token)
    if pr.get("draft"):
        print(f"PR #{args.pr} is a draft; skipping (will fire on ready_for_review).")
        return 0
    if NOTIFIED_LABEL in {label["name"] for label in pr.get("labels") or []}:
        print(f"{NOTIFIED_LABEL} already on PR #{args.pr}; skipping.")
        return 0

    files = github_paginate(f"{API_ROOT}/repos/{repo}/pulls/{args.pr}/files", token)
    touched = kernels_for_paths(root, [f["filename"] for f in files])
    maintained = [name for name in touched if maintainers_of(registry, name)]
    if not maintained:
        print(f"PR #{args.pr} touches no maintained kernel; nothing to do.")
        return 0

    message = format_message(registry, maintained, pr)
    if args.dry_run:
        print(f"--- dry run (would post to ${WEBHOOK_ENV})")
        print(message)
        return 0

    webhook = os.environ.get(WEBHOOK_ENV)
    if not webhook:
        print(f"error: ${WEBHOOK_ENV} is not set", file=sys.stderr)
        return 1
    try:
        post_to_slack(webhook, message)
    except (urllib.error.URLError, OSError) as err:
        print(f"error: failed to post to Slack: {err}", file=sys.stderr)
        return 1
    print(f"notified maintainers of {', '.join(maintained)}")

    # Label last: a failed post must be retryable.
    try:
        github_request(
            f"{API_ROOT}/repos/{repo}/issues/{args.pr}/labels",
            token,
            method="POST",
            body={"labels": [NOTIFIED_LABEL]},
        )
    except (urllib.error.URLError, OSError) as err:
        print(f"warning: could not add {NOTIFIED_LABEL}: {err}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kernel maintainer registry and Slack pings."
    )
    parser.add_argument(
        "--registry", default=DEFAULT_REGISTRY, help="registry JSON path"
    )
    parser.add_argument("--root", default=".", help="repository root")
    sub = parser.add_subparsers(dest="command", required=True)

    check_parser = sub.add_parser(
        "check", help="every kernel is maintained or explicitly unowned"
    )
    check_parser.set_defaults(func=cmd_check)

    notify_parser = sub.add_parser(
        "notify", help="ping the maintainers of the kernels a PR touches"
    )
    notify_parser.add_argument("--pr", required=True, type=int, help="PR number")
    notify_parser.add_argument(
        "--dry-run", action="store_true", help="print the message instead of posting it"
    )
    notify_parser.set_defaults(func=cmd_notify)

    args = parser.parse_args()
    try:
        return args.func(args)
    except (FileNotFoundError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
