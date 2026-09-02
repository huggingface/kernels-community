#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API_ROOT = "https://api.github.com"
REGISTRY = Path(".github/kernel-maintainers.json")
WEBHOOK_ENV = "SLACK_WEBHOOK_URL_MAINTAINERS"
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def load_registry() -> dict:
    with REGISTRY.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_registry() -> None:
    registry = load_registry()
    kernels = {path.parent.name for path in Path(".").glob("*/build.toml")}
    registered = set(registry)

    problems = [
        f"{kernel}: maintainers must be a list"
        for kernel, maintainers in registry.items()
        if not isinstance(maintainers, list)
    ]
    problems.extend(f"{kernel}: missing from registry" for kernel in kernels - registered)
    problems.extend(
        f"{kernel}: registry entry has no kernel directory"
        for kernel in registered - kernels
    )

    if problems:
        raise ValueError("\n".join(sorted(problems)))
    owned = sum(bool(maintainers) for maintainers in registry.values())
    print(f"All {len(kernels)} kernels are registered ({owned} with maintainers)")


def request(url: str, token: str):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "kernels-community-maintainer-ping",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def pr_files(repo: str, number: int, token: str) -> list[dict]:
    files = []
    page = 1
    while True:
        query = urllib.parse.urlencode({"per_page": 100, "page": page})
        batch = request(
            f"{API_ROOT}/repos/{repo}/pulls/{number}/files?{query}", token
        )
        files.extend(batch)
        if len(batch) < 100:
            return files
        page += 1


def slack_escape(text: str) -> str:
    text = CONTROL_CHARS_RE.sub("", text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_message(pr: dict, maintainers: dict[str, list[str]]) -> str:
    lines = [
        f":eyes: PR #{pr['number']} touches maintained kernel(s):",
        f"*{slack_escape(pr['title'])}* — by `{slack_escape(pr['user']['login'])}`",
    ]
    for kernel, members in maintainers.items():
        mentions = " ".join(f"<@{member}>" for member in members)
        lines.append(f"• `{kernel}` — {mentions}")
    lines.append(pr["html_url"])
    return "\n".join(lines)


def notify(number: int, dry_run: bool) -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    maintainers_by_kernel = load_registry()

    pr = request(f"{API_ROOT}/repos/{repo}/pulls/{number}", token)
    if pr["draft"]:
        print(f"PR #{number} is a draft; skipping")
        return

    touched = {
        item["filename"].split("/", 1)[0] for item in pr_files(repo, number, token)
    }
    maintainers = {
        kernel: maintainers_by_kernel[kernel]
        for kernel in sorted(touched)
        if maintainers_by_kernel.get(kernel)
    }
    if not maintainers:
        print(f"PR #{number} touches no maintained kernel; skipping")
        return

    message = format_message(pr, maintainers)
    if dry_run:
        print(message)
        return

    payload = json.dumps({"text": message}).encode()
    req = urllib.request.Request(
        os.environ[WEBHOOK_ENV],
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    notify_parser = commands.add_parser("notify")
    notify_parser.add_argument("--pr", required=True, type=int)
    notify_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        if args.command == "check":
            check_registry()
        else:
            notify(args.pr, args.dry_run)
    except (KeyError, OSError, ValueError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
