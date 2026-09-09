#!/usr/bin/env python3

import json
import os
import urllib.request
from pathlib import Path, PurePosixPath


API_ROOT = "https://api.github.com"
API_TIMEOUT = 30
BOT_LOGIN = "github-actions[bot]"
COMMENT_MARKER = "<!-- kernel-bot-build-guidance -->"
DOCUMENTATION_STEMS = {"authors", "card", "license", "readme", "upstream"}


def request_json(url: str, token: str, method: str = "GET", data: dict | None = None):
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8") if data is not None else None,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "kernels-community-build-guidance",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=API_TIMEOUT) as response:
        return json.load(response)


def paginate(url: str, token: str) -> list[dict]:
    items = []
    page = 1
    while True:
        separator = "&" if "?" in url else "?"
        batch = request_json(f"{url}{separator}per_page=100&page={page}", token)
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def kernel_directories(root: Path) -> set[str]:
    return {manifest.parent.name for manifest in root.glob("*/build.toml")}


def is_documentation(path: PurePosixPath) -> bool:
    return "docs" in path.parts[1:-1] or path.stem.lower() in DOCUMENTATION_STEMS


def touched_source_kernels(files: list[dict], existing_kernels: set[str]) -> list[str]:
    paths = [PurePosixPath(item["filename"]) for item in files]
    new_kernels = {
        path.parts[0]
        for path in paths
        if len(path.parts) == 2 and path.name == "build.toml"
    }
    kernels = existing_kernels | new_kernels
    touched = set(new_kernels)

    for path in paths:
        if len(path.parts) < 2 or path.parts[0] not in kernels:
            continue
        if is_documentation(path):
            continue
        touched.add(path.parts[0])

    return sorted(touched)


def format_comment(kernels: list[str]) -> str:
    kernel_names = " ".join(kernels)
    changed = ", ".join(f"`{kernel}`" for kernel in kernels)
    return (
        f"{COMMENT_MARKER}\n"
        f"This PR changes source code for the following kernel(s): {changed}.\n\n"
        "If you have the kernel-bot build permissions, then trigger "
        f"`/kernel-bot build {kernel_names}` yourself. Otherwise, a maintainer "
        "will review the PR and will take an appropriate action. Thanks for your patience."
    )


def upsert_comment(repo: str, number: int, token: str, body: str) -> None:
    comments_url = f"{API_ROOT}/repos/{repo}/issues/{number}/comments"
    existing = next(
        (
            comment
            for comment in paginate(comments_url, token)
            if COMMENT_MARKER in (comment.get("body") or "")
            and (comment.get("user") or {}).get("login") == BOT_LOGIN
        ),
        None,
    )
    if existing is None:
        request_json(comments_url, token, method="POST", data={"body": body})
        print(f"Posted build guidance on PR #{number}")
    elif existing.get("body") != body:
        request_json(existing["url"], token, method="PATCH", data={"body": body})
        print(f"Updated build guidance on PR #{number}")
    else:
        print(f"Build guidance already exists on PR #{number}")


def run(number: int, root: Path = Path(".")) -> None:
    repo = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    files = paginate(f"{API_ROOT}/repos/{repo}/pulls/{number}/files", token)
    kernels = touched_source_kernels(files, kernel_directories(root))
    if not kernels:
        print(f"PR #{number} does not add a kernel or change kernel source; skipping")
        return

    upsert_comment(repo, number, token, format_comment(kernels))


def main() -> None:
    run(int(os.environ["PR_NUMBER"]))


if __name__ == "__main__":
    main()
