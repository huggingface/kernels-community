#!/usr/bin/env python3
# Input:
#   [--base <comment-body>] <kernel> <repo-prefix> <upload-json-or-path>...
#
# Output:
#   A Hub upload section, or <comment-body> with that section updated.
#
# Example:
#   python3 .github/scripts/format_hub_upload_comment.py msa kernels-community "$RUNNER_TEMP"/upload*.json
#   python3 .github/scripts/format_hub_upload_comment.py --base "$BODY" msa kernels-community "$RUNNER_TEMP"/upload*.json
#   python3 .github/scripts/format_hub_upload_comment.py msa kernels-community '{"pull_requests":[{"url":"https://hf.co/kernels/MiniMaxAI/msa/discussions/1"}]}'
import argparse
import json
from pathlib import Path

from hub_pr_upload_args import repo_id

SECTION = "Hub uploads:"


def load_json(value):
    if not value.lstrip().startswith(("{", "[")):
        path = Path(value)
        if path.is_file():
            with open(path) as f:
                return json.load(f)
    return json.loads(value)


def dedupe(lines):
    return list(dict.fromkeys(lines))


def upload_lines(kernel, repo_prefix, uploads):
    urls = [
        pr["url"]
        for upload in uploads
        for pr in load_json(upload).get("pull_requests", [])
    ]
    lines = [
        f"- Hub repo: https://huggingface.co/kernels/{repo_id(kernel, repo_prefix)}"
    ]
    lines.extend(f"- Hub pull request: {url}" for url in urls)
    return dedupe(lines)


def update_comment(base, lines):
    body_lines = base.rstrip().splitlines()
    if SECTION in body_lines:
        section = body_lines.index(SECTION)
        prefix = body_lines[:section]
        existing = [line for line in body_lines[section + 1 :] if line.startswith("- ")]
    else:
        prefix = body_lines
        existing = []
    body = "\n".join(prefix).rstrip()
    section = "\n".join([SECTION, *dedupe(existing + lines)])
    return f"{body}\n\n{section}".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("kernel")
    parser.add_argument("repo_prefix")
    parser.add_argument("uploads", nargs="+")
    args = parser.parse_args()

    lines = upload_lines(args.kernel, args.repo_prefix, args.uploads)
    if args.base is not None:
        print(update_comment(args.base, lines))
    else:
        print("\n".join([SECTION, *lines]))


if __name__ == "__main__":
    raise SystemExit(main())
