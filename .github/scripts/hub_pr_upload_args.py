#!/usr/bin/env python3
# Input:
#   <repo-id|upload-args> <kernel> <repo-prefix>
#
# Output:
#   The effective Hub repo-id, or upload args using --create-pr for external repos.
#
# The kernel is resolved against the working directory, not this file's location:
# workflows run the helpers from a copy of the default branch while the kernel
# itself comes from the PR checkout. A kernel without a build.toml has no
# external repo-id.
#
# Example:
#   python3 .github/scripts/hub_pr_upload_args.py upload-args msa kernels-community
import sys
import tomllib
from pathlib import Path

COMMUNITY = "kernels-community"


def external_repo_id(kernel):
    build_toml = Path(kernel) / "build.toml"
    if not build_toml.is_file():
        return ""
    with open(build_toml, "rb") as f:
        repo_id = tomllib.load(f).get("general", {}).get("hub", {}).get("repo-id")
    return repo_id if isinstance(repo_id, str) and repo_id and not repo_id.startswith(f"{COMMUNITY}/") else ""


def repo_id(kernel, repo_prefix):
    return external_repo_id(kernel) or f"{repo_prefix}/{kernel}"


if __name__ == "__main__":
    mode, kernel, repo_prefix = sys.argv[1:]
    if mode == "repo-id":
        print(repo_id(kernel, repo_prefix))
    elif mode == "upload-args":
        print("--create-pr" if external_repo_id(kernel) else f"--repo-id {repo_prefix}/{kernel}")
    else:
        sys.exit(f"unknown mode: {mode}")
