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
# Only the community org defers to build.toml's repo-id. Every other prefix
# (kernels-staging) uploads into its own org whatever build.toml declares: a
# staging build must never open a pull request against the vendor repo it
# mirrors, and the CI token has no write access there anyway.
#
# Example:
#   python3 .github/scripts/hub_pr_upload_args.py upload-args msa kernels-community
import sys
import tomllib
from pathlib import Path

COMMUNITY = "kernels-community"


# Ported kernels keep their generated tree in <kernel>/src, with flake.nix
# staying at <kernel>. Everything else keeps build.toml beside the flake.
def build_toml_path(kernel):
    nested = Path(kernel) / "src" / "build.toml"
    return nested if nested.is_file() else Path(kernel) / "build.toml"


def external_repo_id(kernel, repo_prefix):
    if repo_prefix != COMMUNITY:
        return ""
    build_toml = build_toml_path(kernel)
    if not build_toml.is_file():
        return ""
    with open(build_toml, "rb") as f:
        repo_id = tomllib.load(f).get("general", {}).get("hub", {}).get("repo-id")
    return repo_id if isinstance(repo_id, str) and repo_id and not repo_id.startswith(f"{COMMUNITY}/") else ""


def repo_id(kernel, repo_prefix):
    return external_repo_id(kernel, repo_prefix) or f"{repo_prefix}/{kernel}"


def upload_args(kernel, repo_prefix):
    # --create-pr carries no --repo-id: the uploader falls back to build.toml's
    # repo-id, so it is only ever correct when that repo-id is the target.
    return "--create-pr" if external_repo_id(kernel, repo_prefix) else f"--repo-id {repo_prefix}/{kernel}"


if __name__ == "__main__":
    mode, kernel, repo_prefix = sys.argv[1:]
    if mode == "repo-id":
        print(repo_id(kernel, repo_prefix))
    elif mode == "upload-args":
        print(upload_args(kernel, repo_prefix))
    else:
        sys.exit(f"unknown mode: {mode}")
