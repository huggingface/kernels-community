"""Upload a built kernel to the Hub and report the outcome back to a PR.

This wraps the `kernel-builder upload` invocation used by the build workflows so
that kernels living outside the `kernels-community` org are uploaded through a
pull request (the CI bot has no write access to third-party orgs) instead of a
direct commit.

Routing is decided from the kernel's `build.toml` `repo-id`:

* org == `kernels-community`  -> direct upload to `<repo_prefix>/<kernel>`
  (unchanged behaviour; `repo_prefix` lets `build-and-stage` target
  `kernels-staging`).
* org != `kernels-community`  -> upload to the real `repo-id` from `build.toml`
  with `--create-pr`, which opens a pull request on the Hub. `repo_prefix` is
  ignored: an external kernel is always published to its own repository.

When `--create-pr` is used and a `--comment-pr-number` is given, the Hub pull
request links (printed by the uploader as `Pull request created: <url>`) are
posted back as a comment on the originating GitHub PR. If the upload fails, a
failure comment is posted instead. Direct (`kernels-community`) uploads are left
untouched and never comment.

The process exit code mirrors the uploader's, so a failed upload still fails the
build step.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

COMMUNITY_ORG = "kernels-community"
PR_URL_RE = re.compile(r"^Pull request created:\s*(\S+)\s*$", re.MULTILINE)


def read_repo_id(kernel_dir: Path) -> str | None:
    """Return the `[general.hub] repo-id` declared in the kernel's build.toml."""
    build_toml = kernel_dir / "build.toml"
    if not build_toml.exists():
        return None
    with open(build_toml, "rb") as f:
        config = tomllib.load(f)
    repo_id = config.get("general", {}).get("hub", {}).get("repo-id")
    return repo_id if isinstance(repo_id, str) and repo_id else None


def repo_org(repo_id: str) -> str:
    return repo_id.split("/", 1)[0]


def resolve_upload_target(
    repo_id: str | None, kernel: str, repo_prefix: str
) -> tuple[str, bool]:
    """Return (upload_repo_id, create_pr) for the upload.

    External kernels (repo-id org != kernels-community) upload to their own
    repo-id via a pull request; everything else keeps the direct
    `<repo_prefix>/<kernel>` behaviour.
    """
    if repo_id and repo_org(repo_id) != COMMUNITY_ORG:
        return repo_id, True
    return f"{repo_prefix}/{kernel}", False


def parse_pr_urls(stdout: str) -> list[str]:
    return PR_URL_RE.findall(stdout)


def github_api_post(url: str, token: str, data: dict) -> None:
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        resp.read()


def post_pr_comment(
    repo: str, token: str, pr_number: str, message: str
) -> bool:
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    try:
        github_api_post(url, token, {"body": message})
        return True
    except urllib.error.HTTPError as e:
        err_text = e.read().decode("utf-8", errors="replace")
        print(f"Failed to post PR comment (HTTP {e.code}).", file=sys.stderr)
        print(err_text, file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"Failed to post PR comment: {e}", file=sys.stderr)
        return False


def format_success_comment(
    kernel: str, repo_id: str, label: str, pr_urls: list[str]
) -> str:
    header = "### Kernel upload → pull request opened"
    context = (
        f"Kernel `{kernel}` ({label}) targets `{repo_id}`, which is outside "
        f"`{COMMUNITY_ORG}`, so the build was uploaded via pull request:"
    )
    if pr_urls:
        links = "\n".join(f"- {url}" for url in pr_urls)
        return f"{header}\n\n{context}\n\n{links}"
    # Upload succeeded but the Hub reported nothing to change, so no PR opened.
    return (
        "### Kernel upload\n\n"
        f"Kernel `{kernel}` ({label}) → `{repo_id}`: upload completed, but no "
        "pull request was opened (no changes to upload)."
    )


def format_failure_comment(
    kernel: str, repo_id: str, label: str, run_url: str | None, returncode: int
) -> str:
    details = (
        f" See the [workflow run]({run_url}) for details." if run_url else ""
    )
    return (
        "### Kernel upload failed\n\n"
        f"Uploading kernel `{kernel}` ({label}) to `{repo_id}` via pull request "
        f"failed (exit code {returncode}).{details}"
    )


def run_upload(
    builder_cmd: list[str],
    *,
    upload_repo_id: str,
    upload_path: str | None,
    branch: str | None,
    create_pr: bool,
    cwd: Path,
) -> tuple[int, str]:
    """Run the uploader, streaming stderr live and capturing stdout.

    The uploader prints progress/diagnostics to stderr (streamed to the CI log)
    and the `Pull request created: <url>` lines to stdout (captured here so the
    PR links can be reported back).
    """
    cmd = list(builder_cmd)
    if upload_path:
        cmd.append(upload_path)
    cmd += ["--repo-type", "kernel", "--repo-id", upload_repo_id]
    if branch:
        cmd += ["--branch", branch]
    if create_pr:
        cmd.append("--create-pr")

    print(f"+ {' '.join(cmd)}", file=sys.stderr, flush=True)
    # stderr inherits the console (live progress); stdout is captured for URLs.
    proc = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, text=True)
    if proc.stdout:
        # Echo captured stdout so it still shows up in the workflow logs.
        sys.stdout.write(proc.stdout)
        sys.stdout.flush()
    return proc.returncode, proc.stdout or ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a kernel and report PR links back to a GitHub PR."
    )
    parser.add_argument("--kernel", required=True, help="Kernel directory name")
    parser.add_argument(
        "--repo-prefix",
        default=COMMUNITY_ORG,
        help="Hub org prefix for direct (kernels-community) uploads",
    )
    parser.add_argument(
        "--branch", default="", help="Target branch for the upload (optional)"
    )
    parser.add_argument(
        "--upload-path",
        default="",
        help="Positional path passed to the uploader (e.g. build/ on Windows)",
    )
    parser.add_argument(
        "--comment-pr-number",
        default="",
        help="GitHub PR number to report create-pr links to (empty = no comment)",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub repo (owner/repo) for commenting; defaults to $GITHUB_REPOSITORY",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Human-readable build label for the comment (e.g. 'Build / cuda / x86_64-linux')",
    )
    parser.add_argument(
        "--run-url", default="", help="Workflow run URL (used in failure comments)"
    )
    parser.add_argument(
        "--builder",
        required=True,
        help="Uploader command up to and including `upload` "
        '(e.g. "nix run -L github:huggingface/kernels#kernel-builder -- upload")',
    )
    args = parser.parse_args()

    builder_cmd = shlex.split(args.builder)
    if not builder_cmd:
        print("Error: empty --builder command.", file=sys.stderr)
        return 2

    kernel_dir = Path(args.kernel)
    repo_id = read_repo_id(kernel_dir)
    upload_repo_id, create_pr = resolve_upload_target(
        repo_id, args.kernel, args.repo_prefix
    )

    label = args.label or "Build"
    if create_pr:
        print(
            f"Kernel `{args.kernel}` repo-id `{repo_id}` is outside "
            f"`{COMMUNITY_ORG}`; uploading via pull request.",
            file=sys.stderr,
        )
    else:
        print(
            f"Kernel `{args.kernel}` uploads directly to `{upload_repo_id}`.",
            file=sys.stderr,
        )

    returncode, stdout = run_upload(
        builder_cmd,
        upload_repo_id=upload_repo_id,
        upload_path=args.upload_path or None,
        branch=args.branch or None,
        create_pr=create_pr,
        cwd=kernel_dir,
    )

    # Only create-pr uploads report back; direct uploads keep prior behaviour.
    if not create_pr:
        return returncode

    token = os.environ.get("GITHUB_TOKEN", "")
    if not args.comment_pr_number or not args.repo or not token:
        if not token:
            print(
                "No GITHUB_TOKEN available; skipping PR comment.", file=sys.stderr
            )
        return returncode

    if returncode == 0:
        pr_urls = parse_pr_urls(stdout)
        message = format_success_comment(
            args.kernel, upload_repo_id, label, pr_urls
        )
    else:
        message = format_failure_comment(
            args.kernel, upload_repo_id, label, args.run_url or None, returncode
        )

    post_pr_comment(args.repo, token, args.comment_pr_number, message)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
