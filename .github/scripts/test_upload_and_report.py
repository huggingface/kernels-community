import os
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))

import upload_and_report as uar  # noqa: E402


def test_external_kernel_uses_repo_id_and_create_pr():
    # Repo-id outside kernels-community -> upload to that repo-id via PR,
    # ignoring the repo_prefix (external kernels publish to their own repo).
    assert uar.resolve_upload_target("MiniMaxAI/msa", "msa", "kernels-community") == (
        "MiniMaxAI/msa",
        True,
    )
    assert uar.resolve_upload_target("MiniMaxAI/msa", "msa", "kernels-staging") == (
        "MiniMaxAI/msa",
        True,
    )
    assert uar.resolve_upload_target(
        "sgl-project/sgl-flash-attn3", "sgl-flash-attn3", "kernels-community"
    ) == ("sgl-project/sgl-flash-attn3", True)


def test_community_kernel_uses_prefix_no_create_pr():
    assert uar.resolve_upload_target(
        "kernels-community/relu", "relu", "kernels-community"
    ) == ("kernels-community/relu", False)
    # repo_prefix override (build-and-stage) still routes directly, no PR.
    assert uar.resolve_upload_target(
        "kernels-community/relu", "relu", "kernels-staging"
    ) == ("kernels-staging/relu", False)


def test_missing_repo_id_falls_back_to_prefix():
    assert uar.resolve_upload_target(None, "foo", "kernels-community") == (
        "kernels-community/foo",
        False,
    )


def test_parse_pr_urls():
    out = (
        "progress...\n"
        "Pull request created: https://hf.co/kernels/MiniMaxAI/msa/discussions/3\n"
        "Pull request created: https://hf.co/kernels/MiniMaxAI/msa/discussions/4\n"
    )
    assert uar.parse_pr_urls(out) == [
        "https://hf.co/kernels/MiniMaxAI/msa/discussions/3",
        "https://hf.co/kernels/MiniMaxAI/msa/discussions/4",
    ]
    assert uar.parse_pr_urls("No changes to upload.\n") == []


def test_success_comment_lists_links():
    msg = uar.format_success_comment(
        "msa", "MiniMaxAI/msa", "Build / cuda / x86_64-linux", ["url1", "url2"]
    )
    assert "pull request opened" in msg
    assert "MiniMaxAI/msa" in msg
    assert "- url1" in msg and "- url2" in msg


def test_success_comment_no_changes():
    msg = uar.format_success_comment("msa", "MiniMaxAI/msa", "Build", [])
    assert "no changes to upload" in msg


def test_failure_comment_includes_run_url_and_code():
    msg = uar.format_failure_comment(
        "msa", "MiniMaxAI/msa", "Build", "http://run/1", 2
    )
    assert "failed" in msg
    assert "http://run/1" in msg
    assert "exit code 2" in msg


def _fake_kernel(tmp_path: Path, repo_id: str) -> Path:
    kdir = tmp_path / "kern"
    kdir.mkdir()
    (kdir / "build.toml").write_text(
        f'[general.hub]\nrepo-id = "{repo_id}"\n'
    )
    return kdir


def test_read_repo_id(tmp_path):
    kdir = _fake_kernel(tmp_path, "MiniMaxAI/msa")
    assert uar.read_repo_id(kdir) == "MiniMaxAI/msa"
    assert uar.read_repo_id(tmp_path / "missing") is None


def _run_main(argv, posts):
    def fake_post(url, token, data):
        posts.append((url, data))

    with mock.patch.object(uar, "github_api_post", fake_post), mock.patch.dict(
        os.environ,
        {"GITHUB_TOKEN": "tok", "GITHUB_REPOSITORY": "huggingface/kernels-community"},
    ):
        old = sys.argv
        sys.argv = ["upload_and_report.py"] + argv
        try:
            return uar.main()
        finally:
            sys.argv = old


def _builder_script(tmp_path: Path, name: str, body: str) -> str:
    script = tmp_path / name
    script.write_text("#!/bin/bash\n" + body)
    script.chmod(0o755)
    return str(script)


def test_main_external_success_comments(tmp_path):
    kdir = _fake_kernel(tmp_path, "MiniMaxAI/msa")
    builder = _builder_script(
        tmp_path,
        "ok.sh",
        "echo 'Pull request created: https://hf.co/kernels/MiniMaxAI/msa/discussions/9'\n",
    )
    posts = []
    rc = _run_main(
        [
            "--kernel", str(kdir),
            "--repo-prefix", "kernels-community",
            "--comment-pr-number", "42",
            "--label", "Build / cuda",
            "--run-url", "http://run",
            "--builder", f"{builder} up",
        ],
        posts,
    )
    assert rc == 0
    assert len(posts) == 1
    url, data = posts[0]
    assert url.endswith("/issues/42/comments")
    assert "discussions/9" in data["body"]


def test_main_external_failure_comments_and_propagates(tmp_path):
    kdir = _fake_kernel(tmp_path, "MiniMaxAI/msa")
    builder = _builder_script(tmp_path, "fail.sh", "echo boom >&2\nexit 3\n")
    posts = []
    rc = _run_main(
        [
            "--kernel", str(kdir),
            "--comment-pr-number", "42",
            "--run-url", "http://run/9",
            "--builder", f"{builder} up",
        ],
        posts,
    )
    assert rc == 3
    assert len(posts) == 1
    assert "failed" in posts[0][1]["body"]


def test_main_community_does_not_comment(tmp_path):
    kdir = _fake_kernel(tmp_path, "kernels-community/relu")
    builder = _builder_script(tmp_path, "ok.sh", "echo hi\n")
    posts = []
    rc = _run_main(
        [
            "--kernel", str(kdir),
            "--repo-prefix", "kernels-community",
            "--comment-pr-number", "42",
            "--builder", f"{builder} up",
        ],
        posts,
    )
    assert rc == 0
    assert posts == []


def test_main_external_without_comment_pr_number_does_not_comment(tmp_path):
    kdir = _fake_kernel(tmp_path, "MiniMaxAI/msa")
    builder = _builder_script(tmp_path, "ok.sh", "echo hi\n")
    posts = []
    rc = _run_main(
        ["--kernel", str(kdir), "--builder", f"{builder} up"],
        posts,
    )
    assert rc == 0
    assert posts == []
