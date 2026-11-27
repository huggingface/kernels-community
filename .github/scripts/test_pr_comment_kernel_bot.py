import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import pr_comment_kernel_bot as bot  # noqa: E402


def write_build_toml(
    root: Path,
    kernel: str,
    *,
    repo_id: str,
    version: int = 1,
    branch: str | None = None,
):
    kernel_dir = root / kernel
    kernel_dir.mkdir()
    branch_line = f'branch = "{branch}"\n' if branch is not None else ""
    (kernel_dir / "build.toml").write_text(
        f"""[general]
name = "{kernel}"
version = {version}

[general.hub]
repo-id = "{repo_id}"
{branch_line}"""
    )


def test_external_upload_target_uses_version_branch(tmp_path, monkeypatch):
    write_build_toml(tmp_path, "example", repo_id="vendor/example", version=3)
    monkeypatch.chdir(tmp_path)

    target = bot.external_upload_target("example")

    assert target == bot.ExternalUploadTarget("example", "vendor/example", "v3")


def test_external_upload_target_prefers_requested_branch(tmp_path, monkeypatch):
    write_build_toml(
        tmp_path,
        "example",
        repo_id="vendor/example",
        version=3,
        branch="stable",
    )
    monkeypatch.chdir(tmp_path)

    target = bot.external_upload_target("example", requested_branch="candidate")

    assert target == bot.ExternalUploadTarget("example", "vendor/example", "candidate")


def test_external_upload_target_uses_configured_branch(tmp_path, monkeypatch):
    write_build_toml(
        tmp_path,
        "example",
        repo_id="vendor/example",
        version=3,
        branch="stable",
    )
    monkeypatch.chdir(tmp_path)

    target = bot.external_upload_target("example")

    assert target == bot.ExternalUploadTarget("example", "vendor/example", "stable")


def test_community_upload_does_not_need_branch_preflight(tmp_path, monkeypatch):
    write_build_toml(
        tmp_path, "example", repo_id="kernels-community/example", version=3
    )
    monkeypatch.chdir(tmp_path)

    def unexpected_lookup(repo_id):
        raise AssertionError(f"unexpected Hub lookup for {repo_id}")

    assert (
        bot.preflight_external_uploads(["example"], branch_lookup=unexpected_lookup)
        == []
    )


def test_kernels_sdk_upload_is_not_external(tmp_path, monkeypatch):
    write_build_toml(tmp_path, "example", repo_id="kernels-sdk/example", version=3)
    monkeypatch.chdir(tmp_path)

    assert bot.external_upload_target("example") is None


def test_external_upload_preflight_accepts_existing_branch(tmp_path, monkeypatch):
    write_build_toml(tmp_path, "example", repo_id="vendor/example", version=3)
    monkeypatch.chdir(tmp_path)

    assert (
        bot.preflight_external_uploads(
            ["example"], branch_lookup=lambda repo_id: {"main", "v3"}
        )
        == []
    )


def test_external_upload_preflight_rejects_missing_branch(tmp_path, monkeypatch):
    write_build_toml(tmp_path, "example", repo_id="vendor/example", version=3)
    monkeypatch.chdir(tmp_path)

    failures = bot.preflight_external_uploads(
        ["example"], branch_lookup=lambda repo_id: {"main", "v2"}
    )

    assert len(failures) == 1
    assert "branch `v3` does not exist" in failures[0]
    assert "`vendor/example`" in failures[0]


@pytest.mark.parametrize(
    ("command", "expected_ref"),
    [("release", "main"), ("merge-and-upload", "abc123")],
)
def test_upload_preflight_failure_stops_before_merge_and_dispatch(
    monkeypatch, command, expected_ref
):
    monkeypatch.setattr(
        bot,
        "_resolve_context_from_env",
        lambda: {
            "token": "token",
            "repository": "owner/repo",
            "comment": f"/kernel-bot {command} example",
            "comment_id": 7,
            "issue_number": 42,
            "default_branch": "main",
            "commenter": "maintainer",
        },
    )
    monkeypatch.setattr(bot, "try_post_issue_comment_reaction", lambda *a, **k: True)
    monkeypatch.setattr(bot, "get_user_permission", lambda *a, **k: "admin")
    monkeypatch.setattr(
        bot,
        "get_pull_request",
        lambda *a, **k: {
            "user": {"login": "maintainer"},
            "head": {"sha": "abc123"},
            "state": "open",
            "merged": False,
        },
    )

    checked = {}

    def fail_preflight(kernels, **kwargs):
        checked["kernels"] = kernels
        checked.update(kwargs)
        return ["missing external branch"]

    monkeypatch.setattr(bot, "preflight_external_uploads", fail_preflight)

    def unexpected(*args, **kwargs):
        raise AssertionError("merge or dispatch must not run after preflight failure")

    monkeypatch.setattr(bot, "merge_pull_request", unexpected)
    monkeypatch.setattr(bot, "dispatch", unexpected)
    comments = []
    monkeypatch.setattr(
        bot,
        "try_post_issue_comment",
        lambda api_base, token, issue_number, message: comments.append(message),
    )

    assert bot.main() == 1
    assert checked == {
        "kernels": ["example"],
        "requested_branch": None,
        "ref": expected_ref,
    }
    assert len(comments) == 1
    assert "no merge or build was started" in comments[0]


def test_build_with_kernels():
    parsed = bot.parse_command("/kernel-bot build activation relu")
    assert parsed.error is None
    assert parsed.command == "build"
    assert parsed.kernels == ["activation", "relu"]
    assert parsed.branch is None


def test_security_needs_no_kernels():
    parsed = bot.parse_command("/kernel-bot security")
    assert parsed.error is None
    assert parsed.command == "security"
    assert parsed.kernels == []


def test_build_without_kernels_is_error():
    parsed = bot.parse_command("/kernel-bot build")
    assert parsed.error is not None
    assert parsed.command is None


def test_unknown_command_is_error():
    assert bot.parse_command("/kernel-bot frobnicate foo").error is not None


def test_wrong_prefix_is_error():
    assert bot.parse_command("/not-kernel-bot build foo").error is not None


def test_too_few_tokens_is_error():
    assert bot.parse_command("/kernel-bot").error is not None


def test_branch_at_end_is_parsed():
    parsed = bot.parse_command("/kernel-bot build foo --branch dev")
    assert parsed.error is None
    assert parsed.kernels == ["foo"]
    assert parsed.branch == "dev"


def test_branch_not_at_end_is_error():
    parsed = bot.parse_command("/kernel-bot build --branch dev foo")
    assert parsed.error is not None


def test_security_with_branch_only():
    parsed = bot.parse_command("/kernel-bot security --branch dev")
    assert parsed.error is None
    assert parsed.command == "security"
    assert parsed.kernels == []
    assert parsed.branch == "dev"


def test_duplicate_kernels_are_deduped():
    parsed = bot.parse_command("/kernel-bot build foo foo bar")
    assert parsed.kernels == ["foo", "bar"]


def test_invalid_kernel_name_is_error():
    assert bot.parse_command("/kernel-bot build bad/name").error is not None


def test_build_with_backend_scope():
    parsed = bot.parse_command("/kernel-bot build flash-attn2[xpu,cpu]")
    assert parsed.error is None
    assert parsed.kernels == ["flash-attn2"]
    assert parsed.backends == {"flash-attn2": ["xpu", "cpu"]}


def test_backend_scope_mixes_with_plain_kernels():
    parsed = bot.parse_command("/kernel-bot build activation flash-attn2[xpu]")
    assert parsed.error is None
    assert parsed.kernels == ["activation", "flash-attn2"]
    assert parsed.backends == {"flash-attn2": ["xpu"]}


def test_backend_scope_unknown_backend_is_error():
    parsed = bot.parse_command("/kernel-bot build flash-attn2[gpu]")
    assert parsed.error is not None
    assert "gpu" in parsed.error


def test_backend_scopes_union_across_repeats():
    parsed = bot.parse_command("/kernel-bot build relu[cpu] relu[xpu]")
    assert parsed.kernels == ["relu"]
    assert parsed.backends == {"relu": ["cpu", "xpu"]}


def test_supported_characters_allows_backend_scope():
    assert bot.comment_has_only_supported_characters(
        "/kernel-bot build flash-attn2[xpu,cpu]"
    )


def test_invalid_branch_name_is_error():
    parsed = bot.parse_command("/kernel-bot build foo --branch bad~branch")
    assert parsed.error is not None


def test_supported_characters():
    assert bot.comment_has_only_supported_characters("/kernel-bot build activation")
    assert not bot.comment_has_only_supported_characters("/kernel-bot build `rm -rf`")


def test_make_dispatch_key_format():
    key = bot.make_dispatch_key(42, "activation")
    assert key.startswith("pr42-activation-")


def test_pending_comment_lists_security_workflows():
    text = bot.format_pending_comment(
        "/kernel-bot security",
        "security audit only",
        "pr-42",
        "abc123",
        include_security=True,
    )
    assert "Security audit" in text
    for wf in bot.WORKFLOWS["security"]:
        assert wf in text


def test_result_comment_renders_security_dispatch_and_failure():
    dispatched = bot.DispatchResult(
        kernel_name="security (security-audit.yml)",
        dispatch_key="pr42-security-x",
        action_url="https://example/run/1",
    )
    text = bot.format_result_comment(
        "/kernel-bot security",
        "security audit only",
        "pr-42",
        "abc123",
        security_dispatches=[dispatched],
        security_failed=[("security-audit.yml", 500)],
    )
    assert "https://example/run/1" in text
    assert "Security audit failed" in text
    assert "security-audit.yml (HTTP 500)" in text


def _patch_membership(monkeypatch, states):
    """Route get_team_membership_state through a dict of {username: state}.

    An int value raises an HTTPError with that status instead.
    """

    def fake(org, team_slug, username, token):
        state = states.get(username)
        if isinstance(state, int):
            raise bot.urllib.error.HTTPError("url", state, "boom", None, None)
        return state

    monkeypatch.setattr(bot, "get_team_membership_state", fake)


def _patch_pr_author(monkeypatch, login):
    monkeypatch.setattr(
        bot, "get_pull_request", lambda *a, **k: {"user": {"login": login}}
    )


def test_active_team_member_is_authorized(monkeypatch):
    _patch_membership(monkeypatch, {"alice": "active"})
    assert bot.is_authorized_team_member("alice", "tok")


def test_pending_and_unknown_members_are_not_authorized(monkeypatch):
    _patch_membership(monkeypatch, {"bob": "pending"})
    assert not bot.is_authorized_team_member("bob", "tok")
    assert not bot.is_authorized_team_member("carol", "tok")


def test_membership_lookup_error_denies(monkeypatch):
    _patch_membership(monkeypatch, {"dave": 500})
    assert not bot.is_authorized_team_member("dave", "tok")


def test_membership_without_a_token_denies(monkeypatch):
    _patch_membership(monkeypatch, {"alice": "active"})
    assert not bot.is_authorized_team_member("alice", "")


def test_team_member_is_authorized_for_build(monkeypatch):
    monkeypatch.setenv("TEAM_READ_TOKEN", "tok")
    _patch_membership(monkeypatch, {"alice": "active"})
    assert bot.team_grants_access("api", "t", "build", "alice", 42)


def test_team_membership_does_not_unlock_other_commands(monkeypatch):
    monkeypatch.setenv("TEAM_READ_TOKEN", "tok")
    _patch_membership(monkeypatch, {"alice": "active"})
    for command in ("release", "merge-and-upload"):
        assert not bot.team_grants_access("api", "t", command, "alice", 42)


def test_missing_team_read_token_denies(monkeypatch):
    monkeypatch.delenv("TEAM_READ_TOKEN", raising=False)
    _patch_membership(monkeypatch, {"alice": "active"})
    assert not bot.team_grants_access("api", "t", "build", "alice", 42)


def test_authorized_via_pr_author_membership(monkeypatch):
    monkeypatch.setenv("TEAM_READ_TOKEN", "tok")
    _patch_membership(monkeypatch, {"alice": "active"})
    _patch_pr_author(monkeypatch, "alice")
    # Commenter is not on the team; the PR author is.
    assert bot.team_grants_access("api", "t", "build", "outsider", 42)


HEAD = "a" * 40
OLD = "b" * 40


def _review(login, commit_id, state="APPROVED"):
    return {"user": {"login": login}, "commit_id": commit_id, "state": state}


def _external_pr(author="outsider", head=HEAD):
    return {"user": {"login": author}, "head": {"sha": head}}


def _patch_review_gate(monkeypatch, *, reviews, permissions=None):
    """Untrusted author by default; `permissions` maps login -> repo permission."""
    permissions = permissions or {}
    monkeypatch.delenv("TEAM_READ_TOKEN", raising=False)
    monkeypatch.setattr(
        bot,
        "get_user_permission",
        lambda api_base, token, username: permissions.get(username),
    )

    def fake_reviews(*args):
        if isinstance(reviews, int):
            raise bot.urllib.error.HTTPError("url", reviews, "boom", None, None)
        return reviews

    monkeypatch.setattr(bot, "list_pull_request_reviews", fake_reviews)


def test_review_on_head_by_commenter_counts():
    reviews = [_review("maintainer", OLD), _review("maintainer", HEAD, "COMMENTED")]
    assert bot.has_review_on_commit(reviews, "maintainer", HEAD)


@pytest.mark.parametrize(
    "review",
    [
        _review("maintainer", OLD),
        _review("someone-else", HEAD),
        _review("maintainer", HEAD, "DISMISSED"),
        _review("maintainer", HEAD, "PENDING"),
    ],
)
def test_review_not_on_head_by_commenter_does_not_count(review):
    assert not bot.has_review_on_commit([review], "maintainer", HEAD)


def test_gate_blocks_build_when_pushed_after_review(monkeypatch):
    _patch_review_gate(monkeypatch, reviews=[_review("maintainer", OLD)])
    error = bot.review_gate_error("api", "t", "build", "maintainer", _external_pr(), 42)
    assert error is not None
    assert HEAD[:12] in error


def test_gate_allows_build_reviewed_on_head(monkeypatch):
    _patch_review_gate(monkeypatch, reviews=[_review("maintainer", HEAD)])
    assert bot.review_gate_error("api", "t", "build", "maintainer", _external_pr(), 42) is None


@pytest.mark.parametrize("permission", ["write", "admin"])
def test_gate_skips_trusted_author(monkeypatch, permission):
    _patch_review_gate(monkeypatch, reviews=[], permissions={"insider": permission})
    pr = _external_pr(author="insider")
    assert bot.review_gate_error("api", "t", "build", "maintainer", pr, 42) is None


def test_gate_skips_team_member_author(monkeypatch):
    _patch_review_gate(monkeypatch, reviews=[])
    monkeypatch.setenv("TEAM_READ_TOKEN", "tok")
    _patch_membership(monkeypatch, {"teammate": "active"})
    pr = _external_pr(author="teammate")
    assert bot.review_gate_error("api", "t", "build", "maintainer", pr, 42) is None


def test_gate_requires_review_when_author_lookup_fails(monkeypatch):
    _patch_review_gate(monkeypatch, reviews=[])

    def boom(*args):
        raise bot.urllib.error.HTTPError("url", 500, "boom", None, None)

    monkeypatch.setattr(bot, "get_user_permission", boom)
    assert bot.review_gate_error("api", "t", "build", "maintainer", _external_pr(), 42)


def test_gate_ignores_security_only(monkeypatch):
    _patch_review_gate(monkeypatch, reviews=[])
    assert bot.review_gate_error("api", "t", "security", "maintainer", _external_pr(), 42) is None


@pytest.mark.parametrize(
    "command", ["build", "build-and-stage", "security-and-build", "merge-and-upload"]
)
def test_gate_covers_commands_that_run_pr_code(monkeypatch, command):
    _patch_review_gate(monkeypatch, reviews=[])
    assert bot.review_gate_error("api", "t", command, "maintainer", _external_pr(), 42)


def test_gate_denies_when_reviews_cannot_be_listed(monkeypatch):
    _patch_review_gate(monkeypatch, reviews=500)
    error = bot.review_gate_error("api", "t", "build", "maintainer", _external_pr(), 42)
    assert error is not None
    assert "Could not verify PR reviews" in error


def test_unreviewed_head_stops_before_dispatch(monkeypatch):
    monkeypatch.setattr(
        bot,
        "_resolve_context_from_env",
        lambda: {
            "token": "token",
            "repository": "owner/repo",
            "comment": "/kernel-bot build example",
            "comment_id": 7,
            "issue_number": 42,
            "default_branch": "main",
            "commenter": "maintainer",
        },
    )
    monkeypatch.setattr(bot, "try_post_issue_comment_reaction", lambda *a, **k: True)
    _patch_review_gate(
        monkeypatch,
        reviews=[_review("maintainer", OLD)],
        permissions={"maintainer": "write"},
    )
    monkeypatch.setattr(bot, "get_pull_request", lambda *a, **k: _external_pr())

    def unexpected(*args, **kwargs):
        raise AssertionError("dispatch must not run on an unreviewed head")

    monkeypatch.setattr(bot, "dispatch", unexpected)
    comments = []
    monkeypatch.setattr(
        bot,
        "try_post_issue_comment",
        lambda api_base, token, issue_number, message: comments.append(message),
    )

    assert bot.main() == 0
    assert len(comments) == 1
    assert "has no review on the current head commit" in comments[0]


def test_merge_is_pinned_to_checked_head(monkeypatch):
    sent = {}

    def fake_request(url, token, method="GET", data=None):
        sent.update(url=url, method=method, data=data)
        return 200, "{}"

    monkeypatch.setattr(bot, "github_api_request", fake_request)
    bot.merge_pull_request("api", "t", 42, HEAD)
    assert sent["method"] == "PUT"
    assert sent["data"] == {"merge_method": "squash", "sha": HEAD}


def test_reviews_are_paginated(monkeypatch):
    pages = {1: [_review("u", OLD)] * bot.REVIEW_PAGE_SIZE, 2: [_review("u", HEAD)]}

    def fake_request(url, token, method="GET", data=None):
        page = int(url.rsplit("page=", 1)[1])
        return 200, bot.json.dumps(pages.get(page, []))

    monkeypatch.setattr(bot, "github_api_request", fake_request)
    reviews = bot.list_pull_request_reviews("api", "t", 42)
    assert len(reviews) == bot.REVIEW_PAGE_SIZE + 1
    assert bot.has_review_on_commit(reviews, "u", HEAD)


def _run_dry(comment, *extra):
    return subprocess.run(
        [
            sys.executable,
            os.path.join(SCRIPT_DIR, "pr_comment_kernel_bot.py"),
            "--dry-run",
            comment,
            "--pr-number",
            "42",
            "--repo",
            "owner/repo",
            "--head-sha",
            "abc123",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_security_only_dry_run():
    proc = _run_dry("/kernel-bot security")
    assert proc.returncode == 0, proc.stderr
    assert "security-audit.yml" in proc.stdout


def test_parse_error_dry_run_does_not_crash():
    proc = _run_dry("/kernel-bot frobnicate")
    assert proc.returncode == 0, proc.stderr
    assert "Parse error" in proc.stderr


def _assert_dry_run_mode(comment, expected):
    proc = _run_dry(comment)
    assert proc.returncode == 0, proc.stderr
    assert f'"mode": "{expected}"' in proc.stdout
    other = {"pr", "stage", "release"} - {expected}
    assert all(f'"mode": "{m}"' not in proc.stdout for m in other)


def test_build_dispatches_pr_mode():
    _assert_dry_run_mode("/kernel-bot build relu", "pr")


def test_security_and_build_dispatches_pr_mode():
    _assert_dry_run_mode("/kernel-bot security-and-build relu", "pr")


def test_build_and_stage_dispatches_stage_mode():
    _assert_dry_run_mode("/kernel-bot build-and-stage relu", "stage")


def test_release_dispatches_release_mode():
    _assert_dry_run_mode("/kernel-bot release relu", "release")


def test_merge_and_upload_dispatches_release_mode():
    _assert_dry_run_mode("/kernel-bot merge-and-upload relu", "release")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
