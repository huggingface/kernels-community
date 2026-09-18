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
    for command in ("release", "merge-and-upload", "build-and-stage", "security"):
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
