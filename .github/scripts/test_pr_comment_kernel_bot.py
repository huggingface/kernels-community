import os
import subprocess
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import pr_comment_kernel_bot as bot  # noqa: E402


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
